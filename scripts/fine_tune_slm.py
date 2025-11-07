#!/usr/bin/env python3
"""
Fine-tune the Aura planner Small Language Model (SLM).

The script expects JSONL input where each line contains:
{
    "instruction": "... natural language intent ...",
    "actions": [... list of atomic actions ...],
    "view_hierarchy_path": "... optional path ...",
    "screenshot_path": "... optional path ..."
}

Training strategy:
    * Load an INT8-capable base model (e.g., "meta-llama/Llama-3.2-1B-Instruct").
    * Apply QLoRA adapters for low-memory fine-tuning.
    * Optimise with standard causal LM objective using HuggingFace Trainer.

Example:
    python fine_tune_slm.py \
        --base-model meta-llama/Llama-3.2-1B-Instruct \
        --train-file ../data/processed/slm_training/episodes.jsonl \
        --output-dir ../models/slm/aura-planner
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

try:
    from peft import LoraConfig, get_peft_model
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise SystemExit("peft is required. Install with `pip install peft`.") from exc

logger = logging.getLogger("aura.fine_tune_slm")
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRAIN_FILE = REPO_ROOT / "data/processed/slm_training/episodes.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune the Aura planner SLM.")
    parser.add_argument(
        "--base-model",
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Base HF model id or local path (default: meta-llama/Llama-3.2-1B-Instruct).",
    )
    parser.add_argument(
        "--train-file",
        type=Path,
        default=DEFAULT_TRAIN_FILE,
        help=f"Path to JSONL training file (default: {DEFAULT_TRAIN_FILE}).",
    )
    parser.add_argument("--eval-file", type=Path, help="Optional evaluation JSONL file.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=2, help="Per-device batch size.")
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def format_example(example: Dict) -> str:
    action_text = []
    for idx, action in enumerate(example.get("actions", []), start=1):
        action_text.append(f"{idx}. {json.dumps(action, ensure_ascii=True)}")
    action_block = "\n".join(action_text) if action_text else "1. NO_ACTIONS"

    prompt = (
        "You are Aura, an on-device mobile assistant.\n"
        "Instruction:\n"
        f"{example['instruction']}\n\n"
        "Plan:\n"
        f"{action_block}\n"
        "### End of plan ###"
    )
    return prompt


def load_dataset(path: Path) -> Dataset:
    records: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            records.append({"text": format_example(payload)})
    if not records:
        raise ValueError(f"No records found in {path}")
    return Dataset.from_list(records)


def tokenize_dataset(dataset: Dataset, tokenizer: AutoTokenizer, max_length: int) -> Dataset:
    def _tokenize(batch: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
        return tokenizer(
            batch["text"],
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )

    return dataset.map(_tokenize, batched=True, remove_columns=["text"])


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="[%(levelname)s] %(message)s")

    logger.info("Loading tokenizer and base model: %s", args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if args.fp16 else torch.bfloat16 if args.bf16 else torch.float32,
        device_map="auto",
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()  # noqa: T201 - informative print

    logger.info("Loading training data from %s", args.train_file)
    train_dataset = tokenize_dataset(load_dataset(args.train_file), tokenizer, args.max_length)

    eval_dataset = None
    if args.eval_file and args.eval_file.exists():
        eval_dataset = tokenize_dataset(load_dataset(args.eval_file), tokenizer, args.max_length)

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        evaluation_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=200,
        learning_rate=args.lr,
        logging_steps=50,
        save_steps=500,
        save_total_limit=3,
        fp16=args.fp16,
        bf16=args.bf16,
        report_to="none",
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    logger.info("Starting training...")
    trainer.train()
    logger.info("Saving adapter weights to %s", args.output_dir)
    trainer.save_model()

    tokenizer.save_pretrained(args.output_dir)
    logger.info("Fine-tuning complete.")


if __name__ == "__main__":
    main()

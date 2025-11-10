#!/usr/bin/env python3
"""
Fine-tune the Aura perception Vision-Language Model (VLM).

Input format (JSONL):
{
    "question": "...",
    "answer": {"text": "...", "id": "...", "bounds": [x0, y0, x1, y1]},
    "xml_path": ".../episode_000001.json",
    "screenshot_path": ".../episode_000001.png"   # optional
}

The script adapts a compact BLIP-style model using LoRA adapters.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
from datasets import Dataset
from PIL import Image
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    Trainer,
    TrainingArguments,
)

try:
    from peft import LoraConfig, get_peft_model
except ImportError as exc:  # pragma: no cover
    raise SystemExit("peft is required. Install with `pip install peft`.") from exc

logger = logging.getLogger("aura.fine_tune_vlm")
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRAIN_FILE = REPO_ROOT / "data/processed/vlm_training/examples.jsonl"
DEFAULT_VLM_MODEL = "defog/smol-vlm-2b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Aura perception VLM.")
    parser.add_argument(
        "--base-model",
        default=DEFAULT_VLM_MODEL,
        help=(
            "Vision-language model id/path (default: defog/smol-vlm-2b). "
            "Other suggested option: google/gemini-nano-2 for AI Edge SDK devices."
        ),
    )
    parser.add_argument(
        "--train-file",
        type=Path,
        default=DEFAULT_TRAIN_FILE,
        help=f"VLM training JSONL (default: {DEFAULT_TRAIN_FILE}).",
    )
    parser.add_argument("--eval-file", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--max-context-chars", type=int, default=2048)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def load_examples(path: Path, max_context_chars: int) -> List[Dict[str, Optional[str]]]:
    examples = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            xml_text = ""
            xml_path = record.get("xml_path")
            if xml_path and Path(xml_path).exists():
                with Path(xml_path).open("r", encoding="utf-8") as xml_file:
                    xml_text = xml_file.read(max_context_chars)

            examples.append(
                {
                    "question": record["question"],
                    "context": xml_text,
                    "answer": record["answer"]["text"],
                    "image_path": record.get("screenshot_path"),
                }
            )
    if not examples:
        raise ValueError(f"No records found in {path}")
    return examples


def prepare_dataset(records: Iterable[Dict[str, Optional[str]]]) -> Dataset:
    return Dataset.from_list(list(records))


def load_image(image_path: Optional[str]) -> Image.Image:
    if image_path and Path(image_path).exists():
        return Image.open(image_path).convert("RGB")
    # Fallback blank canvas for datasets without screenshots
    return Image.new("RGB", (512, 512), color=(255, 255, 255))


def tokenize_dataset(
    dataset: Dataset,
    processor: AutoProcessor,
) -> Dataset:
    def _process(batch: Dict[str, List[Optional[str]]]) -> Dict[str, List[torch.Tensor]]:
        prompts = []
        images: List[Image.Image] = []
        for question, context, image_path in zip(batch["question"], batch["context"], batch["image_path"]):
            context = context or ""
            prompts.append(f"Question: {question}\nContext:\n{context}")
            images.append(load_image(image_path))

        answer_texts = batch["answer"]
        inputs = processor(images=images, text=prompts, return_tensors="pt", padding=True)
        with processor.as_target_processor():
            labels = processor(text=answer_texts, return_tensors="pt", padding=True).input_ids

        # Replace padding token id with -100 so that they are ignored by loss
        labels[labels == processor.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": inputs["pixel_values"].detach().cpu().numpy(),
            "input_ids": inputs["input_ids"].detach().cpu().numpy(),
            "attention_mask": inputs["attention_mask"].detach().cpu().numpy(),
            "labels": labels.detach().cpu().numpy(),
        }

    return dataset.map(_process, batched=True, remove_columns=dataset.column_names)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="[%(levelname)s] %(message)s")

    logger.info("Loading processor and base model: %s", args.base_model)
    processor = AutoProcessor.from_pretrained(args.base_model)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    base_model = AutoModelForVision2Seq.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if args.fp16 else torch.bfloat16 if args.bf16 else torch.float32,
        device_map="auto",
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="SEQ_2_SEQ_LM",
    )
    model = get_peft_model(base_model, lora_config)

    logger.info("Loading training examples from %s", args.train_file)
    train_dataset = prepare_dataset(load_examples(args.train_file, args.max_context_chars))
    train_dataset = tokenize_dataset(train_dataset, processor)

    eval_dataset = None
    if args.eval_file and args.eval_file.exists():
        eval_dataset = prepare_dataset(load_examples(args.eval_file, args.max_context_chars))
        eval_dataset = tokenize_dataset(eval_dataset, processor)

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
        save_total_limit=2,
        fp16=args.fp16,
        bf16=args.bf16,
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    logger.info("Starting VLM fine-tuning...")
    trainer.train()
    logger.info("Saving fine-tuned adapters to %s", args.output_dir)
    trainer.save_model()
    processor.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Evaluation utilities for Aura agents.

Supports two evaluation modes:
  * slm – benchmark planner model perplexity and token throughput.
  * vlm – measure answer accuracy on perception QA examples.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from statistics import mean
from typing import Dict, List

import torch
from datasets import Dataset
from tqdm.auto import tqdm

logger = logging.getLogger("aura.eval_pipeline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Aura models.")
    parser.add_argument("--mode", choices={"slm", "vlm"}, required=True)
    parser.add_argument("--model-path", required=True, help="HF model path or ExecuTorch checkpoint.")
    parser.add_argument("--tokenizer", help="Tokenizer id/path (required for slm).")
    parser.add_argument("--data-file", type=Path, required=True, help="JSONL evaluation file.")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


# --------------------------------------------------------------------------------------
# Planner evaluation
# --------------------------------------------------------------------------------------


def load_slm_dataset(path: Path) -> Dataset:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            records.append(payload["instruction"])
    if not records:
        raise ValueError(f"No records in {path}")
    return Dataset.from_dict({"text": records})


def evaluate_slm(model_path: str, tokenizer_path: str, data_file: Path, device: str, max_length: int) -> Dict[str, float]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16 if device != "cpu" else torch.float32)
    model.to(device)
    model.eval()

    dataset = load_slm_dataset(data_file)

    losses: List[float] = []
    tok_per_sec: List[float] = []

    for batch_text in tqdm(dataset["text"], desc="Evaluating planner"):
        encoded = tokenizer(batch_text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
        with torch.no_grad():
            start = time.perf_counter()
            output = model(**encoded, labels=encoded["input_ids"])
            elapsed = time.perf_counter() - start
        losses.append(output.loss.item())
        tokens = encoded["input_ids"].numel()
        tok_per_sec.append(tokens / elapsed if elapsed > 0 else 0.0)

    perplexity = torch.exp(torch.tensor(losses).mean()).item()
    return {
        "perplexity": perplexity,
        "mean_loss": mean(losses),
        "avg_tokens_per_sec": mean(tok_per_sec),
    }


# --------------------------------------------------------------------------------------
# Perception evaluation
# --------------------------------------------------------------------------------------


def load_vlm_dataset(path: Path) -> Dataset:
    questions: List[str] = []
    answers: List[str] = []
    image_paths: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            questions.append(payload["question"])
            answers.append(payload["answer"]["text"])
            image_paths.append(payload.get("screenshot_path"))
    if not questions:
        raise ValueError(f"No records in {path}")
    return Dataset.from_dict({"question": questions, "answer": answers, "image_path": image_paths})


def evaluate_vlm(model_path: str, data_file: Path, device: str) -> Dict[str, float]:
    from PIL import Image
    from transformers import AutoModelForVision2Seq, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForVision2Seq.from_pretrained(model_path)
    model.to(device)
    model.eval()

    dataset = load_vlm_dataset(data_file)
    correct = 0
    total = len(dataset)
    latencies: List[float] = []

    for sample in tqdm(dataset, desc="Evaluating perception"):
        question = sample["question"]
        answer = sample["answer"]
        image_path = sample["image_path"]
        if image_path and Path(image_path).exists():
            image = Image.open(image_path).convert("RGB")
        else:
            image = Image.new("RGB", (512, 512), color=(255, 255, 255))

        inputs = processor(images=image, text=question, return_tensors="pt").to(device)
        with torch.no_grad():
            start = time.perf_counter()
            generated = model.generate(**inputs, max_new_tokens=32)
            elapsed = time.perf_counter() - start
        latencies.append(elapsed)
        decoded = processor.batch_decode(generated, skip_special_tokens=True)[0].strip().lower()
        correct += int(answer.lower() in decoded)

    accuracy = correct / total
    mean_latency = mean(latencies)
    return {"accuracy": accuracy, "mean_latency_sec": mean_latency}


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="[%(levelname)s] %(message)s")

    if args.mode == "slm" and not args.tokenizer:
        raise SystemExit("--tokenizer must be provided for SLM evaluation.")

    if args.mode == "slm":
        metrics = evaluate_slm(args.model_path, args.tokenizer, args.data_file, args.device, args.max_length)
    else:
        metrics = evaluate_vlm(args.model_path, args.data_file, args.device)

    print(json.dumps(metrics, indent=2))  # noqa: T201


if __name__ == "__main__":
    main()

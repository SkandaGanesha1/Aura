#!/usr/bin/env python3
"""
Export Aura models to ExecuTorch portable packages.

Example usage:
    python export_to_executorch.py \
        --model-type slm \
        --checkpoint ../models/slm/aura-planner \
        --tokenizer meta-llama/Llama-3.2-1B-Instruct \
        --output-dir ../models/compiled/slm
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from torch.export import export

logger = logging.getLogger("aura.export_to_executorch")

try:
    from executorch.exir import EdgeProgramManager, to_edge
except ImportError as exc:  # pragma: no cover
    raise SystemExit("ExecuTorch 1.0+ is required. Install official wheels before exporting.") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export an Aura model to ExecuTorch format.")
    parser.add_argument("--model-type", choices={"slm", "vlm"}, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to fine-tuned model directory or file.")
    parser.add_argument("--tokenizer", help="Tokenizer id/path (required for SLM).")
    parser.add_argument("--example-prompt", default="Summarise this conversation.")
    parser.add_argument("--example-question", default="Where is the submit button?")
    parser.add_argument("--image-path", type=Path, help="Optional example image for VLM export.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--delegate-preferences", type=Path, help="Optional YAML manifest to merge with compiled output.")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def load_slm_model(checkpoint: Path, tokenizer_id: str) -> Tuple[torch.nn.Module, Dict[str, torch.Tensor]]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id or checkpoint)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.float16).cpu()

    encoded = tokenizer(["Summarise the latest calendar updates."], return_tensors="pt")
    inputs = {key: value.to(torch.int64) for key, value in encoded.items()}
    return model, inputs


def load_vlm_model(checkpoint: Path, example_question: str, image_path: Path | None) -> Tuple[torch.nn.Module, Dict[str, torch.Tensor]]:
    from PIL import Image
    from transformers import AutoModelForVision2Seq, AutoProcessor

    processor = AutoProcessor.from_pretrained(checkpoint)
    model = AutoModelForVision2Seq.from_pretrained(checkpoint, torch_dtype=torch.float16).cpu()

    if image_path and image_path.exists():
        image = Image.open(image_path).convert("RGB")
    else:
        image = Image.new("RGB", (512, 512), color=(255, 255, 255))

    inputs = processor(images=image, text=example_question, return_tensors="pt")
    input_tensors = {key: value.to(torch.float32 if value.dtype.is_floating_point else torch.int64) for key, value in inputs.items()}
    return model, input_tensors


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="[%(levelname)s] %(message)s")

    if args.model_type == "slm" and not args.tokenizer:
        raise SystemExit("--tokenizer is required when exporting the planner SLM.")

    if args.model_type == "slm":
        model, example_inputs = load_slm_model(args.checkpoint, args.tokenizer)
    else:
        model, example_inputs = load_vlm_model(args.checkpoint, args.example_question, args.image_path)

    model.eval()
    logger.info("Tracing model with torch.export ...")
    exported_program = export(model, tuple(example_inputs.values()))
    logger.info("Converting exported program to ExecuTorch edge format...")
    edge_program = to_edge(exported_program)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    package_path = output_dir / "model.pte"
    logger.info("Saving ExecuTorch package to %s", package_path)

    manager = EdgeProgramManager(edge_program)
    manager.export_to_dir(package_path.parent, package_path.stem)

    metadata = {
        "model_type": args.model_type,
        "source_checkpoint": str(args.checkpoint),
        "delegate_preferences": None,
    }
    if args.delegate_preferences and args.delegate_preferences.exists():
        metadata["delegate_preferences"] = json.loads(args.delegate_preferences.read_text(encoding="utf-8"))

    manifest_path = output_dir / "manifest.json"
    logger.info("Writing manifest to %s", manifest_path)
    manifest_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Export complete.")


if __name__ == "__main__":
    main()

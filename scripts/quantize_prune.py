#!/usr/bin/env python3
"""
Quantize and prune Aura models for efficient on-device deployment.

This script supports dynamic INT8 quantization for causal LM weights and magnitude-based
structured pruning. Outputs are saved as standard PyTorch checkpoints that can be consumed
by ExecuTorch exporters.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, Tuple

import torch
from torch.nn.utils import prune

logger = logging.getLogger("aura.quantize_prune")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantize and prune models for Aura.")
    parser.add_argument("--model-path", type=Path, required=True, help="Path to base model or adapter weights.")
    parser.add_argument("--output-path", type=Path, required=True, help="Destination for quantized model.")
    parser.add_argument("--quantize", action="store_true", help="Apply dynamic INT8 quantization.")
    parser.add_argument("--prune", action="store_true", help="Apply magnitude pruning.")
    parser.add_argument("--prune-amount", type=float, default=0.1, help="Fraction of weights to prune.")
    parser.add_argument("--linear-only", action="store_true", help="Restrict pruning to linear layers.")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def iter_linear_modules(model: torch.nn.Module) -> Iterable[Tuple[str, torch.nn.Module]]:
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            yield name, module


def apply_pruning(model: torch.nn.Module, amount: float, linear_only: bool) -> None:
    logger.info("Applying global magnitude pruning (amount=%.2f)", amount)
    parameters_to_prune = []
    for name, module in model.named_modules():
        if linear_only and not isinstance(module, torch.nn.Linear):
            continue
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            parameters_to_prune.append((module, "weight"))

    if not parameters_to_prune:
        logger.warning("No modules matched the pruning criteria.")
        return

    prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=amount)

    # Remove pruning reparameterisation to materialise pruned weights
    for module, _ in parameters_to_prune:
        prune.remove(module, "weight")


def apply_quantization(model: torch.nn.Module) -> torch.nn.Module:
    logger.info("Applying dynamic INT8 quantization to Linear layers.")
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8,
    )
    return quantized_model


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="[%(levelname)s] %(message)s")

    logger.info("Loading model from %s", args.model_path)
    model = torch.load(args.model_path, map_location="cpu")
    if isinstance(model, dict) and "model_state_dict" in model:
        state_dict = model["model_state_dict"]
        model_class = model.get("model_class")
        if model_class is None:
            raise ValueError("model_class field missing in checkpoint dictionary.")
        module = model_class()
        module.load_state_dict(state_dict)
        model = module

    if args.prune:
        apply_pruning(model, args.prune_amount, args.linear_only)

    if args.quantize:
        model = apply_quantization(model)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Saving optimised model to %s", args.output_path)
    torch.save(model, args.output_path)
    logger.info("Quantization/pruning complete.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Dataset preprocessing utilities for Aura.

This script ingests raw mobile UI datasets (e.g. Android In The Wild, AndroidArena),
normalises view hierarchy trees, and produces training corpora for:

1. The planner SLM – instruction → action-sequence examples.
2. The perception VLM – (XML context, question) → answer triples and screenshot assets.

Usage:
    python dataset_preprocess.py \
        --dataset aitw \
        --raw-dir ../data/raw/aitw \
        --output-dir ../data/processed \
        --min-instruction-length 5

Expected directory layout:
    data/
      raw/
        aitw/           # Android In The Wild episodes
        androidarena/   # AndroidArena (A3) episodes
      processed/
        slm_training/episodes.jsonl
        vlm_training/examples.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

logger = logging.getLogger("aura.dataset_preprocess")


# --------------------------------------------------------------------------------------
# Data containers
# --------------------------------------------------------------------------------------


@dataclass
class UIAction:
    """Represents a single action the agent can replay."""

    type: str
    target_id: Optional[str]
    text: Optional[str]
    coordinates: Optional[List[int]]

    def serialize(self) -> Dict[str, Optional[str]]:
        return {
            "type": self.type,
            "target_id": self.target_id,
            "text": self.text,
            "coordinates": self.coordinates,
        }


@dataclass
class Episode:
    """Container for a single recorded task episode."""

    instruction: str
    view_hierarchy_path: Path
    screenshot_path: Optional[Path]
    actions: List[UIAction]


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def safe_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        logger.debug("Skipping copy; destination already exists: %s", destination)
        return
    shutil.copy2(source, destination)


def flatten_view_hierarchy(node: Dict, nodes: Optional[List[Dict]] = None) -> List[Dict]:
    """Flattens the Android view hierarchy into a list of nodes with useful metadata."""
    if nodes is None:
        nodes = []

    node_id = node.get("resource_id") or node.get("id") or f"node_{len(nodes)}"
    bounds = node.get("bounds") or node.get("boundsInScreen")
    text = node.get("text") or node.get("contentDescription") or ""
    class_name = node.get("class") or node.get("className")

    nodes.append(
        {
            "id": node_id,
            "class": class_name,
            "text": text,
            "content_desc": node.get("contentDescription") or "",
            "clickable": bool(node.get("clickable") or node.get("focusable")),
            "bounds": bounds,
        }
    )

    for child in node.get("children", []):
        flatten_view_hierarchy(child, nodes)

    return nodes


def parse_episode(directory: Path) -> Episode:
    """Parses a directory containing one recorded episode."""
    meta_path = directory / "metadata.json"
    hierarchy_path = directory / "view_hierarchy.json"
    actions_path = directory / "actions.json"
    screenshot_path = next(directory.glob("screenshot*.png"), None)

    if not hierarchy_path.exists():
        raise FileNotFoundError(f"{hierarchy_path} missing")
    if not actions_path.exists():
        raise FileNotFoundError(f"{actions_path} missing")

    metadata = load_json(meta_path) if meta_path.exists() else {}
    instruction = metadata.get("instruction") or metadata.get("goal") or ""
    if not instruction:
        # Some public datasets encode instruction in actions.json
        actions_payload = load_json(actions_path)
        instruction = actions_payload.get("instruction") or actions_payload.get("prompt") or ""

    if not instruction:
        raise ValueError(f"Instruction could not be found in {directory}")

    actions: List[UIAction] = []
    for raw_action in load_json(actions_path).get("actions", []):
        actions.append(
            UIAction(
                type=raw_action.get("type") or raw_action.get("action"),
                target_id=raw_action.get("target_id"),
                text=raw_action.get("text"),
                coordinates=raw_action.get("coordinates"),
            )
        )

    return Episode(
        instruction=instruction.strip(),
        view_hierarchy_path=hierarchy_path,
        screenshot_path=screenshot_path,
        actions=actions,
    )


def discover_episodes(raw_dir: Path) -> Iterable[Episode]:
    seen_directories: set[Path] = set()

    for path in sorted(raw_dir.rglob("episode.json")):
        # Some datasets use a single JSON file per episode
        data = load_json(path)
        hierarchy_path = path.with_name("view_hierarchy.json")
        if hierarchy_path.exists():
            episode_dir = path.parent
        else:
            hierarchy_path = path
            episode_dir = path.parent

        resolved_dir = episode_dir.resolve()
        if resolved_dir in seen_directories:
            continue
        seen_directories.add(resolved_dir)

        try:
            yield parse_episode(episode_dir)
        except (ValueError, FileNotFoundError) as exc:
            logger.warning("Skipping episode %s: %s", episode_dir, exc)

    # Fallback: assume each subdirectory is an episode
    for directory in raw_dir.iterdir():
        if directory.is_dir():
            if directory.resolve() in seen_directories:
                continue
            try:
                yield parse_episode(directory)
            except (ValueError, FileNotFoundError):
                continue


# --------------------------------------------------------------------------------------
# Output writers
# --------------------------------------------------------------------------------------


def write_slm_training_data(episodes: Iterable[Episode], output_dir: Path, min_instruction_length: int) -> int:
    target = output_dir / "slm_training" / "episodes.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with target.open("w", encoding="utf-8") as handle:
        for episode in episodes:
            if len(episode.instruction.split()) < min_instruction_length:
                continue

            serialized_actions = [action.serialize() for action in episode.actions]
            record = {
                "instruction": episode.instruction,
                "actions": serialized_actions,
                "view_hierarchy_path": str(episode.view_hierarchy_path),
                "screenshot_path": str(episode.screenshot_path) if episode.screenshot_path else None,
            }
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
            count += 1

    return count


def write_vlm_training_data(episodes: Iterable[Episode], output_dir: Path, sample_questions: int) -> int:
    target = output_dir / "vlm_training" / "examples.jsonl"
    images_dir = output_dir / "vlm_training" / "images"
    xml_dir = output_dir / "vlm_training" / "xml"
    images_dir.mkdir(parents=True, exist_ok=True)
    xml_dir.mkdir(parents=True, exist_ok=True)

    random.seed(13)
    examples = 0

    with target.open("w", encoding="utf-8") as handle:
        for idx, episode in enumerate(episodes):
            nodes = flatten_view_hierarchy(load_json(episode.view_hierarchy_path))
            xml_copy = xml_dir / f"episode_{idx:06d}.json"
            safe_copy(episode.view_hierarchy_path, xml_copy)

            if episode.screenshot_path:
                image_copy = images_dir / f"episode_{idx:06d}{episode.screenshot_path.suffix}"
                safe_copy(episode.screenshot_path, image_copy)
            else:
                image_copy = None

            sampled_nodes = random.sample(nodes, min(sample_questions, len(nodes)))

            for node in sampled_nodes:
                question = f"Where is the UI element with text '{node['text']}'?" if node["text"] else \
                    f"What class is the element with id {node['id']}?"
                answer = {
                    "text": node["text"] or node["content_desc"],
                    "bounds": node["bounds"],
                    "id": node["id"],
                    "class": node["class"],
                }
                record = {
                    "question": question,
                    "answer": answer,
                    "xml_path": str(xml_copy),
                    "screenshot_path": str(image_copy) if image_copy else None,
                }
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")
                examples += 1

    return examples


def write_splits(processed_dir: Path, train_ratio: float, val_ratio: float) -> None:
    episodes_file = processed_dir / "slm_training" / "episodes.jsonl"
    examples_file = processed_dir / "vlm_training" / "examples.jsonl"
    splits_dir = processed_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    def _split(path: Path) -> None:
        with path.open("r", encoding="utf-8") as handle:
            lines = handle.readlines()
        total = len(lines)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)

        mapping = {
            "train": lines[:train_end],
            "val": lines[train_end:val_end],
            "test": lines[val_end:],
        }

        for split, split_lines in mapping.items():
            target = splits_dir / f"{path.stem}_{split}.jsonl"
            with target.open("w", encoding="utf-8") as handle:
                handle.writelines(split_lines)

    if episodes_file.exists():
        _split(episodes_file)
    if examples_file.exists():
        _split(examples_file)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Aura datasets.")
    parser.add_argument("--dataset", choices={"aitw", "androidarena", "custom"}, default="aitw")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        required=True,
        help="Root directory for raw episodes (e.g. data/raw/aitw or data/raw/androidarena).",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Destination for processed artifacts.")
    parser.add_argument("--min-instruction-length", type=int, default=4, help="Filter short instructions.")
    parser.add_argument("--vlm-questions-per-episode", type=int, default=4, help="Number of perception QA samples per episode.")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="[%(levelname)s] %(message)s")

    logger.info("Discovering episodes in %s", args.raw_dir)
    episodes = list(discover_episodes(args.raw_dir))
    if not episodes:
        raise SystemExit("No episodes detected – ensure --raw-dir points to the dataset root.")

    logger.info("Preparing SLM dataset...")
    slm_count = write_slm_training_data(episodes, args.output_dir, args.min_instruction_length)
    logger.info("Prepared %d SLM instruction examples", slm_count)

    logger.info("Preparing VLM dataset...")
    vlm_count = write_vlm_training_data(episodes, args.output_dir, args.vlm_questions_per_episode)
    logger.info("Prepared %d VLM perception examples", vlm_count)

    logger.info("Writing train/val/test splits...")
    write_splits(args.output_dir, args.train_ratio, args.val_ratio)
    logger.info("Dataset preparation complete. Processed artifacts saved to %s", args.output_dir)


if __name__ == "__main__":
    main()

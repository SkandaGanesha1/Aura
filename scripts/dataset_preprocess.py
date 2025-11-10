# #!/usr/bin/env python3
# """
# Dataset preprocessing utilities for Aura.

# This script ingests raw mobile UI datasets (e.g. Android In The Wild, AndroidArena, DroidTask, MoTiF),
# normalises view hierarchy trees, and produces training corpora for:

# 1. The planner SLM – instruction → action-sequence examples.
# 2. The perception VLM – (XML context, question) → answer triples and screenshot assets.

# Usage:
#     python dataset_preprocess.py \
#         --dataset aitw \
#         --raw-dir ../data/raw/aitw \
#         --output-dir ../data/processed \
#         --min-instruction-length 5

# Expected directory layout:
#     data/
#       raw/
#         aitw/           # Android In The Wild episodes
#         androidarena/   # AndroidArena (A3) episodes
#         droidtask/      # Optional DroidTask benchmark
#         motif/          # Optional MoTiF benchmark
#       processed/
#         slm_training/episodes.jsonl
#         vlm_training/examples.jsonl
# """

# from __future__ import annotations

# import argparse
# import json
# import logging
# import random
# import shutil
# from dataclasses import dataclass
# from pathlib import Path
# from typing import Dict, Iterable, List, Optional

# logger = logging.getLogger("aura.dataset_preprocess")


# # --------------------------------------------------------------------------------------
# # Data containers
# # --------------------------------------------------------------------------------------


# @dataclass
# class UIAction:
#     """Represents a single action the agent can replay."""

#     type: str
#     target_id: Optional[str]
#     text: Optional[str]
#     coordinates: Optional[List[int]]

#     def serialize(self) -> Dict[str, Optional[str]]:
#         return {
#             "type": self.type,
#             "target_id": self.target_id,
#             "text": self.text,
#             "coordinates": self.coordinates,
#         }


# @dataclass
# class Episode:
#     """Container for a single recorded task episode."""

#     instruction: str
#     view_hierarchy_path: Path
#     screenshot_path: Optional[Path]
#     actions: List[UIAction]


# # --------------------------------------------------------------------------------------
# # Helpers
# # --------------------------------------------------------------------------------------


# def load_json(path: Path) -> Dict:
#     with path.open("r", encoding="utf-8") as handle:
#         return json.load(handle)


# def safe_copy(source: Path, destination: Path) -> None:
#     destination.parent.mkdir(parents=True, exist_ok=True)
#     if destination.exists():
#         logger.debug("Skipping copy; destination already exists: %s", destination)
#         return
#     shutil.copy2(source, destination)


# def flatten_view_hierarchy(node: Dict, nodes: Optional[List[Dict]] = None) -> List[Dict]:
#     """Flattens the Android view hierarchy into a list of nodes with useful metadata."""
#     if nodes is None:
#         nodes = []

#     node_id = node.get("resource_id") or node.get("id") or f"node_{len(nodes)}"
#     bounds = node.get("bounds") or node.get("boundsInScreen")
#     text = node.get("text") or node.get("contentDescription") or ""
#     class_name = node.get("class") or node.get("className")

#     nodes.append(
#         {
#             "id": node_id,
#             "class": class_name,
#             "text": text,
#             "content_desc": node.get("contentDescription") or "",
#             "clickable": bool(node.get("clickable") or node.get("focusable")),
#             "bounds": bounds,
#         }
#     )

#     for child in node.get("children", []):
#         flatten_view_hierarchy(child, nodes)

#     return nodes


# def parse_episode(directory: Path) -> Episode:
#     """Parses a directory containing one recorded episode."""
#     meta_path = directory / "metadata.json"
#     hierarchy_path = directory / "view_hierarchy.json"
#     actions_path = directory / "actions.json"
#     screenshot_path = next(directory.glob("screenshot*.png"), None)

#     if not hierarchy_path.exists():
#         raise FileNotFoundError(f"{hierarchy_path} missing")
#     if not actions_path.exists():
#         raise FileNotFoundError(f"{actions_path} missing")

#     metadata = load_json(meta_path) if meta_path.exists() else {}
#     instruction = metadata.get("instruction") or metadata.get("goal") or ""
#     if not instruction:
#         # Some public datasets encode instruction in actions.json
#         actions_payload = load_json(actions_path)
#         instruction = actions_payload.get("instruction") or actions_payload.get("prompt") or ""

#     if not instruction:
#         raise ValueError(f"Instruction could not be found in {directory}")

#     actions: List[UIAction] = []
#     for raw_action in load_json(actions_path).get("actions", []):
#         actions.append(
#             UIAction(
#                 type=raw_action.get("type") or raw_action.get("action"),
#                 target_id=raw_action.get("target_id"),
#                 text=raw_action.get("text"),
#                 coordinates=raw_action.get("coordinates"),
#             )
#         )

#     return Episode(
#         instruction=instruction.strip(),
#         view_hierarchy_path=hierarchy_path,
#         screenshot_path=screenshot_path,
#         actions=actions,
#     )


# def discover_episodes(raw_dir: Path) -> Iterable[Episode]:
#     seen_directories: set[Path] = set()

#     for path in sorted(raw_dir.rglob("episode.json")):
#         # Some datasets use a single JSON file per episode
#         data = load_json(path)
#         hierarchy_path = path.with_name("view_hierarchy.json")
#         if hierarchy_path.exists():
#             episode_dir = path.parent
#         else:
#             hierarchy_path = path
#             episode_dir = path.parent

#         resolved_dir = episode_dir.resolve()
#         if resolved_dir in seen_directories:
#             continue
#         seen_directories.add(resolved_dir)

#         try:
#             yield parse_episode(episode_dir)
#         except (ValueError, FileNotFoundError) as exc:
#             logger.warning("Skipping episode %s: %s", episode_dir, exc)

#     # Fallback: assume each subdirectory is an episode
#     for directory in raw_dir.iterdir():
#         if directory.is_dir():
#             if directory.resolve() in seen_directories:
#                 continue
#             try:
#                 yield parse_episode(directory)
#             except (ValueError, FileNotFoundError):
#                 continue


# # --------------------------------------------------------------------------------------
# # Output writers
# # --------------------------------------------------------------------------------------


# def write_slm_training_data(episodes: Iterable[Episode], output_dir: Path, min_instruction_length: int) -> int:
#     target = output_dir / "slm_training" / "episodes.jsonl"
#     target.parent.mkdir(parents=True, exist_ok=True)
#     count = 0

#     with target.open("w", encoding="utf-8") as handle:
#         for episode in episodes:
#             if len(episode.instruction.split()) < min_instruction_length:
#                 continue

#             serialized_actions = [action.serialize() for action in episode.actions]
#             record = {
#                 "instruction": episode.instruction,
#                 "actions": serialized_actions,
#                 "view_hierarchy_path": str(episode.view_hierarchy_path),
#                 "screenshot_path": str(episode.screenshot_path) if episode.screenshot_path else None,
#             }
#             handle.write(json.dumps(record, ensure_ascii=True) + "\n")
#             count += 1

#     return count


# def write_vlm_training_data(episodes: Iterable[Episode], output_dir: Path, sample_questions: int) -> int:
#     target = output_dir / "vlm_training" / "examples.jsonl"
#     images_dir = output_dir / "vlm_training" / "images"
#     xml_dir = output_dir / "vlm_training" / "xml"
#     images_dir.mkdir(parents=True, exist_ok=True)
#     xml_dir.mkdir(parents=True, exist_ok=True)

#     random.seed(13)
#     examples = 0

#     with target.open("w", encoding="utf-8") as handle:
#         for idx, episode in enumerate(episodes):
#             nodes = flatten_view_hierarchy(load_json(episode.view_hierarchy_path))
#             xml_copy = xml_dir / f"episode_{idx:06d}.json"
#             safe_copy(episode.view_hierarchy_path, xml_copy)

#             if episode.screenshot_path:
#                 image_copy = images_dir / f"episode_{idx:06d}{episode.screenshot_path.suffix}"
#                 safe_copy(episode.screenshot_path, image_copy)
#             else:
#                 image_copy = None

#             sampled_nodes = random.sample(nodes, min(sample_questions, len(nodes)))

#             for node in sampled_nodes:
#                 question = f"Where is the UI element with text '{node['text']}'?" if node["text"] else \
#                     f"What class is the element with id {node['id']}?"
#                 answer = {
#                     "text": node["text"] or node["content_desc"],
#                     "bounds": node["bounds"],
#                     "id": node["id"],
#                     "class": node["class"],
#                 }
#                 record = {
#                     "question": question,
#                     "answer": answer,
#                     "xml_path": str(xml_copy),
#                     "screenshot_path": str(image_copy) if image_copy else None,
#                 }
#                 handle.write(json.dumps(record, ensure_ascii=True) + "\n")
#                 examples += 1

#     return examples


# def write_splits(processed_dir: Path, train_ratio: float, val_ratio: float) -> None:
#     episodes_file = processed_dir / "slm_training" / "episodes.jsonl"
#     examples_file = processed_dir / "vlm_training" / "examples.jsonl"
#     splits_dir = processed_dir / "splits"
#     splits_dir.mkdir(parents=True, exist_ok=True)

#     def _split(path: Path) -> None:
#         with path.open("r", encoding="utf-8") as handle:
#             lines = handle.readlines()
#         total = len(lines)
#         train_end = int(total * train_ratio)
#         val_end = train_end + int(total * val_ratio)

#         mapping = {
#             "train": lines[:train_end],
#             "val": lines[train_end:val_end],
#             "test": lines[val_end:],
#         }

#         for split, split_lines in mapping.items():
#             target = splits_dir / f"{path.stem}_{split}.jsonl"
#             with target.open("w", encoding="utf-8") as handle:
#                 handle.writelines(split_lines)

#     if episodes_file.exists():
#         _split(episodes_file)
#     if examples_file.exists():
#         _split(examples_file)


# # --------------------------------------------------------------------------------------
# # CLI
# # --------------------------------------------------------------------------------------


# def parse_args() -> argparse.Namespace:
#     parser = argparse.ArgumentParser(description="Prepare Aura datasets.")
#     parser.add_argument(
#         "--dataset",
#         type=str,
#         choices={"aitw", "androidarena", "droidtask", "motif", "custom"},
#         default="aitw",
#         help=(
#             "Dataset name. Supported: "
#             "aitw (Android-In-The-Wild, 715k episodes), "
#             "androidarena (A3, 21 apps), "
#             "droidtask (158 tasks, 13 apps), "
#             "motif (>4.7k tasks), or custom."
#         ),
#     )
#     parser.add_argument(
#         "--raw-dir",
#         type=Path,
#         required=True,
#         help="Root directory for raw episodes (e.g. data/raw/aitw, data/raw/androidarena, data/raw/droidtask).",
#     )
#     parser.add_argument("--output-dir", type=Path, required=True, help="Destination for processed artifacts.")
#     parser.add_argument("--min-instruction-length", type=int, default=4, help="Filter short instructions.")
#     parser.add_argument("--vlm-questions-per-episode", type=int, default=4, help="Number of perception QA samples per episode.")
#     parser.add_argument("--train-ratio", type=float, default=0.8)
#     parser.add_argument("--val-ratio", type=float, default=0.1)
#     parser.add_argument("--log-level", default="INFO")
#     return parser.parse_args()


# def main() -> None:
#     args = parse_args()
#     logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="[%(levelname)s] %(message)s")

#     if not args.raw_dir.exists():
#         raise SystemExit(
#             f"Raw directory {args.raw_dir} does not exist. "
#             f"Expected layout: data/raw/{args.dataset} (e.g. data/raw/aitw)."
#         )

#     logger.info("Discovering episodes in %s", args.raw_dir)
#     episodes = list(discover_episodes(args.raw_dir))
#     if not episodes:
#         raise SystemExit("No episodes detected – ensure --raw-dir points to the dataset root.")

#     logger.info("Preparing SLM dataset...")
#     slm_count = write_slm_training_data(episodes, args.output_dir, args.min_instruction_length)
#     logger.info("Prepared %d SLM instruction examples", slm_count)

#     logger.info("Preparing VLM dataset...")
#     vlm_count = write_vlm_training_data(episodes, args.output_dir, args.vlm_questions_per_episode)
#     logger.info("Prepared %d VLM perception examples", vlm_count)

#     logger.info("Writing train/val/test splits...")
#     write_splits(args.output_dir, args.train_ratio, args.val_ratio)
#     logger.info("Dataset preparation complete. Processed artifacts saved to %s", args.output_dir)


# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
#!/usr/bin/env python3
"""
data_preprocess.py
===================

This script reads the task definitions from the AndroidArena `tasks/` directory
and produces a JSONL file containing instruction–action pairs for fine‑tuning
an Aura planner (small language model).  Each instruction in the YAML task
files is passed to a generative language model (LLM) to synthesise a sequence
of high‑level actions.  The output can then be used with the `fine_tune_slm.py`
script for supervised fine‑tuning.

Usage:
  python data_preprocess.py \
    --tasks-dir /path/to/AndroidArena/tasks \
    --output episodes.jsonl \
    --use-llm

When the `--use-llm` flag is specified, the script calls a generative
model (e.g. Gemini 1.5 Pro) to decompose each instruction into a list
of action dictionaries.  Without `--use-llm`, it writes empty action
lists; these can be filled later or used to evaluate zero‑shot planning.

LLM integration requires installing the `google-generativeai` package and
configuring a valid API key via the `GOOGLE_API_KEY` environment variable.

Example Gemini usage:

    import google.generativeai as genai
    genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
    model = genai.GenerativeModel('gemini-1.5-pro-latest')
    response = model.generate_content(prompt)

The prompt instructs Gemini to break down a mobile task into a JSON list of
actions.  The returned text is parsed as JSON.  See the `call_llm` function
below for details.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml  # PyYAML is required.

# Optional import of Google Generative AI.  Only required when --use-llm is set.
import google.generativeai as genai


def call_llm(model: Any, instruction: str) -> List[Dict[str, Any]]:
    """Invoke the generative model to decompose an instruction into actions.

    The prompt asks the model to return a JSON list of actions.  Each action
    should be a dict with the following keys: `type` (tap, swipe, type, etc.),
    `target` (a brief description or resource ID of the UI element),
    `text` (optional text to type), and `coordinates` (optional x,y screen
    coordinates for taps or swipes).

    Args:
        model: A configured generative model (e.g. Gemini).
        instruction: The natural language instruction to decompose.

    Returns:
        A list of action dictionaries.
    """
    prompt = (
        "You are an expert mobile assistant. Your task is to decompose the "
        "following natural language instruction into a sequence of high-level "
        "mobile UI actions. Respond strictly in JSON format as a list of "
        "objects. Each object must include: 'type' (tap, swipe, type, open_app, etc.), "
        "'target' (description or resource_id of the element), 'text' (for typing actions), "
        "and 'coordinates' (list [x, y] for taps or [x1, y1, x2, y2] for swipes) if relevant. "
        "Do not include any extra text outside of the JSON list.\n\n"
        f"Instruction: {instruction}"
    )
    # Generate content using Gemini 1.5 Pro.  We request a JSON response.
    try:
        response = model.generate_content(prompt)
    except Exception as exc:
        raise RuntimeError(f"LLM call failed for instruction '{instruction}': {exc}")
    content = response.text.strip()
    try:
        actions = json.loads(content)
        assert isinstance(actions, list)
    except Exception as exc:
        raise ValueError(
            f"Expected the model to return a JSON list of actions, but got: {content}\nError: {exc}"
        )
    return actions


def iter_instructions(tasks_dir: Path) -> Iterable[str]:
    """Yield all instructions from YAML task files in the given directory."""
    for yaml_file in sorted(tasks_dir.glob("*.yaml")):
        with yaml_file.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        tasks = data.get("tasks", [])
        for item in tasks:
            instr = item.get("instruction")
            if instr:
                yield instr.strip()


def process_tasks(tasks_dir: Path, output_path: Path, use_llm: bool) -> None:
    """Process instructions and write them to a JSONL file for training.

    Args:
        tasks_dir: Path to the AndroidArena tasks directory.
        output_path: Where to write the JSONL records.
        use_llm: Whether to call a generative model to produce action plans.
    """
    # Configure the generative model if requested
    model = None
    if use_llm:
        if genai is None:
            raise ImportError(
                "google-generativeai is not installed. Install it with 'pip install google-generativeai'."
            )
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GOOGLE_API_KEY environment variable must be set to use the generative model."
            )
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-pro")

    records_written = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out_file:
        for instr in iter_instructions(tasks_dir):
            if use_llm:
                try:
                    actions = call_llm(model, instr)
                except Exception as exc:
                    # If the LLM fails, write an empty action list and log the error.
                    actions = []
                    print(f"Warning: LLM failed on instruction '{instr}': {exc}")
            else:
                actions = []
            record = {
                "instruction": instr,
                "actions": actions,
            }
            out_file.write(json.dumps(record, ensure_ascii=True) + "\n")
            records_written += 1
    print(f"Wrote {records_written} records to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process AndroidArena tasks into training data")
    parser.add_argument(
        "--tasks-dir",
        type=Path,
        required=True,
        help="Path to the AndroidArena tasks directory (contains YAML files).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination JSONL file for the processed records.",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Whether to call a generative model (Gemini) to generate action plans.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.tasks_dir.exists():
        raise SystemExit(f"Tasks directory does not exist: {args.tasks_dir}")
    process_tasks(args.tasks_dir, args.output, use_llm=args.use_llm)


if __name__ == "__main__":
    main()
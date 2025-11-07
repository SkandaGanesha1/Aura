# Data Setup

Aura's training pipelines expect the canonical Android UI datasets referenced in the project proposal to be unpacked inside `data/raw/`. The preprocessing script then emits processed JSONL corpora under `data/processed/`.

## Android In The Wild (AITW)
1. Request access and download AITW from the official provider.
2. Extract the archive into `data/raw/aitw`.
3. Each episode folder should contain `metadata.json`, `view_hierarchy.json`, `actions.json`, and at least one `screenshot.png`.

## AndroidArena (A3)
1. Download AndroidArena (A3) from its provider.
2. Extract into `data/raw/androidarena`.
3. The directory layout mirrors AITW, so the same preprocessing script can parse it.

## Processed Outputs
Run:
```bash
python scripts/dataset_preprocess.py \
  --dataset aitw \
  --raw-dir data/raw/aitw \
  --output-dir data/processed
```

By default the script writes:
- Planner SLM training data to `data/processed/slm_training/episodes.jsonl`
- Perception VLM training data to `data/processed/vlm_training/examples.jsonl`

These files are the defaults consumed by `scripts/fine_tune_slm.py` and `scripts/fine_tune_vlm.py`.

# Data & Training Strategy

Aura depends on a diverse corpus of mobile UI trajectories and natural language instructions to behave reliably across applications. This document outlines the staged data plan, focusing on practicality for hackathons while preserving a path to production-quality personalization.

## Phase 0 – Environment Setup
- Provision a workstation (or Arm server) with **Python 3.10+**, **PyTorch 2.3+**, **ExecuTorch 1.0**, and the Arm AI software stack (KleidiAI, Arm Compute Library, Arm NN).
- Ensure sufficient storage (≥ 250 GB) for raw datasets, synthetic expansions, and checkpoints.
- Configure `scripts/requirements.txt` to pin compatible versions of `torch`, `transformers`, `datasets`, `peft`, `accelerate`, and ExecuTorch tooling.

## Phase 1 – Foundation Fine-Tuning
1. **Datasets**  
   - *Android In The Wild (AITW)* – 715k episodes with screenshots and `view_hierarchy.json`.  
   - *Android Arena (A3)* – curated multi-step tasks across 21 popular apps.  
   - *MoTIF* & *DroidTask* – optional datasets to diversify app categories.

2. **Preprocessing** (see `scripts/dataset_preprocess.py`)  
   - Normalize view hierarchy trees into a consistent schema (`node_id`, `bounds`, `text`, `class`, `content_desc`).  
   - Generate paired instruction → action traces for the SLM by flattening event sequences.  
   - Produce Perception labelled samples: `context (XML)` + `(question, answer, bounding_box)` triples.

3. **Training**  
   - Run `scripts/fine_tune_slm.py` with LoRA/QLoRA adapters on INT8 base weights.  
   - Train the VLM with contrastive objectives (text ↔ UI element) while preserving low-rank adapters that are compatible with quantization.

## Phase 2 – Synthetic Teach Mode
Designed for fast iteration during demos or user onboarding.

1. **Capture** – Use a lightweight recorder (planned under `scripts/record_workflow.py`) to log 3–5 full task executions. Each run stores:
   - Sequential screenshots (`PNG`)
   - Accessibility XML snapshots
   - Action metadata (tap coordinates, text input)

2. **Synthesize** – `scripts/generate_synthetic_instructions.py` expands each trajectory into hundreds of paraphrased instructions and perturbed UI states:
   - Apply templated perturbations (e.g., alternate restaurant types, meeting times).
   - Use a large external LLM (if permitted) to diversify intents while preserving outcome constraints.

3. **Augment** – Mix synthetic trajectories with base datasets using curriculum sampling (generalist → specialist).

4. **Rapid Fine-Tune** – Run short, parameter-efficient finetuning cycles (≤ 30 min) targeting adapter layers only. Export updated adapters to `models/slm/adapters/` or `models/vlm/adapters/`.

## Phase 3 – Continuous Improvement
- Integrate opt-in telemetry (success/failure traces, anonymized) to enrich replay buffers.
- Plan for **federated learning** to fine-tune adapters on-device without sharing raw data.
- Automate dataset curation pipelines (`scripts/data_health_check.py`, future work) that validate schema, class balance, and distribution drift.

## Validation
- Use `scripts/eval_pipeline.py` to score agents on:
  - Task success rate
  - Latency per step and total completion
  - Token/s generation speed for the SLM
  - Battery consumption (through Android power stats APIs)
- Compose evaluation suites in `tests/evaluation_scenarios/` mirroring real user flows (booking travel, expense approval, messaging).

## Storage & Versioning
- Large artifacts remain outside Git. Use `git-annex`, `DVC`, or object storage with versioned manifests.
- Each dataset slice includes a manifest with SHA256 hashes, license metadata, and provenance.

## Privacy Considerations
- User-generated trajectories are encrypted at rest and tagged with retention policies.
- Synthetic data is labeled to avoid mixing with sensitive logs.
- Consent flows inform users when interactions may contribute to personalization models.

This staged approach ensures Aura can be bootstrapped quickly for demos while paving the road toward scalable, privacy-preserving model updates in production.

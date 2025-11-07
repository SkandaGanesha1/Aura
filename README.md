# Aura: On-Device Agentic Orchestration Layer

Aura is an Arm-optimized, on-device agent that turns a mobile device into a proactive assistant. It runs a full **Sense → Plan → Act** loop locally: perceiving the UI through the Android Accessibility APIs, reasoning over user intent with a quantized small language model, and executing taps, swipes, or cross-device transfers at machine speed. The system is built as a hierarchical multi-agent architecture, mapping each component to the Arm compute unit best suited for the job.

The project is designed for the *Arm AI Developer Challenge 2025* and serves as a blueprint for building an ambient, agent-first experience without relying on cloud inference.

## Key Capabilities
- **On-device planning** using an INT8-quantized Llama 3.2 1B (or similar SLM) accelerated with Arm KleidiAI.
- **Hybrid perception** that prefers fast Accessibility XML parsing and falls back to a compact VLM on Mali GPU / Ethos-N NPU via ExecuTorch.
- **Secure actuation** through a guarded Accessibility Service that automates UI interactions with user confirmations for sensitive commands.
- **Cross-device continuity** powered by Google’s Cross-Device SDK, enabling seamless task migration between phone, tablet, and laptop.
- **Synthetic Teach Mode** pipeline for rapidly adapting to new apps during hackathons via automated data generation and fine-tuning.

## Repository Layout
```
aura-project/
├── android/          # Multi-module Android app (orchestrator, perception, actuator, continuity)
├── data/             # Raw/processed datasets and splits (not versioned for large files)
├── desktop/          # Optional desktop companion for cross-device demos
├── docs/             # Design documents, architecture notes, datasets strategy
├── models/           # Fine-tuned checkpoints and ExecuTorch exports (tracked via metadata)
├── scripts/          # Python tooling for training, quantization, export, evaluation
├── tests/            # Kotlin, instrumentation, and Python test scaffolding
└── README.md
```

## Getting Started
### 1. Environment Setup
1. Install **Python 3.10+** and **Poetry** or `pip`.
2. Install ExecuTorch tooling and PyTorch (match the Arm AI toolkit version).
3. Install Android Studio Iguana+ with Android SDK 34, NDK r26+, and the Google Cross-Device SDK.

```bash
cd aura-project/scripts
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Datasets
- Download **Android In The Wild (AITW)** into `data/raw/aitw` and **Android Arena (A3)** into `data/raw/androidarena`. Each episode directory should contain `metadata.json`, `view_hierarchy.json`, `actions.json`, and at least one screenshot (`screenshot.png`).
- Run `scripts/dataset_preprocess.py --raw-dir data/raw/<dataset>` to normalise the data. The script writes planner data to `data/processed/slm_training/episodes.jsonl` and perception data to `data/processed/vlm_training/examples.jsonl`.
- Record app-specific trajectories using the `Teach Mode` recorder (see `docs/data_strategy.md` for the seven-step workflow) to augment the public datasets when needed.

### 3. Fine-Tune Models
The provided scripts default to the proposal's recommended on-device models: `meta-llama/Llama-3.2-1B-Instruct` for the planner SLM and `defog/smol-vlm-7b` for the VLM fallback. Override the `--base-model` flag if you need a different checkpoint (e.g., experiments with Llama-3.2-3B or Gemini Nano when available).

```bash
# Intent planner SLM
python scripts/fine_tune_slm.py --config configs/slm_default.yaml

# Perception VLM hybrid assets
python scripts/fine_tune_vlm.py --config configs/vlm_hybrid.yaml

# Quantize and prune
python scripts/quantize_prune.py --model-path models/slm/latest.pt
```

### 4. Export to ExecuTorch
```bash
python scripts/export_to_executorch.py \
  --model-path models/slm/quantized/aura_slm_int8.pt \
  --output-dir models/compiled/slm

bash scripts/build_mobile_models.sh
```
Exports create `.pte` packages for ExecuTorch, along with manifest YAML files describing tensor formats and delegate preferences. When exporting planner models you can omit `--tokenizer`; the script automatically falls back to `meta-llama/Llama-3.2-1B-Instruct`.

### 5. Build Android App
1. Open `android/` in Android Studio.
2. Sync Gradle; ensure `models/compiled/` assets are linked via `app/src/main/assets`.
3. Run the `Aura` application on an Arm-based device with Accessibility permissions granted.
4. Trigger a sample command (“Book an Uber to my next meeting and tell Slack I’m on my way”) to watch the orchestrated workflow.

### 6. Cross-Device Demo (Optional)
1. Build the `desktop/` companion (Kotlin Multiplatform or JVM).
2. Start the desktop listener: `./gradlew :desktop:run`.
3. On the phone, issue a task better suited for larger screens (e.g., “Draft Q3 board report from this sheet”); Aura will offer to transfer the session.

## Synthetic Teach Mode
During hackathons, reliably demoing multi-app workflows demands targeted data. Aura includes a synthetic data pipeline:
1. Record 3–5 manual runs through the target workflow (`scripts/record_workflow.py`).
2. Generate synthetic instruction variants with `scripts/generate_synthetic_instructions.py`.
3. Fine-tune SLM/VLM adapters in under an hour on a workstation with the provided scripts.
4. Deploy the freshly tuned models via ExecuTorch without changing application code.

## Security & Privacy
- All inference stays **on device**; no user data leaves the handset unless the user approves a cross-device transfer.
- Sensitive actions (payments, data deletion) invoke confirmation prompts enforced in the Actuator and Planner guardrails.
- Accessibility permissions are clearly explained and revocable.

## Roadmap
- Add speech-to-intent via on-device ASR.
- Expand Teach Mode into federated personalization.
- Explore iOS support via App Intents and Model Context Protocol once public SDKs are available.

## Contributing
1. Fork the repository and create feature branches.
2. Follow the included linting and formatting patterns (`./gradlew lintKotlin`, `ruff`, `black`).
3. Submit PRs with test results and benchmarks.

## License
This project is released under the terms of the included `LICENSE`. Ensure datasets used comply with their respective licenses.

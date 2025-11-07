# Aura Architecture Overview

Aura transforms a commodity Arm-powered mobile device into an agentic operating layer that performs the complete **Sense → Plan → Act** loop locally. This document captures the major components, their responsibilities, and the runtime interactions.

## System Goals
- Eliminate "app silos" by letting the agent orchestrate multiple apps on behalf of the user.
- Preserve privacy and responsiveness by running all inference on-device.
- Exploit Arm's heterogeneous compute stack (CPU + GPU + NPU) through ExecuTorch delegation.
- Maintain a modular, extensible agent hierarchy that can absorb new specialist skills.

## High-Level Flow
```
          ┌────────────┐
          │   Speech   │  (Optional on-device ASR)
          └─────┬──────┘
                │ user intent
        ┌───────▼────────┐
        │ Orchestrator   │  Small Language Model + Guardrails
        │ (Planner Agent)│
        └───────┬────────┘
   information  │ commands
        ┌───────▼─────────┐          ┌───────────────────┐
        │ Perception Agent │          │ Continuity Agent  │
        │  (Hybrid XML/VLM)│          │  (Cross-device)   │
        └───────┬─────────┘          └───────┬───────────┘
                │ answers                        │
        ┌───────▼─────────┐                      │
        │ Actuator Agent  │◄─────────────────────┘
        │ (Accessibility) │   execute taps/swipes/text
        └─────────────────┘
```

1. **Intent ingestion** (voice or text) enters the Orchestrator.
2. The Orchestrator decomposes the goal into steps using the SLM and policy rules.
3. Perception Agent resolves UI state queries via XML parsing or VLM fallback.
4. The Actuator executes guarded UI commands via the Accessibility Service.
5. The Continuity Agent monitors for tasks that merit device handoff.

## Component Responsibilities

### Orchestrator (Planner)
- Runs on the **Arm Cortex-A CPU** with **KleidiAI**-optimized kernels.
- Implements a planning state machine: goal parsing, subtask expansion, execution monitoring, and error recovery.
- Coordinates Perception and Actuator agents via gRPC-like in-process calls.
- Enforces safety policies (confirmation prompts, domain restrictions).
- Maintains task graphs that track dependencies and progress.

### Perception Agent
- Offers `queryScreen()` APIs to answer questions like *"Is there a 'Reserve' button visible?"*.
- **XML-first**: Parses the `AccessibilityNodeInfo` tree for text, bounds, hints.
- **Vision-fallback**: Captures a screenshot and runs a quantized VLM via ExecuTorch on the **Mali GPU**. Heavy subgraphs are delegated to **Ethos-N** through **Arm NN**.
- Returns structured responses (element IDs, coordinates, confidence scores).

### Actuator Agent
- Minimal reasoning: executes `CLICK`, `SWIPE`, `INPUT_TEXT`, and `SYSTEM_INTENT` actions specified by the Orchestrator.
- Wraps Android’s `AccessibilityService#performAction` and `GestureDescription`.
- Maintains a safety layer for sensitive actions (requires explicit Planner flag + user consent).
- Generates execution telemetry for diagnostics and learning.

### Continuity Agent
- Uses Google’s **Cross-Device SDK** to discover nearby devices and initiate multi-device sessions.
- Receives cues from the Orchestrator when a task should migrate (e.g., large text composition, spreadsheet editing).
- Serializes task intent and relevant context (documents, state) and hands off to the companion Aura client.

## ExecuTorch Deployment Strategy
- Models are exported with `torch.export()` and compiled into `.pte` assets.
- Runtime preferences specify delegate order: `CPU(KleidiAI) → GPU(ACL) → NPU(Arm NN)`.
- Execution contexts are pre-warmed at application startup to minimize cold latency.
- Tensor utilities handle NHWC ↔ NCHW conversions consistent with Arm’s optimized kernels.

## Data Flow & Storage
- Task histories and telemetry are stored locally in encrypted form (e.g., Room + SQLCipher).
- Model caches are stored under `models/compiled/` and packaged into the APK as assets.
- No cloud storage is required. Optional synchronization for personalization is a future roadmap item.

## Extensibility Points
- **Specialist Agents**: Add new modules (e.g., calendar agent) that register with the Orchestrator dispatcher.
- **Teach Mode**: Provides hooks for uploading newly fine-tuned SLM/VLM checkpoints without code changes.
- **Policy Updates**: Guardrail logic is data-driven, enabling rapid updates through configuration files.

## Error Handling
- Planner monitors each subtask; on failure, it either retries, replans, or surfaces the error to the user.
- Perception fallbacks log missing XML nodes for dataset improvement.
- Actuator validates post-conditions (e.g., verify text input succeeded via Perception).

## Future Work
- Integrate on-device speech recognition and TTS for hands-free operation.
- Build federated Teach Mode to enable privacy-preserving personalization.
- Adapt architecture for iOS using App Intents and Model Context Protocol.

Aura’s architecture blends advanced AI planning with pragmatic mobile engineering. By aligning model workloads with Arm hardware strengths, Aura achieves the responsiveness and reliability necessary for a next-generation agentic experience.

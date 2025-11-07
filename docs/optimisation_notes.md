# Arm Optimisation Notes

Aura is engineered to exploit the full Arm heterogeneous compute stack. This cheat sheet lists the critical optimisations, tuning knobs, and verification steps that keep latency within the threshold for real-time agentic interactions.

## ExecuTorch Configuration
- Export models with `torch.export()` and bundle with **ExecuTorch 1.0**.
- Preferred delegate order: `cpu` (KleidiAI) → `gpu` (Arm Compute Library / Vulkan) → `npu` (Arm NN). Configure using the model’s runtime YAML manifest.
- Enable operator fusion passes (`executorch.optimize.fuse_linear_bias`) before export.
- Preallocate tensor arenas to avoid heap churn on Android (configured in Kotlin via `ExecutionConfig`).

## CPU (Arm Cortex-A + KleidiAI)
- Quantize SLM weights to **INT8** using static calibration; keep KV cache residuals in FP16 for accuracy.
- Use **KleidiAI** kernels for attention, RMSNorm, and GEMM – delivers up to 30× uplift over naive FP32.
- Pin planner inference threads to big cores using `setThreadAffinity` in JNI/NDK layer exposed to Kotlin.
- Measure token throughput with `scripts/eval_pipeline.py --benchmark slm`.

## GPU (Arm Mali + Arm Compute Library)
- For VLM fallback, map convolutional / vision subgraphs to the GPU delegate.
- Prefetch textures via Vulkan to warm caches before running the VLM; reuse command buffers.
- Monitor GPU utilization with Android’s `systrace` to ensure execution stays under thermal throttling limits.

## NPU (Arm Ethos-N + Arm NN)
- Identify INT8-friendly subgraphs (e.g., image encoders) and mark them as preferred for NPU offload.
- Use Arm NN’s delegate profiler to verify operator placement; adjust quantization ranges when ops fall back unexpectedly.
- Keep batch size at 1 with dynamic shapes disabled for deterministic performance.

## Memory Optimisation
- Apply **structured pruning** (magnitude or movement) to reduce model footprint before quantization.
- Use **KV cache reuse** for the SLM planner to avoid recomputation during replanning loops.
- On Android, store compiled models under `app/src/main/assets/models/` and memory-map them instead of fully loading into RAM.

## Battery & Thermal Management
- Gate heavy perception queries: only invoke the VLM when XML parsing returns low-confidence answers.
- Pause inference when device temperature crosses configured thresholds; surface notification to the user.
- Coalesce UI actions from the Actuator to minimize wake-lock churn.

## Tooling & Verification
- Profiling:
  - Android Studio Profiler for CPU/GPU.
  - `perfetto` / `systrace` for system-wide traces.
  - ExecuTorch built-in profiling hooks (`ExecutionResult.profile`).
- Regression Benchmarks:
  - Maintain golden latencies in `tests/evaluation_scenarios/benchmarks.json`.
  - Fail CI when latency regressions >10% or battery drain >5% baseline.

## Future Research Ideas
- Evaluate **SME2/SVE2** accelerations in next-gen Armv9 cores via KleidiAI experimental kernels.
- Experiment with **FOX-NAS** or similar HW-aware NAS to co-design perception architectures.
- Investigate **low-rank adaptation caching** for on-device personalization without retraining full models.

These optimisations allow Aura to remain responsive while executing complex multi-app workflows locally, satisfying both user experience and power constraints on mobile Arm platforms.

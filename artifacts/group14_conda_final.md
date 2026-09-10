# FastWAM Group 14: Sequential Five-Op Fusion

Measured on 2026-09-07 in the FastWAM conda environment, on one H100 80GB
using BF16 and the release checkpoint. Both groups ran in the same process,
with 10 warmups and 100 measured requests each. All latencies include VAE.

| Group | Action Steps | Compile (incl. VAE) | GPU Count | Execution | Fusion | Mean Latency | Speedup |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 13 | 1 | CUDA Graph | 1 | Asynchronous | Yes | 12.222 ms | 24.52x |
| 14 | 1 | CUDA Graph | 1 | Sequential | Yes | 14.676 ms | 20.42x |

Speedup uses the historical original FastWAM baseline of 299.658 ms; group 1
was not rerun. Group 13 is a fresh measurement in this comparison. Group 14
is 20.08% slower than group 13; asynchronous execution reduces latency by
16.72% compared with sequential execution.

Both groups use `five_ops_pair_norm`: paired Q/K RMSNorm with FP32 statistics,
FP64 RoPE, BF16 modulation/gate, all-True mask removal, and packed QKV/KV
projections. Group 14 executes all Video blocks, retaining per-layer KV, then
all Action blocks on one compute stream. It retains the final Video block's
computation. LayerNorm and VAE use the same implementations as group 13.

Timed: CPU input copies and RNG, VAE, preparation, Video/Action, scheduler,
and CPU output. Text encoding, model loading, graph setup and validation are
excluded. No image-latent caching is used. Group 14's graph captures the
sequential fused implementation, rather than the original unfused MoT methods.

Validation passed for original inputs, changed image/proprio and changed
noise seed. Both groups have maximum absolute error 0.015625 versus original
eager (atol=0.02, rtol=0), and exact graph/eager parity for their implementations.
Configuration and a focused control-flow check also verified the new group,
all-Video-before-Action order, and correct per-layer KV handoff.

## Reproduce

```bash
conda activate FastWAM
PYTHONDONTWRITEBYTECODE=1 python scripts/bench_latency.py \
  ckpt=checkpoints/fastwam_release/libero_uncond_2cam224.pt \
  '+BENCH.groups=[13,14]' \
  +BENCH.output_json=artifacts/group14_conda_final.json
```

Interpreter: `/mnt/miaohua/charles/envs/miniconda3/envs/FastWAM/bin/python`.
Raw samples/configuration: `group14_conda_final.json`.
Log: `group14_conda_final.log`.

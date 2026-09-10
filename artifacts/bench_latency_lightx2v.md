# FastWAM TP2/SP2 Optimization

Measured on 2026-09-07 in the FastWAM environment, PyTorch 2.7.1+cu128,
BF16, H100 80 GB GPUs 0/1, with
`checkpoints/fastwam_release/libero_uncond_2cam224.pt`.
All measurements include input copies, noise generation, VAE, preparation,
Video/Action, communication, scheduler updates, and CPU output. Prompt encoding,
model loading, validation, graph setup, and optional profiling are excluded.
Each variant uses 10 warmups and 100 measured requests.

## Retained Results

| Group | Before mean ms | Optimized mean ms | Optimized p50 ms | Optimized p90 ms | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| 9: TP2 | 44.124 | 27.493 | 25.806 | 31.403 | 1.605x |
| 10: Ulysses SP2 | 24.142 | 18.537 | 17.122 | 21.449 | 1.302x |

The before/after pairs were measured in the same workers with the same loaded
checkpoint. TP mean latency decreased by 37.7%, SP by 23.2%. The distributions
still show substantial variation; all 100 samples, including outliers, are
retained. Minimum latency also improved: TP 38.318 to 25.179 ms, SP 21.293 to
16.742 ms. These changes do not provide a 2x end-to-end speedup over one GPU.

Group 7 single GPU fusion measured 17.321 ms; group 8 two GPU expert pipeline
fusion measured 15.431 ms in the same run. Group 8 remains the fastest measured
configuration. Groups 7/8 retain their existing implementation and numbering.

`bench_latency_lightx2v.json` selects the retained PyTorch NCCL variants from
the raw `bench_latency_lightx2v_v2.json` report's `operator_only` records. It is
a projection of measured samples, not an additional benchmark run. The raw
V2 top-level TP/SP results correspond to the rejected direct-NCCL experiment.

## Changes

The reference was the local LightX2V_fastwam working tree at revision
`101a70dd`, particularly:

- `lightx2v/common/ops/attn/ulysses_prepost.py`
- `lightx2v/common/ops/attn/kernels/ulysses_layout.py`
- `lightx2v/common/ops/mm/mm_weight.py` (`MMWeightTP`)
- `lightx2v/models/networks/wan/weights/transformer_weights.py` (TP RMSNorm)

FastWAM now follows the fused Ulysses pre/post approach with batch-aware Triton
wire-layout kernels. Q/K/V use one packed all-to-all before attention and one
all-to-all after attention. No communication quantization is used.

Both optimized runners combine self-attention QKV projections and cross-attention
KV projections, fuse RMSNorm with RoPE where applicable, and reuse the existing
modulation/gate fusion. RoPE retains the original FP64 arithmetic, with explicit
BF16 intermediate rounding and FP fusion disabled. Per-request computation is
still timed; projected context K/V are not cached between observations.

TP combines Q/K square sums into one FP32 all-reduce per attention module.
This reduces collective calls from 420 to 300 per rank per request while retaining
global normalization across all heads. SP retains 120 all-to-all calls and one
final output all-gather. Each GPU still runs Video and Action asynchronously on
two logical compute streams, with separate expert NCCL groups, in one local
CUDA Graph. Each rank retains its own VAE computation.

In a single traced request on rank 0, the kernel count fell from 6142 to 3082
for TP and from 5553 to 2936 for SP. Trace timings include profiler effects;
the table above uses the separate unprofiled measurement loop.

## Correctness and Experiments

The original input, changed image/proprio, and changed noise seed all pass.
Maximum absolute error against the unsharded eager model remains 0.015625 for
both optimized methods, with the existing absolute tolerance 0.02 and relative
tolerance zero. Captured outputs match the corresponding optimized eager
outputs exactly in all three cases. This is numerical validation, not a rollout
success-rate evaluation.

Three CUDA tests cover strided QKV layouts, batch sizes 1/2, attention layouts,
and fused normalization/RoPE. Syntax and whitespace checks also pass.

The first optimization round, before RMSNorm/RoPE fusion, measured TP
41.787 to 31.900 ms and SP 25.576 to 23.733 ms. Raw data and traces are in
`bench_latency_lightx2v_v1.json` and its trace files.

A current-stream NCCL experiment was also checked with nonzero all-reduce and
all-to-all inputs, then tested inside the real model graphs. It measured TP
30.471 ms and SP 18.603 ms, compared with 27.493/18.537 ms using PyTorch NCCL.
It was removed from the implementation because it offered no end-to-end benefit.
Its measurements and traces remain in the raw V2 report for comparison.

## Reproduce

```bash
conda activate FastWAM
python scripts/bench_latency.py \
  ckpt=checkpoints/fastwam_release/libero_uncond_2cam224.pt \
  '+BENCH.groups=[9,10]' \
  +BENCH.parallel_compare=true \
  +BENCH.output_json=artifacts/bench_latency_lightx2v_repeat.json
```

The optimized TP/SP runners are enabled by default. Set
`+BENCH.parallel_optimized=false` to run the original implementations alone.
`+BENCH.parallel_compare=true` runs both in the same workers and stores the
original result under `baseline`. `+BENCH.parallel_profile=true` exports one
trace per variant/rank after its timed measurements; it is off by default.

```bash
python tests/test_parallel_layout.py
```

# LightX2V FastWAM: Fourteen Groups

Groups 1-10 retain the user's historical results. Groups 11-13 use the native,
custom, and best hybrid backend measurements from the three-round comparison
on 2026-09-07 in the LightX2V uv environment. This update fixes their group IDs;
it does not rerun or alter those measurements. Group 14 adds a fresh sequential
measurement with the same operators as group 13. All request latencies include
VAE. This table combines measurement batches.

Speedup = 260.883 / mean latency. Compared to Original FastWAM = 299.658 / mean latency.

| Group | Action Steps | Compile (incl. VAE) | GPU Count | Execution | Fusion | Mean Latency | Speedup | Compared to Original FastWAM |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 10 | None | 1 | Sequential | No | 260.883 ms | 1.00x | 1.15x |
| 2 | 10 | CUDA Graph | 1 | Sequential | No | 104.759 ms | 2.49x | 2.86x |
| 3 | 1 | None | 1 | Sequential | No | 50.770 ms | 5.14x | 5.90x |
| 4 | 1 | CUDA Graph | 1 | Sequential | No | 25.336 ms | 10.30x | 11.83x |
| 5 | 1 | CUDA Graph | 1 | Asynchronous | No | 18.541 ms | 14.07x | 16.16x |
| 6 | 1 | CUDA Graph | 2 | Asynchronous | No | 16.472 ms | 15.84x | 18.19x |
| 7 | 1 | CUDA Graph | 1 | Asynchronous | Yes | 13.709 ms | 19.03x | 21.86x |
| 8 | 1 | CUDA Graph | 2 | Asynchronous | Yes | 12.241 ms | 21.31x | 24.48x |
| 9 | 1 | CUDA Graph | 2 | SP2, LightX2V optimized | Yes | 18.808 ms | 13.87x | 15.93x |
| 10 | 1 | CUDA Graph | 2 | TP2, LightX2V optimized | Yes | 32.100 ms | 8.13x | 9.34x |
| 11 | 1 | CUDA Graph | 1 | Asynchronous, five-op native | Yes | 12.619 ms | 20.67x | 23.75x |
| 12 | 1 | CUDA Graph | 1 | Asynchronous, five-op custom | Yes | 12.247 ms | 21.30x | 24.47x |
| 13 | 1 | CUDA Graph | 1 | Asynchronous, five-op best combination | Yes | 11.875 ms | 21.97x | 25.23x |
| 14 | 1 | CUDA Graph | 1 | Sequential, five-op best combination | Yes | 14.270 ms | 18.28x | 21.00x |

## Same-Process Backend Comparison

One H100 80GB, BF16, the same checkpoint, real prompt encoding cached once,
and the original native LightX2V VAE. Each backend ran three rounds, each with
10 warmups and 100 measured requests. Order alternated forward/reverse/forward.
Primary latency uses synchronized wall time and includes input copies, CPU RNG,
VAE, preparation, Video/Action, scheduler and CPU output. Model loading, text
encoding, graph capture and validation are outside timing.

| Backend | Q/K RMSNorm | RoPE | LayerNorm | Modulation | Mean incl. VAE |
| --- | --- | --- | --- | --- | --- |
| native | LightX2V | LightX2V FP32 | LightX2V | LightX2V | 12.619 ms |
| custom | Local strided | Local FP64 | Native torch | Local BF16 | 12.247 ms |
| hybrid | Local strided | Local FP64 | LightX2V | LightX2V | 11.914 ms |
| hybrid_affine (group 13) | Local strided | Local FP64 | LightX2V | Local BF16 | 11.875 ms |

Every backend uses packed native MMWeight QKV/KV projections, static all-True
mask removal, and a local BF16 residual-gate kernel. The available fused
GEMM/gate implementation requires MXFP8, so it does not apply to this BF16 run.
The strided Q/K kernel avoids contiguous copies made by the native wrapper.
The FP64 RoPE kernel uses flat element indexing and preserves the original
rotation precision. No VAE or per-request context-KV caching was introduced.

Selected backend round means: 11.868413, 11.872474, 11.884415 ms. The pooled
300-sample mean is 11.875101 ms; p50 11.871114 ms, p90 11.892061 ms.
Group 7 in this same process measured 13.732 ms, so group 13 is 1.156x
faster and reduces request latency by 13.52%. The table above retains the
historical group 7 value as requested.

All twelve backend trials passed the original, changed-observation, and
changed-noise cases. Selected backend maximum absolute error versus native
eager is 0.015625 (atol=0.02, rtol=0); graph/eager parity is exact.
Focused checks also covered native and custom Q/K normalization on packed
views, FP64 RoPE, BF16/FP16 gate, mixed/optional projection biases, and masks.
Policy task success was not measured.

## Sequential Group 14

Group 14 uses exactly the same `hybrid_affine` operators as group 13, including
native LightX2V LayerNorm, MMWeight, attention and VAE. All Video blocks finish
before Action starts on one compute stream. Per-layer Video KV is retained for
Action, and the final Video block is still executed. VAE is captured and runs
on every request; no cached image latents are used.

The new uv run compared groups 13/14 in one process on one H100 80GB, BF16,
with 10 warmups and 100 measured requests each:

| Group | Execution | Mean incl. VAE | p50 | p90 |
| --- | --- | --- | --- | --- |
| 13 | Asynchronous | 11.900 ms | 11.897 ms | 11.924 ms |
| 14 | Sequential | 14.270 ms | 14.270 ms | 14.340 ms |

Sequential execution increases latency by 19.92% in this comparison. The full
table retains the previous three-round group 13 mean of 11.875 ms.
Both groups passed three input cases with maximum absolute error 0.015625
versus native eager (atol=0.02, rtol=0), and exact graph/eager parity. Kernel
metadata matches between the groups. A focused check also verified sequential
ordering and per-layer KV handoff. Raw data: `lightx2v_group14_final.json`;
log: `lightx2v_group14_final.log`.

## Code and Reproduction

Only `scripts/fastwam/bench_latency.py` was changed inside LightX2V. It now
defaults to fourteen groups, with groups 11-14 on a single GPU. Existing groups
1-10 keep their operator paths. Results record the kernel backend and both
speedup denominators. Report files remain outside the LightX2V repository.

Interpreter: `/mnt/miaohua/charles/codes/LightX2V_fastwam/.venv/bin/python`.
Checkpoint: `/mnt/lm_data_afs/charles/codes/FastWAM/checkpoints/fastwam_release/libero_uncond_2cam224.pt`.

Run from `/mnt/miaohua/charles/codes/LightX2V_fastwam`:

```bash
uv run --no-project --python .venv/bin/python scripts/fastwam/bench_latency.py \
  '+BENCH.groups=[11,12,13,14]' \
  +BENCH.output_json=/tmp/lightx2v_five_ops.json
```

Groups are fixed: 11=`native`, 12=`custom`, 13/14=`hybrid_affine` with
asynchronous/sequential execution respectively. Use `+BENCH.groups=[13,14]`
for the scheduling comparison. The former
`group11_backend`, `group11_compare`, and `group11_rounds` overrides have been
removed; use group selection instead. Configuration and dispatch checks passed
in the uv environment, including a non-default group order and single-GPU metadata.

The historical measurements in this report used the former `group11_compare`
mode. Each backend's raw samples and checks are retained unchanged in
`lightx2v_group11_compared.json` and `lightx2v_group11_compared.group11_trials.json`.
Their original group ID was 11; map trial `backend` to the new fixed group IDs.
Log: `lightx2v_group11_compared.log`. The intermediate `hybrid` candidate is
retained in this historical comparison only, without a public group ID.

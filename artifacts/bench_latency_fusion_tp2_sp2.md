# FastWAM Fusion, TP2, and SP2 Latency

Measured on 2026-09-07 using the FastWAM conda environment, PyTorch 2.7.1+cu128,
BF16, H100 80 GB GPUs 0/1 with NVLink, and
`checkpoints/fastwam_release/libero_uncond_2cam224.pt`.
Input image: `[1, 3, 224, 448]`; Video tokens: 98; Action tokens: 32;
cached text context: `[1, 128, 4096]`. Text encoder remains resident.
Each group uses 10 warmups and 100 measured requests.

## Results

| Group | Execution | Mean ms | p50 ms | p90 ms | Speedup vs 5 |
| --- | --- | ---: | ---: | ---: | ---: |
| 5 | Single GPU async | 18.162 | 18.142 | 18.172 | 1.000x |
| 6 | Two GPU expert pipeline | 16.058 | 16.058 | 16.072 | 1.131x |
| 7 | Single GPU async + fusion | 17.393 | 17.336 | 17.395 | 1.044x |
| 8 | Two GPU expert pipeline + fusion | 15.455 | 15.446 | 15.468 | 1.175x |
| 9 | TP2, local Video/Action async | 42.016 | 39.872 | 45.901 | 0.432x |
| 10 | Ulysses SP2, local Video/Action async | 23.090 | 21.768 | 25.961 | 0.787x |

Groups 7/8 are the fusion groups; TP2/SP2 are groups 9/10.
Fusion reduces mean latency by 4.2% on one GPU and 3.8% on two GPUs.
The two GPU fused pipeline is 1.125x faster than the single GPU fused pipeline.
TP2/SP2 execute correctly but are slower for this workload and implementation.

## Timing and Correctness

All groups use one action denoising step, no_grad, and explicit CUDA Graphs.
Input copies, noise generation, VAE, proprio/context preparation, transformer
layers, scheduler updates, communication, and CPU action output are timed.
Model loading, prompt encoding, validation, and graph setup are excluded.

Groups 6/8 contain 63 graphs per request: two preparation graphs, 30 Video
phases, 30 Action phases, and the final Video block completion. Video K/V
copies and cross-device events execute outside capture.

Groups 9/10 use two NCCL processes. Each GPU runs Video and Action on separate
streams, with separate expert communicators, inside one local CUDA Graph.
Each rank executes its own VAE and input/output staging. Reported distributed
time is the maximum rank wall time; the coordination barrier and timing
reduction are excluded. Both ranks produce the complete action output.
Workers execute SP2 before TP2 because TP mutates model weights; the result
table retains the requested group order.

Validation changes observations independently from noise, for three cases.
Groups 5-8 match the unsharded eager reference exactly in all cases.
Groups 9/10 have maximum absolute error 0.015625 in each case, using an
explicit absolute tolerance of 0.02 and relative tolerance of zero, consistent
with the reference parallel benchmark. This is a looser absolute tolerance
than the original benchmark's 0.01 and is recorded in JSON. Graph outputs
match the corresponding sharded eager outputs exactly in all cases.
This numerical check does not establish downstream policy success rates.

## Interpretation

Splitting the Video and Action experts across GPUs leaves the entire Video
branch on one GPU. Previous component measurements in `pipeline_analysis.md`
found roughly 4.16 ms for VAE/preparation and 11.73 ms for the Video branch;
Action and KV copies were mostly overlapped. Those historical measurements
explain why this expert split does not approach 2x over an already asynchronous
single GPU pipeline. They used the previous 123-graph layout.

A short verification of the updated profiling script used 10 measured samples
with the 63-graph layout (`pipeline_profile_merged_graphs.json`). The baseline
and fixed-input ablations all passed exact output checks. VAE/preparation was
4.161 ms and the live Video branch was 11.696 ms. Compute-only latency was
15.872 ms; the matching instrumented control was 15.878 ms; omitting KV copies
and waits with fixed, preloaded KV was 15.864 ms. VAE plus Video alone on its
original stream took 15.859 ms. This repeats the earlier critical-path finding:
KV handoffs are largely hidden, while the Video branch and VAE remain exposed.
These no-copy ablations are not valid inference paths for changing observations.

The new TP implementation performs seven all-reduces per expert block:
four Q/K normalization reductions, two attention output reductions, and one
FFN output reduction. Across two experts and 30 layers this is 420 collective
calls per request. SP packs Q/K/V into one all-to-all and applies another
after attention: 120 all-to-all calls plus one final output all-gather.
These counts are per rank and exclude untimed benchmark coordination.

With SP2, each rank has only 49 Video tokens and 16 Action tokens. It retains
the full expert weights and repeats context K/V projections and VAE work.
Communication, synchronization, small matrix efficiency, and repeated work
are plausible reasons that splitting computation loses to the local pipeline.
The measured end-to-end times do not isolate the contribution of each cause;
kernel traces would be needed for a quantitative attribution. They do not
establish a general limit on optimized TP/SP implementations.

## Reproduce

```bash
conda activate FastWAM
python scripts/bench_latency.py \
  ckpt=checkpoints/fastwam_release/libero_uncond_2cam224.pt \
  '+BENCH.groups=[5,6,7,8,9,10]' \
  +BENCH.output_json=artifacts/bench_latency_fusion_tp2_sp2_repeat.json
```

Omit the group override to run all ten groups. To measure only fusion, select
`'+BENCH.groups=[7,8]'`. Raw samples and configuration are stored in
`bench_latency_fusion_tp2_sp2.json`.

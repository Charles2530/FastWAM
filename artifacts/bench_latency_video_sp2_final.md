# Final Video-SP2 / Full-Action Experiment

Measured on 2026-09-07 in the FastWAM environment on H100 80 GB GPUs 0/1,
PyTorch 2.7.1+cu128, BF16, using
`checkpoints/fastwam_release/libero_uncond_2cam224.pt`.
All five groups were measured in the same invocation with 10 warmups and
100 samples per group. Each uses one action denoising step, explicit CUDA Graphs
including VAE, and fusion. Input copies, RNG, VAE, preparation, expert computation,
communication, scheduler updates, and CPU output are timed. Prompt encoding,
model loading, validation, and graph capture are excluded.

## Results

| Group | Execution | Mean ms | P50 ms | P90 ms | Collectives per rank |
| --- | --- | ---: | ---: | ---: | ---: |
| 7 | Single GPU async + fusion | 17.368 | 17.328 | 17.366 | 0 |
| 8 | Two GPU expert pipeline + fusion | 15.449 | 15.449 | 15.468 | 0 (peer KV copies) |
| 11 | Video and Action SP2, optimized | 18.232 | 16.825 | 21.254 | 121 |
| 13 | Video SP2 + full Action, separate KV gather | 17.977 | 16.708 | 21.040 | 90 |
| 14 | Video SP2 + full Action, shared QKV gather | 18.782 | 18.275 | 21.131 | 60 |

Group 8 remains the fastest measured configuration. Group 13's mean is only
1.4% below group 11's, with substantial sample variation, so this run does not
establish a stable improvement. Neither new group beats single GPU fusion or
the two GPU expert pipeline in mean latency. Group 14 reduces collective count
but does not improve end-to-end latency. Its minimum is 16.400 ms, still above
group 8's mean of 15.449 ms. No additional optimization variants were pursued.

## The Two New Groups

Both 13 and 14 retain all existing QKV/KV projection, RMSNorm/RoPE, affine,
and communication-layout fusion. Both partition Video's 98 tokens across two
GPUs while each GPU executes the complete 32-token Action expert. Each rank
still runs its own VAE. Video and Action execute on separate local streams.
Action tokens and action outputs need no sequence-parallel communication.
The complete Video KV required by Action is freshly assembled on every request.

Group 13 releases local Video KV before Video's QKV all-to-all. The Action
stream gathers the two sequence partitions while Video performs its Ulysses
attention using a separate communicator. Each of 30 layers performs two Video
all-to-alls and one packed KV all-gather, for 90 collectives per rank.

Group 14 replaces the Video QKV all-to-all and separate KV all-gather with one
packed QKV all-gather. A fused unpack kernel produces both the head-partitioned
QKV for Video attention and the complete KV for Action. Only Video's attention
output still needs an all-to-all, for 60 collectives per rank across 30 layers.

NCCL coordination used for validation and timing is excluded from these counts
and from request timing. Distributed latency is the maximum rank request time.

## Validation

Groups 13/14 pass the base input, changed image/proprio, and changed noise seed
checks. Maximum absolute error against unsharded eager inference is 0.015625
for all three cases, within the existing 0.02 absolute tolerance. Graph output
matches each new runner's eager output exactly. Group 11 has the same errors;
groups 7/8 match the unsharded reference exactly.

Five CUDA tests pass, including the new KV gather layouts, shared QKV unpack,
both rank head selections, batch sizes 1/2, existing Ulysses wire layouts,
and normalization/RoPE. Group-order checks ensure all SP variants execute before
TP modifies model weights. These are numerical tests, not policy rollout tests.

## Reproduce

```bash
conda activate FastWAM
python scripts/bench_latency.py \
  ckpt=checkpoints/fastwam_release/libero_uncond_2cam224.pt \
  '+BENCH.groups=[7,8,11,13,14]' \
  +BENCH.output_json=artifacts/bench_latency_video_sp2_repeat.json
```

Select `'+BENCH.groups=[13,14]'` for the new experiments only, or
`'+BENCH.groups=[8]'` for the fastest measured configuration. The default now
includes all 14 groups. Raw samples and configuration:
`bench_latency_video_sp2_final.json`.

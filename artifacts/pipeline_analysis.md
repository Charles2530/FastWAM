# FastWAM Pipeline Latency Diagnosis

Measured on 2026-09-07 in the FastWAM conda environment, PyTorch 2.7.1+cu128,
BF16, H100 80 GB GPUs 0 and 1 connected by NV18. Checkpoint:
`checkpoints/fastwam_release/libero_uncond_2cam224.pt`.
Synthetic image shape: [1, 3, 224, 448]; video tokens: 98; action tokens: 32.
Text embeddings are cached. Baselines include input staging and CPU output.
Each baseline uses 10 warmups and 50 measured requests. Microbenchmarks use
5 warmups and 50 samples. All baseline changed-input correctness checks passed
with zero maximum absolute error. Fixed-input ablations also matched exactly.

## End-to-End Results

| One-step mode | Mean (ms) | p50 (ms) |
| --- | ---: | ---: |
| Single GPU sequential graph | 24.543 | 24.535 |
| Single GPU pipeline graph | 18.164 | 18.155 |
| Two GPU pipeline | 16.091 | 16.092 |
| Two GPU pipeline, second process | 16.093 | 16.087 |

Two GPUs give 1.129x speedup over the existing single GPU pipeline, or 1.525x
over single GPU sequential execution. The single GPU pipeline already overlaps
Video and Action. A 2x improvement over 18.164 ms would require 9.082 ms.

## Critical Path

The second run records CUDA events inside the live two GPU pipeline:

| Interval | Mean (ms) |
| --- | ---: |
| VAE and video preparation | 4.162 |
| Video branch after preparation | 11.729 |
| Action branch, including KV waits | 11.657 |
| Entire pipeline, stream event interval | 15.899 |
| VAE/preparation + Video only, original stream | 15.890 |

The last row omits Action entirely yet takes almost the same time as the full
pipeline. Video and Action both remain sequential across their own 30 layers.
Moving Action to another GPU does not divide the Video computation between GPUs.

Isolated single-graph stages measured approximately 3.96 ms for VAE alone,
4.17 ms for VAE plus preparation, 11.67 ms for Video, and 8.39 ms for Action
with its head and preloaded KV. A simple perfect-overlap estimate is
`4.17 + max(11.67, 8.39) = 15.84 ms`, before input/output overhead. This is
an approximation from isolated measurements, not a hardware lower bound.
It explains the measured 16.09 ms without assuming large communication losses.
Input staging and CPU output measured approximately 0.14 and 0.03 ms separately.

## Communication and Submission Controls

| Compute-only experiment, first run | Mean (ms) |
| --- | ---: |
| Live two GPU pipeline | 15.913 |
| Reimplemented live control | 15.911 |
| Omit KV copies but keep waits | 15.936 |
| Preload KV and omit KV waits | 15.906 |

The second process reproduced the same result: 15.909, 15.909, 15.927, and
15.901 ms respectively. Omitting copies/waits produces no material improvement.
These ablations only apply to the same fixed observation with matching preloaded
KV; they are not valid inference methods for changing observations.

Each request transfers 36,126,720 bytes (34.45 MiB) of Video KV in 60 copies.
The 60 isolated copies cost about 0.66-0.67 ms, but their cost is almost fully
overlapped in the live pipeline. This isolated wall time includes host submission
and synchronization; it is not a measurement of peak NVLink bandwidth.

The dual pipeline submits 123 graphs and spends about 3.7-3.9 ms in its host
call. That interval overlaps GPU execution and must not be added to GPU time.
VAE/preparation + Video alone costs 15.890 ms with only 0.351 ms of host calls,
versus 15.888 ms for the full pipeline measured again after the components.
This argues against host submission being the current critical-path bottleneck.

## Stream Context Matters

Video's 60 small graphs took 13.24 ms in an isolated default-stream test but
11.748 ms on the original video stream. The latter matches the live 11.729 ms
branch and is close to the 11.663 ms monolithic graph. Default-stream fragment
measurements therefore must not be used to claim a 1.5-2 ms removable overhead
in the existing pipeline. Action's default-stream fragment measurement also
varied between runs; use its single-graph result for the stage-cost estimate.

## Next Experiments

Prioritize VAE and Video kernel profiling and acceleration, which shorten the
measured critical path. A more balanced distribution of Video computation,
such as tensor parallelism, needs its own latency and correctness evaluation.
Optimizing KV copies alone is unlikely to materially improve this workload.
Even after accelerating Video, the approximately 8.39 ms Action stage can become
the bottleneck; reaching 9.08 ms total requires improving more than communication.
Request-level pipelining or replication can improve throughput but does not
automatically halve the latency of one observation-to-action request.

## Reproduce

```bash
conda activate FastWAM
python scripts/profile_pipeline.py \
  ckpt=checkpoints/fastwam_release/libero_uncond_2cam224.pt \
  '+BENCH.groups=[4,5,6]' +BENCH.iters=50 \
  +BENCH.output_json=artifacts/pipeline_profile_repeat.json
```

Data: `pipeline_profile.json` and `pipeline_timeline.json`.
Chart: `pipeline_analysis.png`.

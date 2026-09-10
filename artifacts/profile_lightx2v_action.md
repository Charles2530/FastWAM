# LightX2V Action Component Timing

Measured 2026-09-07 in the LightX2V uv environment, PyTorch 2.7.1+cu128,
one H100 80GB, BF16, release checkpoint, horizon 32. Each component used
10 warmups and 100 CUDA-event measurements of explicit CUDA Graph replay.
All groups share one loaded model and the same observation/context/noise.

| Group | Operators | Complete Action | Action 30 blocks | Speedup vs group 5 |
| --- | --- | --- | --- | --- |
| 5 | Unfused | 8.850 ms | 8.738 ms | 1.00x |
| 7 | LightX2V kernel fusion | 5.023 ms | 4.914 ms | 1.76x |
| 13 | Best mixed five-op | 3.593 ms | 3.467 ms | 2.46x |

Complete Action includes schedule construction, Action preparation, all 30
blocks (including attention over Video KV), output head and scheduler update.
Video KV is precomputed from the same observation before this diagnostic timer.
VAE, Video computation, KV waiting, CPU noise generation/input copies/output
copies are excluded from the isolated Action measurement. These numbers do not
replace the official VAE-inclusive end-to-end benchmark.

Group 13 reduces isolated complete-Action time by 59.40% versus group 5 and
28.47% versus group 7. Isolated results match each implementation's full
pipeline output exactly; maximum error versus native eager is 0 for group 5
and 0.015625 for groups 7/13, within atol=0.02, rtol=0.

## Live Pipeline and Other Components

| Group | Video 30 blocks alone | VAE alone | Action body span in live pipeline |
| --- | --- | --- | --- |
| 5 | 12.020 ms | 4.000 ms | 14.024 ms |
| 7 | 7.793 ms | 4.006 ms | 9.165 ms |
| 13 | 6.292 ms | 4.004 ms | 7.442 ms |

The live Action span starts before its first block projection and ends after
its final block. It includes Video-KV waits and GPU resource contention, while
excluding Action preparation/head/scheduler. Two external CUDA Graph event
nodes instrument that interval. PyTorch 2.7 does not expose Event(external=True),
so the diagnostic records these events through cudaEventRecordWithFlags.

Do not interpret the difference between live span and isolated block time as
pure KV waiting: concurrent Video changes compute execution time too. Do not
sum isolated Video/Action costs to estimate the asynchronous end-to-end latency.
For group 13, Video's isolated blocks still take longer than Action's, and VAE
cost remains outside their overlap, limiting gains from optimizing Action alone.

Only diagnostic artifacts were added for this measurement; inference source and
the production benchmark timer were not changed. Raw data:
`profile_lightx2v_action.json`; log: `profile_lightx2v_action.log`.

Reproduce from the LightX2V repository:

```bash
uv run --no-project --python .venv/bin/python \
  /mnt/lm_data_afs/charles/codes/FastWAM/artifacts/profile_lightx2v_action.py
```

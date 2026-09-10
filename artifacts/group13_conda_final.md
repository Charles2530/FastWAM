# FastWAM Single-GPU Group 13

Final comparison on 2026-09-07. Both groups include the original VAE on every
request, captured in CUDA Graph. Group 14 has been removed. Group 13 uses one
GPU with asynchronous Video/Action streams and no distributed collectives.

| Group | Action Steps | Compile (incl. VAE) | GPU Count | Execution | Fusion | Mean Latency | Speedup vs Group 7 | p50 | p90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 7 | 1 | CUDA Graph | 1 | Asynchronous | Modulation/gate | 17.327 ms | 1.00x | 17.319 ms | 17.359 ms |
| 13 | 1 | CUDA Graph | 1 | Asynchronous | Five operations + paired Q/K norm | 12.208 ms | 1.42x | 12.206 ms | 12.235 ms |

Group 13 reduces mean request latency by 29.54%. These are measurements from
one process using the same checkpoint and observations, with 10 warmups and
100 measured requests per group. Group 1 was not rerun in this comparison.

## Timing and Environment

- Python: `/mnt/miaohua/charles/envs/miniconda3/envs/FastWAM/bin/python`.
- PyTorch 2.7.1+cu128, BF16, NVIDIA H100 80GB HBM3, cuda:0.
- Checkpoint: `checkpoints/fastwam_release/libero_uncond_2cam224.pt`.
- Image: [1, 3, 224, 448]; action horizon: 32; output: [32, 7].
- Original prompt encoder runs once; its weights stay resident.
- Timed: CPU noise generation, input copies, VAE, proprio/context preparation,
  Video/Action, scheduler, and CPU output. Devices synchronize around wall timing.
- Excluded: text encoding, model loading, graph setup, and validation.
- `group13_without_vae=false`; primary results never cache VAE latents.
- Peak allocated memory: group 7 23.253 GiB, group 13 26.864 GiB. Packed
  projection buffers coexist with original weights in this benchmark.

## Final Implementation

The default variant is `five_ops_pair_norm` in
`scripts/fastwam_single_gpu_ops.py`:

1. Fused RMSNorm with FP32 statistics and BF16 intermediate rounding. Q/K
   normalization shares one launch, including unequal cross-attention lengths.
2. Fused RoPE with FP64 rotation and the original output conversion order.
3. Fused modulation and residual gate with BF16 intermediate rounding.
4. Remove static boolean all-True attention masks. Masks with blocked positions
   and additive masks remain intact. Decisions occur before graph capture.
5. Pack self-attention QKV and cross-attention KV projections into GEMMs.
   Linear calls decrease from 600 to 420 per forward; LayerNorm stays separate.

Packed weights belong to the runner; the original model is not patched.

## Validation and Earlier Tuning

Three cases were checked: original inputs, changed image/proprio, and changed
noise seed. Group 7 matches the original eager output exactly. Group 13's
maximum absolute error is 0.015625 in all three cases, within atol=0.02,
rtol=0. Its captured and uncaptured implementations match exactly.
The five single-GPU kernel/mask/projection tests and five existing parallel
layout tests passed in the FastWAM environment. Configuration checks confirm
13 groups, single-GPU group 13, VAE-inclusive defaults, and rejection of group 14.
This validates numerical outputs; policy task success was not measured.

Earlier three-round tuning of the selected variant measured 12.409 ms mean
end-to-end including VAE, and 8.187 ms mean GPU graph time excluding VAE.
The latter also excludes CPU input/noise/output and is diagnostic only; it is
not the latency used in the comparison table. Its trace contains 1692 kernels
excluding VAE. Raw tuning data: `group13_conda_tuned.group13_trials.json`.

## Reproduce

```bash
conda activate FastWAM
PYTHONDONTWRITEBYTECODE=1 python scripts/bench_latency.py \
  ckpt=checkpoints/fastwam_release/libero_uncond_2cam224.pt \
  '+BENCH.groups=[7,13]' \
  +BENCH.output_json=artifacts/group13_conda_final.json
```

Raw final samples and configuration: `group13_conda_final.json`.
Log: `group13_conda_final.log`.

# LightX2V Native FastWAM Latency

Tested using `/mnt/miaohua/charles/codes/LightX2V_fastwam/.venv/bin/python` (Python 3.11, PyTorch 2.7.1+cu128), two H100 80GB GPUs with peer access.

Checkpoint: `libero_uncond_2cam224.pt`. BF16, no_grad, 224x448 images, 32 actions, cached real prompt `pick up the object`, action sigma shift 1.0. Each group uses 10 warmups and 100 measured requests. The text encoder remains resident.

End-to-end times include CPU input copies, CPU noise generation, VAE, preparation, Video/Action, scheduler, communication, and CPU action output. CUDA Graph setup and validation are excluded. Distributed measurements use the slower rank; timing barriers/reductions are excluded.

| Group | Action Steps | Compile (incl. VAE) | GPU Count | Execution | Fusion | Mean Latency | Speedup |
|---|---|---|---|---|---|---|---|
| 1 | 10 | None | 1 | Sequential | No | 260.883 ms | 1.00x |
| 2 | 10 | CUDA Graph | 1 | Sequential | No | 104.759 ms | 2.49x |
| 3 | 1 | None | 1 | Sequential | No | 50.770 ms | 5.14x |
| 4 | 1 | CUDA Graph | 1 | Sequential | No | 25.336 ms | 10.30x |
| 5 | 1 | CUDA Graph | 1 | Asynchronous | No | 18.541 ms | 14.07x |
| 6 | 1 | CUDA Graph | 2 | Asynchronous (split experts) | No | 16.472 ms | 15.84x |
| 7 | 1 | CUDA Graph | 1 | Asynchronous | Yes | 13.709 ms | 19.03x |
| 8 | 1 | CUDA Graph | 2 | Asynchronous (split experts) | Yes | 12.241 ms | 21.31x |
| 9 | 1 | CUDA Graph | 2 | SP2 native | No | 24.705 ms | 10.56x |
| 10 | 1 | CUDA Graph | 2 | TP2 native | No | 38.526 ms | 6.77x |
| 11 | 1 | CUDA Graph | 2 | SP2 LightX2V | Yes | 18.808 ms | 13.87x |
| 12 | 1 | CUDA Graph | 2 | TP2 LightX2V | Yes | 32.100 ms | 8.13x |
| 13 | 1 | CUDA Graph | 2 | Video SP2 + full Action / KV gather | Yes | 18.569 ms | 14.05x |
| 14 | 1 | CUDA Graph | 2 | Video SP2 + full Action / shared QKV | Yes | 17.640 ms | 14.79x |

## Interpretation

- Group 7 reduces single-GPU asynchronous latency by 26.1% relative to group 5.
- Group 8 reduces split-expert pipeline latency by 25.7% relative to group 6 and is the fastest measured configuration.
- Group 8 is 1.12x faster than the fused single-GPU pipeline; two GPUs do not deliver a 2x latency reduction.
- SP2/TP2 retain per-layer collectives and replicated VAE/preparation. With only 98 Video tokens and 32 Action tokens, communication dominates the saved compute. Groups 13/14 did not beat group 8.

## Validation And Scope

- All 14 retained results passed against native LightX2V serial inference on base inputs, changed image/proprio, and changed noise seed.
- Groups 1-6 match the native reference exactly; groups 7-14 have maximum absolute error 0.015625, below atol=0.02 with rtol=0.
- Captured output matches the same implementation's eager output exactly for single-device and distributed graphs. Split-expert groups 6/8 are checked against serial inference.
- The reference is LightX2V's native serial FastWAM path. This is not a new measurement of the original FastWAM `model.infer_action()` baseline; its per-call module traversal differs.
- Fused groups call LightX2V's QK RMSNorm, LayerNorm, affine and RoPE kernels. SP uses its Torch/Triton Ulysses layouts. TP uses MMWeightTP and packed FP32 normalization reductions. No quantization is enabled. The installed uv environment has no FlashAttention/FlashInfer/sgl-kernel package, so attention uses the repository's torch SDPA backend.
- Native and fused SP2 both pack QKV into one collective. Group 9/11 use 121 collectives per rank/request, TP groups 10/12 use 420/300, and Video-SP2 groups 13/14 use 90/60.
- Synthetic-context diagnostics failed the same strict tolerance; they are excluded from the retained results. The default real-prompt path passes.

## Provenance

Groups 1-8 are from the local run; 10/11/13/14 from the parallel run; 9/12 from the final targeted retest. All use the same uv environment, checkpoint, prompt, input seed, 10 warmups and 100 samples. Speedups use group 1 from the local run, not one all-groups invocation.

- [lightx2v_native_local.json](lightx2v_native_local.json)
- [lightx2v_native_parallel.json](lightx2v_native_parallel.json)
- [lightx2v_native_parallel_final.json](lightx2v_native_parallel_final.json)
- [Combined samples and configuration](lightx2v_native_summary.json)

Only `scripts/fastwam/bench_latency.py` was added to LightX2V. Existing LightX2V source and configuration files were not modified.

```bash
cd /mnt/miaohua/charles/codes/LightX2V_fastwam
PYTHONDONTWRITEBYTECODE=1 uv run --no-project --python .venv/bin/python \
  scripts/fastwam/bench_latency.py \
  ckpt=/mnt/lm_data_afs/charles/codes/FastWAM/checkpoints/fastwam_release/libero_uncond_2cam224.pt \
  '+BENCH.groups=[1,2,3,4,5,6,7,8,9,10,11,12,13,14]' \
  +BENCH.output_json=/mnt/lm_data_afs/charles/codes/FastWAM/artifacts/lightx2v_native_all.json
```


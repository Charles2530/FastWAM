"""Groups 13/14: fused single-GPU asynchronous/sequential Video/Action.

No model weights or global attention functions are patched. Packed projections
belong to each runner. Mask decisions are made before capture; real masked
positions are preserved. The primary result always includes the original VAE.
"""

from contextlib import nullcontext
import gc
import json
import logging
from pathlib import Path
import statistics
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
import triton
import triton.language as tl

from bench_latency import ActionRunner, FusedOpsMixin, _benchmark, _percentile
from fastwam_parallel_ops import norm_rope
from fastwam.models.wan22.wan_video_dit import flash_attention, rope_apply

logger = logging.getLogger(__name__)


@triton.jit
def _rms_pair(Q, K, WQ, WK, OQ, OK, NQ: tl.constexpr, D: tl.constexpr,
              QS: tl.constexpr, KS: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    is_q = row < NQ
    local_row = row if is_q else row - NQ
    x_ptr = Q if is_q else K
    w_ptr = WQ if is_q else WK
    y_ptr = OQ if is_q else OK
    stride = QS if is_q else KS
    col = tl.arange(0, BLOCK)
    dtype = Q.dtype.element_ty
    x = tl.load(x_ptr + local_row * stride + col, col < D, 0).to(tl.float32)
    scale = tl.rsqrt(tl.sum(x * x, axis=0) / D + EPS)
    normalized = (x * scale).to(dtype).to(tl.float32)
    weight = tl.load(w_ptr + col, col < D, 0).to(tl.float32)
    tl.store(y_ptr + local_row * D + col, normalized * weight, col < D)


def rms_pair(q, k, nq, nk):
    if q.shape[-1] != k.shape[-1] or nq.eps != nk.eps:
        raise ValueError("Paired RMSNorm requires equal widths and eps")
    oq, ok = (torch.empty(x.shape, device=x.device, dtype=x.dtype) for x in (q, k))
    d = q.shape[-1]
    _rms_pair[(q.numel() // d + k.numel() // d,)](
        q, k, nq.weight, nk.weight, oq, ok, q.numel() // d, d,
        q.stride(1), k.stride(1), nq.eps, triton.next_power_of_2(d),
        enable_fp_fusion=False)
    return oq, ok


@triton.jit
def _rope_fp64(X, FREQ, Y, PAIRS: tl.constexpr, D: tl.constexpr,
               S: tl.constexpr, XS: tl.constexpr, HD: tl.constexpr,
               BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < PAIRS
    row, col = i // (D // 2), (i % (D // 2)) * 2
    offset = (row % S) * HD + col % HD
    real = tl.load(FREQ + offset, valid, 0).to(tl.float64)
    imag = tl.load(FREQ + offset + 1, valid, 0).to(tl.float64)
    x0 = tl.load(X + row * XS + col, valid, 0).to(tl.float64)
    x1 = tl.load(X + row * XS + col + 1, valid, 0).to(tl.float64)
    # PyTorch's CUDA cast from double to BF16/FP16 goes through float32.
    tl.store(Y + row * D + col, (x0 * real - x1 * imag).to(tl.float32), valid)
    tl.store(Y + row * D + col + 1, (x0 * imag + x1 * real).to(tl.float32), valid)


def rope(x, freqs, head_dim=128):
    if x.ndim != 3 or x.stride(-1) != 1 or not freqs.is_complex():
        raise ValueError("RoPE requires [B,S,D] row-contiguous input and complex frequencies")
    output = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    _rope_fp64[(triton.cdiv(x.numel() // 2, 256),)](
        x, torch.view_as_real(freqs), output, x.numel() // 2, x.shape[-1],
        x.shape[1], x.stride(1), head_dim, 256, enable_fp_fusion=False)
    return output


class FusedRMSNorm(nn.Module):
    """One Triton reduction with FP32 statistics and BF16 intermediate casts."""

    def __init__(self, norm):
        super().__init__()
        self.register_buffer("weight", norm.weight.detach())
        self.eps = norm.eps

    def forward(self, x):
        return norm_rope(x, self)


class PackedLinear(nn.Module):
    """One GEMM, with strided views into its output instead of split copies."""

    def __init__(self, linears):
        super().__init__()
        self.widths = tuple(layer.out_features for layer in linears)
        self.register_buffer("weight", torch.cat([layer.weight.detach() for layer in linears]))
        biases = [layer.bias for layer in linears]
        bias = None if all(b is None for b in biases) else torch.cat([
            layer.weight.new_zeros(layer.out_features) if layer.bias is None else layer.bias.detach()
            for layer in linears])
        self.register_buffer("bias", bias)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias).split(self.widths, dim=-1)


def is_unmasked(mask):
    """Only boolean all-True masks are removable; call outside graph capture."""
    return mask is None or (mask.dtype == torch.bool and bool(mask.all().item()))


VARIANTS = {
    "affine_only": dict(norm=False, rotary=False, packed=False, masks=False, combined=False),
    "four_ops": dict(norm=True, rotary=True, packed=True, masks=False, combined=True),
    "five_ops_separate": dict(norm=True, rotary=True, packed=True, masks=True, combined=False),
    "five_ops_combined": dict(norm=True, rotary=True, packed=True, masks=True, combined=True),
    "five_ops_pair_norm": dict(norm=True, rotary=True, packed=True, masks=True, combined=False, pair_norm=True),
    "five_ops_flash": dict(norm=True, rotary=True, packed=True, masks=True, combined=True, backend="flash"),
    "five_ops_single_stream": dict(norm=True, rotary=True, packed=True, masks=True, combined=True, single_stream=True),
}


class SingleGPUFiveOpRunner(FusedOpsMixin, ActionRunner):
    def __init__(self, *args, variant="five_ops_pair_norm", **kwargs):
        self.options = VARIANTS[variant]
        self.cached_latents = None
        super().__init__(*args, **kwargs)
        if len(self.devices) != 1:
            raise ValueError("Groups 13/14 require one GPU")
        self.variant = variant
        self.projections, self.norms = {}, {}
        for expert in (self.model.video_expert, self.model.action_expert):
            for block in expert.blocks:
                for attn, names in ((block.self_attn, ("q", "k", "v")),
                                    (block.cross_attn, ("k", "v"))):
                    if self.options["packed"]:
                        self.projections[id(attn)] = PackedLinear([getattr(attn, name) for name in names])
                    for norm in (attn.norm_q, attn.norm_k):
                        self.norms[id(norm)] = FusedRMSNorm(norm)
        video, context, mask, attention = self.prepare_video()
        ts, _ = self.schedule()
        action = self.prepare_action(self.noise, ts[0], context, mask, attention)
        self.drop_masks = {}
        for expert, prepared in ((self.model.video_expert, video), (self.model.action_expert, action)):
            self.drop_masks[id(expert)] = (
                self.options["masks"] and is_unmasked(prepared.attention_mask),
                self.options["masks"] and is_unmasked(prepared.context_mask))
        if self.options.get("single_stream"):
            self.action_stream = self.video_stream

    def encode_image(self):
        return super().encode_image() if self.cached_latents is None else self.cached_latents

    def cache_image_latents(self):
        self.stage_inputs()
        latents = self.model._encode_input_image_latents_tensor(self.image)
        if self.cached_latents is None:
            self.cached_latents = latents.clone()
        else:
            self.cached_latents.copy_(latents)
        torch.cuda.synchronize(self.video_device)

    def _norm_pair(self, attention, q, k, freqs=None):
        if self.options.get("pair_norm"):
            q, k = rms_pair(q, k, attention.norm_q, attention.norm_k)
            if freqs is not None:
                q, k = (rope(x, freqs, attention.attn_head_dim) for x in (q, k))
            return q, k
        outputs = []
        for x, norm in ((q, attention.norm_q), (k, attention.norm_k)):
            if self.options["combined"] and freqs is not None:
                x = norm_rope(x, norm, freqs=freqs, head_dim=attention.attn_head_dim,
                              match_output_cast=True)
            else:
                x = self.norms[id(norm)](x) if self.options["norm"] else norm(x)
                if freqs is not None:
                    x = (rope(x, freqs, attention.attn_head_dim) if self.options["rotary"]
                         else rope_apply(x, freqs, attention.num_heads))
            outputs.append(x)
        return tuple(outputs)

    def project(self, expert, index, x, prepared):
        block = expert.blocks[index]
        shift, scale, gate, shift_mlp, scale_mlp, gate_mlp = self.model.mot._split_modulation(block, prepared.t_mod)
        z = self._modulate(block.norm1(x), shift, scale)
        attn = block.self_attn
        q, k, v = (self.projections[id(attn)](z) if self.options["packed"]
                   else (attn.q(z), attn.k(z), attn.v(z)))
        q, k = self._norm_pair(attn, q, k, prepared.freqs)
        return q, k, v, x, gate, shift_mlp, scale_mlp, gate_mlp, False

    def finish(self, expert, index, io, prepared, kv=None):
        block = expert.blocks[index]
        k, v = io[1:3] if kv is None else (
            torch.cat((kv[0], io[1]), dim=1), torch.cat((kv[1], io[2]), dim=1))
        self_mask = None if self.drop_masks[id(expert)][0] else prepared.attention_mask
        cross_mask = None if self.drop_masks[id(expert)][1] else prepared.context_mask.unsqueeze(1)
        backend = (sdpa_kernel(SDPBackend.FLASH_ATTENTION)
                   if self.options.get("backend") == "flash" else nullcontext())
        with backend:
            mixed = flash_attention(io[0], k, v, block.num_heads, self_mask)
        x = self._gate(io[3], io[4], block.self_attn.o(mixed))
        attn = block.cross_attn
        q = attn.q(block.norm3(x))
        k, v = (self.projections[id(attn)](prepared.context) if self.options["packed"]
                else (attn.k(prepared.context), attn.v(prepared.context)))
        q, k = self._norm_pair(attn, q, k)
        backend = (sdpa_kernel(SDPBackend.FLASH_ATTENTION)
                   if self.options.get("backend") == "flash" else nullcontext())
        with backend:
            mixed = flash_attention(q, k, v, attn.num_heads, cross_mask)
        x = x + attn.o(mixed)
        z = self._modulate(block.norm2(x), io[5], io[6])
        return self._gate(x, io[7], block.ffn(z))


class SequentialFiveOpRunner(SingleGPUFiveOpRunner):
    """Complete Video prefill before Action, using the same fused operators."""

    def sequential(self):
        video, context, mask, attention = self.prepare_video()
        ve, ae = self.model.video_expert, self.model.action_expert
        vx, cache = video.tokens, []
        for index in range(self.model.mot.num_layers):
            io = self.project(ve, index, vx, video)
            cache.append(io[1:3])
            vx = self.finish(ve, index, io, video)
        timesteps, deltas = self.schedule()
        x = self.noise
        for timestep, delta in zip(timesteps, deltas):
            action = self.prepare_action(x, timestep, context, mask, attention)
            ax = action.tokens
            for index, kv in enumerate(cache):
                io = self.project(ae, index, ax, action)
                ax = self.finish(ae, index, io, action, kv)
            x = self.post_step(ax, delta, x)
        return x[0].float()


def validate(runner, cases, references, *, atol):
    original = runner.image_cpu, runner.proprio_cpu, runner.seed
    errors, outputs = [], []
    try:
        for (image, proprio, seed), reference in zip(cases, references):
            runner.image_cpu, runner.proprio_cpu, runner.seed = image, proprio, seed
            if runner.cached_latents is not None:
                runner.cache_image_latents()
            actual = runner()
            torch.cuda.synchronize(runner.video_device)
            torch.testing.assert_close(actual, reference, rtol=0, atol=atol)
            errors.append((actual - reference).abs().max().item())
            outputs.append(actual.clone())
    finally:
        runner.image_cpu, runner.proprio_cpu, runner.seed = original
        if runner.cached_latents is not None:
            runner.cache_image_latents()
    return errors, outputs


def gpu_time(runner, bench):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    samples = []
    runner.stage_inputs()
    for i in range(int(bench.warmup) + int(bench.iters)):
        start.record()
        runner.run_gpu()
        end.record()
        end.synchronize()
        if i >= int(bench.warmup):
            samples.append(start.elapsed_time(end))
    return {"mean_ms": statistics.fmean(samples), "p50_ms": _percentile(samples, 50),
            "p90_ms": _percentile(samples, 90), "samples_ms": samples,
            "scope": "GPU graph only; excludes CPU input/noise/output; includes prepare and scheduler"}


def measure(runner, bench, cases, references, label, *, pipeline=True):
    runner.run_gpu = runner.single_gpu_pipeline if pipeline else runner.sequential
    eager_errors, eager_outputs = (validate(runner, cases, references, atol=float(bench.parallel_atol))
                                    if bench.verify else (None, None))
    runner.stage_inputs()
    start = time.perf_counter()
    runner.capture(pipeline, int(bench.graph_warmup))
    setup = time.perf_counter() - start
    errors, _ = validate(runner, cases, references, atol=float(bench.parallel_atol)) if bench.verify else (None, None)
    graph_errors, _ = validate(runner, cases, eager_outputs, atol=0) if bench.verify else (None, None)
    result = _benchmark(runner, label, bench)
    result.update(validation_max_abs_errors=errors, eager_validation_max_abs_errors=eager_errors,
                  graph_eager_max_abs_errors=graph_errors, graph_setup_seconds=setup,
                  reference_validation_atol=float(bench.parallel_atol), reference_validation_rtol=0,
                  vae_included=runner.cached_latents is None, gpu_graph_only=gpu_time(runner, bench))
    return result


def run_group14(model, image, proprio, context, mask, horizon, seed, rand_device,
                sigma_shift, bench, cases, references, label):
    variant = "five_ops_pair_norm"
    runner = SequentialFiveOpRunner(model, image, proprio, context, mask, horizon, 1,
                                    model.device, seed, rand_device, sigma_shift=sigma_shift,
                                    variant=variant)
    try:
        result = measure(runner, bench, cases, references, label, pipeline=False)
        result.update(group_id=14, label=label, cuda_graph=True, pipeline=False,
                      operator_fusion=True, operator_backend="triton_five_ops_fp64_rope",
                      parallel_mode=None, vae_backend="cuda_graph", variant=variant,
                      execution_order="all_video_then_all_action", local_compute_streams=1,
                      attention_masks_removed={name: runner.drop_masks[id(expert)] for name, expert in (
                          ("video", model.video_expert), ("action", model.action_expert))},
                      linear_calls_per_forward=420, rmsnorm_operations_per_forward=240,
                      collectives_per_rank=0)
        logger.info("Group 14 sequential end-to-end: %s", result["summary"])
        return result
    finally:
        torch.cuda.synchronize(model.device)
        del runner
        gc.collect()
        torch.cuda.empty_cache()


def run_group13(model, image, proprio, context, mask, horizon, seed, rand_device,
                sigma_shift, bench, cases, references, label):
    selected = str(bench.group13_variant)
    if selected not in VARIANTS:
        raise ValueError(f"Unknown group13_variant={selected}; choose {list(VARIANTS)}")
    variants = (list(bench.group13_variants) if bench.group13_variants is not None else list(VARIANTS)) if bench.group13_tune else [selected]
    if selected not in variants or len(set(variants)) != len(variants) or any(v not in VARIANTS for v in variants):
        raise ValueError("group13_variants must contain distinct known variants, including group13_variant")
    rounds = int(bench.group13_rounds) if bench.group13_tune else 1
    trials = []
    for round_index in range(rounds):
        # Alternate ordering to expose clock/thermal/order bias.
        order = variants if round_index % 2 == 0 else list(reversed(variants))
        for variant in order:
            logger.info("Group 13 round %d/%d: %s", round_index + 1, rounds, variant)
            runner = SingleGPUFiveOpRunner(model, image, proprio, context, mask, horizon, 1,
                                           model.device, seed, rand_device, sigma_shift=sigma_shift,
                                           variant=variant)
            trial = {"variant": variant, "round": round_index + 1, "options": VARIANTS[variant]}
            try:
                trial["end_to_end"] = measure(runner, bench, cases, references, f"{label}_{variant}")
                trial["removed_masks"] = {name: runner.drop_masks[id(expert)] for name, expert in (
                    ("video", model.video_expert), ("action", model.action_expert))}
                if bench.group13_without_vae:
                    runner.cache_image_latents()
                    trial["without_vae"] = measure(runner, bench, cases, references, f"{label}_{variant}_without_vae")
                if bench.group13_profile and round_index == 0:
                    scope = "without_vae" if runner.cached_latents is not None else "end_to_end"
                    path = Path(str(bench.output_json)).with_suffix(f".group13_{variant}_{scope}.trace.json")
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                           torch.profiler.ProfilerActivity.CUDA]) as prof:
                        runner()
                        torch.cuda.synchronize(model.device)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    prof.export_chrome_trace(str(path))
                    trial["profile_trace"] = str(path)
                    events = json.loads(path.read_text())["traceEvents"]
                    kernels = [e for e in events if e.get("cat") == "kernel" and e.get("ph") == "X"]
                    trial["profile_kernel_count"] = len(kernels)
                    trial["profile_kernel_sum_ms"] = sum(e["dur"] for e in kernels) / 1000
            except (AssertionError, RuntimeError) as error:
                if not bench.group13_tune:
                    raise
                trial["error"] = str(error)
                logger.error("Variant %s rejected: %s", variant, error)
            trials.append(trial)
            torch.cuda.synchronize(model.device)
            del runner
            gc.collect()
            torch.cuda.empty_cache()
            if bench.output_json:
                path = Path(str(bench.output_json)).with_suffix(".group13_trials.json")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(trials, indent=2) + "\n")
    chosen = [t for t in trials if t["variant"] == selected and "error" not in t]
    if len(chosen) != rounds:
        raise RuntimeError(f"Selected group13 variant {selected} did not pass every round")
    result = dict(chosen[0]["end_to_end"])
    all_samples = [s for t in chosen for s in t["end_to_end"]["samples_ms"]]
    result.update(samples_ms=all_samples, summary={"mean_ms": statistics.fmean(all_samples),
                  "p50_ms": _percentile(all_samples, 50), "p90_ms": _percentile(all_samples, 90),
                  "min_ms": min(all_samples), "max_ms": max(all_samples)})
    gpu_samples = [s for t in chosen for s in t["end_to_end"]["gpu_graph_only"]["samples_ms"]]
    result["gpu_graph_only"] = dict(result["gpu_graph_only"],
        mean_ms=statistics.fmean(gpu_samples), p50_ms=_percentile(gpu_samples, 50),
        p90_ms=_percentile(gpu_samples, 90), samples_ms=gpu_samples)
    result.update(group_id=13, label=label, cuda_graph=True,
                  pipeline=not VARIANTS[selected].get("single_stream", False), operator_fusion=True,
                  operator_backend="triton_five_ops_fp64_rope", parallel_mode=None,
                  vae_backend="cuda_graph", variant=selected, tuning_trials=trials,
                  rounds=rounds, attention_masks_removed=chosen[0]["removed_masks"],
                  linear_calls_per_forward=420 if VARIANTS[selected]["packed"] else 600,
                  rmsnorm_operations_per_forward=240, collectives_per_rank=0)
    if bench.group13_without_vae:
        diagnostic = dict(chosen[0]["without_vae"])
        samples = [s for t in chosen for s in t["without_vae"]["samples_ms"]]
        diagnostic.update(samples_ms=samples, summary={"mean_ms": statistics.fmean(samples),
                          "p50_ms": _percentile(samples, 50), "p90_ms": _percentile(samples, 90),
                          "min_ms": min(samples), "max_ms": max(samples)})
        gpu_samples = [s for t in chosen for s in t["without_vae"]["gpu_graph_only"]["samples_ms"]]
        diagnostic["gpu_graph_only"] = dict(diagnostic["gpu_graph_only"],
            mean_ms=statistics.fmean(gpu_samples), p50_ms=_percentile(gpu_samples, 50),
            p90_ms=_percentile(gpu_samples, 90), samples_ms=gpu_samples)
        result["without_vae"] = diagnostic
    logger.info("Group 13 %s end-to-end: %s", selected, result["summary"])
    return result

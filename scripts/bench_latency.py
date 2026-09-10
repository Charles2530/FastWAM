"""Fourteen end-to-end FastWAM latency groups (10 warmups, 100 measured samples).

    python scripts/bench_latency.py \
      ckpt=checkpoints/fastwam_release/libero_uncond_2cam224.pt

Groups: 10-step eager, 10-step sequential CUDA Graph, 1-step eager,
1-step sequential CUDA Graph, 1-step single-GPU pipeline CUDA Graph,
1-step two-GPU expert pipeline CUDA Graph, matching single-/two-GPU pipeline
groups with Triton affine fusion, native SP2/TP2, optimized SP2/TP2, a five-op
single-GPU pipeline, and its fully sequential five-op counterpart.
Groups 5/6 use PyTorch operators; groups 7/8 add modulation/gate fusion.
Compile means explicit CUDA Graph capture/replay (including VAE), not Inductor. Only prompt encoding
is cached outside the timer. Input copies, noise generation, VAE, proprio,
prepare, Video/Action, scheduler updates, KV transfers and CPU output are timed.
Graph setup and validation are untimed. Groups 6/8 use device-local graphs, with
cross-device copies/events outside capture, following the two reference scripts.
All groups use no_grad(), matching dryrun_fastwam.py. Eager groups call the
original model.infer_action(), including its per-call setup and input allocation.
Groups 9-12 automatically spawn two NCCL ranks. Each rank runs Video and Action
on separate streams in one CUDA Graph, including its replicated VAE. Distributed
latency is the slower rank's request time; timing barriers/reductions are excluded.
TP/SP use the reference parallel script's 0.02 absolute tolerance against the
unsharded policy, plus exact graph/eager parity within each sharded implementation.
Groups 9/10 are native Ulysses SP2/TP2. Groups 11/12 use the LightX2V-inspired
optimized SP2/TP2 path, which fuses QKV/KV projection, RMSNorm/RoPE and affine
operators, combines TP Q/K normalization reductions, and fuses SP wire layouts.
Group 13 is a single-GPU asynchronous pipeline with fused RMSNorm, FP64 RoPE,
BF16 affine/gate, static all-True mask removal and packed QKV/KV projections.
Group 14 uses the same five_ops_pair_norm operators as default group 13, but
executes all Video blocks before Action on one stream, including VAE in capture.
Use +BENCH.group13_tune=true for repeated operator/SDPA/stream comparisons.
Its without_vae diagnostic is off by default. When explicitly enabled, it caches
only image latents outside timing; the primary result always includes VAE.
Use +BENCH.groups=[9,10,11,12] to compare all four in the same workers. Optional
+BENCH.parallel_profile=true exports traces AFTER timing each variant.
Speedup uses group 1 from the same report; it is unavailable when group 1 is omitted.

Overrides: +BENCH.action_device=cuda:1, +BENCH.warmup=10, +BENCH.iters=100,
+BENCH.output_json=artifacts/latency.json. Synthetic text context is available
with model.load_text_encoder=false +BENCH.use_random_context=true.
Use +BENCH.groups=[1] to compare the 10-step eager baseline on one visible GPU.
For the corresponding dryrun, explicitly disable compile_vae_encode and enable
cache_text_embeddings; compile_action_infer=false alone does not disable its VAE
compilation. Component profiling should be off when comparing latency.
"""

from __future__ import annotations

import gc
import json
import logging
from datetime import datetime
from functools import partial
from pathlib import Path
import statistics
import sys
import time
from typing import NamedTuple

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
logger = logging.getLogger(__name__)


class BenchmarkGroup(NamedTuple):
    label: str
    steps: int
    graph: bool
    pipeline: bool
    dual: bool
    fusion: bool = False
    parallel_mode: str | None = None


GROUPS = [
    BenchmarkGroup("1_eager_10step_sequential", 10, False, False, False),
    BenchmarkGroup("2_graph_10step_sequential", 10, True, False, False),
    BenchmarkGroup("3_eager_1step_sequential", 1, False, False, False),
    BenchmarkGroup("4_graph_1step_sequential", 1, True, False, False),
    BenchmarkGroup("5_graph_1step_1gpu_pipeline", 1, True, True, False),
    BenchmarkGroup("6_graph_1step_2gpu_pipeline", 1, True, True, True),
    BenchmarkGroup("7_graph_1step_1gpu_pipeline_fused", 1, True, True, False, True),
    BenchmarkGroup("8_graph_1step_2gpu_pipeline_fused", 1, True, True, True, True),
    BenchmarkGroup("9_graph_1step_sp2_native_pipeline", 1, True, True, True, parallel_mode="ulysses_sp2"),
    BenchmarkGroup("10_graph_1step_tp2_native_pipeline", 1, True, True, True, parallel_mode="tp2"),
    BenchmarkGroup("11_graph_1step_sp2_lightx2v_pipeline", 1, True, True, True, True, "ulysses_sp2"),
    BenchmarkGroup("12_graph_1step_tp2_lightx2v_pipeline", 1, True, True, True, True, "tp2"),
    BenchmarkGroup("13_graph_1step_1gpu_five_ops_pipeline", 1, True, True, False, True),
    BenchmarkGroup("14_graph_1step_1gpu_five_ops_sequential", 1, True, False, False, True),
]


def _config(cfg: DictConfig) -> DictConfig:
    defaults = OmegaConf.create({
        "warmup": 10, "iters": 100, "graph_warmup": 3, "seed": 42,
        "action_horizon": None, "action_device": None,
        "prompt": "pick up the object", "use_random_context": False,
        "verify": True, "rtol": 0.01, "atol": 0.01,
        "parallel_atol": 0.02,
        "parallel_profile": False,
        "group13_tune": False, "group13_variant": "five_ops_pair_norm",
        "group13_variants": None,
        "group13_rounds": 3, "group13_without_vae": False,
        "group13_profile": False,
        "groups": list(range(1, len(GROUPS) + 1)),
        "output_json": f"artifacts/bench_latency_{datetime.now():%Y%m%d_%H%M%S}.json",
    })
    # Preserve useful input/count overrides from the former dryrun script.
    legacy = {k: v for k, v in cfg.get("DRYRUN", {}).items() if k in defaults}
    result = OmegaConf.merge(defaults, legacy, cfg.get("BENCH", {}))
    if any(key in result for key in ("parallel_optimized", "parallel_compare")):
        raise ValueError("parallel_optimized/parallel_compare are replaced by fixed groups: "
                         "9/10=native SP2/TP2, 11/12=optimized SP2/TP2; "
                         "select +BENCH.groups=[9,10,11,12]")
    if int(result.warmup) < 0 or int(result.iters) <= 0 or int(result.graph_warmup) <= 0:
        raise ValueError("warmup must be >= 0; iters and graph_warmup must be > 0")
    if int(result.group13_rounds) <= 0:
        raise ValueError("group13_rounds must be > 0")
    if (not result.groups or len(set(result.groups)) != len(result.groups)
            or any(group not in range(1, len(GROUPS) + 1) for group in result.groups)):
        raise ValueError(f"BENCH.groups must contain distinct group IDs from 1 to {len(GROUPS)}")
    return result


def _sync(devices):
    for device in devices:
        torch.cuda.synchronize(device)


def _percentile(values, percent):
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _write_report(report, bench):
    if bench.output_json is not None:
        path = Path(str(bench.output_json))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n")


class ExpertInputs(NamedTuple):
    tokens: torch.Tensor
    freqs: torch.Tensor
    t_mod: torch.Tensor
    context: torch.Tensor
    context_mask: torch.Tensor
    attention_mask: torch.Tensor


class CapturedCall:
    """Retain stable graph outputs; initialize them before downstream capture."""

    def __init__(self, function, device, warmup):
        self.device = device
        self.stream = torch.cuda.Stream(device=device)
        self.stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.device(device), torch.cuda.stream(self.stream):
            for _ in range(warmup):
                self.output = function()
        self.stream.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.device(device), torch.cuda.graph(self.graph, stream=self.stream):
            self.output = function()
        with torch.cuda.device(device), torch.cuda.stream(self.stream):
            self.graph.replay()
        self.stream.synchronize()

    def __call__(self):
        with torch.cuda.device(self.device):
            self.graph.replay()
        return self.output


class EagerActionRunner:
    """Use the public inference path timed by dryrun, with cached text only."""

    def __init__(self, model, image, proprio, context, context_mask, horizon,
                 steps, action_device, seed, rand_device, sigma_shift=None):
        self.model = model
        self.devices = [model.device]
        self.steps, self.seed = steps, seed
        self.image_cpu, self.proprio_cpu = image, proprio
        self.kwargs = dict(prompt=None, context=context, context_mask=context_mask,
                           action_horizon=horizon, num_inference_steps=steps,
                           rand_device=rand_device, sigma_shift=sigma_shift,
                           compile_action_infer=False)

    def __call__(self):
        return self.model.infer_action(input_image=self.image_cpu,
                                       proprio=self.proprio_cpu, seed=self.seed,
                                       **self.kwargs)["action"]


class ActionRunner:
    """The same first-frame-causal action computation in all execution modes."""

    def __init__(self, model, image, proprio, context, context_mask, horizon,
                 steps, action_device, seed, rand_device, sigma_shift=None):
        self.model = model
        self.video_device = model.device
        self.action_device = action_device
        self.devices = list(dict.fromkeys([self.video_device, action_device]))
        self.steps, self.seed, self.rand_device = steps, seed, rand_device
        self.sigma_shift = sigma_shift
        self.image_cpu, self.proprio_cpu = image, proprio
        self.context, self.context_mask = context, context_mask
        self.image = torch.empty_like(image, device=self.video_device, dtype=model.torch_dtype)
        self.proprio = (None if proprio is None else torch.empty_like(
            proprio, device=self.video_device, dtype=model.torch_dtype))
        self.noise = torch.empty((1, horizon, model.action_expert.action_dim),
                                 device=action_device, dtype=model.torch_dtype)
        self.video_stream = torch.cuda.Stream(device=self.video_device)
        self.action_stream = torch.cuda.Stream(device=action_device)
        self.ready = [torch.cuda.Event() for _ in range(model.mot.num_layers)]
        self.run_gpu = self.sequential
        self.stage_inputs()
        _sync(self.devices)

    def stage_inputs(self):
        # Refresh graph inputs on EVERY call; host RNG and copies are timed.
        generator = torch.Generator(device=self.rand_device).manual_seed(self.seed)
        noise = torch.randn(self.noise.shape, generator=generator,
                            device=self.rand_device, dtype=torch.float32)
        self.image.copy_(self.image_cpu)
        if self.proprio is not None:
            self.proprio.copy_(self.proprio_cpu)
        self.noise.copy_(noise)

    def __call__(self):
        self.stage_inputs()
        return self.run_gpu().to(device="cpu")

    def encode_image(self):
        return self.model._encode_input_image_latents_tensor(self.image)

    def prepare_video(self):
        model = self.model
        latents = self.encode_image()
        context, mask = model._append_proprio_to_context(
            self.context, self.context_mask, self.proprio)
        prepared = model.video_expert.prepare(
            x=latents, timestep=torch.zeros(latents.shape[0], device=self.video_device,
                                           dtype=latents.dtype),
            context=context, context_mask=mask, action=None,
            fuse_vae_embedding_in_latents=bool(model.video_expert.fuse_vae_embedding_in_latents))
        tokens, _, t_mod, v_context, v_mask, freqs, _, _, _, per_frame = prepared
        seq = tokens.shape[1]
        attention = model._build_mot_attention_mask(
            seq, self.noise.shape[1], per_frame, self.video_device)
        video = ExpertInputs(tokens, freqs, t_mod, v_context, v_mask, attention[:seq, :seq])
        return video, context, mask, attention[seq:, :]

    def schedule(self):
        return self.model.infer_action_scheduler.build_inference_schedule(
            self.steps, self.action_device, self.noise.dtype,
            shift_override=self.sigma_shift)

    def prepare_action(self, x, timestep, context, mask, attention):
        tokens, _, t_mod, context, mask, freqs = self.model.action_expert.prepare(
            action_tokens=x, timestep=timestep.unsqueeze(0),
            context=context, context_mask=mask)
        return ExpertInputs(tokens, freqs, t_mod, context, mask, attention)

    def project(self, expert, index, x, prepared):
        return self.model.mot._build_expert_attention_io(
            expert=expert, block=expert.blocks[index], x=x,
            freqs=prepared.freqs, t_mod=prepared.t_mod)

    def finish(self, expert, index, io, prepared, kv=None):
        k, v = (io[1], io[2]) if kv is None else (
            torch.cat([kv[0], io[1]], dim=1), torch.cat([kv[1], io[2]], dim=1))
        mixed = self.model.mot._mixed_attention(
            q_cat=io[0], k_cat=k, v_cat=v, attention_mask=prepared.attention_mask)
        return self.model.mot._apply_expert_post_block_tensor(
            block=expert.blocks[index], residual_x=io[3], mixed_attn_out=mixed,
            gate_msa=io[4], shift_mlp=io[5], scale_mlp=io[6], gate_mlp=io[7],
            context=prepared.context, context_mask=prepared.context_mask)

    def post_step(self, hidden, delta, latents):
        prediction = self.model.action_expert.post(hidden)
        return self.model.infer_action_scheduler.step(prediction, delta, latents)

    def sequential(self):
        video, context, mask, attention = self.prepare_video()
        cache_k, cache_v = self.model.mot.prefill_video_cache_tensor(*video)
        timesteps, deltas = self.schedule()
        x = self.noise
        for timestep, delta in zip(timesteps, deltas):
            action = self.prepare_action(x, timestep, context, mask, attention)
            hidden = self.model.mot.forward_action_with_video_cache_tensor(
                *action[:5], cache_k, cache_v, action.attention_mask)
            x = self.post_step(hidden, delta, x)
        return x[0].float()

    def single_gpu_pipeline(self):
        video, context, mask, attention = self.prepare_video()
        timesteps, deltas = self.schedule()
        action = self.prepare_action(self.noise, timesteps[0], context, mask, attention)
        caller = torch.cuda.current_stream(self.video_device)
        self.video_stream.wait_stream(caller)
        self.action_stream.wait_stream(caller)
        vx, ax = video.tokens, action.tokens
        ve, ae = self.model.video_expert, self.model.action_expert
        for index in range(self.model.mot.num_layers):
            with torch.cuda.stream(self.video_stream):
                vio = self.project(ve, index, vx, video)
                self.ready[index].record(self.video_stream)
                vx = self.finish(ve, index, vio, video)
            with torch.cuda.stream(self.action_stream):
                aio = self.project(ae, index, ax, action)
                self.action_stream.wait_event(self.ready[index])
                vio[1].record_stream(self.action_stream)
                vio[2].record_stream(self.action_stream)
                ax = self.finish(ae, index, aio, action, (vio[1], vio[2]))
        caller.wait_stream(self.video_stream)
        caller.wait_stream(self.action_stream)
        ax.record_stream(caller)
        return self.post_step(ax, deltas[0], self.noise)[0].float()

    def capture(self, pipeline, warmup):
        if pipeline and self.steps != 1:
            raise ValueError("Pipeline graphs require exactly one action denoising step")
        # Sequential capture includes VAE, one Video prefill, and every Action
        # denoising/scheduler step. Only the pipeline path is limited to one step.
        function = self.single_gpu_pipeline if pipeline else self.sequential
        self.captured = CapturedCall(function, self.video_device, warmup)
        self.run_gpu = self.captured


class FusedOpsMixin:
    """Use the same affine kernels in the single- and two-GPU fused groups."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._setup_fused_ops()

    def _setup_fused_ops(self):
        # Triton resolves kernel symbols in module globals; import only for fusion.
        global tl
        import triton
        import triton.language as tl

        @triton.jit
        def affine_kernel(X, S, G, Y, N: tl.constexpr, D: tl.constexpr,
                          SROWS: tl.constexpr, GROWS: tl.constexpr,
                          SSTRIDE: tl.constexpr, GSTRIDE: tl.constexpr,
                          MODULATE: tl.constexpr, BLOCK: tl.constexpr):
            i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
            valid = i < N
            dtype = X.dtype.element_ty
            x = tl.load(X + i, valid, 0).to(tl.float32)
            s = tl.load(S + (i // D % SROWS) * SSTRIDE + i % D, valid, 0).to(tl.float32)
            g = tl.load(G + (i // D % GROWS) * GSTRIDE + i % D, valid, 0).to(tl.float32)
            # Preserve eager BF16/FP16 rounding after every arithmetic step.
            if MODULATE:
                factor = (1.0 + g).to(dtype).to(tl.float32)
                product = (x * factor).to(dtype).to(tl.float32)
                y = product + s
            else:
                product = (s * g).to(dtype).to(tl.float32)
                y = x + product
            tl.store(Y + i, y.to(dtype), valid)

        def supported(x, *values):
            # Modulation tensors can be strided slices with contiguous rows.
            return (x.dtype in (torch.bfloat16, torch.float16)
                    and x.ndim == 3 and x.shape[0] == 1 and x.is_contiguous()
                    and all(v.dtype == x.dtype and v.device == x.device
                            and v.ndim in (2, 3) and v.shape[-1] == x.shape[-1]
                            and v.stride(-1) == 1
                            and (v.ndim == 2 or v.shape[0] == 1)
                            and v.numel() // x.shape[-1] in (1, x.shape[1])
                            for v in values))

        def modulate(x, shift, scale):
            if not supported(x, shift, scale):
                return x * (1 + scale) + shift
            y = torch.empty_like(x)
            affine_kernel[(triton.cdiv(x.numel(), 256),)](
                x, shift, scale, y, x.numel(), x.shape[-1], shift.numel() // x.shape[-1],
                scale.numel() // x.shape[-1], shift.stride(-2), scale.stride(-2), True, 256,
                enable_fp_fusion=False)
            return y

        def gate(x, g, residual):
            if not supported(x, residual, g):
                return x + g * residual
            y = torch.empty_like(x)
            affine_kernel[(triton.cdiv(x.numel(), 256),)](
                x, residual, g, y, x.numel(), x.shape[-1], residual.numel() // x.shape[-1],
                g.numel() // x.shape[-1], residual.stride(-2), g.stride(-2), False, 256,
                enable_fp_fusion=False)
            return y

        self._modulate, self._gate = modulate, gate

    def project(self, expert, index, x, prepared):
        from fastwam.models.wan22.wan_video_dit import rope_apply

        block = expert.blocks[index]
        shift, scale, gate, shift_mlp, scale_mlp, gate_mlp = self.model.mot._split_modulation(
            block, prepared.t_mod)
        z = self._modulate(block.norm1(x), shift, scale)
        q = block.self_attn.norm_q(block.self_attn.q(z))
        k = block.self_attn.norm_k(block.self_attn.k(z))
        v = block.self_attn.v(z)
        q = rope_apply(q, prepared.freqs, block.num_heads)
        k = rope_apply(k, prepared.freqs, block.num_heads)
        return q, k, v, x, gate, shift_mlp, scale_mlp, gate_mlp, bool(
            getattr(expert, "use_gradient_checkpointing", False))

    def finish(self, expert, index, io, prepared, kv=None):
        k, v = io[1:3] if kv is None else (
            torch.cat([kv[0], io[1]], dim=1), torch.cat([kv[1], io[2]], dim=1))
        mixed = self.model.mot._mixed_attention(
            q_cat=io[0], k_cat=k, v_cat=v, attention_mask=prepared.attention_mask)
        block = expert.blocks[index]
        x = self._gate(io[3], io[4], block.self_attn.o(mixed))
        x = x + block.cross_attn(block.norm3(x), prepared.context,
                                ctx_mask=prepared.context_mask.unsqueeze(1))
        z = self._modulate(block.norm2(x), io[5], io[6])
        return self._gate(x, io[7], block.ffn(z))


class TwoGPUActionRunner(ActionRunner):
    """Two-device pipeline with graph boundaries only at Video K/V handoffs."""

    def capture_two_gpu(self, warmup):
        self.copy_stream = torch.cuda.Stream(device=self.video_device)
        self.copy_ready = [torch.cuda.Event() for _ in self.ready]
        self.context_ready = torch.cuda.Event()
        self.pre_video = CapturedCall(self.prepare_video, self.video_device, warmup)
        video, context, mask, attention = self.pre_video.output
        self.remote_context = context.to(self.action_device)
        self.remote_mask = mask.to(self.action_device)
        self.remote_attention = attention.to(self.action_device)
        ve, ae = self.model.video_expert, self.model.action_expert
        count = self.model.mot.num_layers

        def prepare_one_action():
            timesteps, deltas = self.schedule()
            action = self.prepare_action(self.noise, timesteps[0], self.remote_context,
                                         self.remote_mask, self.remote_attention)
            io = self.project(ae, 0, action.tokens, action)
            return action, deltas[0], io

        _sync(self.devices)
        self.pre_action = CapturedCall(prepare_one_action, self.action_device, warmup)
        action, delta, aio = self.pre_action.output
        shape = (self.noise.shape[0], video.tokens.shape[1],
                 self.model.mot.num_heads * self.model.mot.attn_head_dim)
        self.received_k = [torch.empty(shape, device=self.action_device, dtype=self.noise.dtype)
                           for _ in self.ready]
        self.received_v = [torch.empty_like(k) for k in self.received_k]
        self.layers = []
        vio = None
        for index in range(count):
            # Finish Video i-1 and produce Video i K/V without delaying its handoff.
            def video_phase(index=index, previous=vio):
                x = video.tokens if previous is None else self.finish(ve, index - 1, previous, video)
                return self.project(ve, index, x, video)

            vg = CapturedCall(video_phase, self.video_device, warmup)
            vio = vg.output
            self.received_k[index].copy_(vio[1])
            self.received_v[index].copy_(vio[2])
            _sync(self.devices)

            def action_phase(index=index, current=aio):
                x = self.finish(ae, index, current, action,
                                (self.received_k[index], self.received_v[index]))
                if index + 1 < count:
                    return self.project(ae, index + 1, x, action)
                return self.post_step(x, delta, self.noise)[0].float()

            ag = CapturedCall(action_phase, self.action_device, warmup)
            aio = ag.output
            self.layers.append((vg, ag))
        # Retain the last Video block, matching the other benchmark groups.
        self.video_tail = CapturedCall(partial(self.finish, ve, count - 1, vio, video),
                                       self.video_device, warmup)
        self.run_gpu = self.two_gpu_pipeline
        _sync(self.devices)

    def two_gpu_pipeline(self):
        caller = torch.cuda.current_stream(self.video_device)
        action_caller = torch.cuda.current_stream(self.action_device)
        _, context, mask, attention = self.pre_video()
        self.video_stream.wait_stream(caller)
        self.copy_stream.wait_stream(caller)
        self.action_stream.wait_stream(action_caller)
        # copy_ uses the SOURCE current stream. Select both streams so its
        # destination synchronization stays on the Action branch.
        with torch.cuda.stream(self.action_stream), torch.cuda.stream(self.copy_stream):
            self.remote_context.copy_(context, non_blocking=True)
            self.remote_mask.copy_(mask, non_blocking=True)
            self.remote_attention.copy_(attention, non_blocking=True)
            self.context_ready.record(self.copy_stream)
        with torch.cuda.stream(self.action_stream):
            self.action_stream.wait_event(self.context_ready)
            self.pre_action()
        for index, (vg, ag) in enumerate(self.layers):
            with torch.cuda.stream(self.video_stream):
                vio = vg()
                self.ready[index].record(self.video_stream)
                if index + 1 == len(self.layers):
                    self.video_tail()
            with torch.cuda.stream(self.action_stream), torch.cuda.stream(self.copy_stream):
                self.copy_stream.wait_event(self.ready[index])
                self.received_k[index].copy_(vio[1], non_blocking=True)
                self.received_v[index].copy_(vio[2], non_blocking=True)
                self.copy_ready[index].record(self.copy_stream)
            with torch.cuda.stream(self.action_stream):
                self.action_stream.wait_event(self.copy_ready[index])
                output = ag()
        caller.wait_stream(self.video_stream)
        caller.wait_stream(self.action_stream)
        action_caller.wait_stream(self.action_stream)
        return output


class FusedActionRunner(FusedOpsMixin, ActionRunner):
    """Group 5 execution with affine fusion (group 7)."""


class FusedTwoGPUActionRunner(FusedOpsMixin, TwoGPUActionRunner):
    """Group 6 execution with affine fusion (group 8)."""


def _benchmark(runner, label, bench):
    def timed():
        _sync(runner.devices)
        start = time.perf_counter()
        output = runner()
        _sync(runner.devices)
        return (time.perf_counter() - start) * 1000, output

    for index in range(int(bench.warmup)):
        elapsed, _ = timed()
        logger.info("%s warmup %d/%d: %.3f ms", label, index + 1, bench.warmup, elapsed)
    for device in runner.devices:
        torch.cuda.reset_peak_memory_stats(device)
    samples = []
    for index in range(int(bench.iters)):
        elapsed, output = timed()
        samples.append(elapsed)
        if (index + 1) % 10 == 0 or index + 1 == int(bench.iters):
            logger.info("%s sample %d/%d: %.3f ms", label, index + 1, bench.iters, elapsed)
    summary = {
        "mean_ms": statistics.fmean(samples), "p50_ms": _percentile(samples, 50),
        "p90_ms": _percentile(samples, 90), "min_ms": min(samples), "max_ms": max(samples),
    }
    return {"label": label, "action_steps": runner.steps,
            "devices": [str(d) for d in runner.devices], "summary": summary,
            "samples_ms": samples, "action_shape": list(output.shape),
            "peak_allocated_gib": {str(d): torch.cuda.max_memory_allocated(d) / 1024**3
                                   for d in runner.devices}}


def _verify(runner, cases, references, bench):
    errors = []
    original = runner.image_cpu, runner.proprio_cpu, runner.seed
    try:
        for (image, proprio, seed), reference in zip(cases, references):
            runner.image_cpu, runner.proprio_cpu, runner.seed = image, proprio, seed
            actual = runner()
            _sync(runner.devices)
            torch.testing.assert_close(actual, reference, rtol=float(bench.rtol), atol=float(bench.atol))
            errors.append((actual - reference).abs().max().item())
    finally:
        runner.image_cpu, runner.proprio_cpu, runner.seed = original
    logger.info("Validation passed (base and changed inputs), max abs errors=%s", errors)
    return errors


@hydra.main(config_path="../configs", config_name="sim_libero.yaml", version_base="1.3")
@torch.no_grad()
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    bench = _config(cfg)
    groups = [GROUPS[index - 1] for index in bench.groups]
    needs_dual = any(group.dual for group in groups)
    device = torch.device(str(cfg.EVALUATION.device))
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("The benchmark requires CUDA")
    device = torch.device("cuda", torch.cuda.current_device() if device.index is None else device.index)
    if not needs_dual:
        action_device = device
    elif bench.action_device is None:
        candidates = [i for i in range(torch.cuda.device_count()) if i != device.index]
        if not candidates:
            raise ValueError("Groups 6, 8, 9-12 require two visible GPUs")
        action_device = torch.device("cuda", candidates[0])
    else:
        action_device = torch.device(str(bench.action_device))
    if needs_dual and (action_device.type != "cuda" or action_device.index is None
            or action_device == device or not 0 <= action_device.index < torch.cuda.device_count()):
        raise ValueError("Groups 6, 8, 9-12 require a different, available +BENCH.action_device=cuda:N")
    torch.cuda.set_device(device)
    torch.manual_seed(int(bench.seed))
    dtype = {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[str(cfg.mixed_precision)]
    model = instantiate(cfg.model, model_dtype=dtype, device=str(device))
    if cfg.ckpt is not None:
        model.load_checkpoint(str(cfg.ckpt))
    else:
        logger.warning("ckpt=null: measuring initialized weights, not a trained policy")
    model.eval()
    from fastwam.models.wan22.fastwam import FastWAM
    if type(model).infer_action is not FastWAM.infer_action:
        raise ValueError("Use model=fastwam; IDM/joint inference has different computation")
    if str(model.video_expert.video_attention_mask_mode) != "first_frame_causal":
        raise ValueError("The benchmark requires first_frame_causal video attention")
    height, width = map(int, cfg.data.train.video_size)
    horizon = int(cfg.data.train.num_frames) - 1 if bench.action_horizon is None else int(bench.action_horizon)
    if horizon <= 0 or height % 16 or width % 16:
        raise ValueError("action_horizon must be positive; image dimensions must be multiples of 16")
    # dryrun seeds synthetic observations AFTER model construction/loading.
    torch.manual_seed(int(bench.seed))
    image = torch.rand(1, 3, height, width) * 2 - 1
    proprio = None if model.proprio_dim is None else torch.randn(1, model.proprio_dim)
    if bool(bench.use_random_context):
        context = torch.randn(1, int(cfg.data.train.context_len), model.text_dim, device=device, dtype=dtype)
        context_mask = torch.ones(context.shape[:2], device=device, dtype=torch.bool)
        context_source = "synthetic"
    else:
        context, context_mask = model.encode_prompt(str(bench.prompt))
        context = context.to(device=device, dtype=dtype)
        context_mask = context_mask.to(device=device, dtype=torch.bool)
        context_source = "encode_prompt_once"
    # Retain the encoder like dryrun: caching skips its forward, not residency
    # or the module traversal in infer_action()'s per-call self.eval().
    gc.collect()
    torch.cuda.empty_cache()
    _sync([device, action_device])
    logger.info("Text context cached (%s); VAE/prepare/denoising/transfers/CPU output are timed", context_source)
    logger.info("Video GPU=%s (%s); execution mode=no_grad; eager path=model.infer_action",
                device, torch.cuda.get_device_name(device))
    if needs_dual:
        logger.info("Second GPU for groups 6/8/9-12=%s (%s); peer access=%s", action_device,
                    torch.cuda.get_device_name(action_device),
                    torch.cuda.can_device_access_peer(device, action_device))
    sigma_shift = cfg.EVALUATION.get("sigma_shift")
    logger.info("sigma_shift=%s; eager VAE is uncompiled; graph VAE is captured", sigma_shift)

    seed = int(bench.seed)
    cases = [(image, proprio, seed)]
    # Change observations separately from noise to detect stale captured VAE/KV.
    cases.append((image * 0.5, None if proprio is None else proprio + 0.25, seed))
    cases.append((image, proprio, seed + 1))
    references = {}
    if bool(bench.verify):
        for steps in sorted({group[1] for group in groups}):
            references[steps] = [model.infer_action(
                prompt=None, input_image=im, proprio=pr, action_horizon=horizon,
                context=context, context_mask=context_mask, seed=s,
                rand_device=str(cfg.EVALUATION.rand_device), num_inference_steps=steps,
                sigma_shift=sigma_shift, compile_action_infer=False)["action"]
                for im, pr, s in cases]
            if torch.equal(references[steps][0], references[steps][1]):
                raise RuntimeError("Changed-observation validation is ineffective: action did not change")
    report = {"torch": torch.__version__, "python": sys.executable,
              "dtype": str(dtype), "checkpoint": cfg.ckpt,
              "gpu_names": {str(d): torch.cuda.get_device_name(d) for d in (device, action_device)},
              "peer_access": torch.cuda.can_device_access_peer(device, action_device) if needs_dual else None,
              "image_shape": list(image.shape), "action_horizon": horizon,
              "context_shape": list(context.shape), "rand_device": str(cfg.EVALUATION.rand_device),
              "execution_mode": "no_grad", "eager_path": "model.infer_action",
              "text_encoder_retained": model.text_encoder is not None,
              "sigma_shift": sigma_shift, "cpu_threads": torch.get_num_threads(),
              "cpu_interop_threads": torch.get_num_interop_threads(),
              "warmup": int(bench.warmup), "iterations": int(bench.iters),
              "graph_setup_warmup": int(bench.graph_warmup), "context_source": context_source,
              "compile_backend": "explicit_cuda_graph", "latency": "synchronized_end_to_end_wall_ms",
              "excluded": ["text_encoder", "model_load", "graph_setup", "validation"],
              "config": OmegaConf.to_container(bench, resolve=True), "results": []}
    parallel_results = None
    for group in groups:
        label, steps, graph, pipeline, dual, fusion, parallel_mode = group
        logger.info("Setting up %s; operator_fusion=%s", label, fusion)
        if GROUPS.index(group) + 1 in (13, 14):
            from fastwam_single_gpu_ops import run_group13, run_group14
            model.action_expert.to(device)
            run_group = run_group13 if GROUPS.index(group) + 1 == 13 else run_group14
            result = run_group(model, image, proprio, context, context_mask, horizon,
                               seed, str(cfg.EVALUATION.rand_device), sigma_shift,
                               bench, cases, references.get(1, []), label)
            report["results"].append(result)
            _write_report(report, bench)
            continue
        if parallel_mode is not None:
            if parallel_results is None:
                from bench_parallel_latency import run_parallel_groups
                parallel_results = run_parallel_groups(
                    cfg, bench, [index for index in bench.groups if GROUPS[index - 1].parallel_mode],
                    [device, action_device], image, proprio, context, context_mask, references.get(1, []))
            result = parallel_results[label]
            report["results"].append(result)
            _write_report(report, bench)
            logger.info("%s: %s", label, result["summary"])
            continue
        # Support custom group orders by moving Action to each group's device.
        model.action_expert.to(action_device if dual else device)
        if dual:
            runner_type = FusedTwoGPUActionRunner if fusion else TwoGPUActionRunner
        elif fusion:
            runner_type = FusedActionRunner
        else:
            runner_type = ActionRunner if graph else EagerActionRunner
        runner = runner_type(model, image, proprio, context, context_mask, horizon, steps,
                             action_device if dual else device, seed, str(cfg.EVALUATION.rand_device),
                             sigma_shift=sigma_shift)
        setup_start = time.perf_counter()
        if dual:
            runner.capture_two_gpu(int(bench.graph_warmup))
        elif graph:
            runner.capture(pipeline, int(bench.graph_warmup))
        setup_seconds = time.perf_counter() - setup_start
        errors = (_verify(runner, cases, references[steps], bench)
                  if bool(bench.verify) else None)
        result = _benchmark(runner, label, bench)
        result.update(group_id=GROUPS.index(group) + 1,
                      cuda_graph=graph, pipeline=pipeline, operator_fusion=fusion,
                      operator_backend="triton_affine" if fusion else "pytorch",
                      graph_setup_seconds=setup_seconds,
                      vae_backend="cuda_graph" if graph else "eager",
                      validation_max_abs_errors=errors)
        report["results"].append(result)
        logger.info("%s: %s", label, result["summary"])
        _write_report(report, bench)
        _sync(runner.devices)
        del runner
        gc.collect()
        for used_device in (device, action_device):
            with torch.cuda.device(used_device):
                torch.cuda.empty_cache()

    print("\nEnd-to-end latency (ms), text encoding excluded; "
          f"warmup={bench.warmup}, samples={bench.iters}")
    _print_results(report)
    if bench.output_json is not None:
        print(f"Saved samples and configuration: {bench.output_json}")


def _print_results(report):
    baseline = next((r["summary"]["mean_ms"] for r in report["results"]
                     if r["group_id"] == 1), None)
    print("| Group | Action Steps | Compile (incl. VAE) | GPU Count | Execution | Fusion | Mean Latency | Speedup |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for result in report["results"]:
        mode = result.get("parallel_mode")
        if mode:
            execution = "SP2" if mode == "ulysses_sp2" else "TP2"
            execution += ", LightX2V optimized" if result["parallel_optimized"] else ", Native"
        else:
            execution = "Asynchronous" if result["pipeline"] else "Sequential"
        mean = result["summary"]["mean_ms"]
        speedup = "-" if baseline is None else f"{baseline / mean:.2f}x"
        compile_mode = "CUDA Graph" if result["cuda_graph"] else "None"
        fusion = "Yes" if result["operator_fusion"] else "No"
        print(f"| {result['group_id']} | {result['action_steps']} | {compile_mode} | "
              f"{len(result['devices'])} | {execution} | {fusion} | {mean:.3f} ms | {speedup} |")


if __name__ == "__main__":
    main()

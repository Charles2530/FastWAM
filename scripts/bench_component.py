"""Five-mode FastWAM component benchmark, including VAE and prompt encoding.

    conda activate FastWAM
    python scripts/bench_component.py \
      ckpt=checkpoints/fastwam_release/libero_uncond_2cam224.pt

Columns: 10-step eager, 1-step eager, 1-step CUDA Graph, asynchronous CUDA
Graph, and asynchronous CUDA Graph with group 13's five_ops_pair_norm kernels.
Compile includes VAE but excludes text encoding, which runs on every request
by default. +COMPONENT.cache_text_embeddings=true matches bench_latency's
cached-prompt scope and reports text_encode=0. Synthetic context is opt-in.

Components are CUDA-event intervals in the real request, not isolated kernels.
Async Video/Action intervals overlap; overlap_correction subtracts their shared
time. other = profiled wall time - interval union. Component rows plus that
correction and other equal total. total_uninstrumented is measured separately
without profiling events and is the latency to use for performance comparisons.
Profiling adds graph event nodes, so its total can exceed the uninstrumented one.
No synchronization is inserted between Video/Action blocks.

Overrides: +COMPONENT.modes=[1,2,3,4,5], +COMPONENT.warmup=10,
+COMPONENT.iters=100, +COMPONENT.graph_warmup=3,
+COMPONENT.output_json=artifacts/bench_component.json. Matching count/input
overrides under the former DRYRUN section remain accepted. Results include a
Markdown table, raw samples, timing semantics, and three-case output checks.
"""

from __future__ import annotations

from contextlib import contextmanager
import ctypes
from datetime import datetime
from functools import wraps
import gc
import json
import logging
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
from bench_latency import ActionRunner, CapturedCall, EagerActionRunner, _percentile
from fastwam_single_gpu_ops import SingleGPUFiveOpRunner

logger = logging.getLogger(__name__)
COMPONENTS = ("vae_encode", "text_encode", "video_prepare", "video_dit_prefill",
              "action_dit_denoise", "action_scheduler")
ROWS = (*COMPONENTS, "overlap_correction", "other", "total", "total_uninstrumented")


class Mode(NamedTuple):
    label: str
    steps: int
    graph: bool
    asynchronous: bool
    kernels: bool


MODES = {
    1: Mode("10 steps", 10, False, False, False),
    2: Mode("1 step", 1, False, False, False),
    3: Mode("1 step compile", 1, True, False, False),
    4: Mode("1 step compile async", 1, True, True, False),
    5: Mode("1 step compile async kernel", 1, True, True, True),
}


def component_config(cfg):
    defaults = OmegaConf.create({
        "warmup": 10, "iters": 100, "graph_warmup": 3, "seed": 42,
        "action_horizon": None, "prompt": "pick up the object",
        "cache_text_embeddings": False, "use_random_context": False,
        "verify": True, "atol": 0.02, "rtol": 0.0,
        "modes": list(MODES),
        "output_json": f"artifacts/bench_component_{datetime.now():%Y%m%d_%H%M%S}.json",
    })
    legacy = {k: v for k, v in cfg.get("DRYRUN", {}).items() if k in defaults}
    overrides = cfg.get("COMPONENT", {})
    unknown = set(overrides) - set(defaults)
    if unknown:
        raise ValueError(f"Unknown COMPONENT options: {sorted(unknown)}")
    result = OmegaConf.merge(defaults, legacy, overrides)
    if result.warmup < 0 or result.iters <= 0 or result.graph_warmup <= 0:
        raise ValueError("warmup must be >= 0; iters and graph_warmup must be > 0")
    if not result.modes or len(set(result.modes)) != len(result.modes) or any(i not in MODES for i in result.modes):
        raise ValueError("COMPONENT.modes must contain distinct IDs from 1 to 5")
    return result


def interval_accounting(intervals, wall_ms):
    """Partition measured wall time without counting concurrent intervals twice."""
    components = dict.fromkeys(COMPONENTS, 0.0)
    ordered = []
    for name, start, end in intervals:
        if end < start:
            raise ValueError(f"Negative CUDA interval for {name}")
        components[name] += end - start
        ordered.append((start, end))
    union, right = 0.0, float("-inf")
    for start, end in sorted(ordered):
        union += max(0.0, end - max(start, right))
        right = max(right, end)
    overlap = sum(components.values()) - union
    return dict(components, overlap_correction=-overlap, other=wall_ms - union, total=wall_ms)


class ComponentTimer:
    """Stable events are reused by capture and replay, including on PyTorch 2.7."""

    def __init__(self):
        try:
            runtime = ctypes.CDLL("libcudart.so.12")
        except OSError:
            runtime = ctypes.CDLL("libcudart.so")
        self.record_external = runtime.cudaEventRecordWithFlags
        self.record_external.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint)
        self.record_external.restype = ctypes.c_int
        self.origin = torch.cuda.Event(enable_timing=True)
        self.phase = "host"
        self.pool, self.counts = {}, {}
        self.used = {"host": [], "graph": []}

    def reset(self, phase):
        self.phase = phase
        self.counts = {}
        self.used[phase] = []

    def record(self, event):
        if torch.cuda.is_current_stream_capturing():
            error = self.record_external(event.cuda_event, torch.cuda.current_stream().cuda_stream, 1)
            if error:
                raise RuntimeError(f"cudaEventRecordWithFlags failed: {error}")
        else:
            event.record()

    def wrap(self, name, function):
        @wraps(function)
        def timed(*args, **kwargs):
            component = name(*args, **kwargs) if callable(name) else name
            index = self.counts.get(component, 0)
            self.counts[component] = index + 1
            key = self.phase, component, index
            if key not in self.pool:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("Component call structure changed during graph capture")
                self.pool[key] = tuple(torch.cuda.Event(enable_timing=True) for _ in range(2))
            start, end = self.pool[key]
            self.record(start)
            output = function(*args, **kwargs)
            self.record(end)
            self.used[self.phase].append((component, start, end))
            return output
        return timed

    def values(self, wall_ms):
        intervals = [(name, self.origin.elapsed_time(start), self.origin.elapsed_time(end))
                     for phase in ("host", "graph") for name, start, end in self.used[phase]]
        return interval_accounting(intervals, wall_ms)


@contextmanager
def instrument(model, runner, mode, timer):
    originals = []

    def install(obj, attr, component):
        owned = attr in vars(obj)
        original = getattr(obj, attr)
        originals.append((obj, attr, original, owned))
        setattr(obj, attr, timer.wrap(component, original))

    try:
        install(model, "_encode_input_image_latents_tensor", "vae_encode")
        install(model, "encode_prompt", "text_encode")
        install(model.video_expert, "prepare", "video_prepare")
        install(model.action_expert, "prepare", "action_dit_denoise")
        install(model.action_expert, "post", "action_dit_denoise")
        install(model.infer_action_scheduler, "step", "action_scheduler")
        if mode.asynchronous:
            def expert_component(expert, *args, **kwargs):
                return "video_dit_prefill" if expert is model.video_expert else "action_dit_denoise"
            install(runner, "project", expert_component)
            install(runner, "finish", expert_component)
        else:
            install(model.mot, "prefill_video_cache_tensor", "video_dit_prefill")
            install(model.mot, "forward_action_with_video_cache_tensor", "action_dit_denoise")
        yield
    finally:
        for obj, attr, original, owned in reversed(originals):
            if owned:
                setattr(obj, attr, original)
            else:
                delattr(obj, attr)


def run_samples(request, device, options, timer=None):
    samples, components = [], []
    for index in range(int(options.warmup) + int(options.iters)):
        torch.cuda.synchronize(device)
        if timer:
            timer.reset("host")
        start = time.perf_counter()
        if timer:
            timer.origin.record()
        output = request()
        torch.cuda.synchronize(device)
        elapsed = (time.perf_counter() - start) * 1000
        if index >= options.warmup:
            samples.append(elapsed)
            if timer:
                components.append(timer.values(elapsed))
    return samples, components, output


def validate(request, runner, cases, references, *, atol, rtol):
    original = runner.image_cpu, runner.proprio_cpu, runner.seed
    errors, outputs = [], []
    try:
        for (image, proprio, seed), reference in zip(cases, references):
            runner.image_cpu, runner.proprio_cpu, runner.seed = image, proprio, seed
            actual = request()
            torch.testing.assert_close(actual, reference, atol=atol, rtol=rtol)
            errors.append((actual - reference).abs().max().item())
            outputs.append(actual.clone())
    finally:
        runner.image_cpu, runner.proprio_cpu, runner.seed = original
    return errors, outputs


def benchmark_mode(model, mode_id, options, inputs, cases, references, rand_device, sigma_shift):
    mode = MODES[mode_id]
    image, proprio, context, mask, horizon = inputs
    cls = SingleGPUFiveOpRunner if mode.kernels else ActionRunner if mode.graph else EagerActionRunner
    runner = cls(model, image, proprio, context.clone(), mask.clone(), horizon, mode.steps,
                 model.device, int(options.seed), rand_device, sigma_shift=sigma_shift)
    encode_each_call = not options.cache_text_embeddings and not options.use_random_context
    if mode.graph:
        runner.run_gpu = runner.single_gpu_pipeline if mode.asynchronous else runner.sequential
    elif encode_each_call:
        runner.kwargs.update(prompt=str(options.prompt), context=None, context_mask=None)

    def request():
        if mode.graph and encode_each_call:
            current_context, current_mask = model.encode_prompt(str(options.prompt))
            runner.context.copy_(current_context)
            runner.context_mask.copy_(current_mask)
        return runner()

    result = {"mode": mode_id, "label": mode.label, "steps": mode.steps,
              "cuda_graph": mode.graph, "asynchronous": mode.asynchronous,
              "kernel_variant": "five_ops_pair_norm" if mode.kernels else None}
    eager_outputs = None
    if options.verify:
        result["eager_reference_errors"], eager_outputs = validate(
            request, runner, cases, references, atol=float(options.atol), rtol=float(options.rtol))
    if mode.graph:
        runner.capture(mode.asynchronous, int(options.graph_warmup))
    if options.verify:
        result["reference_errors"], _ = validate(
            request, runner, cases, references, atol=float(options.atol), rtol=float(options.rtol))
        result["graph_eager_errors"], clean_outputs = validate(
            request, runner, cases, eager_outputs, atol=0, rtol=0)
    clean_samples, _, _ = run_samples(request, model.device, options)
    result["uninstrumented_samples_ms"] = clean_samples
    timer = ComponentTimer()
    with instrument(model, runner, mode, timer):
        if mode.graph:
            function = runner.single_gpu_pipeline if mode.asynchronous else runner.sequential

            def profiled_graph():
                timer.reset("graph")
                return function()

            runner.stage_inputs()
            runner.captured = CapturedCall(profiled_graph, model.device, int(options.graph_warmup))
            runner.run_gpu = runner.captured
            timer.phase = "host"
        if options.verify:
            timer.reset("host")
            result["instrumented_reference_errors"], _ = validate(
                request, runner, cases, references, atol=float(options.atol), rtol=float(options.rtol))
            result["instrumentation_errors"], _ = validate(
                request, runner, cases, clean_outputs, atol=0, rtol=0)
        profile_samples, component_samples, _ = run_samples(request, model.device, options, timer)
    result["profiled_samples_ms"] = profile_samples
    result["component_samples_ms"] = component_samples
    result["components_ms"] = {name: statistics.fmean(row[name] for row in component_samples)
                               for name in ROWS if name != "total_uninstrumented"}
    result["components_ms"]["total_uninstrumented"] = statistics.fmean(clean_samples)
    result["profiling_overhead_ms"] = statistics.fmean(profile_samples) - statistics.fmean(clean_samples)
    result["uninstrumented_p50_ms"] = _percentile(clean_samples, 50)
    result["uninstrumented_p90_ms"] = _percentile(clean_samples, 90)
    torch.cuda.synchronize(model.device)
    del runner, timer
    gc.collect()
    torch.cuda.empty_cache()
    return result


def markdown_table(report):
    results = report["results"]
    lines = ["| Component | " + " | ".join(r["label"] for r in results) + " |",
             "| --- | " + " | ".join("---" for _ in results) + " |"]
    for row in ROWS:
        lines.append(f"| {row} | " + " | ".join(f"{r['components_ms'][row]:.3f} ms" for r in results) + " |")
    return "\n".join(lines)


def save_report(report, options):
    if options.output_json is None:
        return
    path = Path(str(options.output_json))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    note = ("Component CUDA intervals are measured in a separate instrumented pass. "
            "overlap_correction removes double-counted concurrent time; other is profiled wall time "
            "minus interval union. Components + overlap_correction + other = total. "
            "Use total_uninstrumented for latency comparisons. All totals include VAE. "
            f"Text encoding per request: {report['text_encoded_per_request']}. "
            "Text is not compiled. Setup and validation are excluded.\n")
    path.with_suffix(".md").write_text("# FastWAM Components\n\n" + markdown_table(report) + "\n\n" + note)


@hydra.main(config_path="../configs", config_name="sim_libero.yaml", version_base="1.3")
@torch.no_grad()
def main(cfg: DictConfig):
    logging.basicConfig(level=logging.INFO)
    options = component_config(cfg)
    device = torch.device(str(cfg.EVALUATION.device))
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("CUDA is required")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.set_device(device)
    torch.manual_seed(int(options.seed))
    dtype = {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[str(cfg.mixed_precision)]
    model = instantiate(cfg.model, model_dtype=dtype, device=str(device))
    if cfg.ckpt is not None:
        model.load_checkpoint(str(cfg.ckpt))
    model.eval()
    from fastwam.models.wan22.fastwam import FastWAM
    if type(model).infer_action is not FastWAM.infer_action:
        raise ValueError("This benchmark requires model=fastwam")
    if model.video_expert.video_attention_mask_mode != "first_frame_causal":
        raise ValueError("first_frame_causal video attention is required")
    height, width = map(int, cfg.data.train.video_size)
    horizon = int(cfg.data.train.num_frames) - 1 if options.action_horizon is None else int(options.action_horizon)
    if horizon <= 0 or height % 16 or width % 16:
        raise ValueError("Positive horizon and image dimensions divisible by 16 are required")
    torch.manual_seed(int(options.seed))
    image = torch.rand(1, 3, height, width) * 2 - 1
    proprio = None if model.proprio_dim is None else torch.randn(1, model.proprio_dim)
    if options.use_random_context:
        context = torch.randn(1, int(cfg.data.train.context_len), model.text_dim, device=device, dtype=dtype)
        mask = torch.ones(context.shape[:2], device=device, dtype=torch.bool)
    else:
        context, mask = model.encode_prompt(str(options.prompt))
        context, mask = context.to(dtype=dtype), mask.to(dtype=torch.bool)
    cases = [(image, proprio, int(options.seed)),
             (image * 0.5, None if proprio is None else proprio + 0.25, int(options.seed)),
             (image, proprio, int(options.seed) + 1)]
    sigma_shift, rand_device = cfg.EVALUATION.get("sigma_shift"), str(cfg.EVALUATION.rand_device)
    references = {}
    if options.verify:
        for steps in sorted({MODES[i].steps for i in options.modes}):
            references[steps] = [model.infer_action(
                prompt=None, context=context, context_mask=mask, input_image=im, proprio=pr,
                action_horizon=horizon, num_inference_steps=steps, seed=seed, rand_device=rand_device,
                sigma_shift=sigma_shift, compile_action_infer=False)["action"] for im, pr, seed in cases]
            if torch.equal(references[steps][0], references[steps][1]):
                raise RuntimeError("Changed-observation validation is ineffective")
    report = {"python": sys.executable, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(device),
              "checkpoint": cfg.ckpt, "dtype": str(dtype), "image_shape": list(image.shape),
              "action_horizon": horizon, "context_shape": list(context.shape),
              "text_encoded_per_request": not options.cache_text_embeddings and not options.use_random_context,
              "vae_included": True, "compile_backend": "explicit_cuda_graph",
              "component_timer": "CUDA event intervals with explicit overlap correction",
              "latency_timer": "synchronized wall time; uninstrumented and profiled passes",
              "cpu_threads": torch.get_num_threads(), "sigma_shift": sigma_shift, "rand_device": rand_device,
              "config": OmegaConf.to_container(options, resolve=True), "results": []}
    for mode_id in options.modes:
        logger.info("Benchmarking %s; text encoding per request=%s", MODES[mode_id].label, report["text_encoded_per_request"])
        result = benchmark_mode(model, mode_id, options, (image, proprio, context, mask, horizon),
                                cases, references.get(MODES[mode_id].steps, []), rand_device, sigma_shift)
        report["results"].append(result)
        save_report(report, options)
        logger.info("%s: %s", result["label"], result["components_ms"])
    print("\n" + markdown_table(report))
    print("total includes profiling events; total_uninstrumented is the performance reference.")
    print(f"Saved: {options.output_json}")


if __name__ == "__main__":
    main()

"""Diagnose the existing one-step pipeline with fixed-input microbenchmarks.

    python scripts/profile_pipeline.py \
      ckpt=checkpoints/fastwam_release/libero_uncond_2cam224.pt \
      +BENCH.groups=[4,5,6] +BENCH.iters=50 \
      +BENCH.output_json=artifacts/pipeline_profile.json

The no-copy/no-wait experiments reuse KV from the SAME observation. They are
diagnostic ablations, not valid inference paths for changing observations.
GPU event intervals include stream waits and host starvation, not just kernels.
"""

from functools import partial
import logging
import statistics
import time

import hydra
import torch

import bench_latency as bench_module
from bench_latency import CapturedCall, TwoGPUActionRunner, _percentile, _sync


logger = logging.getLogger(__name__)
original_benchmark = bench_module._benchmark


def measure(function, devices, event_device, iterations, warmup=5, intervals=None):
    for _ in range(warmup):
        function()
        _sync(devices)
    with torch.cuda.device(event_device):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
    wall, host, gpu = [], [], []
    segments = {name: [] for name in intervals or {}}
    for _ in range(iterations):
        _sync(devices)
        with torch.cuda.device(event_device):
            start_event.record()
        start = time.perf_counter()
        output = function()
        submitted = time.perf_counter()
        with torch.cuda.device(event_device):
            end_event.record()
        _sync(devices)
        wall.append((time.perf_counter() - start) * 1000)
        host.append((submitted - start) * 1000)
        gpu.append(start_event.elapsed_time(end_event))
        for name, (begin, end) in (intervals or {}).items():
            segments[name].append(begin.elapsed_time(end))
    result = {
        "wall_mean_ms": statistics.fmean(wall),
        "wall_p50_ms": _percentile(wall, 50),
        "wall_p90_ms": _percentile(wall, 90),
        "host_call_mean_ms": statistics.fmean(host),
        "stream_interval_mean_ms": statistics.fmean(gpu),
        "samples_ms": wall,
    }
    if segments:
        result["stream_segments_mean_ms"] = {
            name: statistics.fmean(values) for name, values in segments.items()}
    return result


def variant(runner, copy_kv=True, wait_kv=True, marks=None):
    caller = torch.cuda.current_stream(runner.video_device)
    action_caller = torch.cuda.current_stream(runner.action_device)
    if marks:
        marks["begin"].record(caller)
    _, context, mask, attention = runner.pre_video()
    if marks:
        marks["prepared"].record(caller)
    runner.video_stream.wait_stream(caller)
    if marks:
        marks["video_begin"].record(runner.video_stream)
    runner.copy_stream.wait_stream(caller)
    runner.action_stream.wait_stream(action_caller)
    with torch.cuda.stream(runner.action_stream), torch.cuda.stream(runner.copy_stream):
        runner.remote_context.copy_(context, non_blocking=True)
        runner.remote_mask.copy_(mask, non_blocking=True)
        runner.remote_attention.copy_(attention, non_blocking=True)
        runner.context_ready.record(runner.copy_stream)
    with torch.cuda.stream(runner.action_stream):
        runner.action_stream.wait_event(runner.context_ready)
        if marks:
            marks["action_begin"].record(runner.action_stream)
        runner.pre_action()
    for index, (vg, ag) in enumerate(runner.layers):
        with torch.cuda.stream(runner.video_stream):
            vio = vg()
            runner.ready[index].record(runner.video_stream)
            if index + 1 == len(runner.layers):
                runner.video_tail()
        with torch.cuda.stream(runner.action_stream), torch.cuda.stream(runner.copy_stream):
            if wait_kv:
                runner.copy_stream.wait_event(runner.ready[index])
            if copy_kv:
                runner.received_k[index].copy_(vio[1], non_blocking=True)
                runner.received_v[index].copy_(vio[2], non_blocking=True)
            runner.copy_ready[index].record(runner.copy_stream)
        with torch.cuda.stream(runner.action_stream):
            if wait_kv:
                runner.action_stream.wait_event(runner.copy_ready[index])
            output = ag()
    with torch.cuda.stream(runner.action_stream):
        if marks:
            marks["action_end"].record(runner.action_stream)
    if marks:
        marks["video_end"].record(runner.video_stream)
    caller.wait_stream(runner.video_stream)
    caller.wait_stream(runner.copy_stream)
    caller.wait_stream(runner.action_stream)
    action_caller.wait_stream(runner.action_stream)
    if marks:
        marks["end"].record(caller)
    return output


def dual_diagnostics(runner, iterations):
    results = {}
    reference = runner().clone()
    _sync(runner.devices)

    def record(name, function, devices=None, event_device=None, intervals=None):
        result = measure(function, devices or runner.devices,
                         event_device or runner.video_device, iterations, intervals=intervals)
        results[name] = result
        logger.info("%s: wall=%.3f ms, host call=%.3f ms, stream interval=%.3f ms",
                    name, result["wall_mean_ms"], result["host_call_mean_ms"],
                    result["stream_interval_mean_ms"])

    marks = {name: torch.cuda.Event(enable_timing=True) for name in
             ("begin", "prepared", "video_begin", "video_end", "action_begin", "action_end", "end")}
    intervals = {name: (marks[begin], marks[end]) for name, begin, end in (
        ("vae_and_prepare", "begin", "prepared"),
        ("video_branch_in_live_pipeline", "video_begin", "video_end"),
        ("action_branch_including_kv_waits", "action_begin", "action_end"),
        ("whole_pipeline", "begin", "end"),
    )}
    record("live_pipeline_timeline", partial(variant, runner, marks=marks), intervals=intervals)
    logger.info("Live pipeline segments: %s", results["live_pipeline_timeline"]["stream_segments_mean_ms"])

    record("live_pipeline_compute", runner.run_gpu)
    for name, copy_kv, wait_kv in [
        ("control_live_variant", True, True),
        ("fixed_input_no_kv_copy_keep_waits", False, True),
        ("fixed_input_preloaded_kv_no_waits", False, False),
    ]:
        runner()
        function = partial(variant, runner, copy_kv, wait_kv)
        actual = function().cpu()
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        record(name, function)
        results[name]["fixed_input_max_abs_error"] = (actual - reference).abs().max().item()

    runner()
    video, context, mask, attention = runner.pre_video.output
    action, delta = runner.pre_action.output[:2]
    model = runner.model
    local = [runner.video_device]
    remote = [runner.action_device]
    record("vae_plus_video_prepare", runner.pre_video, local)
    vae = CapturedCall(lambda: model._encode_input_image_latents_tensor(runner.image),
                       runner.video_device, 3)
    record("vae_only", vae, local)

    def video_body():
        return model.mot.prefill_video_cache_tensor(*video)

    video_graph = CapturedCall(video_body, runner.video_device, 3)
    record("video_30_layers_one_graph", video_graph, local)

    def video_fragments():
        for vg, _ in runner.layers:
            vg()
        runner.video_tail()

    record("video_30_layers_31_graphs", video_fragments, local)

    def video_fragments_original_stream(include_prepare=False):
        caller = torch.cuda.current_stream(runner.video_device)
        if include_prepare:
            runner.pre_video()
        runner.video_stream.wait_stream(caller)
        with torch.cuda.stream(runner.video_stream):
            video_fragments()
        caller.wait_stream(runner.video_stream)

    record("video_30_layers_31_graphs_original_stream", video_fragments_original_stream, local)
    record("vae_prepare_and_video_original_stream", partial(video_fragments_original_stream, True), local)

    def action_body():
        hidden = model.mot.forward_action_with_video_cache_tensor(
            *action[:5], runner.received_k, runner.received_v, action.attention_mask)
        return runner.post_step(hidden, delta, runner.noise)[0].float()

    action_graph = CapturedCall(action_body, runner.action_device, 3)
    record("action_30_layers_plus_head_one_graph_cached_kv", action_graph,
           remote, runner.action_device)
    torch.testing.assert_close(action_graph().cpu(), reference, rtol=0, atol=0)

    def action_fragments():
        with torch.cuda.device(runner.action_device):
            for _, ag in runner.layers:
                output = ag()
            return output

    record("action_after_first_qkv_plus_head_30_graphs_cached_kv", action_fragments,
           remote, runner.action_device)
    record("action_prepare_plus_first_qkv", runner.pre_action, remote, runner.action_device)

    def copy_all_kv():
        caller = torch.cuda.current_stream(runner.video_device)
        action_caller = torch.cuda.current_stream(runner.action_device)
        runner.copy_stream.wait_stream(caller)
        with torch.cuda.stream(runner.action_stream), torch.cuda.stream(runner.copy_stream):
            for index, (vg, _) in enumerate(runner.layers):
                runner.received_k[index].copy_(vg.output[1], non_blocking=True)
                runner.received_v[index].copy_(vg.output[2], non_blocking=True)
        caller.wait_stream(runner.copy_stream)
        action_caller.wait_stream(runner.copy_stream)

    record("kv_60_copies_no_compute", copy_all_kv)
    kv_bytes = sum(t.numel() * t.element_size()
                   for vg, _ in runner.layers for t in vg.output[1:3])
    results["kv_bytes_per_request"] = kv_bytes
    results["video_tokens"] = video.tokens.shape[1]
    results["action_tokens"] = action.tokens.shape[1]
    results["graphs_per_dual_request"] = 3 + 2 * len(runner.layers)
    results["video_parameters"] = sum(p.numel() for p in model.video_expert.parameters())
    results["action_parameters"] = sum(p.numel() for p in model.action_expert.parameters())
    results["notes"] = [
        "Ablations only validate a fixed observation with preloaded matching KV.",
        "Isolated stage times need not add up to concurrent end-to-end time.",
        "Host call time can overlap GPU work; do not add it to GPU intervals.",
        "Stream intervals include waits and host submission gaps.",
    ]
    runner()
    record("live_pipeline_compute_after_components", runner.run_gpu)
    return results


def profiled_benchmark(runner, label, bench):
    result = original_benchmark(runner, label, bench)
    iterations = min(50, int(bench.iters))
    runner()
    diagnostics = {
        "compute_only": measure(runner.run_gpu, runner.devices, runner.video_device, iterations),
        "input_staging_only": measure(runner.stage_inputs, runner.devices, runner.video_device, iterations),
    }
    output = runner.run_gpu()
    _sync(runner.devices)
    diagnostics["output_to_cpu_only"] = measure(
        lambda: output.cpu(), runner.devices, runner.action_device, iterations)
    if isinstance(runner, TwoGPUActionRunner):
        diagnostics.update(dual_diagnostics(runner, iterations))
    result["diagnostics"] = diagnostics
    logger.info("%s compute-only: %s", label,
                {k: v for k, v in diagnostics["compute_only"].items() if k != "samples_ms"})
    return result


@hydra.main(config_path="../configs", config_name="sim_libero.yaml", version_base="1.3")
def main(cfg):
    bench = bench_module._config(cfg)
    if any(group not in (4, 5, 6) for group in bench.groups):
        raise ValueError("Select graph groups only: +BENCH.groups=[4,5,6]")
    bench_module._benchmark = profiled_benchmark
    try:
        bench_module.main(cfg)
    finally:
        bench_module._benchmark = original_benchmark


if __name__ == "__main__":
    main()

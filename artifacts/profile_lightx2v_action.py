"""Fixed-input component diagnostics; official request timing still includes VAE."""

import ctypes
import gc
import json
import logging
from pathlib import Path
import statistics
import sys

ROOT = Path("/mnt/miaohua/charles/codes/LightX2V_fastwam")
sys.path.insert(0, str(ROOT / "scripts/fastwam"))
import bench_latency as b
import torch


def event_recorder():
    # PyTorch 2.7 lacks Event(external=True); use CUDA's graph event flag.
    runtime = ctypes.CDLL("libcudart.so.12")
    record = runtime.cudaEventRecordWithFlags
    record.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint)
    record.restype = ctypes.c_int

    def mark(event):
        flags = 1 if torch.cuda.is_current_stream_capturing() else 0
        error = record(event.cuda_event, torch.cuda.current_stream().cuda_stream, flags)
        if error:
            raise RuntimeError(f"cudaEventRecordWithFlags failed: {error}")
    return mark


def measure(graph, warmup=10, iterations=100, span=None):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples, spans = [], []
    for index in range(warmup + iterations):
        torch.cuda.synchronize()
        start.record()
        graph()
        end.record()
        end.synchronize()
        if index >= warmup:
            samples.append(start.elapsed_time(end))
            if span is not None:
                spans.append(span[0].elapsed_time(span[1]))
    result = {"mean_ms": statistics.fmean(samples), "p50_ms": b.percentile(samples, 50),
              "p90_ms": b.percentile(samples, 90), "samples_ms": samples}
    if spans:
        result["action_body_span_including_waits_ms"] = statistics.fmean(spans)
    return result


@torch.no_grad()
def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = b.config()
    cfg.BENCH.groups = [5, 7, 13]
    device = torch.device(cfg.BENCH.device)
    torch.cuda.set_device(device)
    model, vae = b.load_model(cfg, device)
    cases, context, mask, encoder = b.inputs(cfg, device)
    report = {"python": sys.executable, "torch": torch.__version__,
              "gpu": torch.cuda.get_device_name(device), "checkpoint": cfg.ckpt,
              "warmup": 10, "iterations": 100, "results": [],
              "scope": "Fixed-input GPU CUDA Graph components; not end-to-end request latency",
              "action_scope": "prepare + schedule + 30 blocks + head + scheduler; preloaded Video KV; no KV waits, VAE or CPU copies/RNG/output",
              "span_scope": "Action 30-block stream interval in full pipeline; includes KV waits and resource contention; excludes Action prepare/head/scheduler"}
    reference_runner = b.ActionRunner(model, vae, context, mask, cases[0], cfg.BENCH, 3)
    reference = reference_runner()
    del reference_runner
    output_path = Path(__file__).with_suffix(".json")
    record_event = event_recorder()
    for group in cfg.BENCH.groups:
        kwargs = {"backend": b.FIVE_OP_GROUPS[group]} if group in b.FIVE_OP_GROUPS else {}
        cls = b.FiveOpActionRunner if kwargs else b.ActionRunner
        runner = cls(model, vae, context, mask, cases[0], cfg.BENCH, group, **kwargs)
        own_reference = runner()
        video, ctx, ctx_mask, attention = runner.prepare_video()
        count = model.transformer_infer.num_layers

        def video_body():
            x, cache = video.tokens, []
            for index in range(count):
                io = runner.project("video", index, x, video)
                cache.append(io[1:3])
                x = runner.finish("video", index, io, video)
            return cache, x

        video_graph = b.CapturedCall(video_body, device, 3)
        cache = video_graph.output[0]
        ts, ds = runner.schedule()
        prepared_action = runner.prepare_action(runner.noise, ts[0], ctx, ctx_mask, attention)

        def action_blocks(prepared):
            x = prepared.tokens
            for index, kv in enumerate(cache):
                io = runner.project("action", index, x, prepared)
                x = runner.finish("action", index, io, prepared, kv)
            return x

        def action_complete():
            timesteps, deltas = runner.schedule()
            prepared = runner.prepare_action(runner.noise, timesteps[0], ctx, ctx_mask, attention)
            hidden = action_blocks(prepared)
            return runner.post_step(hidden, deltas[0], runner.noise)[0].float()

        action_graph = b.CapturedCall(action_complete, device, 3)
        actual = action_graph().cpu()
        torch.testing.assert_close(actual, reference, atol=0.02, rtol=0)
        torch.testing.assert_close(actual, own_reference, atol=0, rtol=0)
        body_graph = b.CapturedCall(lambda: action_blocks(prepared_action), device, 3)
        vae_graph = b.CapturedCall(lambda: runner.vae.encode(runner.image.unsqueeze(2)), device, 3)
        result = {"group": group, "backend": kwargs.get("backend", "lightx2v_fused" if group == 7 else "unfused"),
                  "action_complete": measure(action_graph), "action_30_blocks": measure(body_graph),
                  "video_30_blocks": measure(video_graph), "vae": measure(vae_graph),
                  "validation_max_abs_error": (actual - reference).abs().max().item(),
                  "isolated_vs_pipeline_max_abs_error": (actual - own_reference).abs().max().item()}

        marks = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        for event in marks:
            event.record()
        torch.cuda.synchronize()
        project, finish = runner.project, runner.finish

        def marked_project(expert, index, *args):
            if expert == "action" and index == 0:
                record_event(marks[0])
            return project(expert, index, *args)

        def marked_finish(expert, index, *args):
            output = finish(expert, index, *args)
            if expert == "action" and index == count - 1:
                record_event(marks[1])
            return output

        runner.project, runner.finish = marked_project, marked_finish
        pipeline_graph = b.CapturedCall(runner.pipeline, device, 3)
        torch.testing.assert_close(pipeline_graph().cpu(), own_reference, atol=0, rtol=0)
        result["pipeline_gpu_including_vae"] = measure(pipeline_graph, span=marks)
        report["results"].append(result)
        output_path.write_text(json.dumps(report, indent=2) + "\n")
        logging.info("Group %d: Action %.3f ms, Action blocks %.3f ms, Video %.3f ms, VAE %.3f ms, Action live span %.3f ms",
                     group, result["action_complete"]["mean_ms"], result["action_30_blocks"]["mean_ms"],
                     result["video_30_blocks"]["mean_ms"], result["vae"]["mean_ms"],
                     result["pipeline_gpu_including_vae"]["action_body_span_including_waits_ms"])
        runner.project, runner.finish = project, finish
        torch.cuda.synchronize()
        del pipeline_graph, vae_graph, body_graph, action_graph, video_graph, runner
        del project, finish, cache, prepared_action
        gc.collect()
        torch.cuda.empty_cache()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()

"""Native/optimized SP2/TP2 pipelines for groups 9-12."""

from datetime import timedelta
import gc
import json
import logging
from pathlib import Path
import statistics
import tempfile
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from hydra.utils import instantiate
from omegaconf import OmegaConf

from bench_latency import ActionRunner, CapturedCall, GROUPS, _percentile
from benchmark_fastwam_parallel import UlyssesSP2, apply_tensor_parallel
from fastwam.models.wan22.wan_video_dit import flash_attention
from fastwam_parallel_ops import FusedUlyssesSP2, OptimizedParallelMixin


logger = logging.getLogger(__name__)


class SequenceParallelActionRunner(ActionRunner):
    def __init__(self, *args, rank, groups, **kwargs):
        super().__init__(*args, **kwargs)
        self.sp_video = UlyssesSP2(self.model, rank, 2, groups["video"])
        self.sp_action = UlyssesSP2(self.model, rank, 2, groups["action"])

    @staticmethod
    def partition(prepared, sp):
        return prepared._replace(
            tokens=sp._split_sequence(prepared.tokens, 1),
            freqs=sp._split_sequence(prepared.freqs, 0),
            t_mod=(sp._split_sequence(prepared.t_mod, 1)
                   if prepared.t_mod.ndim == 4 else prepared.t_mod),
            context_mask=sp._split_sequence(prepared.context_mask, 1),
        )

    def project(self, expert, index, x, prepared):
        io = super().project(expert, index, x, prepared)
        sp = self.sp_video if expert is self.model.video_expert else self.sp_action
        return (*sp._sequence_to_heads(*io[:3]), *io[3:])

    def finish(self, expert, index, io, prepared, kv=None):
        sp = self.sp_video if expert is self.model.video_expert else self.sp_action
        k, v = io[1:3] if kv is None else (
            torch.cat([kv[0], io[1]], dim=1), torch.cat([kv[1], io[2]], dim=1))
        mixed = flash_attention(io[0], k, v, sp.local_heads, prepared.attention_mask)
        mixed = sp._heads_to_sequence(mixed)
        return self.model.mot._apply_expert_post_block_tensor(
            block=expert.blocks[index], residual_x=io[3], mixed_attn_out=mixed,
            gate_msa=io[4], shift_mlp=io[5], scale_mlp=io[6], gate_mlp=io[7],
            context=prepared.context, context_mask=prepared.context_mask)

    def prepare_video(self):
        video, context, mask, attention = super().prepare_video()
        return self.partition(video, self.sp_video), context, mask, attention

    def prepare_action(self, *args):
        return self.partition(super().prepare_action(*args), self.sp_action)

    def post_step(self, hidden, delta, latents):
        local_latents = self.sp_action._split_sequence(latents, 1)
        local = super().post_step(hidden, delta, local_latents)
        gathered = [torch.empty_like(local) for _ in range(2)]
        dist.all_gather(gathered, local)
        return torch.cat(gathered, dim=1)


class OptimizedTPRunner(OptimizedParallelMixin, ActionRunner):
    pass


class OptimizedSPRunner(OptimizedParallelMixin, SequenceParallelActionRunner):
    def __init__(self, *args, rank, groups, **kwargs):
        super().__init__(*args, rank=rank, groups=groups, **kwargs)
        self.sp_video = FusedUlyssesSP2(self.model, rank, 2, groups["video"])
        self.sp_action = FusedUlyssesSP2(self.model, rank, 2, groups["action"])

    def project(self, expert, index, x, prepared):
        io = super().project(expert, index, x, prepared)
        sp = self.sp_video if expert is self.model.video_expert else self.sp_action
        return (*sp._sequence_to_heads(*io[:3]), *io[3:])

    def attention(self, expert, io, prepared, kv):
        sp = self.sp_video if expert is self.model.video_expert else self.sp_action
        k, v = io[1:3] if kv is None else (
            torch.cat((kv[0], io[1]), dim=1), torch.cat((kv[1], io[2]), dim=1))
        return sp._heads_to_sequence(
            flash_attention(io[0], k, v, sp.local_heads, prepared.attention_mask))


def validate(runner, cases, references, bench, *, atol=None, rtol=None, outputs=None):
    original = runner.image_cpu, runner.proprio_cpu, runner.seed
    errors = []
    try:
        for (image, proprio, seed), reference in zip(cases, references):
            runner.image_cpu, runner.proprio_cpu, runner.seed = image, proprio, seed
            actual = runner()
            torch.cuda.synchronize(runner.video_device)
            failure = ""
            try:
                torch.testing.assert_close(actual, reference,
                                           rtol=float(bench.rtol) if rtol is None else rtol,
                                           atol=float(bench.atol) if atol is None else atol)
            except AssertionError as error:
                failure = str(error)
            status = torch.tensor([bool(failure), (actual - reference).abs().max().item()],
                                  device=runner.video_device, dtype=torch.float64)
            dist.all_reduce(status, op=dist.ReduceOp.MAX)
            if status[0].item():
                raise AssertionError(f"Distributed validation failed: {failure}; max error={status[1].item()}")
            errors.append(status[1].item())
            if outputs is not None:
                outputs.append(actual.clone())
    finally:
        runner.image_cpu, runner.proprio_cpu, runner.seed = original
    return errors


def benchmark(runner, bench, rank):
    device = runner.video_device

    def timed():
        # Rank coordination and the MAX reduction are outside the request timer.
        dist.barrier()
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        output = runner()
        torch.cuda.synchronize(device)
        elapsed = torch.tensor((time.perf_counter() - start) * 1000,
                               device=device, dtype=torch.float64)
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        return elapsed.item(), output

    for _ in range(int(bench.warmup)):
        timed()
    torch.cuda.reset_peak_memory_stats(device)
    samples = []
    for index in range(int(bench.iters)):
        elapsed, output = timed()
        samples.append(elapsed)
        if rank == 0 and ((index + 1) % 10 == 0 or index + 1 == int(bench.iters)):
            logger.info("sample %d/%d: %.3f ms", index + 1, bench.iters, elapsed)
    memory = [None, None]
    dist.all_gather_object(memory, (str(device), torch.cuda.max_memory_allocated(device) / 1024**3))
    return {
        "summary": {"mean_ms": statistics.fmean(samples), "p50_ms": _percentile(samples, 50),
                    "p90_ms": _percentile(samples, 90), "min_ms": min(samples), "max_ms": max(samples)},
        "samples_ms": samples, "action_shape": list(output.shape),
        "peak_allocated_gib": dict(memory),
    }


def run_variant(runner, cases, references, bench, rank, label, optimized):
    device = runner.video_device
    runner.run_gpu = runner.single_gpu_pipeline
    eager_outputs = []
    eager_errors = (validate(runner, cases, references, bench, rtol=0,
                             atol=float(bench.parallel_atol), outputs=eager_outputs)
                    if bool(bench.verify) else None)
    runner.stage_inputs()
    torch.cuda.synchronize(device)
    dist.barrier()
    setup_start = time.perf_counter()
    runner.captured = CapturedCall(runner.single_gpu_pipeline, device, int(bench.graph_warmup))
    runner.run_gpu = runner.captured
    setup_seconds = time.perf_counter() - setup_start
    errors = (validate(runner, cases, references, bench, rtol=0,
                       atol=float(bench.parallel_atol)) if bool(bench.verify) else None)
    graph_errors = (validate(runner, cases, eager_outputs, bench, rtol=0, atol=0)
                    if bool(bench.verify) else None)
    if rank == 0:
        logger.info("%s optimized=%s validation: eager=%s; graph=%s; graph/eager=%s",
                    label, optimized, eager_errors, errors, graph_errors)
    result = benchmark(runner, bench, rank)
    result.update(operator_fusion=optimized, parallel_optimized=optimized, communication_backend="torch",
                  operator_backend="triton_affine_norm_rope_fused_projection_layout" if optimized else "pytorch",
                  graph_setup_seconds=setup_seconds,
                  validation_max_abs_errors=errors, eager_validation_max_abs_errors=eager_errors,
                  graph_vs_sharded_eager_max_abs_errors=graph_errors,
                  reference_validation_rtol=0, reference_validation_atol=float(bench.parallel_atol))
    if bool(bench.parallel_profile):
        dist.barrier()
        torch.cuda.synchronize(device)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as profile:
            runner()
            torch.cuda.synchronize(device)
        path = Path(str(bench.output_json or "artifacts/parallel.json")).with_suffix("")
        path = Path(f"{path}_{label}_{'optimized' if optimized else 'baseline'}_rank{rank}.trace.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        profile.export_chrome_trace(str(path))
        result["profile_trace_rank0"] = str(path.with_name(path.name.replace(f"rank{rank}", "rank0")))
    if rank == 0:
        logger.info("%s optimized=%s: %s", label, optimized, result["summary"])
    return result


@torch.no_grad()
def worker(rank, device_indices, rendezvous, output_path, cfg_data, bench_data,
           group_ids, image, proprio, context_cpu, mask_cpu, references):
    logging.basicConfig(level=logging.INFO,
                        format=f"[rank {rank}] %(asctime)s %(name)s: %(message)s")
    device = torch.device("cuda", device_indices[rank])
    torch.cuda.set_device(device)
    torch.manual_seed(int(bench_data["seed"]))
    dist.init_process_group("nccl", init_method=rendezvous, rank=rank, world_size=2,
                            device_id=device, timeout=timedelta(minutes=5))
    try:
        # Separate communicators keep Video and Action collectives on independent streams.
        groups = {name: dist.new_group([0, 1], backend="nccl") for name in ("video", "action")}
        cfg, bench = OmegaConf.create(cfg_data), OmegaConf.create(bench_data)
        dtype = {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[str(cfg.mixed_precision)]
        model = instantiate(cfg.model, model_dtype=dtype, device=str(device))
        if cfg.ckpt is not None:
            model.load_checkpoint(str(cfg.ckpt))
        model.eval()
        context = context_cpu.to(device=device, dtype=dtype)
        context_mask = mask_cpu.to(device=device, dtype=torch.bool)
        horizon = int(cfg.data.train.num_frames) - 1 if bench.action_horizon is None else int(bench.action_horizon)
        seed = int(bench.seed)
        cases = [(image, proprio, seed),
                 (image * 0.5, None if proprio is None else proprio + 0.25, seed),
                 (image, proprio, seed + 1)]
        rand_device = str(cfg.EVALUATION.rand_device)
        if torch.device(rand_device).type == "cuda":
            rand_device = str(device)
        results = []
        # TP changes weights in place, so complete SP first when both are requested.
        execution_order = sorted(group_ids, key=lambda i: (
            GROUPS[i - 1].parallel_mode == "tp2", GROUPS[i - 1].fusion))
        tp_applied = False
        for group_id in execution_order:
            group = GROUPS[group_id - 1]
            label = group.label
            mode = group.parallel_mode
            if mode not in ("tp2", "ulysses_sp2"):
                raise ValueError(f"Not a distributed benchmark group: {group_id}")
            if rank == 0:
                logger.info("Setting up %s (worker execution order %s)", label, execution_order)
            if mode == "tp2" and not tp_applied:
                apply_tensor_parallel(model, rank, 2, groups)
                tp_applied = True
            kwargs = {"rank": rank, "groups": groups} if mode != "tp2" else {}
            optimized = group.fusion
            if mode == "ulysses_sp2":
                runner_class = OptimizedSPRunner if optimized else SequenceParallelActionRunner
            else:
                runner_class = OptimizedTPRunner if optimized else ActionRunner
            runner = runner_class(model, image, proprio, context, context_mask, horizon, 1,
                                  device, seed, rand_device, sigma_shift=cfg.EVALUATION.get("sigma_shift"),
                                  **kwargs)
            result = run_variant(runner, cases, references, bench, rank, label, optimized)
            torch.cuda.synchronize(device)
            del runner
            gc.collect()
            torch.cuda.empty_cache()
            dist.barrier()
            layers = model.mot.num_layers
            collective_count = ((5 if optimized else 7) * 2 * layers if mode == "tp2"
                                else 4 * layers + 1)
            result.update(group_id=group_id, label=label, action_steps=1,
                          devices=[f"cuda:{i}" for i in device_indices],
                          cuda_graph=True, pipeline=True, parallel_mode=mode,
                          vae_backend="cuda_graph_replicated_per_rank",
                          collectives_per_rank=collective_count,
                          action_replicated=False, video_kv_handoff=None,
                          local_compute_streams=2, local_cuda_graphs_per_request=1,
                          distributed_timing="max_rank_wall_ms; barrier and timing reduction excluded",
                          worker_execution_order=execution_order)
            results.append(result)
            if rank == 0:
                logger.info("%s: %s", label, result["summary"])
                Path(output_path).write_text(json.dumps(results, indent=2) + "\n")
    finally:
        dist.destroy_process_group()


def run_parallel_groups(cfg, bench, group_ids, devices, image, proprio, context, mask, references):
    with tempfile.TemporaryDirectory(prefix="fastwam_parallel_") as directory:
        output_path = str(Path(directory) / "results.json")
        rendezvous = (Path(directory) / "rendezvous").as_uri()
        mp.spawn(worker, nprocs=2, join=True, args=(
            [device.index for device in devices], rendezvous, output_path,
            OmegaConf.to_container(cfg, resolve=True), OmegaConf.to_container(bench, resolve=True),
            group_ids, image, proprio, context.cpu(), mask.cpu(), references))
        return {result["label"]: result for result in json.loads(Path(output_path).read_text())}

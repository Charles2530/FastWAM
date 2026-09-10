"""Benchmark eager FastWAM with expert pipeline, tensor parallelism, or Ulysses."""

from __future__ import annotations

import logging
import os
import statistics
import time

import hydra
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from fastwam.models.wan22.wan_video_dit import flash_attention


logger = logging.getLogger(__name__)


def _config(cfg: DictConfig) -> DictConfig:
    defaults = OmegaConf.create(
        {
            "warmup": 10,
            "iters": 30,
            "seed": 42,
            "action_horizon": None,
            "mode": "expert_pipeline",
            "tune_pipeline": False,
            "compute_priority": 0,
            "comm_priority": -1,
            "pack_kv": False,
            "action_qkv_first": False,
        }
    )
    return OmegaConf.merge(defaults, cfg.get("PARALLEL", {}))


def _dtype(name: str) -> torch.dtype:
    return {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[str(name)]


def _sync(devices) -> None:
    for device in devices:
        torch.cuda.synchronize(device)


def _time_call(fn, devices) -> tuple[float, torch.Tensor]:
    _sync(devices)
    start = time.perf_counter()
    output = fn()
    _sync(devices)
    return time.perf_counter() - start, output


def _benchmark(label: str, fn, devices, warmup: int, iters: int):
    for index in range(warmup):
        elapsed, _ = _time_call(fn, devices)
        logger.info("%s warmup %d/%d: %.3f ms", label, index + 1, warmup, elapsed * 1e3)
    times = []
    output = None
    for index in range(iters):
        elapsed, output = _time_call(fn, devices)
        times.append(elapsed)
        logger.info("%s iteration %d/%d: %.3f ms", label, index + 1, iters, elapsed * 1e3)
    logger.info(
        "%s mean=%.3f ms median=%.3f ms min=%.3f ms max=%.3f ms",
        label,
        statistics.mean(times) * 1e3,
        statistics.median(times) * 1e3,
        min(times) * 1e3,
        max(times) * 1e3,
    )
    return times, output


class ColumnParallelLinear(nn.Module):
    def __init__(self, linear: nn.Linear, rank: int, world_size: int):
        super().__init__()
        if linear.out_features % world_size:
            raise ValueError(
                f"Cannot column-shard {linear.out_features} across {world_size} ranks."
            )
        self.in_features = linear.in_features
        self.out_features = linear.out_features // world_size
        self.weight = nn.Parameter(
            linear.weight.chunk(world_size, dim=0)[rank].contiguous(), requires_grad=False
        )
        self.bias = (
            None
            if linear.bias is None
            else nn.Parameter(
                linear.bias.chunk(world_size, dim=0)[rank].contiguous(),
                requires_grad=False,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class RowParallelLinear(nn.Module):
    def __init__(self, linear: nn.Linear, rank: int, world_size: int, group=None):
        super().__init__()
        self.group = group
        if linear.in_features % world_size:
            raise ValueError(f"Cannot row-shard {linear.in_features} across {world_size} ranks.")
        self.in_features = linear.in_features // world_size
        self.out_features = linear.out_features
        self.weight = nn.Parameter(
            linear.weight.chunk(world_size, dim=1)[rank].contiguous(), requires_grad=False
        )
        self.bias = (
            None
            if linear.bias is None
            else nn.Parameter(linear.bias.detach().clone(), requires_grad=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = F.linear(x, self.weight, None)
        dist.all_reduce(output, group=self.group)
        if self.bias is not None:
            output = output + self.bias
        return output


class TensorParallelRMSNorm(nn.Module):
    def __init__(self, norm: nn.Module, rank: int, world_size: int, group=None):
        super().__init__()
        self.group = group
        if norm.weight.numel() % world_size:
            raise ValueError(
                f"Cannot shard RMSNorm dim {norm.weight.numel()} across {world_size} ranks."
            )
        self.eps = norm.eps
        self.global_dim = norm.weight.numel()
        self.weight = nn.Parameter(
            norm.weight.chunk(world_size, dim=0)[rank].contiguous(), requires_grad=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        square_sum = x.float().square().sum(dim=-1, keepdim=True)
        dist.all_reduce(square_sum, group=self.group)
        normalized = x.float() * torch.rsqrt(square_sum / self.global_dim + self.eps)
        return normalized.to(dtype) * self.weight


def apply_tensor_parallel(model, rank: int, world_size: int, groups=None) -> None:
    for name, expert in (("video", model.video_expert), ("action", model.action_expert)):
        if expert.num_heads % world_size:
            raise ValueError("Attention heads must be divisible by the TP size.")
        group = None if groups is None else groups[name]
        expert.num_heads //= world_size
        for block in expert.blocks:
            block.num_heads //= world_size
            for attention in (block.self_attn, block.cross_attn):
                attention.q = ColumnParallelLinear(attention.q, rank, world_size)
                attention.k = ColumnParallelLinear(attention.k, rank, world_size)
                attention.v = ColumnParallelLinear(attention.v, rank, world_size)
                attention.o = RowParallelLinear(attention.o, rank, world_size, group)
                attention.norm_q = TensorParallelRMSNorm(
                    attention.norm_q, rank, world_size, group
                )
                attention.norm_k = TensorParallelRMSNorm(
                    attention.norm_k, rank, world_size, group
                )
                attention.num_heads //= world_size
                attention.attn_hidden_dim //= world_size
            block.ffn[0] = ColumnParallelLinear(block.ffn[0], rank, world_size)
            block.ffn[2] = RowParallelLinear(block.ffn[2], rank, world_size, group)
    model.mot.num_heads //= world_size


class DistributedExpertPipeline:
    def __init__(
        self,
        model,
        rank: int,
        video_seq_len: int,
        compute_priority: int = -1,
        comm_priority: int = -1,
        pack_kv: bool = False,
        action_qkv_first: bool = True,
    ):
        self.model = model
        self.rank = rank
        self.device = torch.device(f"cuda:{rank}")
        self.compute_stream = torch.cuda.Stream(device=self.device, priority=compute_priority)
        self.comm_stream = torch.cuda.Stream(device=self.device, priority=comm_priority)
        self.ready = [torch.cuda.Event() for _ in range(model.mot.num_layers)]
        self.video_seq_len = video_seq_len
        self.pack_kv = pack_kv
        self.action_qkv_first = action_qkv_first
        if rank == 1:
            shape = (1, video_seq_len, model.mot.num_heads * model.mot.attn_head_dim)
            if pack_kv:
                self.cache_kv = [
                    torch.empty((2, *shape), device=self.device, dtype=model.torch_dtype)
                    for _ in range(model.mot.num_layers)
                ]
            else:
                self.cache_k = [
                    torch.empty(shape, device=self.device, dtype=model.torch_dtype)
                    for _ in range(model.mot.num_layers)
                ]
                self.cache_v = [torch.empty_like(tensor) for tensor in self.cache_k]

    def _prepare_video(self, input_image, proprio, prompt):
        input_image = input_image.to(device=self.device, dtype=self.model.torch_dtype)
        first_frame_latents = self.model._encode_input_image_latents_tensor(input_image=input_image)
        context, context_mask = self.model.encode_prompt(prompt)
        context, context_mask = self.model._append_proprio_to_context(
            context=context, context_mask=context_mask, proprio=proprio
        )
        timestep = torch.zeros(
            (first_frame_latents.shape[0],),
            dtype=first_frame_latents.dtype,
            device=self.device,
        )
        tokens, _, t_mod, context, context_mask, freqs, _, _, _, _ = (
            self.model.video_expert.prepare(
                x=first_frame_latents,
                timestep=timestep,
                context=context,
                context_mask=context_mask,
                action=None,
                fuse_vae_embedding_in_latents=bool(
                    self.model.video_expert.fuse_vae_embedding_in_latents
                ),
            )
        )
        mask = self.model.video_expert.build_video_to_video_mask(
            video_seq_len=tokens.shape[1],
            video_tokens_per_frame=tokens.shape[1],
            device=self.device,
        )
        return tokens, t_mod, context, context_mask, freqs, mask

    def _prepare_action(self, proprio, prompt, horizon, sigma_shift, seed):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        latents = torch.randn(
            (1, horizon, self.model.action_expert.action_dim),
            generator=generator,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.model.torch_dtype)
        context, context_mask = self.model.encode_prompt(prompt)
        context, context_mask = self.model._append_proprio_to_context(
            context=context, context_mask=context_mask, proprio=proprio
        )
        timesteps, deltas = self.model.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=1,
            device=self.device,
            dtype=latents.dtype,
            shift_override=sigma_shift,
        )
        timestep = timesteps[0].unsqueeze(0).to(device=self.device, dtype=latents.dtype)
        tokens, _, t_mod, context, context_mask, freqs = self.model.action_expert.prepare(
            action_tokens=latents,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
        )
        mask = self.model._build_mot_attention_mask(
            video_seq_len=self.video_seq_len,
            action_seq_len=horizon,
            video_tokens_per_frame=self.video_seq_len,
            device=self.device,
        )[self.video_seq_len :, :]
        return tokens, t_mod, context, context_mask, freqs, mask, latents, deltas[0]

    def _run_video(self, prepared) -> None:
        x, t_mod, context, context_mask, freqs, mask = prepared
        works = []
        source_kv = []
        self.compute_stream.wait_stream(torch.cuda.current_stream(self.device))
        for layer_idx, block in enumerate(self.model.video_expert.blocks):
            with torch.cuda.stream(self.compute_stream):
                q, k, v, residual, gate, shift, scale, gate_mlp, _ = (
                    self.model.mot._build_expert_attention_io(
                        self.model.video_expert, block, x, freqs, t_mod
                    )
                )
                self.ready[layer_idx].record(self.compute_stream)
                if layer_idx + 1 < self.model.mot.num_layers:
                    mixed = flash_attention(q, k, v, self.model.mot.num_heads, mask)
                    x = self.model.mot._apply_expert_post_block_tensor(
                        block,
                        residual,
                        mixed,
                        gate,
                        shift,
                        scale,
                        gate_mlp,
                        context,
                        context_mask,
                    )
            with torch.cuda.stream(self.comm_stream):
                self.comm_stream.wait_event(self.ready[layer_idx])
                if self.pack_kv:
                    packed = torch.stack((k, v))
                    works.extend(
                        dist.batch_isend_irecv([dist.P2POp(dist.isend, packed, 1)])
                    )
                    source_kv.append(packed)
                else:
                    works.extend(
                        dist.batch_isend_irecv(
                            [
                                dist.P2POp(dist.isend, k, 1),
                                dist.P2POp(dist.isend, v, 1),
                            ]
                        )
                    )
                    source_kv.append((k, v))
        self.compute_stream.synchronize()
        self.comm_stream.synchronize()
        for work in works:
            work.wait()

    def _receive_kv(self, layer_idx: int):
        with torch.cuda.stream(self.comm_stream):
            if self.pack_kv:
                layer_works = dist.batch_isend_irecv(
                    [dist.P2POp(dist.irecv, self.cache_kv[layer_idx], 0)]
                )
                k_video, v_video = self.cache_kv[layer_idx]
            else:
                layer_works = dist.batch_isend_irecv(
                    [
                        dist.P2POp(dist.irecv, self.cache_k[layer_idx], 0),
                        dist.P2POp(dist.irecv, self.cache_v[layer_idx], 0),
                    ]
                )
                k_video = self.cache_k[layer_idx]
                v_video = self.cache_v[layer_idx]
            for work in layer_works:
                work.wait()
            self.ready[layer_idx].record(self.comm_stream)
        return layer_works, k_video, v_video

    def _run_action(self, prepared) -> torch.Tensor:
        x, t_mod, context, context_mask, freqs, mask, latents, delta = prepared
        works = []
        self.compute_stream.wait_stream(torch.cuda.current_stream(self.device))
        for layer_idx, block in enumerate(self.model.action_expert.blocks):
            if self.action_qkv_first:
                with torch.cuda.stream(self.compute_stream):
                    aq, ak, av, residual, gate, shift, scale, gate_mlp, _ = (
                        self.model.mot._build_expert_attention_io(
                            self.model.action_expert, block, x, freqs, t_mod
                        )
                    )
                layer_works, k_video, v_video = self._receive_kv(layer_idx)
            else:
                layer_works, k_video, v_video = self._receive_kv(layer_idx)
                with torch.cuda.stream(self.compute_stream):
                    aq, ak, av, residual, gate, shift, scale, gate_mlp, _ = (
                        self.model.mot._build_expert_attention_io(
                            self.model.action_expert, block, x, freqs, t_mod
                        )
                    )
            works.extend(layer_works)
            with torch.cuda.stream(self.compute_stream):
                self.compute_stream.wait_event(self.ready[layer_idx])
                mixed = flash_attention(
                    aq,
                    torch.cat([k_video, ak], dim=1),
                    torch.cat([v_video, av], dim=1),
                    self.model.mot.num_heads,
                    mask,
                )
                x = self.model.mot._apply_expert_post_block_tensor(
                    block,
                    residual,
                    mixed,
                    gate,
                    shift,
                    scale,
                    gate_mlp,
                    context,
                    context_mask,
                )
        with torch.cuda.stream(self.compute_stream):
            prediction = self.model.action_expert.post(x)
            output = self.model.infer_action_scheduler.step(prediction, delta, latents)
        self.compute_stream.synchronize()
        self.comm_stream.synchronize()
        for work in works:
            work.wait()
        return output[0].to(device="cpu", dtype=torch.float32)

    @torch.no_grad()
    def infer(self, input_image, proprio, prompt, horizon, sigma_shift, seed):
        if self.rank == 0:
            self._run_video(self._prepare_video(input_image, proprio, prompt))
            return None
        return self._run_action(
            self._prepare_action(proprio, prompt, horizon, sigma_shift, seed)
        )


class UlyssesSP2:
    def __init__(self, model, rank: int, world_size: int, group=None):
        if world_size != 2:
            raise ValueError("This benchmark implements Ulysses SP2 only.")
        if model.mot.num_heads % world_size:
            raise ValueError("Attention heads must be divisible by the SP size.")
        self.model = model
        self.group = group
        self.rank = rank
        self.world_size = world_size
        self.num_heads = model.mot.num_heads
        self.local_heads = self.num_heads // world_size
        self.head_dim = model.mot.attn_head_dim
        self.device = torch.device(f"cuda:{rank}")

    def _split_sequence(self, tensor: torch.Tensor, dim: int) -> torch.Tensor:
        if tensor.shape[dim] % self.world_size:
            raise ValueError(
                f"Sequence length {tensor.shape[dim]} is not divisible by SP{self.world_size}."
            )
        return tensor.chunk(self.world_size, dim=dim)[self.rank].contiguous()

    def _sequence_to_heads(
        self, *tensors: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        batch, local_seq, width = tensors[0].shape
        expected_width = self.num_heads * self.head_dim
        if width != expected_width:
            raise ValueError(f"Expected QKV width {expected_width}, got {width}.")
        packed = torch.stack(tensors).view(
            len(tensors),
            batch,
            local_seq,
            self.world_size,
            self.local_heads,
            self.head_dim,
        )
        send = packed.permute(3, 0, 1, 2, 4, 5).contiguous()
        received = torch.empty_like(send)
        dist.all_to_all_single(received, send, group=self.group)
        received = received.permute(1, 2, 0, 3, 4, 5).contiguous()
        full_seq = local_seq * self.world_size
        return tuple(
            tensor.reshape(batch, full_seq, self.local_heads * self.head_dim)
            for tensor in received
        )

    def _heads_to_sequence(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, full_seq, width = tensor.shape
        local_seq = full_seq // self.world_size
        expected_width = self.local_heads * self.head_dim
        if full_seq % self.world_size or width != expected_width:
            raise ValueError(f"Invalid Ulysses attention output shape {tuple(tensor.shape)}.")
        send = (
            tensor.view(
                batch,
                self.world_size,
                local_seq,
                self.local_heads,
                self.head_dim,
            )
            .permute(1, 0, 2, 3, 4)
            .contiguous()
        )
        received = torch.empty_like(send)
        dist.all_to_all_single(received, send, group=self.group)
        return (
            received.permute(1, 2, 0, 3, 4)
            .contiguous()
            .view(batch, local_seq, self.num_heads * self.head_dim)
        )

    def _prefill_video(
        self,
        tokens,
        freqs,
        t_mod,
        context,
        context_mask,
        attention_mask,
    ):
        x = self._split_sequence(tokens, dim=1)
        local_freqs = self._split_sequence(freqs, dim=0)
        local_t_mod = self._split_sequence(t_mod, dim=1)
        local_context_mask = self._split_sequence(context_mask, dim=1)
        cache_k = []
        cache_v = []
        for layer_idx, block in enumerate(self.model.video_expert.blocks):
            q, k, v, residual, gate, shift, scale, gate_mlp, _ = (
                self.model.mot._build_expert_attention_io(
                    self.model.video_expert,
                    block,
                    x,
                    local_freqs,
                    local_t_mod,
                )
            )
            q, k, v = self._sequence_to_heads(q, k, v)
            cache_k.append(k)
            cache_v.append(v)
            if layer_idx + 1 == self.model.mot.num_layers:
                continue
            mixed = flash_attention(
                q,
                k,
                v,
                self.local_heads,
                attention_mask,
            )
            mixed = self._heads_to_sequence(mixed)
            x = self.model.mot._apply_expert_post_block_tensor(
                block,
                residual,
                mixed,
                gate,
                shift,
                scale,
                gate_mlp,
                context,
                local_context_mask,
            )
        return cache_k, cache_v

    def _denoise_action(
        self,
        tokens,
        freqs,
        t_mod,
        context,
        context_mask,
        attention_mask,
        video_cache_k,
        video_cache_v,
    ):
        x = self._split_sequence(tokens, dim=1)
        local_freqs = self._split_sequence(freqs, dim=0)
        local_context_mask = self._split_sequence(context_mask, dim=1)
        for layer_idx, block in enumerate(self.model.action_expert.blocks):
            q, k, v, residual, gate, shift, scale, gate_mlp, _ = (
                self.model.mot._build_expert_attention_io(
                    self.model.action_expert,
                    block,
                    x,
                    local_freqs,
                    t_mod,
                )
            )
            q, k, v = self._sequence_to_heads(q, k, v)
            mixed = flash_attention(
                q,
                torch.cat([video_cache_k[layer_idx], k], dim=1),
                torch.cat([video_cache_v[layer_idx], v], dim=1),
                self.local_heads,
                attention_mask,
            )
            mixed = self._heads_to_sequence(mixed)
            x = self.model.mot._apply_expert_post_block_tensor(
                block,
                residual,
                mixed,
                gate,
                shift,
                scale,
                gate_mlp,
                context,
                local_context_mask,
            )
        return self.model.action_expert.post(x)

    @torch.no_grad()
    def infer(self, input_image, proprio, prompt, horizon, sigma_shift, seed):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        latents = torch.randn(
            (1, horizon, self.model.action_expert.action_dim),
            generator=generator,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.model.torch_dtype)
        input_image = input_image.to(device=self.device, dtype=self.model.torch_dtype)
        proprio = proprio.to(device=self.device, dtype=self.model.torch_dtype)
        first_frame_latents = self.model._encode_input_image_latents_tensor(input_image)
        context, context_mask = self.model.encode_prompt(prompt)
        context, context_mask = self.model._append_proprio_to_context(
            context=context,
            context_mask=context_mask,
            proprio=proprio,
        )

        timestep_video = torch.zeros(
            (first_frame_latents.shape[0],),
            device=self.device,
            dtype=first_frame_latents.dtype,
        )
        (
            video_tokens,
            _,
            video_t_mod,
            video_context,
            video_context_mask,
            video_freqs,
            _,
            _,
            _,
            tokens_per_frame,
        ) = self.model.video_expert.prepare(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=bool(
                self.model.video_expert.fuse_vae_embedding_in_latents
            ),
        )
        video_seq_len = video_tokens.shape[1]
        attention_mask = self.model._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=horizon,
            video_tokens_per_frame=tokens_per_frame,
            device=self.device,
        )
        cache_k, cache_v = self._prefill_video(
            video_tokens,
            video_freqs,
            video_t_mod,
            video_context,
            video_context_mask,
            attention_mask[:video_seq_len, :video_seq_len],
        )

        timesteps, deltas = self.model.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=1,
            device=self.device,
            dtype=latents.dtype,
            shift_override=sigma_shift,
        )
        timestep_action = timesteps[0].unsqueeze(0).to(
            device=self.device, dtype=latents.dtype
        )
        action_tokens, _, action_t_mod, action_context, action_context_mask, action_freqs = (
            self.model.action_expert.prepare(
                action_tokens=latents,
                timestep=timestep_action,
                context=context,
                context_mask=context_mask,
            )
        )
        prediction = self._denoise_action(
            action_tokens,
            action_freqs,
            action_t_mod,
            action_context,
            action_context_mask,
            attention_mask[video_seq_len:, :],
            cache_k,
            cache_v,
        )
        local_latents = self._split_sequence(latents, dim=1)
        local_output = self.model.infer_action_scheduler.step(
            prediction, deltas[0], local_latents
        )
        gathered = [torch.empty_like(local_output) for _ in range(self.world_size)]
        dist.all_gather(gathered, local_output, group=self.group)
        return torch.cat(gathered, dim=1)[0].to(device="cpu", dtype=torch.float32)


def _distributed_time_call(fn, device):
    dist.barrier()
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    output = fn()
    torch.cuda.synchronize(device)
    elapsed = torch.tensor(time.perf_counter() - start, device=device)
    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    return float(elapsed.item()), output


def _distributed_pipeline_main(cfg: DictConfig, bench: DictConfig) -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    if dist.get_world_size() != 2:
        raise ValueError("Distributed expert pipeline requires exactly two ranks.")
    if rank != local_rank:
        raise ValueError(f"Expected single-node rank==local_rank, got {rank} and {local_rank}.")
    # Deliberately load the complete checkpoint on both ranks; execution is expert-specific.
    model = instantiate(cfg.model, model_dtype=_dtype(cfg.mixed_precision), device=str(device))
    if cfg.ckpt is not None:
        model.load_checkpoint(str(cfg.ckpt))
    model.eval()

    height, width = (int(value) for value in cfg.data.train.video_size)
    horizon = (
        int(cfg.data.train.num_frames) - 1
        if bench.action_horizon is None
        else int(bench.action_horizon)
    )
    if rank == 0:
        logger.info("benchmark action_horizon=%d", horizon)
    torch.manual_seed(int(bench.seed))
    input_image = torch.rand(1, 3, height, width) * 2 - 1
    proprio = torch.randn(1, int(cfg.data.train.processor.proprio_output_dim))
    prompt = "pick up the object"
    serial_kwargs = {
        "prompt": prompt,
        "input_image": input_image,
        "action_horizon": horizon,
        "proprio": proprio,
        "num_inference_steps": 1,
        "sigma_shift": cfg.EVALUATION.get("sigma_shift"),
        "seed": int(bench.seed),
        "rand_device": "cpu",
        "compile_action_infer": False,
    }
    if rank == 0:
        serial_times, reference = _benchmark(
            "single_gpu_serial",
            lambda: model.infer_action(**serial_kwargs)["action"],
            [device],
            int(bench.warmup),
            int(bench.iters),
        )
    else:
        serial_times = []
        reference = torch.empty((horizon, model.action_expert.action_dim), dtype=torch.float32)
    dist.barrier()
    reference_device = reference.to(device)
    dist.broadcast(reference_device, src=0)
    reference = reference_device.cpu()

    latent_h = height // int(model.vae.upsampling_factor)
    latent_w = width // int(model.vae.upsampling_factor)
    video_seq_len = (
        latent_h // int(model.video_expert.patch_size[1])
    ) * (latent_w // int(model.video_expert.patch_size[2]))

    def create_pipeline(params):
        return DistributedExpertPipeline(
            model,
            rank,
            video_seq_len,
            compute_priority=params[0],
            comm_priority=params[1],
            pack_kv=params[2],
            action_qkv_first=params[3],
        )

    def create_call(pipeline):
        return lambda: pipeline.infer(
            input_image,
            proprio,
            prompt,
            horizon,
            cfg.EVALUATION.get("sigma_shift"),
            int(bench.seed),
        )

    def check_output(label, candidate):
        if rank == 1:
            value = float((reference - candidate).abs().max())
        else:
            value = 0.0
        max_diff = torch.tensor(value, device=device)
        dist.all_reduce(max_diff, op=dist.ReduceOp.MAX)
        if rank == 0:
            logger.info("%s correctness max_abs_diff=%.8f", label, float(max_diff.item()))
        if float(max_diff.item()) > 1e-5:
            raise AssertionError(f"{label} max diff is {float(max_diff.item())}")

    selected = (
        int(bench.compute_priority),
        int(bench.comm_priority),
        bool(bench.pack_kv),
        bool(bench.action_qkv_first),
    )
    if bool(bench.tune_pipeline):
        variants = [
            ("high_compute_high_comm", (-1, -1, False, True)),
            ("default_priorities", (0, 0, False, True)),
            ("high_comm", (0, -1, False, True)),
            ("packed_kv", (0, -1, True, True)),
            ("receive_first", (0, -1, False, False)),
        ]
        means = []
        for label, params in variants:
            pipeline = create_pipeline(params)
            pipeline_fn = create_call(pipeline)
            _, candidate = _distributed_time_call(pipeline_fn, device)
            check_output(label, candidate)
            _distributed_time_call(pipeline_fn, device)
            times = [_distributed_time_call(pipeline_fn, device)[0] for _ in range(5)]
            mean = statistics.mean(times)
            means.append((mean, params, label))
            if rank == 0:
                logger.info("tune %s mean=%.3f ms", label, mean * 1e3)
            del pipeline_fn, pipeline
            torch.cuda.empty_cache()
        _, selected, selected_label = min(means, key=lambda item: item[0])
        if rank == 0:
            logger.info("tune selected=%s params=%s", selected_label, selected)

    pipeline = create_pipeline(selected)
    pipeline_fn = create_call(pipeline)
    _, candidate = _distributed_time_call(pipeline_fn, device)
    check_output("distributed_pipeline", candidate)

    for index in range(int(bench.warmup)):
        elapsed, _ = _distributed_time_call(pipeline_fn, device)
        if rank == 0:
            logger.info(
                "distributed_pipeline warmup %d/%d: %.3f ms",
                index + 1,
                int(bench.warmup),
                elapsed * 1e3,
            )
    pipeline_times = []
    for index in range(int(bench.iters)):
        elapsed, _ = _distributed_time_call(pipeline_fn, device)
        pipeline_times.append(elapsed)
        if rank == 0:
            logger.info(
                "distributed_pipeline iteration %d/%d: %.3f ms",
                index + 1,
                int(bench.iters),
                elapsed * 1e3,
            )
    if rank == 0:
        logger.info(
            "distributed result serial=%.3f ms pipeline=%.3f ms speedup=%.3fx",
            statistics.mean(serial_times) * 1e3,
            statistics.mean(pipeline_times) * 1e3,
            statistics.mean(serial_times) / statistics.mean(pipeline_times),
        )
    dist.destroy_process_group()


def _distributed_tp_main(cfg: DictConfig, bench: DictConfig) -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 2:
        raise ValueError("Tensor-parallel benchmark requires exactly two ranks.")
    if rank != local_rank:
        raise ValueError(f"Expected single-node rank==local_rank, got {rank} and {local_rank}.")

    model = instantiate(cfg.model, model_dtype=_dtype(cfg.mixed_precision), device=str(device))
    if cfg.ckpt is not None:
        model.load_checkpoint(str(cfg.ckpt))
    model.eval()

    height, width = (int(value) for value in cfg.data.train.video_size)
    horizon = (
        int(cfg.data.train.num_frames) - 1
        if bench.action_horizon is None
        else int(bench.action_horizon)
    )
    if rank == 0:
        logger.info("benchmark action_horizon=%d", horizon)
    torch.manual_seed(int(bench.seed))
    input_image = torch.rand(1, 3, height, width) * 2 - 1
    proprio = torch.randn(1, int(cfg.data.train.processor.proprio_output_dim))
    infer_kwargs = {
        "prompt": "pick up the object",
        "input_image": input_image,
        "action_horizon": horizon,
        "proprio": proprio,
        "num_inference_steps": 1,
        "sigma_shift": cfg.EVALUATION.get("sigma_shift"),
        "seed": int(bench.seed),
        "rand_device": "cpu",
        "compile_action_infer": False,
    }

    if rank == 0:
        serial_times, reference = _benchmark(
            "single_gpu_serial",
            lambda: model.infer_action(**infer_kwargs)["action"],
            [device],
            int(bench.warmup),
            int(bench.iters),
        )
    else:
        serial_times = []
        reference = torch.empty((horizon, model.action_expert.action_dim), dtype=torch.float32)
    dist.barrier()
    reference_device = reference.to(device)
    dist.broadcast(reference_device, src=0)
    reference = reference_device.cpu()

    apply_tensor_parallel(model, rank, world_size)
    torch.cuda.empty_cache()
    tp_call = lambda: model.infer_action(**infer_kwargs)["action"]
    _, candidate = _distributed_time_call(tp_call, device)

    local_max_diff = float((reference - candidate).abs().max())
    local_mean_diff = float((reference - candidate).abs().mean())
    differences = torch.tensor([local_max_diff, local_mean_diff], device=device)
    dist.all_reduce(differences, op=dist.ReduceOp.MAX)
    if rank == 0:
        logger.info(
            "tensor_parallel correctness max_abs_diff=%.8f mean_abs_diff=%.8f",
            float(differences[0]),
            float(differences[1]),
        )
    if not torch.isfinite(differences).all() or float(differences[0]) > 0.02:
        raise AssertionError(f"Tensor-parallel max diff is {float(differences[0])}")

    for index in range(int(bench.warmup)):
        elapsed, _ = _distributed_time_call(tp_call, device)
        if rank == 0:
            logger.info(
                "tensor_parallel warmup %d/%d: %.3f ms",
                index + 1,
                int(bench.warmup),
                elapsed * 1e3,
            )
    tp_times = []
    for index in range(int(bench.iters)):
        elapsed, _ = _distributed_time_call(tp_call, device)
        tp_times.append(elapsed)
        if rank == 0:
            logger.info(
                "tensor_parallel iteration %d/%d: %.3f ms",
                index + 1,
                int(bench.iters),
                elapsed * 1e3,
            )
    if rank == 0:
        logger.info(
            "tensor-parallel result serial=%.3f ms tp=%.3f ms speedup=%.3fx",
            statistics.mean(serial_times) * 1e3,
            statistics.mean(tp_times) * 1e3,
            statistics.mean(serial_times) / statistics.mean(tp_times),
        )
    dist.destroy_process_group()


def _distributed_sp_main(cfg: DictConfig, bench: DictConfig) -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 2:
        raise ValueError("Ulysses SP2 benchmark requires exactly two ranks.")
    if rank != local_rank:
        raise ValueError(f"Expected single-node rank==local_rank, got {rank} and {local_rank}.")

    model = instantiate(cfg.model, model_dtype=_dtype(cfg.mixed_precision), device=str(device))
    if cfg.ckpt is not None:
        model.load_checkpoint(str(cfg.ckpt))
    model.eval()

    height, width = (int(value) for value in cfg.data.train.video_size)
    horizon = (
        int(cfg.data.train.num_frames) - 1
        if bench.action_horizon is None
        else int(bench.action_horizon)
    )
    torch.manual_seed(int(bench.seed))
    input_image = torch.rand(1, 3, height, width) * 2 - 1
    proprio = torch.randn(1, int(cfg.data.train.processor.proprio_output_dim))
    prompt = "pick up the object"
    infer_kwargs = {
        "prompt": prompt,
        "input_image": input_image,
        "action_horizon": horizon,
        "proprio": proprio,
        "num_inference_steps": 1,
        "sigma_shift": cfg.EVALUATION.get("sigma_shift"),
        "seed": int(bench.seed),
        "rand_device": "cpu",
        "compile_action_infer": False,
    }
    if rank == 0:
        logger.info("benchmark action_horizon=%d", horizon)
        serial_times, reference = _benchmark(
            "single_gpu_serial",
            lambda: model.infer_action(**infer_kwargs)["action"],
            [device],
            int(bench.warmup),
            int(bench.iters),
        )
    else:
        serial_times = []
        reference = torch.empty((horizon, model.action_expert.action_dim), dtype=torch.float32)
    dist.barrier()
    reference_device = reference.to(device)
    dist.broadcast(reference_device, src=0)
    reference = reference_device.cpu()

    sp = UlyssesSP2(model, rank, world_size)
    sp_call = lambda: sp.infer(
        input_image,
        proprio,
        prompt,
        horizon,
        cfg.EVALUATION.get("sigma_shift"),
        int(bench.seed),
    )
    _, candidate = _distributed_time_call(sp_call, device)
    local_diff = torch.tensor(
        [
            float((reference - candidate).abs().max()),
            float((reference - candidate).abs().mean()),
        ],
        device=device,
    )
    dist.all_reduce(local_diff, op=dist.ReduceOp.MAX)
    if rank == 0:
        logger.info(
            "ulysses_sp2 correctness max_abs_diff=%.8f mean_abs_diff=%.8f",
            float(local_diff[0]),
            float(local_diff[1]),
        )
    if not torch.isfinite(local_diff).all() or float(local_diff[0]) > 0.02:
        raise AssertionError(f"Ulysses SP2 max diff is {float(local_diff[0])}")

    for index in range(int(bench.warmup)):
        elapsed, _ = _distributed_time_call(sp_call, device)
        if rank == 0:
            logger.info(
                "ulysses_sp2 warmup %d/%d: %.3f ms",
                index + 1,
                int(bench.warmup),
                elapsed * 1e3,
            )
    sp_times = []
    for index in range(int(bench.iters)):
        elapsed, _ = _distributed_time_call(sp_call, device)
        sp_times.append(elapsed)
        if rank == 0:
            logger.info(
                "ulysses_sp2 iteration %d/%d: %.3f ms",
                index + 1,
                int(bench.iters),
                elapsed * 1e3,
            )
    if rank == 0:
        logger.info(
            "ulysses-sp2 result serial=%.3f ms sp2=%.3f ms speedup=%.3fx",
            statistics.mean(serial_times) * 1e3,
            statistics.mean(sp_times) * 1e3,
            statistics.mean(serial_times) / statistics.mean(sp_times),
        )
    dist.destroy_process_group()


@hydra.main(config_path="../configs", config_name="sim_libero.yaml", version_base="1.3")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    bench = _config(cfg)
    if int(os.environ.get("WORLD_SIZE", "1")) != 2:
        raise ValueError("Run this benchmark with torchrun --nproc_per_node=2.")
    if str(bench.mode) == "expert_pipeline":
        _distributed_pipeline_main(cfg, bench)
    elif str(bench.mode) == "tp":
        _distributed_tp_main(cfg, bench)
    elif str(bench.mode) in {"sp2", "ulysses"}:
        _distributed_sp_main(cfg, bench)
    else:
        raise ValueError(f"Unknown PARALLEL.mode={bench.mode!s}")


if __name__ == "__main__":
    main()

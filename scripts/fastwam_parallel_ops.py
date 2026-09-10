"""Fused parallel operators, following LightX2V's Ulysses pre/post design.

The batch-aware layout kernels use FastWAM's [rank, qkv, batch, sequence,
width] wire format. No quantization is applied to communication payloads.
"""

import torch
import torch.distributed as dist
import torch.nn.functional as F
import triton
import triton.language as tl

from bench_latency import FusedOpsMixin
from benchmark_fastwam_parallel import RowParallelLinear, TensorParallelRMSNorm, UlyssesSP2


@triton.jit
def _pack_qkv(Q, K, V, O, N: tl.constexpr, S: tl.constexpr,
              B: tl.constexpr, H: tl.constexpr, QS: tl.constexpr,
              KS: tl.constexpr, VS: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < N
    col = i % H
    row = i // H % (B * S)
    rank = i // (H * B * S)
    base = rank * 3 * B * S * H + row * H + col
    tl.store(O + base, tl.load(Q + row * QS + rank * H + col, valid), valid)
    tl.store(O + base + B * S * H, tl.load(K + row * KS + rank * H + col, valid), valid)
    tl.store(O + base + 2 * B * S * H, tl.load(V + row * VS + rank * H + col, valid), valid)


@triton.jit
def _unpack_qkv(X, Y, N: tl.constexpr, S: tl.constexpr,
                B: tl.constexpr, H: tl.constexpr, BLOCK: tl.constexpr,
                COMPONENTS: tl.constexpr = 3):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    col = i % H
    seq = i // H % (2 * S)
    batch = i // (H * 2 * S) % B
    qkv = i // (H * 2 * S * B)
    source = (((seq // S * COMPONENTS + qkv) * B + batch) * S + seq % S) * H + col
    tl.store(Y + i, tl.load(X + source, i < N), i < N)


@triton.jit
def _unpack_shared_video(X, VIDEO, KV, N: tl.constexpr, S: tl.constexpr,
                         B: tl.constexpr, D: tl.constexpr, RANK: tl.constexpr,
                         BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    col = i % D
    seq = i // D % (2 * S)
    batch = i // (D * 2 * S)
    source = (((seq // S * 3) * B + batch) * S + seq % S) * D + col
    k = tl.load(X + source + B * S * D, i < N, 0)
    v = tl.load(X + source + 2 * B * S * D, i < N, 0)
    tl.store(KV + i, k, i < N)
    tl.store(KV + N + i, v, i < N)
    local_col = col - RANK * (D // 2)
    selected = (i < N) & (local_col >= 0) & (local_col < D // 2)
    q = tl.load(X + source, selected, 0)
    dest = i // D * (D // 2) + local_col
    tl.store(VIDEO + dest, q, selected)
    tl.store(VIDEO + N // 2 + dest, k, selected)
    tl.store(VIDEO + N + dest, v, selected)


def gather_video_kv(k, v, group):
    b, s, d = k.shape
    packed = torch.stack((k, v))
    gathered = torch.empty((4, b, s, d), device=k.device, dtype=k.dtype)
    dist.all_gather_into_tensor(gathered, packed, group=group)
    output = torch.empty((2, b, 2 * s, d), device=k.device, dtype=k.dtype)
    _unpack_qkv[(triton.cdiv(output.numel(), 256),)](
        gathered, output, output.numel(), s, b, d, 256, COMPONENTS=2)
    return tuple(output.unbind(0))


def gather_shared_video_qkv(q, k, v, group, rank):
    b, s, d = q.shape
    packed = torch.stack((q, k, v))
    gathered = torch.empty((6, b, s, d), device=q.device, dtype=q.dtype)
    dist.all_gather_into_tensor(gathered, packed, group=group)
    video = torch.empty((3, b, 2 * s, d // 2), device=q.device, dtype=q.dtype)
    kv = torch.empty((2, b, 2 * s, d), device=q.device, dtype=q.dtype)
    n = b * 2 * s * d
    _unpack_shared_video[(triton.cdiv(n, 256),)](gathered, video, kv, n, s, b, d, rank, 256)
    return tuple(video.unbind(0)), tuple(kv.unbind(0))


@triton.jit
def _attn_layout(X, Y, N: tl.constexpr, S: tl.constexpr, B: tl.constexpr,
                 H: tl.constexpr, PACK: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if PACK:
        col = i % H
        seq = i // H % S
        batch = i // (H * S) % B
        rank = i // (H * S * B)
        source = ((batch * 2 + rank) * S + seq) * H + col
    else:
        col = i % (2 * H)
        row = i // (2 * H)
        source = (col // H * B * S + row) * H + col % H
    tl.store(Y + i, tl.load(X + source, i < N), i < N)


class FusedUlyssesSP2(UlyssesSP2):
    def _sequence_to_heads(self, q, k, v):
        b, s, width = q.shape
        h = width // 2
        send = torch.empty((2, 3, b, s, h), device=q.device, dtype=q.dtype)
        _pack_qkv[(triton.cdiv(q.numel(), 256),)](
            q, k, v, send, q.numel(), s, b, h,
            q.stride(1), k.stride(1), v.stride(1), 256)
        received = torch.empty_like(send)
        dist.all_to_all_single(received, send, group=self.group)
        output = torch.empty((3, b, 2 * s, h), device=q.device, dtype=q.dtype)
        _unpack_qkv[(triton.cdiv(output.numel(), 256),)](
            received, output, output.numel(), s, b, h, 256)
        return tuple(output.unbind(0))

    def _heads_to_sequence(self, tensor):
        b, full_s, h = tensor.shape
        s = full_s // 2
        tensor = tensor.contiguous()
        send = torch.empty((2, b, s, h), device=tensor.device, dtype=tensor.dtype)
        _attn_layout[(triton.cdiv(send.numel(), 256),)](
            tensor, send, send.numel(), s, b, h, True, 256)
        received = torch.empty_like(send)
        dist.all_to_all_single(received, send, group=self.group)
        output = torch.empty((b, s, 2 * h), device=tensor.device, dtype=tensor.dtype)
        _attn_layout[(triton.cdiv(output.numel(), 256),)](
            received, output, output.numel(), s, b, h, False, 256)
        return output


@triton.jit
def _pair_square_sum(Q, K, O, NQ: tl.constexpr, D: tl.constexpr,
                     QS: tl.constexpr, KS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    q = tl.load(Q + row * QS + col, (row < NQ) & (col < D), 0).to(tl.float32)
    k = tl.load(K + (row - NQ) * KS + col, (row >= NQ) & (col < D), 0).to(tl.float32)
    x = q + k
    tl.store(O + row, tl.sum(x * x, axis=0))


@triton.jit
def _norm_rope(X, W, SUMS, FREQ, Y, D: tl.constexpr, GLOBAL_D: tl.constexpr,
               STRIDE: tl.constexpr, S: tl.constexpr, HD: tl.constexpr,
               EPS: tl.constexpr, TP: tl.constexpr, ROPE: tl.constexpr,
               BLOCK: tl.constexpr, MATCH_OUTPUT_CAST: tl.constexpr = False):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    dtype = X.dtype.element_ty
    x = tl.load(X + row * STRIDE + col, col < D, 0).to(tl.float32)
    square_sum = tl.load(SUMS + row) if TP else tl.sum(x * x, axis=0)
    norm = (x * tl.rsqrt(square_sum / GLOBAL_D + EPS)).to(dtype).to(tl.float32)
    weight = tl.load(W + col, col < D, 0).to(tl.float32)
    y = (norm * weight).to(dtype)
    if ROPE:
        # Wan's eager RoPE multiplies complex128 values after BF16 normalization.
        y = y.to(tl.float64)
        other = tl.gather(y, col ^ 1, axis=0)
        freq_offset = row % S * HD + (col % HD // 2) * 2
        real = tl.load(FREQ + freq_offset).to(tl.float64)
        imag = tl.load(FREQ + freq_offset + 1).to(tl.float64)
        y = y * real + tl.where(col % 2 == 0, -other, other) * imag
        if MATCH_OUTPUT_CAST:
            y = y.to(tl.float32)
    tl.store(Y + row * D + col, y.to(dtype), col < D)


def norm_rope(x, norm, sums=None, freqs=None, head_dim=128, match_output_cast=False):
    output = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    d = x.shape[-1]
    tp = sums is not None
    real_freqs = None if freqs is None else torch.view_as_real(freqs)
    _norm_rope[(x.numel() // d,)](
        x, norm.weight, sums, real_freqs, output, d,
        norm.global_dim if tp else d, x.stride(1), x.shape[1], head_dim,
        norm.eps, tp, freqs is not None, triton.next_power_of_2(d),
        MATCH_OUTPUT_CAST=match_output_cast, enable_fp_fusion=False)
    return output


class OptimizedParallelMixin(FusedOpsMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.projections = {}
        for expert in (self.model.video_expert, self.model.action_expert):
            for block in expert.blocks:
                for attn, names in ((block.self_attn, ("q", "k", "v")),
                                    (block.cross_attn, ("k", "v"))):
                    linears = [getattr(attn, name) for name in names]
                    self.projections[id(attn)] = (
                        torch.cat([linear.weight for linear in linears]),
                        torch.cat([linear.bias for linear in linears]),
                        [linear.out_features for linear in linears])

    def _projection(self, attention, x):
        weight, bias, widths = self.projections[id(attention)]
        return F.linear(x, weight, bias).split(widths, dim=-1)

    def _row_projection(self, linear, x):
        if not isinstance(linear, RowParallelLinear):
            return linear(x)
        output = F.linear(x, linear.weight)
        dist.all_reduce(output, group=linear.group)
        return output if linear.bias is None else output + linear.bias

    def _norm_pair(self, attention, q, k, freqs=None):
        nq, nk = attention.norm_q, attention.norm_k
        if not isinstance(nq, TensorParallelRMSNorm):
            return (norm_rope(q, nq, freqs=freqs, head_dim=attention.attn_head_dim),
                    norm_rope(k, nk, freqs=freqs, head_dim=attention.attn_head_dim))
        d = q.shape[-1]
        rows_q, rows_k = q.numel() // d, k.numel() // d
        sums = torch.empty(rows_q + rows_k, device=q.device, dtype=torch.float32)
        _pair_square_sum[(rows_q + rows_k,)](
            q, k, sums, rows_q, d, q.stride(1), k.stride(1),
            triton.next_power_of_2(d), enable_fp_fusion=False)
        dist.all_reduce(sums, group=nq.group)
        return (norm_rope(q, nq, sums[:rows_q], freqs, attention.attn_head_dim),
                norm_rope(k, nk, sums[rows_q:], freqs, attention.attn_head_dim))

    def project(self, expert, index, x, prepared):
        block = expert.blocks[index]
        shift, scale, gate, shift_mlp, scale_mlp, gate_mlp = self.model.mot._split_modulation(
            block, prepared.t_mod)
        z = self._modulate(block.norm1(x), shift, scale)
        q, k, v = self._projection(block.self_attn, z)
        q, k = self._norm_pair(block.self_attn, q, k, prepared.freqs)
        return q, k, v, x, gate, shift_mlp, scale_mlp, gate_mlp, False

    def attention(self, expert, io, prepared, kv):
        k, v = io[1:3] if kv is None else (
            torch.cat((kv[0], io[1]), dim=1), torch.cat((kv[1], io[2]), dim=1))
        return self.model.mot._mixed_attention(io[0], k, v, prepared.attention_mask)

    def finish(self, expert, index, io, prepared, kv=None):
        block = expert.blocks[index]
        mixed = self.attention(expert, io, prepared, kv)
        x = self._gate(io[3], io[4], self._row_projection(block.self_attn.o, mixed))
        ca = block.cross_attn
        q = ca.q(block.norm3(x))
        k, v = self._projection(ca, prepared.context)
        q, k = self._norm_pair(ca, q, k)
        from fastwam.models.wan22.wan_video_dit import flash_attention
        cross = flash_attention(q, k, v, ca.num_heads, prepared.context_mask.unsqueeze(1))
        x = x + self._row_projection(ca.o, cross)
        z = self._modulate(block.norm2(x), io[5], io[6])
        z = block.ffn[1](block.ffn[0](z))
        return self._gate(x, io[7], self._row_projection(block.ffn[2], z))

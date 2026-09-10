"""Exact layout checks for Ulysses buffers, including strided fused projections."""

from pathlib import Path
import sys
import unittest

import torch
import triton

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from fastwam_parallel_ops import (
    _attn_layout, _pack_qkv, _unpack_qkv, _unpack_shared_video, norm_rope,
)
from fastwam.models.wan22.wan_video_dit import RMSNorm, precompute_freqs_cis, rope_apply


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class ParallelLayoutTest(unittest.TestCase):
    def test_replicated_action_kv(self):
        for b, s in ((1, 49), (2, 49)):
            with self.subTest(batch=b, sequence=s):
                d = 3072
                wire = torch.randn((2, 2, b, s, d), device="cuda", dtype=torch.bfloat16)
                kv = torch.empty((2, b, 2 * s, d), device="cuda", dtype=wire.dtype)
                _unpack_qkv[(triton.cdiv(kv.numel(), 256),)](
                    wire, kv, kv.numel(), s, b, d, 256, COMPONENTS=2)
                expected = wire.permute(1, 2, 0, 3, 4).reshape(2, b, 2 * s, d)
                torch.testing.assert_close(kv, expected, rtol=0, atol=0)

    def test_shared_video_qkv(self):
        for b, s in ((1, 49), (2, 49)):
            for rank in (0, 1):
                with self.subTest(batch=b, sequence=s, rank=rank):
                    d = 3072
                    wire = torch.randn((2, 3, b, s, d), device="cuda", dtype=torch.bfloat16)
                    video = torch.empty((3, b, 2 * s, d // 2), device="cuda", dtype=wire.dtype)
                    kv = torch.empty((2, b, 2 * s, d), device="cuda", dtype=wire.dtype)
                    n = b * 2 * s * d
                    _unpack_shared_video[(triton.cdiv(n, 256),)](
                        wire, video, kv, n, s, b, d, rank, 256)
                    full = wire.permute(1, 2, 0, 3, 4).reshape(3, b, 2 * s, d)
                    torch.testing.assert_close(kv, full[1:], rtol=0, atol=0)
                    torch.testing.assert_close(video, full[..., rank * (d // 2):(rank + 1) * (d // 2)],
                                               rtol=0, atol=0)

    def test_norm_rope(self):
        torch.manual_seed(42)
        for d, s in ((1536, 16), (3072, 49), (3072, 129)):
            with self.subTest(width=d, sequence=s):
                x = torch.randn((2, s, d * 3), device="cuda", dtype=torch.bfloat16)[..., :d]
                norm = RMSNorm(d).cuda().to(torch.bfloat16)
                freqs = precompute_freqs_cis(128, s).cuda().unsqueeze(1)
                actual = norm_rope(x, norm, freqs=freqs)
                expected = rope_apply(norm(x), freqs, d // 128)
                # Parallel reductions can move a value across a BF16 rounding boundary.
                torch.testing.assert_close(actual, expected, rtol=0.008, atol=0.008)

    def test_wire_layout(self):
        for b, s in ((1, 16), (1, 49), (2, 49)):
            for dtype in (torch.float16, torch.bfloat16):
                with self.subTest(batch=b, sequence=s, dtype=dtype):
                    h = 1536
                    q, k, v = torch.randn((b, s, 6 * h), device="cuda", dtype=dtype).chunk(3, -1)
                    send = torch.empty((2, 3, b, s, h), device="cuda", dtype=dtype)
                    _pack_qkv[(triton.cdiv(q.numel(), 256),)](
                        q, k, v, send, q.numel(), s, b, h,
                        q.stride(1), k.stride(1), v.stride(1), 256)
                    expected = torch.stack((q, k, v)).view(3, b, s, 2, h).permute(3, 0, 1, 2, 4)
                    torch.testing.assert_close(send, expected, rtol=0, atol=0)
                    # Model a received buffer; source rank is its outer dimension.
                    unpacked = torch.empty((3, b, 2 * s, h), device="cuda", dtype=dtype)
                    _unpack_qkv[(triton.cdiv(unpacked.numel(), 256),)](
                        send, unpacked, unpacked.numel(), s, b, h, 256)
                    expected = send.permute(1, 2, 0, 3, 4).reshape(3, b, 2 * s, h)
                    torch.testing.assert_close(unpacked, expected, rtol=0, atol=0)

    def test_attention_layout(self):
        for b, s in ((1, 16), (1, 49), (2, 49)):
            with self.subTest(batch=b, sequence=s):
                h = 1536
                x = torch.randn((b, 2 * s, h), device="cuda", dtype=torch.bfloat16)
                send = torch.empty((2, b, s, h), device="cuda", dtype=x.dtype)
                _attn_layout[(triton.cdiv(x.numel(), 256),)](
                    x, send, x.numel(), s, b, h, True, 256)
                torch.testing.assert_close(send, x.view(b, 2, s, h).permute(1, 0, 2, 3), rtol=0, atol=0)
                output = torch.empty((b, s, 2 * h), device="cuda", dtype=x.dtype)
                _attn_layout[(triton.cdiv(x.numel(), 256),)](
                    send, output, x.numel(), s, b, h, False, 256)
                torch.testing.assert_close(output, send.permute(1, 2, 0, 3).reshape(b, s, 2 * h),
                                           rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()

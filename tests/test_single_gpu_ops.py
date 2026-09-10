"""Precision and mask semantics for the single-GPU five-op runner."""

from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from fastwam_single_gpu_ops import FusedRMSNorm, PackedLinear, is_unmasked, rms_pair, rope
from fastwam.models.wan22.wan_video_dit import RMSNorm, precompute_freqs_cis, rope_apply


class MaskTest(unittest.TestCase):
    def test_mask_semantics(self):
        self.assertTrue(is_unmasked(None))
        self.assertTrue(is_unmasked(torch.ones(3, 4, dtype=torch.bool)))
        self.assertFalse(is_unmasked(torch.tensor([[True, False]])))
        self.assertFalse(is_unmasked(torch.ones(3, 4)))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class KernelTest(unittest.TestCase):
    @torch.no_grad()
    def test_rope_fp64_strided(self):
        for dtype in (torch.bfloat16, torch.float16):
            x = torch.randn(2, 49, 3 * 3072, device="cuda", dtype=dtype)[..., :3072]
            freqs = precompute_freqs_cis(128, 49).cuda().unsqueeze(1)
            torch.testing.assert_close(rope(x, freqs), rope_apply(x, freqs, 24), atol=0, rtol=0)

    @torch.no_grad()
    def test_rmsnorm_rounding(self):
        torch.manual_seed(42)
        for d, s in ((3072, 32), (3072, 98), (3072, 129)):
            x = torch.randn(1, s, d * 3, device="cuda", dtype=torch.bfloat16)[..., :d]
            norm = RMSNorm(d, eps=1e-6).cuda().to(torch.bfloat16)
            norm.weight.normal_(1, 0.2)
            actual, expected = FusedRMSNorm(norm)(x), norm(x)
            # FP32 reduction trees may straddle a BF16 rounding midpoint.
            torch.testing.assert_close(actual, expected, rtol=0.008, atol=0.008)

    @torch.no_grad()
    def test_pair_norm_unequal_sequences(self):
        q = torch.randn(1, 32, 9216, device="cuda", dtype=torch.bfloat16)[..., :3072]
        k = torch.randn(1, 129, 6144, device="cuda", dtype=torch.bfloat16)[..., :3072]
        nq, nk = (RMSNorm(3072, eps=1e-6).cuda().to(torch.bfloat16) for _ in range(2))
        for actual, expected in zip(rms_pair(q, k, nq, nk), (nq(q), nk(k))):
            torch.testing.assert_close(actual, expected, rtol=0.008, atol=0.008)

    @torch.no_grad()
    def test_packed_linear_with_optional_bias(self):
        for biases in ((True, True, True), (False, False, False), (True, False, True)):
            layers = [torch.nn.Linear(1024, width, bias=bias).cuda().to(torch.bfloat16)
                      for width, bias in zip((512, 768, 512), biases)]
            x = torch.randn(1, 32, 1024, device="cuda", dtype=torch.bfloat16)
            for actual, layer in zip(PackedLinear(layers)(x), layers):
                torch.testing.assert_close(actual, layer(x), rtol=0.008, atol=0.008)


if __name__ == "__main__":
    unittest.main()

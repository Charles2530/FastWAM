"""Accounting and replay checks for the component benchmark."""

from pathlib import Path
import sys
import unittest

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from bench_component import COMPONENTS, CapturedCall, ComponentTimer, component_config, interval_accounting


class AccountingTest(unittest.TestCase):
    def test_overlapping_nested_and_disjoint_intervals(self):
        values = interval_accounting([
            ("video_dit_prefill", 1, 6),
            ("action_dit_denoise", 2, 4),
            ("action_dit_denoise", 5, 8),
            ("vae_encode", 10, 12),
        ], 15)
        self.assertEqual(values["overlap_correction"], -3)
        self.assertEqual(values["other"], 6)
        self.assertEqual(sum(values[k] for k in COMPONENTS) + values["overlap_correction"]
                         + values["other"], values["total"])

    def test_sequential_and_empty(self):
        values = interval_accounting([("vae_encode", 0, 2), ("text_encode", 3, 5)], 6)
        self.assertEqual(values["overlap_correction"], 0)
        self.assertEqual(values["other"], 2)
        self.assertEqual(interval_accounting([], 6)["other"], 6)

    def test_invalid_interval(self):
        with self.assertRaises(ValueError):
            interval_accounting([("vae_encode", 2, 1)], 3)

    def test_config_overrides_and_validation(self):
        cfg = OmegaConf.create({"DRYRUN": {"iters": 2}, "COMPONENT": {"iters": 3}})
        self.assertEqual(component_config(cfg).iters, 3)
        self.assertFalse(component_config(cfg).cache_text_embeddings)
        for override in ({"modes": [1, 1]}, {"modes": [6]}, {"iters": 0}, {"typo": 1}):
            with self.assertRaises(ValueError):
                component_config(OmegaConf.create({"COMPONENT": override}))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class GraphTimerTest(unittest.TestCase):
    @torch.no_grad()
    def test_events_replay_with_changed_input_and_two_streams(self):
        device = torch.device("cuda", torch.cuda.current_device())
        x = torch.ones(1024 * 1024, device=device)
        streams = [torch.cuda.Stream(device=device) for _ in range(2)]
        timer = ComponentTimer()
        left = timer.wrap("video_dit_prefill", lambda: x.square())
        right = timer.wrap("action_dit_denoise", lambda: x + 1)

        def function():
            timer.reset("graph")
            caller = torch.cuda.current_stream()
            outputs = []
            for stream, operation in zip(streams, (left, right)):
                stream.wait_stream(caller)
                with torch.cuda.stream(stream):
                    outputs.append(operation())
                caller.wait_stream(stream)
            return outputs

        captured = CapturedCall(function, device, 2)
        for value in (2, 3):
            x.fill_(value)
            timer.reset("host")
            timer.origin.record()
            a, b = captured()
            torch.cuda.synchronize()
            torch.testing.assert_close(a, torch.full_like(x, value ** 2))
            torch.testing.assert_close(b, torch.full_like(x, value + 1))
            values = timer.values(100)
            self.assertGreater(values["video_dit_prefill"], 0)
            self.assertGreater(values["action_dit_denoise"], 0)
            self.assertLessEqual(values["overlap_correction"], 0.000001)
            self.assertEqual(len(timer.pool), 2)


if __name__ == "__main__":
    unittest.main()

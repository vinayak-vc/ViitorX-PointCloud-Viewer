"""ns_helpers/train_resumable.py: the Splatfacto refinement-pause cap (no Nerfstudio needed)."""
import sys
import time
import types
import unittest
from unittest import mock

from pf360.ns_helpers import train_resumable
from pf360.ns_helpers.train_resumable import cap_refine_pause


def _strategy(images, refine_every=100, reset_alpha_every=30):
    # what Nerfstudio 1.1.5 Splatfacto passes to gsplat's DefaultStrategy
    return types.SimpleNamespace(reset_every=reset_alpha_every * refine_every,
                                 pause_refine_after_reset=images + refine_every)


def _refines_at(s, step, start=500, stop=15000, refine_every=100):
    # gsplat DefaultStrategy's grow/prune condition
    return (start < step < stop and step % refine_every == 0
            and step % s.reset_every >= s.pause_refine_after_reset)


class RefinePauseTests(unittest.TestCase):
    def test_large_dataset_never_refined_without_the_cap(self):
        s = _strategy(10849)                                   # the 17 h job: 108,004 Gaussians, unchanged
        self.assertFalse(any(_refines_at(s, k) for k in range(40000)))
        self.assertTrue(cap_refine_pause(s))
        self.assertEqual(s.pause_refine_after_reset, 1500)
        steps = [k for k in range(40000) if _refines_at(s, k)]
        self.assertEqual((steps[0], steps[-1]), (1500, 14900))
        self.assertEqual(len(steps), 75)                       # 15 refinements in each of the 5 cycles

    def test_small_dataset_is_left_alone(self):
        s = _strategy(800)
        self.assertFalse(cap_refine_pause(s))
        self.assertEqual(s.pause_refine_after_reset, 900)

    def test_unknown_strategy_is_ignored(self):
        self.assertFalse(cap_refine_pause(types.SimpleNamespace()))


class TimerTests(unittest.TestCase):
    def test_nerfstudio_timer_uses_perf_counter(self):
        # a fast eval timed with time.time() measured 0.0 s on Windows -> ZeroDivisionError at step ~900
        writer = types.ModuleType("nerfstudio.utils.writer")
        writer.time = time.time
        utils = types.ModuleType("nerfstudio.utils")
        utils.writer = writer
        with mock.patch.dict(sys.modules, {"nerfstudio": types.ModuleType("nerfstudio"),
                                           "nerfstudio.utils": utils, "nerfstudio.utils.writer": writer}):
            train_resumable._patch_timer()
        self.assertIs(writer.time, time.perf_counter)


if __name__ == "__main__":
    unittest.main()

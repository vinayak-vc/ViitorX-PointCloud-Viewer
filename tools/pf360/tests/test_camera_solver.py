"""Camera-solve options: refinement interval, GPU bundle adjustment, loop detection, GLOMAP + fallback.

procrun.run is replaced by a recorder, so neither COLMAP nor GLOMAP is needed."""
import io
import os
import struct
import tempfile
import unittest
from unittest import mock

from pf360 import manifest, procrun
from pf360.backend import StageContext
from pf360.environment import Tools
from pf360.nerfstudio_backend import NerfstudioBackend, mapper_ba_args
from pf360.protocol import Emitter
from pf360.workspace import Workspace

from tests.test_mapper_reuse import _fake_jpeg


class _Stop(Exception):
    pass


def _arg(cmd, name):
    return cmd[cmd.index(name) + 1]


class MapperArgsTests(unittest.TestCase):
    def test_refinement_intervals(self):
        self.assertEqual(mapper_ba_args({"globalRefinement": "thorough"}), [])      # COLMAP defaults
        bal = mapper_ba_args({})                                                   # default = balanced
        self.assertEqual(_arg(bal, "--Mapper.ba_global_images_freq"), "2000")
        fast = mapper_ba_args({"globalRefinement": "fast"})
        self.assertEqual(_arg(fast, "--Mapper.ba_global_images_ratio"), "1.2")

    def test_gpu_bundle_adjustment_only_with_cuda(self):
        self.assertIn("--Mapper.ba_use_gpu", mapper_ba_args({"gpuBundleAdjustment": True}, cuda=True))
        self.assertNotIn("--Mapper.ba_use_gpu", mapper_ba_args({"gpuBundleAdjustment": True}, cuda=False))
        self.assertNotIn("--Mapper.ba_use_gpu", mapper_ba_args({}, cuda=True))


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.video = os.path.join(self.tmp, "v.mp4")
        open(self.video, "wb").close()

    def resolved(self, **pr):
        raw = {"schemaVersion": 1, "jobId": "j1", "input": {"videoPath": self.video},
               "processing": dict({"targetFrames": 4}, **pr), "output": {"directory": os.path.join(self.tmp, "ws")}}
        return manifest.resolve(raw, 10.0)

    def test_defaults_keep_the_proven_path(self):
        pr = manifest.validate(self.resolved())["processing"]
        self.assertEqual((pr["cameraSolver"], pr["loopDetection"], pr["globalRefinement"], pr["gpuBundleAdjustment"]),
                         ("colmap", False, "balanced", False))

    def test_bad_values_are_rejected(self):
        for bad in ({"cameraSolver": "openmvg"}, {"globalRefinement": "max"}, {"loopDetection": "yes"},
                    {"loopDetection": True, "matchingMethod": "exhaustive"}):
            with self.assertRaises(manifest.ManifestError, msg=bad):
                manifest.validate(self.resolved(**bad))


class CameraStageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        video = os.path.join(self.tmp, "v.mp4")
        open(video, "wb").close()
        self.video = video
        exe = os.path.join(self.tmp, "tool.exe")
        open(exe, "wb").close()
        self.exe = exe
        self.vocab = os.path.join(self.tmp, "vocab.bin")
        open(self.vocab, "wb").close()

    def run_stage(self, tools_extra, fake, **pr):
        raw = {"schemaVersion": 1, "jobId": "c1", "input": {"videoPath": self.video},
               "processing": dict({"targetFrames": 4, "imagesPerEquirect": 14}, **pr),
               "training": {"qualityPreset": "high"}, "output": {"directory": os.path.join(self.tmp, "ws")}}
        m = manifest.validate(manifest.resolve(raw, 10.0))
        ws = Workspace(m["output"]["directory"], "out.ply")
        ws.init()
        ws.planar_dir.mkdir(parents=True, exist_ok=True)
        for k in range(1, 3):
            for v in range(14):
                _fake_jpeg(ws.planar_dir / ("frame_%05d_%d.jpg" % (k, v)), 64, 64)
        tools = Tools(ns_python=self.exe, ns_bin=self.tmp, colmap=self.exe, ffmpeg=self.exe, **tools_extra)
        out = io.StringIO()
        ctx = StageContext(manifest=m, tools=tools, workspace=ws, emitter=Emitter(out, "c1"),
                           cancel=procrun.CancelToken(), results={"_env": {"cuda": True}})
        self.sparse = ws.processed_dir / "colmap" / "sparse"
        calls = []

        def fake_run(args, **kw):
            args = [str(a) for a in args]
            calls.append(args)
            return fake(args)

        with mock.patch("pf360.nerfstudio_backend.procrun.run", fake_run):
            with self.assertRaises(_Stop):
                NerfstudioBackend()._camera_reconstruction(ctx)
        return calls, out.getvalue()

    def model(self, name="0"):
        (self.sparse / name).mkdir(parents=True, exist_ok=True)
        (self.sparse / name / "images.bin").write_bytes(struct.pack("<Q", 28))

    @staticmethod
    def joining(args):
        return any(a.endswith("rig_complete.py") for a in args)

    def test_loop_detection_uses_the_vocabulary_tree(self):
        def fake(args):
            if "sequential_matcher" in args:
                raise _Stop()
            return 0
        calls, _ = self.run_stage({"vocab_tree": self.vocab}, fake, loopDetection=True)
        self.assertEqual(_arg(calls[-1], "--SequentialMatching.loop_detection"), "1")
        self.assertEqual(_arg(calls[-1], "--SequentialMatching.vocab_tree_path"), self.vocab)
        self.assertEqual(_arg(calls[-1], "--SequentialMatching.loop_detection_max_num_features"), "512")

    def test_loop_detection_without_tree_warns_and_matches_anyway(self):
        def fake(args):
            if "sequential_matcher" in args:
                raise _Stop()
            return 0
        calls, out = self.run_stage({}, fake, loopDetection=True)
        self.assertNotIn("--SequentialMatching.loop_detection", calls[-1])
        self.assertIn("vocabulary tree engine pack is not installed", out)

    def test_glomap_replaces_the_colmap_mapper(self):
        glomap = os.path.join(self.tmp, "glomap.exe")
        open(glomap, "wb").close()

        def fake(args):
            if args[0] == glomap:
                self.model("0")
            if self.joining(args):
                raise _Stop()
            return 0
        calls, out = self.run_stage({"glomap": glomap}, fake, cameraSolver="glomap")
        names = [c[0] if c[0] == glomap else c[1] for c in calls]
        self.assertEqual(names[:3], ["feature_extractor", "sequential_matcher", glomap])
        self.assertNotIn("mapper", [c[1] for c in calls if c[0] != glomap])
        g = calls[2]
        self.assertEqual(_arg(g, "--BundleAdjustment.optimize_intrinsics"), "0")
        self.assertEqual(_arg(g, "--skip_retriangulation"), "1")
        self.assertIn("GLOMAP finished", out)
        self.assertTrue((self.sparse / "mapper_done.json").is_file())

    def test_failed_glomap_falls_back_to_colmap(self):
        glomap = os.path.join(self.tmp, "glomap.exe")
        open(glomap, "wb").close()

        def fake(args):
            if args[0] == glomap:
                return 3                      # crashed, no model
            if "mapper" in args:
                self.model("0")
            if self.joining(args):
                raise _Stop()
            return 0
        calls, out = self.run_stage({"glomap": glomap}, fake, cameraSolver="glomap")
        self.assertIn("mapper", calls[3])
        self.assertIn("falling back to COLMAP", out)

    def test_missing_glomap_uses_colmap_with_a_warning(self):
        def fake(args):
            if "mapper" in args:
                self.model("0")
            if self.joining(args):
                raise _Stop()
            return 0
        calls, out = self.run_stage({}, fake, cameraSolver="glomap")
        self.assertIn("mapper", calls[2])
        self.assertIn("GLOMAP was chosen but its engine pack is not installed", out)


if __name__ == "__main__":
    unittest.main()

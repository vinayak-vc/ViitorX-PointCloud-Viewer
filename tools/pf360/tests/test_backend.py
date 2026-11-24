"""Nerfstudio backend: command construction + reconstruction checks, with mocked tools."""
import io
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from pf360 import manifest, procrun
from pf360.backend import StageContext, StageFailed
from pf360.environment import Tools
from pf360.nerfstudio_backend import CliFlags, NerfstudioBackend, TRAIN_STEP_RE
from pf360.protocol import Emitter
from pf360.workspace import Workspace

NEW_TYRO = "--viewer.quit-on-train-completion, --viewer.no-quit-on-train-completion  --gpu, --no-gpu --verbose, --no-verbose " \
           "--pipeline.model.use-scale-regularization, --pipeline.model.no-use-scale-regularization "            "--steps-per-eval-all-images INT"
OLD_TYRO = "--viewer.quit-on-train-completion {True,False} --gpu {True,False} --verbose {True,False}"


class CliFlagTests(unittest.TestCase):
    def test_new_style_flags(self):
        f = CliFlags(NEW_TYRO)
        self.assertEqual(f.boolean("--viewer.quit-on-train-completion", True), ["--viewer.quit-on-train-completion"])
        self.assertEqual(f.boolean("--gpu", False), ["--no-gpu"])

    def test_old_style_flags(self):
        f = CliFlags(OLD_TYRO)
        self.assertEqual(f.boolean("--gpu", False), ["--gpu", "False"])
        self.assertEqual(f.boolean("--viewer.quit-on-train-completion", True),
                         ["--viewer.quit-on-train-completion", "True"])

    def test_train_progress_regex(self):
        m = TRAIN_STEP_RE.match("2900 (9.67%)        49.832 ms            24 m, 23 s")
        self.assertEqual((m.group(1), m.group(2)), ("2900", "9.67"))
        self.assertIsNone(TRAIN_STEP_RE.match("Step (% Done)       Train Iter (time)"))


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        video = os.path.join(self.tmp, "v.mp4")
        open(video, "wb").close()
        raw = {"schemaVersion": 1, "jobId": "b1", "input": {"videoPath": video},
               "processing": {"targetFrames": 4}, "training": {"qualityPreset": "high", "cullAlphaThresh": 0.005},
               "output": {"directory": os.path.join(self.tmp, "ws")}}
        self.m = manifest.validate(manifest.resolve(raw, 10.0))
        self.ws = Workspace(self.m["output"]["directory"], "out.ply")
        self.ws.init()
        bindir = os.path.join(self.tmp, "bin")
        os.makedirs(bindir)
        for t in ("ns-process-data", "ns-train", "ns-export", "python"):
            open(os.path.join(bindir, t + (".exe" if os.name == "nt" else "")), "wb").close()
        ext = ".exe" if os.name == "nt" else ""
        self.tools = Tools(ns_python=os.path.join(bindir, "python" + ext), ns_bin=bindir,
                           colmap=os.path.join(bindir, "python" + ext), ffmpeg=os.path.join(bindir, "python" + ext))
        self.buf = io.StringIO()
        self.ctx = StageContext(manifest=self.m, tools=self.tools, workspace=self.ws, emitter=Emitter(self.buf, "b1"),
                                cancel=procrun.CancelToken(),
                                results={"_env": {"cuda": True, "methods": ["splatfacto"]}})

    def test_training_command_and_method_fallback(self):
        be = NerfstudioBackend()
        be._help_cache["train-splatfacto"] = NEW_TYRO
        captured = {}

        def fake_run(args, **kw):
            captured["args"] = args
            kw["on_line"]("1000 (2.50%)   12 ms")
            d = self.ws.training_dir / "pf360" / "splatfacto" / "run"
            (d / "nerfstudio_models").mkdir(parents=True)
            (d / "nerfstudio_models" / "step-000039999.ckpt").write_bytes(b"x")
            (d / "config.yml").write_text("x")
            return 0
        with mock.patch.object(procrun, "run", side_effect=fake_run):
            data = be.run_stage("training", self.ctx)
        a = captured["args"]
        # ns-train runs through the graceful-pause wrapper: python train_resumable.py <pause file> <method> ...
        self.assertEqual(a[0], self.tools.ns_python)
        self.assertTrue(a[1].endswith("train_resumable.py"))
        self.assertTrue(a[2].endswith(".pause-training"))
        self.assertEqual(a[3], "splatfacto")                          # splatfacto-big not installed -> fallback
        self.assertIn("splatfacto-big is not available", self.buf.getvalue())
        self.assertEqual(a[a.index("--max-num-iterations") + 1], "40000")
        self.assertEqual(a[a.index("--steps-per-save") + 1], "1000")
        # full held-out evaluation pushed past the last step (it cost ~4.5 h on a 10k-image job)
        self.assertEqual(a[a.index("--steps-per-eval-all-images") + 1], "40001")
        self.assertNotIn("--load-checkpoint", a)
        self.assertEqual(a[a.index("--machine.seed") + 1], "42")
        self.assertIn("--viewer.quit-on-train-completion", a)
        self.assertIn("--pipeline.model.use-scale-regularization", a)
        self.assertEqual(a[a.index("--pipeline.model.cull-alpha-thresh") + 1], "0.005")
        self.assertEqual(a[-3:], ["nerfstudio-data", "--downscale-factor", "1"])
        self.assertEqual(data["method"], "splatfacto")
        self.assertIn('"percent":2.5', self.buf.getvalue())

    def _ckpt(self, step, complete=True):
        d = self.ws.training_dir / "pf360" / "splatfacto" / "run" / "nerfstudio_models"
        d.mkdir(parents=True, exist_ok=True)
        p = d / ("step-%09d.ckpt" % step)
        # torch.save writes a zip: a complete file ends with the end-of-central-directory record.
        p.write_bytes(b"PK\x03\x04" + b"x" * 100 + (b"PK\x05\x06" + b"\0" * 18 if complete else b""))
        (d.parent / "config.yml").write_text("x")
        return p

    def test_training_resumes_from_latest_complete_checkpoint(self):
        be = NerfstudioBackend()
        be._help_cache["train-splatfacto"] = NEW_TYRO
        self._ckpt(9999)
        truncated = self._ckpt(12999, complete=False)      # killed while saving
        self.ctx.results["_resume_training"] = True
        captured = {}

        def fake_run(args, **kw):
            captured["args"] = args
            kw["on_line"]("20000 (80.00%)   12 ms")         # absolute step after the resume
            return 0
        with mock.patch.object(procrun, "run", side_effect=fake_run):
            data = be.run_stage("training", self.ctx)
        a = captured["args"]
        self.assertTrue(a[a.index("--load-checkpoint") + 1].endswith("step-000009999.ckpt"))
        self.assertEqual(a[a.index("--max-num-iterations") + 1], str(40000 - 10000))   # only the remainder
        self.assertFalse(truncated.exists())                 # unusable partial checkpoint removed
        self.assertEqual(data["resumedFromStep"], 9999)
        self.assertIn('"percent":50.0', self.buf.getvalue())  # 20000 / 40000, not Nerfstudio's 80%

    def test_training_not_resumed_without_restore_point(self):
        be = NerfstudioBackend()
        be._help_cache["train-splatfacto"] = NEW_TYRO
        old = self._ckpt(9999)
        captured = {}

        def fake_run(args, **kw):
            captured["args"] = args
            self._ckpt(39999)
            return 0
        with mock.patch.object(procrun, "run", side_effect=fake_run):
            be.run_stage("training", self.ctx)
        self.assertNotIn("--load-checkpoint", captured["args"])
        self.assertFalse(old.exists())                       # settings changed -> fresh training dir

    def test_training_pause_saves_checkpoint_then_raises_cancelled(self):
        be = NerfstudioBackend()
        be._help_cache["train-splatfacto"] = NEW_TYRO
        seen = {}

        def fake_run(args, **kw):
            pause_file = Path(args[2])
            self.ctx.cancel.cancel()                         # user pressed Pause
            for _ in range(50):                              # wrapper sees the pause request...
                if pause_file.exists():
                    break
                time.sleep(0.05)
            seen["pause"] = pause_file.exists()
            self._ckpt(1500)                                 # ...saves a checkpoint and exits 75
            return 75
        with mock.patch.object(procrun, "run", side_effect=fake_run):
            with self.assertRaises(procrun.Cancelled):
                be.run_stage("training", self.ctx)
        self.assertTrue(seen["pause"])
        self.assertFalse((self.ws.root / ".pause-training").exists())

    def test_training_failure_raises(self):
        be = NerfstudioBackend()
        be._help_cache["train-splatfacto"] = OLD_TYRO
        with mock.patch.object(procrun, "run", return_value=1):
            with self.assertRaises(StageFailed):
                be.run_stage("training", self.ctx)

    def _transforms(self, centers, total):
        (self.ws.planar_dir).mkdir(parents=True, exist_ok=True)
        for i in range(total):
            (self.ws.planar_dir / ("f_%03d.jpg" % i)).write_bytes(b"x")
        self.ws.processed_dir.mkdir(parents=True, exist_ok=True)
        frames = [{"file_path": "images/%d.jpg" % i,
                   "transform_matrix": [[1, 0, 0, c[0]], [0, 1, 0, c[1]], [0, 0, 1, c[2]], [0, 0, 0, 1]]}
                  for i, c in enumerate(centers)]
        (self.ws.processed_dir / "transforms.json").write_text(json.dumps({"frames": frames}))

    def test_reconstruction_ok(self):
        centers = [(i * 0.5, 0, 0) for i in range(10) for _ in range(8)]
        self._transforms(centers, 80)
        d = NerfstudioBackend()._validate_reconstruction(self.ctx, 80)
        self.assertEqual(d["registeredImages"], 80)
        self.assertGreaterEqual(d["distinctCameraPositions"], 10)

    def test_reconstruction_rotation_only_is_degenerate(self):
        self._transforms([(1.0, 2.0, 3.0)] * 80, 80)
        with self.assertRaises(StageFailed) as cm:
            NerfstudioBackend()._validate_reconstruction(self.ctx, 80)
        self.assertIn("barely move", str(cm.exception))

    def test_reconstruction_low_registration(self):
        self._transforms([(i, 0, 0) for i in range(10)], 80)
        with self.assertRaises(StageFailed):
            NerfstudioBackend()._validate_reconstruction(self.ctx, 80)

    def test_missing_transforms(self):
        with self.assertRaises(StageFailed):
            NerfstudioBackend()._validate_reconstruction(self.ctx, 80)

    def test_export_resolves_output_dynamically(self):
        be = NerfstudioBackend()
        d = self.ws.training_dir / "pf360" / "splatfacto" / "run"
        (d / "nerfstudio_models").mkdir(parents=True)
        (d / "nerfstudio_models" / "s.ckpt").write_bytes(b"x")
        (d / "config.yml").write_text("x")

        def fake_run(args, **kw):
            self.assertIn("--load-config", args)
            out = args[args.index("--output-dir") + 1]
            with open(os.path.join(out, "splat.ply"), "wb") as f:   # tool ignored our filename
                f.write(b"ply")
            return 0
        with mock.patch.object(procrun, "run", side_effect=fake_run):
            data = be.run_stage("export", self.ctx)
        self.assertTrue(data["path"].endswith("out.ply"))
        self.assertTrue(os.path.isfile(data["path"]))

    def test_workspace_refuses_foreign_folder(self):
        foreign = os.path.join(self.tmp, "foreign")
        os.makedirs(foreign)
        open(os.path.join(foreign, "important.txt"), "w").close()
        with self.assertRaises(RuntimeError):
            Workspace(foreign, "x.ply").init()


if __name__ == "__main__":
    unittest.main()

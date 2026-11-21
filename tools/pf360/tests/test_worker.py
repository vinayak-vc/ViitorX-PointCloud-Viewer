"""End-to-end worker orchestration with a fake backend (real ffprobe/ffmpeg, no GPU)."""
import io
import json
import os
import tempfile
import unittest
from unittest import mock

from pf360 import environment, worker
from pf360.backend import StageFailed
from pf360.nerfstudio_backend import NerfstudioBackend
from pf360.procrun import CancelToken, Cancelled
from pf360.protocol import Emitter, parse_event
from tests import helpers

HAVE_FF = bool(helpers.ffmpeg() and environment.resolve_tools().ffprobe)


class FakeBackend(NerfstudioBackend):
    def __init__(self, fail_at=None, cancel_at=None, token=None):
        super().__init__()
        self.fail_at, self.cancel_at, self.token, self.ran = fail_at, cancel_at, token, []

    def check_environment(self, tools, manifest):
        return []

    def stage_outputs_exist(self, stage, ctx):
        return stage != "export" or ctx.workspace.export_ply.is_file()

    def run_stage(self, stage, ctx):
        self.ran.append(stage)
        if stage == self.fail_at:
            raise StageFailed("boom at " + stage)
        if stage == self.cancel_at:
            self.token.cancel()
            raise Cancelled()
        if stage == "export":
            ctx.workspace.export_dir.mkdir(parents=True, exist_ok=True)
            helpers.write_ply(str(ctx.workspace.export_ply), n=64)
            return {"path": str(ctx.workspace.export_ply)}
        return {"stage": stage}


@unittest.skipUnless(HAVE_FF, "ffmpeg/ffprobe not installed")
class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="pf360 test ")
        self.video = os.path.join(self.tmp, "clip ü.mp4")
        self.assertTrue(helpers.make_video(self.video, 2048, 1024, 6))
        self.raw = {"schemaVersion": 1, "jobId": "w_001",
                    "input": {"videoPath": self.video, "projection": "equirectangular", "projectionConfirmed": True},
                    "processing": {"targetFrames": 6}, "training": {"qualityPreset": "preview"},
                    "output": {"directory": os.path.join(self.tmp, "ws")},
                    "environment": {"nerfstudioPython": None}}

    def run_job(self, backend, token=None, raw=None):
        buf = io.StringIO()
        token = token or CancelToken()
        job = worker.Job(raw or self.raw, Emitter(buf), token)
        job.backend = backend
        # no GPU probe in tests: pretend CUDA works
        with mock.patch.object(environment, "query_torch", return_value={"cuda": True, "device": "FakeGPU",
                                                                         "freeMB": 99999}), \
                mock.patch.object(environment, "list_methods", return_value=["splatfacto", "splatfacto-big"]), \
                mock.patch.object(environment.Tools, "ns", lambda self, t: "fake"):
            job.tools = None
            orig = environment.resolve_tools

            def rt(cfg=None):
                t = orig(cfg)
                t.ns_python = None        # use in-process PLY validation
                return t
            with mock.patch.object(environment, "resolve_tools", side_effect=rt):
                code = job.run()
        events = [parse_event(l) for l in buf.getvalue().splitlines()]
        return code, events

    def test_full_pipeline_and_cache(self):
        be = FakeBackend()
        code, ev = self.run_job(be)
        types = [e["type"] for e in ev]
        self.assertEqual(code, 0, [e.get("message") for e in ev if e["type"] in ("job_failed", "error")])
        self.assertEqual(types[-1], "job_completed")
        self.assertTrue(all(e["jobId"] == "w_001" for e in ev))
        done = ev[-1]
        self.assertTrue(done["outputPath"].endswith("_splat.ply"))
        self.assertEqual(done["data"]["output"]["gaussianCount"], 64)
        self.assertEqual(done["data"]["processing"]["framesSelected"], 6)
        with open(os.path.join(self.tmp, "ws", "reconstruction_report.json"), encoding="utf-8") as f:
            rep = json.load(f)
        self.assertEqual(rep["status"], "completed")
        # second run: everything cached except validation stages
        be2 = FakeBackend()
        code, ev = self.run_job(be2)
        self.assertEqual(code, 0)
        self.assertEqual(be2.ran, [])
        # frames, projection, masking (off), reconstruction, training, export
        self.assertEqual(sum(1 for e in ev if e["type"] == "stage_skipped"), 6)
        off = [e for e in ev if e["type"] == "stage_skipped" and e.get("stage") == "masking"]
        self.assertEqual(off[0]["data"], {"enabled": False})
        # changing a training setting re-runs training + export only
        raw = json.loads(json.dumps(self.raw))
        raw["training"]["iterations"] = 1234
        be3 = FakeBackend()
        code, ev = self.run_job(be3, raw=raw)
        self.assertEqual(code, 0)
        self.assertEqual(be3.ran, ["training", "export"])

    def test_masking_stage_runs_and_invalidates_downstream(self):
        raw = json.loads(json.dumps(self.raw))
        raw["processing"]["masking"] = {"enabled": True, "groups": ["people"], "dilatePx": 4}
        be = FakeBackend()
        code, ev = self.run_job(be, raw=raw)
        self.assertEqual(code, 0, [e.get("message") for e in ev if e["type"] in ("job_failed", "error")])
        self.assertEqual(be.ran, ["panorama_projection", "masking", "camera_reconstruction", "training", "export"])
        self.assertEqual(ev[-1]["data"]["processing"]["masking"], {"stage": "masking"})
        # Changing the mask categories re-runs masking and everything after it, not the projection.
        raw["processing"]["masking"]["groups"] = ["people", "vehicles"]
        be2 = FakeBackend()
        code, ev = self.run_job(be2, raw=raw)
        self.assertEqual(code, 0)
        self.assertEqual(be2.ran, ["masking", "camera_reconstruction", "training", "export"])
        # Turning masking off again: reconstruction must not reuse the masked result.
        raw["processing"]["masking"]["enabled"] = False
        be3 = FakeBackend()
        code, ev = self.run_job(be3, raw=raw)
        self.assertEqual(be3.ran, ["camera_reconstruction", "training", "export"])

    def test_masking_without_model_fails_before_gpu_work(self):
        raw = json.loads(json.dumps(self.raw))
        raw["processing"]["masking"] = {"enabled": True, "groups": ["sky"]}

        class NoModel(FakeBackend):
            def check_environment(self, tools, manifest):
                tools.mask_model = None
                return NerfstudioBackend.check_environment(self, tools, manifest)[:1]
        be = NoModel()
        code, ev = self.run_job(be, raw=raw)
        self.assertEqual(code, worker.EXIT_FAILED)
        self.assertIn("setup_masking.ps1", ev[-1]["message"])
        self.assertEqual(be.ran, [])

    def test_stage_failure_reports_job_failed(self):
        code, ev = self.run_job(FakeBackend(fail_at="camera_reconstruction"))
        self.assertEqual(code, worker.EXIT_FAILED)
        self.assertEqual(ev[-1]["type"], "job_failed")
        self.assertIn("boom", ev[-1]["message"])
        self.assertTrue(any(e["type"] == "error" and e.get("stage") == "camera_reconstruction" for e in ev))

    def test_cancellation(self):
        token = CancelToken()
        code, ev = self.run_job(FakeBackend(cancel_at="training", token=token), token=token)
        self.assertEqual(code, worker.EXIT_CANCELLED)
        self.assertEqual(ev[-1]["type"], "job_cancelled")
        self.assertTrue(os.path.isdir(os.path.join(self.tmp, "ws", "frames")))   # intermediates kept

    def test_pause_then_resume_continues_the_interrupted_stage(self):
        token = CancelToken()
        code, ev = self.run_job(FakeBackend(cancel_at="training", token=token), token=token)
        self.assertEqual(code, worker.EXIT_CANCELLED)
        with open(os.path.join(self.tmp, "ws", "state.json"), encoding="utf-8") as f:
            st = json.load(f)["stages"]
        self.assertEqual(st["training"]["status"], "cancelled")
        self.assertEqual(st["camera_reconstruction"]["status"], "completed")

        class Spy(FakeBackend):
            def run_stage(self, stage, ctx):
                self.flags = getattr(self, "flags", {})
                self.flags[stage] = ctx.results.get("_resume_" + stage)
                return FakeBackend.run_stage(self, stage, ctx)
        be = Spy()
        code, ev = self.run_job(be)
        self.assertEqual(code, 0)
        self.assertEqual(be.ran, ["training", "export"])     # finished stages reused
        self.assertTrue(be.flags["training"])                 # training continues from its restore point
        self.assertFalse(be.flags["export"])
        started = [e for e in ev if e["type"] == "stage_started" and e.get("stage") == "training"]
        self.assertEqual(started[0].get("data"), {"resume": True})

    def test_crash_counts_as_interrupted_and_settings_change_restarts(self):
        code, _ = self.run_job(FakeBackend())
        self.assertEqual(code, 0)
        path = os.path.join(self.tmp, "ws", "state.json")
        with open(path, encoding="utf-8") as f:
            st = json.load(f)
        st["stages"]["training"]["status"] = "running"       # app / PC died mid-training
        with open(path, "w", encoding="utf-8") as f:
            json.dump(st, f)

        class Spy(FakeBackend):
            def run_stage(self, stage, ctx):
                self.flags = getattr(self, "flags", {})
                self.flags[stage] = ctx.results.get("_resume_" + stage)
                return FakeBackend.run_stage(self, stage, ctx)
        be = Spy()
        self.assertEqual(self.run_job(be)[0], 0)
        self.assertTrue(be.flags["training"])
        # Different training settings: the old partial training must not be continued.
        with open(path, encoding="utf-8") as f:
            st = json.load(f)
        st["stages"]["training"]["status"] = "cancelled"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(st, f)
        raw = json.loads(json.dumps(self.raw))
        raw["training"]["iterations"] = 777
        be2 = Spy()
        self.assertEqual(self.run_job(be2, raw=raw)[0], 0)
        self.assertFalse(be2.flags["training"])

    def test_unconfirmed_projection_fails(self):
        raw = json.loads(json.dumps(self.raw))
        raw["input"]["projectionConfirmed"] = False
        code, ev = self.run_job(FakeBackend(), raw=raw)
        self.assertEqual(code, worker.EXIT_FAILED)
        self.assertIn("Confirm", ev[-1]["message"])

    def test_invalid_manifest(self):
        raw = json.loads(json.dumps(self.raw))
        raw["training"]["method"] = "nerfacto"
        code, ev = self.run_job(FakeBackend(), raw=raw)
        self.assertEqual(code, worker.EXIT_MANIFEST)
        self.assertIn("training.method", ev[-1]["message"])

    def test_frames_only(self):
        raw = json.loads(json.dumps(self.raw))
        raw["options"] = {"stopAfter": "frame_extraction"}
        be = FakeBackend()
        code, ev = self.run_job(be, raw=raw)
        self.assertEqual(code, 0)
        self.assertEqual(be.ran, [])
        self.assertIsNone(ev[-1].get("outputPath"))


if __name__ == "__main__":
    unittest.main()

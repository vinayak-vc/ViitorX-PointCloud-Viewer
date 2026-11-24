"""Camera stage restore point: a FINISHED COLMAP mapper (hours on real footage) is kept on resume.

procrun.run is replaced by a recorder that stops at the first command, so no COLMAP is needed."""
import io
import os
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pf360 import manifest, procrun
from pf360.backend import StageContext
from pf360.environment import Tools
from pf360.nerfstudio_backend import MAPPER_DONE, NerfstudioBackend
from pf360.protocol import Emitter
from pf360.workspace import Workspace


def _fake_jpeg(path: Path, w: int, h: int) -> None:
    app0 = b"\xff\xe0" + struct.pack(">H", 16) + b"JFIF\x00" + b"\x00" * 9
    sof0 = b"\xff\xc0" + struct.pack(">HBHHB", 11, 8, h, w, 1) + b"\x01\x11\x00"
    path.write_bytes(b"\xff\xd8" + app0 + sof0 + b"\xff\xd9")


class _Stop(Exception):
    pass


class MapperReuseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        video = os.path.join(self.tmp, "v.mp4")
        open(video, "wb").close()
        raw = {"schemaVersion": 1, "jobId": "r1", "input": {"videoPath": video},
               "processing": {"targetFrames": 4, "imagesPerEquirect": 14}, "training": {"qualityPreset": "high"},
               "output": {"directory": os.path.join(self.tmp, "ws")}}
        self.m = manifest.validate(manifest.resolve(raw, 10.0))
        self.ws = Workspace(self.m["output"]["directory"], "out.ply")
        self.ws.init()
        self.ws.planar_dir.mkdir(parents=True, exist_ok=True)
        for k in range(1, 3):
            for v in range(14):
                _fake_jpeg(self.ws.planar_dir / ("frame_%05d_%d.jpg" % (k, v)), 64, 64)
        exe = os.path.join(self.tmp, "tool.exe")
        open(exe, "wb").close()
        self.tools = Tools(ns_python=exe, ns_bin=self.tmp, colmap=exe, ffmpeg=exe)
        self.sparse = self.ws.processed_dir / "colmap" / "sparse"

    def ctx(self, resume):
        return StageContext(manifest=self.m, tools=self.tools, workspace=self.ws, emitter=Emitter(io.StringIO(), "r1"),
                            cancel=procrun.CancelToken(),
                            results={"_env": {"cuda": True}, "_resume_camera_reconstruction": resume})

    def previous_run(self, mapper_done):
        """What an interrupted run leaves: database, mapper models, a half-done joined model."""
        (self.sparse / "0").mkdir(parents=True)
        (self.sparse / "0" / "images.bin").write_bytes(struct.pack("<Q", 20))
        (self.sparse / "1").mkdir()
        (self.sparse / "1" / "images.bin").write_bytes(struct.pack("<Q", 7))
        (self.sparse / "joined_rig").mkdir()
        (self.ws.processed_dir / "colmap" / "database.db").write_bytes(b"db")
        if mapper_done:
            (self.sparse / MAPPER_DONE).write_text('{"models": 2}')

    def first_command(self, resume):
        calls = []

        def fake_run(args, **kw):
            calls.append([str(a) for a in args])
            raise _Stop()

        with mock.patch("pf360.nerfstudio_backend.procrun.run", fake_run):
            with self.assertRaises(_Stop):
                NerfstudioBackend()._camera_reconstruction(self.ctx(resume))
        return calls[0]

    def test_finished_mapper_is_reused_on_resume(self):
        self.previous_run(mapper_done=True)
        cmd = self.first_command(resume=True)
        self.assertTrue(any(a.endswith("rig_complete.py") for a in cmd), cmd)     # straight to joining
        self.assertEqual(sorted(p.name for p in self.sparse.iterdir()), ["0", "1", MAPPER_DONE])
        self.assertEqual((self.ws.processed_dir / "colmap" / "database.db").read_bytes(), b"db")

    def test_unfinished_mapper_runs_again_but_keeps_the_database(self):
        self.previous_run(mapper_done=False)
        cmd = self.first_command(resume=True)
        self.assertIn("feature_extractor", cmd)
        self.assertEqual(list(self.sparse.iterdir()), [])
        self.assertEqual((self.ws.processed_dir / "colmap" / "database.db").read_bytes(), b"db")

    def test_marker_is_written_after_the_mapper_succeeds(self):
        calls = []

        def fake_run(args, **kw):
            args = [str(a) for a in args]
            calls.append(args)
            if "mapper" in args:
                (self.sparse / "0").mkdir(parents=True)
                (self.sparse / "0" / "images.bin").write_bytes(struct.pack("<Q", 20))
                self.assertFalse((self.sparse / MAPPER_DONE).exists())     # not before it finished
            if any(a.endswith("rig_complete.py") for a in args):
                raise _Stop()
            return 0

        with mock.patch("pf360.nerfstudio_backend.procrun.run", fake_run):
            with self.assertRaises(_Stop):
                NerfstudioBackend()._camera_reconstruction(self.ctx(False))
        self.assertTrue((self.sparse / MAPPER_DONE).is_file())
        self.assertEqual([c[1] for c in calls[:3]], ["feature_extractor", "sequential_matcher", "mapper"])
        mapper = calls[2]                       # big models: global BA spaced by COLMAP's 10% ratio, not every 500 views
        self.assertEqual(mapper[mapper.index("--Mapper.ba_global_images_freq") + 1], "2000")
        self.assertEqual(mapper[mapper.index("--Mapper.ba_global_points_freq") + 1], "1000000")

    def test_extension_mapper_spaces_global_bundle_adjustment(self):
        calls = []

        def fake_run(args, **kw):
            calls.append([str(a) for a in args])
            raise _Stop()

        rig = {"dir": "joined_rig", "panoramasCompleted": 1, "registeredViews": 14}
        with mock.patch("pf360.nerfstudio_backend.procrun.run", fake_run):
            with self.assertRaises(_Stop):
                NerfstudioBackend()._extend(self.ctx(False), {}, io.StringIO(), "colmap.exe", self.sparse,
                                            self.ws.processed_dir / "colmap" / "database.db", {}, rig)
        cmd = calls[0]
        self.assertIn("--input_path", cmd)
        self.assertEqual(cmd[cmd.index("--Mapper.ba_global_images_freq") + 1], "2000")
        self.assertEqual(cmd[cmd.index("--Mapper.ba_global_points_freq") + 1], "1000000")

    def test_fresh_run_starts_from_scratch(self):
        self.previous_run(mapper_done=True)
        cmd = self.first_command(resume=False)
        self.assertIn("feature_extractor", cmd)
        self.assertFalse((self.ws.processed_dir / "colmap" / "database.db").exists())


if __name__ == "__main__":
    unittest.main()

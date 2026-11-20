import json
import os
import tempfile
import unittest

from pf360 import manifest
from pf360.views import perspective_size, views_per_frame


def base(tmp, video):
    return {"schemaVersion": 1, "jobId": "t_001", "jobType": "360_video_to_gaussian_splat",
            "input": {"videoPath": video, "projection": "equirectangular"},
            "processing": {}, "training": {"qualityPreset": "balanced"},
            "output": {"directory": os.path.join(tmp, "out")}}


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.video = os.path.join(self.tmp, "clip.mp4")
        open(self.video, "wb").close()

    def test_presets_resolve_and_validate(self):
        for preset, method, views in (("preview", "splatfacto", 8), ("balanced", "splatfacto", 8),
                                      ("high", "splatfacto-big", 14)):
            raw = base(self.tmp, self.video)
            raw["training"]["qualityPreset"] = preset
            m = manifest.validate(manifest.resolve(raw, video_duration=60.0))
            self.assertEqual(m["training"]["method"], method)
            self.assertEqual(m["processing"]["imagesPerEquirect"], views)
            self.assertGreaterEqual(m["processing"]["targetFrames"], 2)

    def test_frames_for_window_clamps(self):
        self.assertEqual(manifest.frames_for_window(3.0, 30.0, 30, 300), 90)
        self.assertEqual(manifest.frames_for_window(3.0, 1000.0, 30, 300), 300)
        self.assertEqual(manifest.frames_for_window(3.0, 2.0, 30, 300), 30)

    def test_window_uses_start_end(self):
        raw = base(self.tmp, self.video)
        raw["input"].update(startTimeSeconds=10, endTimeSeconds=40)
        m = manifest.resolve(raw, video_duration=100.0)
        self.assertEqual(m["processing"]["targetFrames"], 90)   # 30 s * 3 fps

    def test_explicit_fields_override_preset(self):
        raw = base(self.tmp, self.video)
        raw["training"].update(iterations=1234, method="splatfacto-big")
        raw["processing"]["targetFrames"] = 17
        m = manifest.validate(manifest.resolve(raw, 60.0))
        self.assertEqual(m["training"]["iterations"], 1234)
        self.assertEqual(m["processing"]["targetFrames"], 17)

    def test_invalid_fields_are_all_reported(self):
        raw = base(self.tmp, self.video)
        raw["training"].update(method="nerfacto", downscaleFactor=3, shDegree=7)
        raw["processing"].update(cropTop=0.9, imagesPerEquirect=10, targetFrames=1)
        raw["output"]["plyFileName"] = "../evil.ply"
        raw["environment"] = {"backend": "docker", "colmapExe": os.path.join(self.tmp, "nope.exe")}
        with self.assertRaises(manifest.ManifestError) as cm:
            manifest.validate(manifest.resolve(raw, 60.0))
        text = " ".join(cm.exception.problems)
        for needle in ("training.method", "downscaleFactor", "shDegree", "cropTop", "imagesPerEquirect",
                       "targetFrames", "plyFileName", "environment.backend", "colmapExe"):
            self.assertIn(needle, text)

    def test_missing_and_unsupported_video(self):
        raw = base(self.tmp, os.path.join(self.tmp, "missing.mp4"))
        with self.assertRaises(manifest.ManifestError) as cm:
            manifest.validate(manifest.resolve(raw, 60.0))
        self.assertTrue(any("does not exist" in p for p in cm.exception.problems))
        raw = base(self.tmp, os.path.join(self.tmp, "clip.avi"))
        with self.assertRaises(manifest.ManifestError):
            manifest.validate(manifest.resolve(raw, 60.0), check_files=False)

    def test_bad_time_window(self):
        raw = base(self.tmp, self.video)
        raw["input"].update(startTimeSeconds=20, endTimeSeconds=10)
        with self.assertRaises(manifest.ManifestError):
            manifest.validate(manifest.resolve(raw, 60.0))

    def test_non_equirect_projection_rejected(self):
        raw = base(self.tmp, self.video)
        raw["input"]["projection"] = "dual_fisheye"
        with self.assertRaises(manifest.ManifestError):
            manifest.validate(manifest.resolve(raw, 60.0))

    def test_load_from_file(self):
        p = os.path.join(self.tmp, "job.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(base(self.tmp, self.video), f)
        m = manifest.load(p, video_duration=30.0)
        self.assertEqual(m["jobId"], "t_001")


class ViewTests(unittest.TestCase):
    def test_view_counts_match_nerfstudio(self):
        self.assertEqual(views_per_frame(8), 8)
        self.assertEqual(views_per_frame(14), 14)
        # left/right crops narrow the yaw span (Nerfstudio semantics)
        self.assertEqual(views_per_frame(8, (0, 0, 0.25, 0.25)), 4)
        self.assertEqual(perspective_size(5760, 2880, 8), 1440)
        self.assertEqual(perspective_size(5760, 2880, 14), 1088)

    def test_bottom_crop_can_drop_a_ring(self):
        # 8 views, 0.45 bottom crop: the +45 ring is removed (2 views) -> 6
        self.assertEqual(views_per_frame(8, (0, 0.45, 0, 0)), 6)

    def test_unsupported_combination_raises(self):
        with self.assertRaises(ValueError):
            views_per_frame(8, (0.45, 0.45, 0, 0))

    def test_invalid_view_count(self):
        with self.assertRaises(ValueError):
            views_per_frame(10)


if __name__ == "__main__":
    unittest.main()

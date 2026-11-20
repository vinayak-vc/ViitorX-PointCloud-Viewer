import os
import tempfile
import unittest

from pf360 import environment, video_probe
from tests import helpers


def probe_json(w=5760, h=2880, side=None, streams=1, rotation=None, tags=None):
    vs = []
    for _ in range(streams):
        s = {"codec_type": "video", "codec_name": "hevc", "width": w, "height": h, "pix_fmt": "yuv420p",
             "avg_frame_rate": "30000/1001", "r_frame_rate": "30000/1001", "duration": "72.0", "nb_frames": "2158"}
        sd = list(side or [])
        if rotation is not None:
            sd.append({"side_data_type": "Display Matrix", "rotation": rotation})
        if sd:
            s["side_data_list"] = sd
        if tags:
            s["tags"] = tags
        vs.append(s)
    vs.append({"codec_type": "audio", "codec_name": "aac"})
    return {"streams": vs, "format": {"format_name": "mov,mp4", "duration": "72.0", "size": "123456"}}


class ParseTests(unittest.TestCase):
    def test_metadata_fields(self):
        info = video_probe.parse_probe(probe_json(), "C:/v/room.mp4")
        self.assertEqual((info["width"], info["height"]), (5760, 2880))
        self.assertAlmostEqual(info["fps"], 29.97, places=2)
        self.assertEqual(info["frameCount"], 2158)
        self.assertEqual(info["codec"], "hevc")
        self.assertTrue(info["hasAudio"])
        self.assertEqual(info["aspectRatio"], 2.0)

    def test_rotation_swaps_dimensions(self):
        info = video_probe.parse_probe(probe_json(w=2880, h=5760, rotation=-90), "a.mp4")
        self.assertEqual((info["width"], info["height"]), (5760, 2880))

    def test_no_video_stream(self):
        info = video_probe.parse_probe({"streams": [{"codec_type": "audio"}], "format": {}}, "a.mp4")
        self.assertFalse(info["readable"])


class ClassifyTests(unittest.TestCase):
    def cls(self, data, path="a.mp4", corners=None):
        return video_probe.classify_projection(video_probe.parse_probe(data, path), corners)

    def test_spherical_metadata_equirect(self):
        c = self.cls(probe_json(side=[{"side_data_type": "Spherical Mapping", "projection": "equirectangular"}]))
        self.assertEqual(c["projection"], "equirectangular")
        self.assertTrue(c["supported"])
        self.assertFalse(c["requiresConfirmation"])

    def test_spherical_metadata_other_projection(self):
        c = self.cls(probe_json(side=[{"side_data_type": "Spherical Mapping", "projection": "cubemap"}]))
        self.assertFalse(c["supported"])

    def test_stereo_rejected(self):
        c = self.cls(probe_json(side=[{"side_data_type": "Spherical Mapping", "projection": "equirectangular"}],
                                tags={"stereo_mode": "top_bottom"}))
        self.assertFalse(c["supported"])

    def test_aspect_only_requires_confirmation(self):
        c = self.cls(probe_json())
        self.assertEqual(c["projection"], "likely_equirectangular")
        self.assertTrue(c["requiresConfirmation"])

    def test_not_two_to_one(self):
        c = self.cls(probe_json(w=3840, h=2160))
        self.assertFalse(c["supported"])

    def test_raw_containers_and_multistream(self):
        self.assertFalse(self.cls(probe_json(), path="VID_001.insv")["supported"])
        self.assertFalse(self.cls(probe_json(w=4096, h=1344, streams=2))["supported"])

    def test_dual_fisheye_corners(self):
        c = self.cls(probe_json(w=3840, h=1920), corners={"cornerMean": 2.0, "centerMean": 120.0, "seamMean": 5.0})
        self.assertEqual(c["projection"], "dual_fisheye")
        self.assertFalse(c["supported"])


class FrameMetricTests(unittest.TestCase):
    def test_flat_vs_textured(self):
        w, h = 32, 16
        flat = bytes([128] * (w * h))
        tex = bytes([(x * 37 + y * 91) % 256 for y in range(h) for x in range(w)])
        self.assertEqual(video_probe.frame_metrics(flat, w, h)["sharpness"], 0.0)
        self.assertGreater(video_probe.frame_metrics(tex, w, h)["sharpness"], 100.0)
        self.assertEqual(video_probe.mean_abs_diff(flat, flat), 0.0)


@unittest.skipUnless(helpers.ffmpeg() and environment.resolve_tools().ffprobe, "ffmpeg/ffprobe not installed")
class RealProbeTests(unittest.TestCase):
    def test_probe_synthetic_video(self):
        tmp = tempfile.mkdtemp()
        v = os.path.join(tmp, "eq.mp4")
        self.assertTrue(helpers.make_video(v))
        t = environment.resolve_tools()
        info = video_probe.probe(t.ffprobe, t.ffmpeg, v, quality=True)
        self.assertEqual((info["width"], info["height"]), (1024, 512))
        self.assertEqual(info["classification"]["projection"], "likely_equirectangular")
        self.assertIn("quality", info)
        self.assertTrue(any("Low resolution" in w for w in info["warnings"]))

    def test_frame_command_writes_rgb(self):
        import io, json, contextlib
        from pf360.__main__ import main
        tmp = tempfile.mkdtemp()
        v = os.path.join(tmp, "eq.mp4")
        self.assertTrue(helpers.make_video(v))
        out = os.path.join(tmp, "f.rgb")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = main(["frame", v, "--time", "1", "--width", "256", "--out", out])
        self.assertEqual(rc, 0)
        info = json.loads(buf.getvalue())
        self.assertEqual((info["width"], info["height"]), (256, 128))
        self.assertEqual(os.path.getsize(out), 256 * 128 * 3)

    def test_corrupt_file(self):
        tmp = tempfile.mkdtemp()
        v = os.path.join(tmp, "bad.mp4")
        with open(v, "wb") as f:
            f.write(b"not a video" * 100)
        t = environment.resolve_tools()
        with self.assertRaises(RuntimeError):
            video_probe.probe(t.ffprobe, t.ffmpeg, v)


if __name__ == "__main__":
    unittest.main()

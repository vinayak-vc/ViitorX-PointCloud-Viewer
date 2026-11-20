"""Fixed-intrinsics COLMAP helpers (nerfstudio_backend): planar pinhole model,
JPEG size sniffing and largest-model selection. No GPU / COLMAP needed."""
import math
import struct
import tempfile
import unittest
from pathlib import Path

from pf360.nerfstudio_backend import PLANAR_FOV_DEG, _colmap_models, _jpeg_size, planar_intrinsics


def _fake_jpeg(path: Path, w: int, h: int) -> None:
    # SOI, an APP0 segment to skip, then SOF0 with height/width.
    app0 = b"\xff\xe0" + struct.pack(">H", 16) + b"JFIF\x00" + b"\x00" * 9
    sof0 = b"\xff\xc0" + struct.pack(">HBHHB", 11, 8, h, w, 1) + b"\x01\x11\x00"
    path.write_bytes(b"\xff\xd8" + app0 + sof0 + b"\xff\xd9")


class PlanarIntrinsicsTest(unittest.TestCase):
    def test_matches_equirect2persp_geometry(self):
        # equirect2persp: pixel i -> ray offset (i - w/2) * 2 tan(fov/2) / w
        w = h = 720
        fx, fy, cx, cy = planar_intrinsics(w, h, 120)
        self.assertAlmostEqual(fx, 720 / (2 * math.tan(math.radians(60))), places=6)  # ~207.85
        self.assertAlmostEqual(fx, fy)
        self.assertEqual((cx, cy), (360.5, 360.5))
        # The edge pixel (index 0, COLMAP centre 0.5) is at -tan(60deg) * (w/2)/(w/2) offset.
        self.assertAlmostEqual((0.5 - cx) / fx, -(w / 2) * 2 * math.tan(math.radians(60)) / w, places=9)

    def test_fov_table_matches_nerfstudio(self):
        self.assertEqual(PLANAR_FOV_DEG, {8: 120, 14: 110})


class JpegSizeTest(unittest.TestCase):
    def test_reads_sof(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.jpg"
            _fake_jpeg(p, 1234, 567)
            self.assertEqual(_jpeg_size(p), (1234, 567))

    def test_not_an_image(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.jpg"
            p.write_bytes(b"not a jpeg")
            self.assertIsNone(_jpeg_size(p))


class ColmapModelsTest(unittest.TestCase):
    def test_counts_registered_images(self):
        with tempfile.TemporaryDirectory() as d:
            sparse = Path(d)
            for name, n in (("0", 12), ("1", 230), ("2", 3)):
                (sparse / name).mkdir()
                (sparse / name / "images.bin").write_bytes(struct.pack("<Q", n))
            (sparse / "broken").mkdir()  # no images.bin -> ignored
            models = _colmap_models(sparse)
            self.assertEqual(models, [("0", 12), ("1", 230), ("2", 3)])
            self.assertEqual(max(models, key=lambda kv: kv[1])[0], "1")

    def test_missing_dir(self):
        self.assertEqual(_colmap_models(Path("does/not/exist")), [])


if __name__ == "__main__":
    unittest.main()

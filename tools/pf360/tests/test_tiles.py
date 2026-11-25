"""Unit tests for spatial tiling support (BT-12)."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pf360 import tiles


class TestTiles(unittest.TestCase):
    def test_box_to_lfs(self):
        # In transforms frame: x in [-10, 10], y in [-6, 5], z in [-1, 2]
        # Under (x, y, z) -> (x, -y, -z):
        # x_lfs in [-10, 10]
        # y_lfs in [-5, 6]
        # z_lfs in [-2, 1]
        box = (-10.0, -6.0, -1.0, 10.0, 5.0, 2.0)
        lfs_box = tiles.box_to_lfs(box)
        self.assertAlmostEqual(lfs_box[0], -10.0)
        self.assertAlmostEqual(lfs_box[1], -5.0)
        self.assertAlmostEqual(lfs_box[2], -2.0)
        self.assertAlmostEqual(lfs_box[3], 10.0)
        self.assertAlmostEqual(lfs_box[4], 6.0)
        self.assertAlmostEqual(lfs_box[5], 1.0)
        # All minima strictly less than maxima
        self.assertLess(lfs_box[0], lfs_box[3])
        self.assertLess(lfs_box[1], lfs_box[4])
        self.assertLess(lfs_box[2], lfs_box[5])

    def test_plan_tiles(self):
        # Synthetic cameras along Y from -6 to +6, small X spread, flat Z
        cams = np.array([
            [0.0, -6.0, 0.0],
            [1.0, 0.0, 0.1],
            [-1.0, 6.0, -0.1],
        ])
        specs = tiles.plan_tiles_from_cameras(cams, num_tiles=2, overlap=1.5, scene_margin=10.0)
        self.assertEqual(len(specs), 2)
        self.assertEqual(specs[0].walk_axis, 1)  # Y axis

        # Midpoint is 0.0
        # Tile 0 in transforms: Y in [-16, 0.0], crop Y in [-16, 1.5]
        # Tile 1 in transforms: Y in [0.0, 16], crop Y in [-1.5, 16]
        self.assertAlmostEqual(specs[0].core_box[4], 0.0)  # y1 of tile 0
        self.assertAlmostEqual(specs[1].core_box[1], 0.0)  # y0 of tile 1
        self.assertAlmostEqual(specs[0].crop_box[4], 1.5)  # crop y1 has overlap

        # In LFS frame:
        # Tile 0 core box in LFS: y_lfs in [0.0, 16.0]
        # Tile 1 core box in LFS: y_lfs in [-16.0, 0.0]
        self.assertAlmostEqual(specs[0].core_box_lfs[1], 0.0)
        self.assertAlmostEqual(specs[1].core_box_lfs[4], 0.0)

        # Formatted cropbox string is non-empty and valid
        self.assertTrue(len(specs[0].crop_box_lfs_str) > 0)
        parts = [float(x) for x in specs[0].crop_box_lfs_str.split(",")]
        self.assertEqual(len(parts), 6)
        self.assertLess(parts[0], parts[3])
        self.assertLess(parts[1], parts[4])
        self.assertLess(parts[2], parts[5])

    def test_merge_tile_plys_deduplication(self):
        # Create two synthetic tile PLYs in LFS frame with overlapping points
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            p1 = tmp / "tile_0.ply"
            p2 = tmp / "tile_1.ply"
            merged = tmp / "merged.ply"

            # Specs with split at y_lfs = 0.0
            # Tile 0: y_lfs in (0.0, 10.0]
            # Tile 1: y_lfs in [-10.0, 0.0]
            spec0 = tiles.TileSpec(
                index=0,
                core_box=(-10, -10, -10, 10, 0, 10),
                crop_box=(-10, -10, -10, 10, 2, 10),
                core_box_lfs=(-10, 0.0, -10, 10, 10.0, 10),
                crop_box_lfs=(-10, -2.0, -10, 10, 10.0, 10),
                walk_axis=1,
            )
            spec1 = tiles.TileSpec(
                index=1,
                core_box=(-10, 0, -10, 10, 10, 10),
                crop_box=(-10, -2, -10, 10, 10, 10),
                core_box_lfs=(-10, -10.0, -10, 10, 0.0, 10),
                crop_box_lfs=(-10, -10.0, -10, 10, 2.0, 10),
                walk_axis=1,
            )

            # Header helper
            def write_ply(path: Path, pts: np.ndarray):
                props = ["x", "y", "z", "scale_0", "scale_1", "scale_2", "opacity", "rot_0", "rot_1", "rot_2", "rot_3"]
                dtype = [(p, "<f4") for p in props]
                arr = np.zeros(len(pts), dtype=dtype)
                arr["x"] = pts[:, 0]
                arr["y"] = pts[:, 1]
                arr["z"] = pts[:, 2]
                arr["opacity"] = 1.0
                arr["rot_0"] = 1.0

                header = [
                    "ply", "format binary_little_endian 1.0",
                    f"element vertex {len(arr)}",
                ] + [f"property float {p}" for p in props] + ["end_header\n"]
                with open(path, "wb") as f:
                    f.write("\n".join(header).encode("utf-8"))
                    f.write(arr.tobytes())

            # Tile 0 has points at y = 1.0, 2.0, and overlap point at y = -1.0
            pts0 = np.array([[0.0, 2.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])
            # Tile 1 has points at y = -2.0, -1.0, and overlap point at y = 1.0
            pts1 = np.array([[0.0, -2.0, 0.0], [0.0, -1.0, 0.0], [0.0, 1.0, 0.0]])

            write_ply(p1, pts0)
            write_ply(p2, pts1)

            kept = tiles.merge_tile_plys([p1, p2], [spec0, spec1], merged)
            # Tile 0 should keep: y=2.0, y=1.0 (y > 0.0) -> 2 points
            # Tile 1 should keep: y=-2.0, y=-1.0 (y <= 0.0) -> 2 points
            # Total kept should be exactly 4 points (the overlap points dropped)
            self.assertEqual(kept, 4)
            self.assertTrue(merged.is_file())


    def test_should_auto_tile_view_count(self):
        # Walk extent no longer gates auto-tiling (COLMAP scale is arbitrary).
        # Only view count matters.
        self.assertFalse(tiles.should_auto_tile(1000.0, 100))   # large extent, low views -> off
        self.assertFalse(tiles.should_auto_tile(1000.0, 2499))  # just below threshold
        self.assertTrue(tiles.should_auto_tile(0.0, 2500))      # exactly at threshold, tiny extent
        self.assertTrue(tiles.should_auto_tile(0.001, 5000))    # well above threshold

    def test_train_tiles_output_name_no_hardcoded_prefix(self):
        """train_tiles() must not embed 'preview10s' in the output file name."""
        import inspect, pf360.tiles as tiles_mod
        src = inspect.getsource(tiles_mod.train_tiles)
        self.assertNotIn("preview10s", src)

    def test_manifest_tile_overlap_fraction(self):
        """manifest tileOverlap should default to 0.15 and validate 0.05-0.5."""
        import os, tempfile
        from pf360 import manifest
        tmp = tempfile.mkdtemp()
        video = os.path.join(tmp, "clip.mp4")
        open(video, "wb").close()
        raw = {"schemaVersion": 1, "jobId": "t_tile", "jobType": "360_video_to_gaussian_splat",
               "input": {"videoPath": video, "projection": "equirectangular"},
               "processing": {}, "training": {"qualityPreset": "balanced"},
               "output": {"directory": os.path.join(tmp, "out")}}
        m = manifest.validate(manifest.resolve(raw, 60.0))
        # Default should be 0.15 (fraction)
        self.assertAlmostEqual(m["training"]["tileOverlap"], 0.15)
        # In-range fraction values should pass
        raw2 = dict(raw)
        raw2["training"] = dict(raw["training"])
        raw2["training"]["tileOverlap"] = 0.3
        m2 = manifest.validate(manifest.resolve(raw2, 60.0))
        self.assertAlmostEqual(m2["training"]["tileOverlap"], 0.3)
        # Out-of-range values should fail
        raw3 = dict(raw)
        raw3["training"] = dict(raw["training"])
        raw3["training"]["tileOverlap"] = 0.9  # > 0.5
        with self.assertRaises(manifest.ManifestError):
            manifest.validate(manifest.resolve(raw3, 60.0))
        raw4 = dict(raw)
        raw4["training"] = dict(raw["training"])
        raw4["training"]["tileOverlap"] = 0.01  # < 0.05
        with self.assertRaises(manifest.ManifestError):
            manifest.validate(manifest.resolve(raw4, 60.0))


if __name__ == "__main__":
    unittest.main()

"""Unit tests for LichtFeld-Studio integration (manifest, discovery, engine, rotation, sidecar)."""
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pf360 import environment, lichtfeld_engine, manifest as manifest_mod
from pf360.backend import StageContext, StageFailed
from pf360.environment import Tools
from pf360.nerfstudio_backend import NerfstudioBackend
from pf360.procrun import CancelToken
from pf360.protocol import Emitter
from pf360.workspace import Workspace


class TestLichtfeldEngine(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_manifest_defaults_and_validation(self):
        raw = {
            "schemaVersion": 1,
            "jobId": "test_job",
            "input": {"videoPath": str(Path(self.tmp) / "input.mp4")},
            "output": {"directory": self.tmp},
            "processing": {"targetFrames": 50},
        }
        # Touch dummy video file so validation passes
        Path(raw["input"]["videoPath"]).touch()

        # 1. Defaults fill engine="nerfstudio", lfsStrategy="mrnf", maxGaussians=2000000
        resolved = manifest_mod.resolve(raw)
        self.assertEqual(resolved["training"]["engine"], "nerfstudio")
        self.assertEqual(resolved["training"]["lfsStrategy"], "mrnf")
        self.assertEqual(resolved["training"]["maxGaussians"], 2_000_000)
        validated = manifest_mod.validate(resolved)
        self.assertEqual(validated["training"]["engine"], "nerfstudio")

        # 2. Explicit LichtFeld settings
        raw_lfs = dict(raw)
        raw_lfs["training"] = {
            "engine": "lichtfeld",
            "lfsStrategy": "mcmc",
            "maxGaussians": 1_660_000,
        }
        res_lfs = manifest_mod.resolve(raw_lfs)
        self.assertEqual(res_lfs["training"]["engine"], "lichtfeld")
        self.assertEqual(res_lfs["training"]["lfsStrategy"], "mcmc")
        self.assertEqual(res_lfs["training"]["maxGaussians"], 1_660_000)
        val_lfs = manifest_mod.validate(res_lfs)
        self.assertEqual(val_lfs["training"]["engine"], "lichtfeld")

        # 3. Invalid engine raises ManifestError
        raw_bad = dict(raw)
        raw_bad["training"] = {"engine": "invalid_engine"}
        with self.assertRaises(manifest_mod.ManifestError):
            manifest_mod.validate(manifest_mod.resolve(raw_bad))

        # 4. Invalid strategy raises ManifestError
        raw_bad_strat = dict(raw)
        raw_bad_strat["training"] = {"engine": "lichtfeld", "lfsStrategy": "unknown"}
        with self.assertRaises(manifest_mod.ManifestError):
            manifest_mod.validate(manifest_mod.resolve(raw_bad_strat))

        # 5. Invalid maxGaussians raises ManifestError
        raw_bad_cap = dict(raw)
        raw_bad_cap["training"] = {"engine": "lichtfeld", "maxGaussians": -50}
        with self.assertRaises(manifest_mod.ManifestError):
            manifest_mod.validate(manifest_mod.resolve(raw_bad_cap))

    def test_environment_discovery_and_checks(self):
        tools = environment.resolve_tools()
        # Verify tools.lichtfeld property exists
        self.assertTrue(hasattr(tools, "lichtfeld"))

        be = NerfstudioBackend()
        manifest_lfs = {
            "training": {"engine": "lichtfeld"},
            "processing": {},
        }
        # Tools without lichtfeld executable reports problem
        empty_tools = Tools()
        empty_tools.ns_python = "python"
        empty_tools.colmap = "colmap"
        empty_tools.ffmpeg = "ffmpeg"
        problems = be.check_environment(empty_tools, manifest_lfs)
        self.assertTrue(any("LichtFeld-Studio executable not found" in p for p in problems))

    def test_command_builder(self):
        processed = Path(self.tmp) / "processed"
        processed.mkdir(parents=True, exist_ok=True)
        out = Path(self.tmp) / "training" / "lfs"
        out.mkdir(parents=True, exist_ok=True)

        tr = {
            "iterations": 15000,
            "downscaleFactor": 2,
            "lfsStrategy": "mrnf",
            "maxGaussians": 1_500_000,
            "shDegree": 3,
        }
        cmd = lichtfeld_engine.build_cmd("LichtFeld-Studio.exe", processed, out, tr)
        self.assertEqual(cmd[0], "LichtFeld-Studio.exe")
        self.assertIn("--headless", cmd)
        self.assertIn("-d", cmd)
        self.assertEqual(cmd[cmd.index("-d") + 1], str(processed))
        self.assertIn("-o", cmd)
        self.assertEqual(cmd[cmd.index("-o") + 1], str(out))
        self.assertIn("--iter", cmd)
        self.assertEqual(cmd[cmd.index("--iter") + 1], "15000")
        self.assertIn("--strategy", cmd)
        self.assertEqual(cmd[cmd.index("--strategy") + 1], "mrnf")
        self.assertIn("--max-cap", cmd)
        self.assertEqual(cmd[cmd.index("--max-cap") + 1], "1500000")
        self.assertIn("-r", cmd)
        self.assertEqual(cmd[cmd.index("-r") + 1], "2")
        self.assertIn("--export", cmd)
        self.assertEqual(cmd[cmd.index("--export") + 1], "ply")

        # Invalid downscale factor (e.g. 3) raises StageFailed
        tr_bad_r = dict(tr, downscaleFactor=3)
        with self.assertRaises(StageFailed):
            lichtfeld_engine.build_cmd("LichtFeld-Studio.exe", processed, out, tr_bad_r)

        # Unexpected transforms_train.json raises StageFailed
        (processed / "transforms_train.json").touch()
        with self.assertRaises(StageFailed):
            lichtfeld_engine.build_cmd("LichtFeld-Studio.exe", processed, out, tr)
        (processed / "transforms_train.json").unlink()

        # Resume command builder
        ckpt = out / "project.licht"
        ckpt.touch()
        resume_cmd = lichtfeld_engine.build_cmd("LichtFeld-Studio.exe", processed, out, tr,
                                                resume=True, project_path=ckpt)
        self.assertIn("--resume", resume_cmd)
        self.assertEqual(resume_cmd[resume_cmd.index("--resume") + 1], str(ckpt))
        self.assertNotIn("-d", resume_cmd)

    def test_progress_parser_regex(self):
        sample_mrnf = "  1200/30000 | Loss: 0.041234 | Splats: 165432"
        m = lichtfeld_engine.LFS_PROGRESS_RE.search(sample_mrnf)
        self.assertIsNotNone(m)
        self.assertEqual(int(m.group(1)), 1200)
        self.assertEqual(int(m.group(2)), 30000)
        self.assertAlmostEqual(float(m.group(3)), 0.041234, places=6)
        self.assertEqual(int(m.group(4)), 165432)

        sample_mcmc = "30000/30000 | Loss: 0.021500 | Splats: 1660000"
        m2 = lichtfeld_engine.LFS_PROGRESS_RE.search(sample_mcmc)
        self.assertIsNotNone(m2)
        self.assertEqual(int(m2.group(1)), 30000)
        self.assertEqual(int(m2.group(4)), 1660000)

    def test_ply_rotation_and_sh_signs(self):
        # Create a synthetic 3DGS binary PLY with 2 vertices
        props = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
        # Add 45 f_rest properties (degree 3)
        for i in range(45):
            props.append(f"f_rest_{i}")
        props += ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]

        dtype = [(p, "<f4") for p in props]
        vertices = np.zeros(2, dtype=dtype)

        # Vertex 0: position (1.0, 2.0, 3.0), quat (1.0, 0.0, 0.0, 0.0), f_rest = 1.0
        vertices[0]["x"] = 1.0
        vertices[0]["y"] = 2.0
        vertices[0]["z"] = 3.0
        vertices[0]["rot_0"] = 1.0  # w
        vertices[0]["rot_1"] = 0.0  # x
        vertices[0]["rot_2"] = 0.0  # y
        vertices[0]["rot_3"] = 0.0  # z
        for i in range(45):
            vertices[0][f"f_rest_{i}"] = 1.0

        # Vertex 1: position (-4.0, -5.0, 6.0), quat (0.0, 1.0, 0.0, 0.0)
        vertices[1]["x"] = -4.0
        vertices[1]["y"] = -5.0
        vertices[1]["z"] = 6.0
        vertices[1]["rot_0"] = 0.0  # w
        vertices[1]["rot_1"] = 1.0  # x
        vertices[1]["rot_2"] = 0.0  # y
        vertices[1]["rot_3"] = 0.0  # z
        for i in range(45):
            vertices[1][f"f_rest_{i}"] = 2.0

        src_ply = Path(self.tmp) / "synthetic.ply"
        dst_ply = Path(self.tmp) / "synthetic_rotated.ply"

        # Write PLY file
        header = ["ply", "format binary_little_endian 1.0", f"element vertex {len(vertices)}"]
        for p in props:
            header.append(f"property float {p}")
        header.append("end_header\n")
        with open(src_ply, "wb") as f:
            f.write("\n".join(header).encode("utf-8"))
            f.write(vertices.tobytes())

        # Rotate
        lichtfeld_engine.rotate_ply_zup(src_ply, dst_ply)
        self.assertTrue(dst_ply.is_file())

        # Read back rotated PLY
        with open(dst_ply, "rb") as f:
            read_hdr = b""
            while b"end_header\n" not in read_hdr:
                line = f.readline()
                if not line:
                    break
                read_hdr += line
            data = np.frombuffer(f.read(), dtype=dtype)

        self.assertEqual(len(data), 2)

        # Check Vertex 0
        # Position: (x, y, z) -> (x, -y, -z)
        self.assertAlmostEqual(data[0]["x"], 1.0)
        self.assertAlmostEqual(data[0]["y"], -2.0)
        self.assertAlmostEqual(data[0]["z"], -3.0)

        # Quaternion (w=1, x=0, y=0, z=0):
        # w' = -x = 0, x' = w = 1, y' = -z = 0, z' = y = 0
        self.assertAlmostEqual(data[0]["rot_0"], 0.0)
        self.assertAlmostEqual(data[0]["rot_1"], 1.0)
        self.assertAlmostEqual(data[0]["rot_2"], 0.0)
        self.assertAlmostEqual(data[0]["rot_3"], 0.0)

        # Check SH signs for Vertex 0 across all 3 channels
        expected_signs = lichtfeld_engine.SH_SIGNS_15
        for ch in range(3):
            for i in range(15):
                prop = f"f_rest_{ch * 15 + i}"
                expected = 1.0 * expected_signs[i]
                self.assertAlmostEqual(data[0][prop], expected, msg=f"Mismatch at {prop}")

    def test_fingerprint_preservation_for_nerfstudio(self):
        be = NerfstudioBackend()
        ctx_ns = StageContext(
            manifest={
                "processing": {"imagesPerEquirect": 8, "cropTop": 0.0, "cropBottom": 0.0, "cropLeft": 0.0,
                               "cropRight": 0.0, "matchingMethod": "sequential"},
                "training": {"engine": "nerfstudio", "method": "splatfacto", "iterations": 30000,
                             "downscaleFactor": 2, "shDegree": 3, "randomSeed": 42,
                             "useScaleRegularization": False},
                "output": {"plyFileName": "test.ply"},
            },
            tools=Tools(),
            workspace=Workspace(self.tmp, "test.ply"),
            emitter=Emitter(),
            cancel=CancelToken(),
        )
        inputs_ns = be.stage_inputs("training", ctx_ns)
        # Old NS fingerprints must not contain engine, lfsStrategy, or maxGaussians
        self.assertNotIn("engine", inputs_ns)
        self.assertNotIn("lfsStrategy", inputs_ns)
        self.assertNotIn("maxGaussians", inputs_ns)
        self.assertEqual(inputs_ns["method"], "splatfacto")
        self.assertEqual(inputs_ns["iterations"], 30000)

        # LFS fingerprints include engine, lfsStrategy, maxGaussians
        ctx_lfs = StageContext(
            manifest={
                "processing": {"imagesPerEquirect": 8, "cropTop": 0.0, "cropBottom": 0.0, "cropLeft": 0.0,
                               "cropRight": 0.0, "matchingMethod": "sequential"},
                "training": {"engine": "lichtfeld", "lfsStrategy": "mrnf", "maxGaussians": 1660000,
                             "iterations": 30000, "downscaleFactor": 1, "shDegree": 3},
                "output": {"plyFileName": "test.ply"},
            },
            tools=Tools(),
            workspace=Workspace(self.tmp, "test.ply"),
            emitter=Emitter(),
            cancel=CancelToken(),
        )
        inputs_lfs = be.stage_inputs("training", ctx_lfs)
        self.assertEqual(inputs_lfs["engine"], "lichtfeld")
        self.assertEqual(inputs_lfs["lfsStrategy"], "mrnf")
        self.assertEqual(inputs_lfs["maxGaussians"], 1660000)

        # LFS fingerprints include seedPointCap, dropMisplacedPanoramas, maxTrainingViews when non-default
        ctx_lfs_clean = StageContext(
            manifest={
                "processing": {"imagesPerEquirect": 8, "cropTop": 0.0, "cropBottom": 0.0, "cropLeft": 0.0,
                               "cropRight": 0.0, "matchingMethod": "sequential"},
                "training": {"engine": "lichtfeld", "lfsStrategy": "mrnf", "maxGaussians": 1660000,
                             "iterations": 30000, "downscaleFactor": 1, "shDegree": 3,
                             "seedPointCap": 200000, "dropMisplacedPanoramas": False, "maxTrainingViews": 500},
                "output": {"plyFileName": "test.ply"},
            },
            tools=Tools(),
            workspace=Workspace(self.tmp, "test.ply"),
            emitter=Emitter(),
            cancel=CancelToken(),
        )
        inputs_lfs_clean = be.stage_inputs("training", ctx_lfs_clean)
        self.assertEqual(inputs_lfs_clean["seedPointCap"], 200000)
        self.assertEqual(inputs_lfs_clean["dropMisplacedPanoramas"], False)
        self.assertEqual(inputs_lfs_clean["maxTrainingViews"], 500)

    def test_export_lfs_fails_when_target_missing(self):
        ws = Workspace(self.tmp, "target.ply")
        ws.init()
        lfs_dir = ws.training_dir / "lfs"
        lfs_dir.mkdir(parents=True, exist_ok=True)

        # Write an interrupted partial (e.g. splat_614.ply) but NOT splat_30000.ply
        (lfs_dir / "splat_614.ply").touch()

        ctx = StageContext(
            manifest={
                "training": {"engine": "lichtfeld", "iterations": 30000, "lfsStrategy": "mrnf"},
                "processing": {},
                "output": {"plyFileName": "target.ply", "directory": self.tmp},
            },
            tools=Tools(),
            workspace=ws,
            emitter=Emitter(),
            cancel=CancelToken(),
        )

        # export_lfs must fail with StageFailed, and must NOT fall back to splat_614.ply
        with self.assertRaises(StageFailed) as cm:
            lichtfeld_engine.export_lfs(ctx)
        self.assertIn("Expected exported PLY splat_30000.ply missing", str(cm.exception))


class TestLevelSceneForLfs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="pf360_level_test_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make_tilted_rig(self, proc_dir: Path, tilt_deg: float = 45.0):
        import math
        proc_dir.mkdir(parents=True, exist_ok=True)
        rad = math.radians(tilt_deg)
        Rx = np.array([
            [1.0, 0.0, 0.0],
            [0.0, math.cos(rad), -math.sin(rad)],
            [0.0, math.sin(rad), math.cos(rad)],
        ], dtype=np.float64)
        offset = np.array([10.0, -5.0, 3.0], dtype=np.float64)

        frames = []
        # 2 panoramas, 4 horizontal views each (yaw 0, 90, 180, 270)
        for p_idx, cx in enumerate((0.0, 2.0)):
            center = np.array([cx, 0.0, 0.0], dtype=np.float64)
            for v_idx in range(4):
                yaw = math.radians(v_idx * 90.0)
                cy, sy = math.cos(yaw), math.sin(yaw)
                # Upright OpenGL c2w: up = (0, 0, 1)
                R_upright = np.array([
                    [cy, 0.0, sy],
                    [sy, 0.0, -cy],
                    [0.0, 1.0, 0.0],
                ], dtype=np.float64)
                # Swap cols 1 and 2 so col 1 = (0,0,1) up
                R_upright = np.array([
                    [cy, 0.0, sy],
                    [sy, 0.0, -cy],
                    [0.0, 1.0, 0.0],
                ], dtype=np.float64)
                # Actually col 1 is up = (0,0,1):
                R_upright = np.array([
                    [cy, 0.0, sy],
                    [sy, 0.0, -cy],
                    [0.0, 0.0, 0.0],
                ], dtype=np.float64)
                R_upright = np.array([
                    [cy, 0.0, sy],
                    [-sy, 0.0, cy],
                    [0.0, 1.0, 0.0],
                ], dtype=np.float64)
                # Build orthonormal basis with col1 = (0,0,1)
                right = np.array([cy, sy, 0.0], dtype=np.float64)
                up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
                back = np.cross(right, up)
                R_upright = np.column_stack([right, up, back])

                mat = np.eye(4, dtype=np.float64)
                mat[:3, :3] = Rx @ R_upright
                mat[:3, 3] = (Rx @ center) + offset
                frames.append({
                    "file_path": f"images/frame_{p_idx:06d}_view_{v_idx:02d}.jpg",
                    "transform_matrix": mat.tolist(),
                })

        # Seed PLY with 2 points along tilted up
        from pf360 import training_prep
        pts = [tuple((Rx @ np.array([0.0, 0.0, 1.0]) + offset).tolist()),
               tuple((Rx @ np.array([2.0, 0.0, 0.0]) + offset).tolist())]
        cols = [(255, 0, 0), (0, 255, 0)]
        training_prep.write_ply(proc_dir / "sparse_pc.ply", pts, cols)

        tf = {"camera_model": "OPENCV", "ply_file_path": "sparse_pc.ply", "frames": frames}
        import json as _json
        with open(proc_dir / "transforms.json", "w", encoding="utf-8") as f:
            _json.dump(tf, f)

    def test_synthetic_tilted_rig_is_levelled_and_idempotent(self):
        import json as _json
        proc = Path(self.tmp) / "proc"
        self._make_tilted_rig(proc, tilt_deg=45.0)

        info = lichtfeld_engine.level_scene_for_lfs(proc)
        self.assertTrue(info["levelled"])
        self.assertAlmostEqual(info["levelTiltDeg"], 45.0, delta=0.5)

        with open(proc / "transforms.json", "r", encoding="utf-8") as f:
            tf = _json.load(f)
        ups = np.array([np.array(fr["transform_matrix"])[:3, 1] for fr in tf["frames"]])
        mean_up = ups.mean(axis=0)
        mean_up /= np.linalg.norm(mean_up)
        err_deg = float(np.degrees(np.arccos(np.clip(mean_up[2], -1.0, 1.0))))
        self.assertLess(err_deg, 0.5)

        # Centred around origin
        centers = np.array([np.array(fr["transform_matrix"])[:3, 3] for fr in tf["frames"]])
        self.assertLess(float(np.linalg.norm(np.median(centers, axis=0))), 1e-6)

        # Seed PLY transformed and updated
        self.assertEqual(tf["ply_file_path"], "sparse_pc_levelled.ply")
        from pf360 import training_prep
        pts, cols = training_prep.read_ply(proc / "sparse_pc_levelled.ply")
        self.assertEqual(len(pts), 2)
        self.assertEqual(cols, [(255, 0, 0), (0, 255, 0)])

        # Idempotent second call
        info2 = lichtfeld_engine.level_scene_for_lfs(proc)
        self.assertEqual(info2, info)

    def test_processed_dir_byte_identical_when_levelling_to_lfs_input(self):
        import hashlib
        proc = Path(self.tmp) / "proc_ro"
        lfs_in = Path(self.tmp) / "training" / "lfs_input"
        self._make_tilted_rig(proc, tilt_deg=45.0)
        (proc / "images").mkdir(exist_ok=True)
        (proc / "images" / "frame_000000_view_00.jpg").write_bytes(b"jpg")

        before_tf = hashlib.sha256((proc / "transforms.json").read_bytes()).hexdigest()
        before_ply = hashlib.sha256((proc / "sparse_pc.ply").read_bytes()).hexdigest()

        info = lichtfeld_engine.level_scene_for_lfs(proc, lfs_input_dir=lfs_in)
        self.assertTrue(info["levelled"])
        self.assertAlmostEqual(info["levelTiltDeg"], 45.0, delta=0.5)

        # processed/ must be byte-for-byte identical
        self.assertEqual(hashlib.sha256((proc / "transforms.json").read_bytes()).hexdigest(), before_tf)
        self.assertEqual(hashlib.sha256((proc / "sparse_pc.ply").read_bytes()).hexdigest(), before_ply)
        self.assertFalse((proc / "sparse_pc_levelled.ply").exists())

        # training/lfs_input/ must have the levelled transforms + ply, NO images junction/reparse point
        import json as _json
        self.assertTrue((lfs_in / "transforms.json").is_file())
        self.assertTrue((lfs_in / "sparse_pc_levelled.ply").is_file())
        self.assertFalse((lfs_in / "images").exists())
        self.assertFalse((lfs_in / "images").is_symlink())
        with open(lfs_in / "transforms.json", "r", encoding="utf-8") as f:
            lfs_tf = _json.load(f)
        for fr in lfs_tf["frames"]:
            resolved = (lfs_in / fr["file_path"]).resolve()
            self.assertTrue(resolved.is_relative_to(proc.resolve()))

    def test_already_level_10s_clip_unchanged(self):
        src_tf = Path(r"C:\Users\ADMIN\Desktop\Video\PointForge360\_preview10s_ab\data\transforms.json")
        if not src_tf.is_file():
            self.skipTest("10s preview clip transforms.json not present")
        proc = Path(self.tmp) / "proc10s"
        proc.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_tf, proc / "transforms.json")
        info = lichtfeld_engine.level_scene_for_lfs(proc)
        self.assertTrue(info["levelled"])
        self.assertLess(info["levelTiltDeg"], 0.5)

    def test_nerfstudio_fingerprint_unchanged(self):
        from pf360.nerfstudio_backend import NerfstudioBackend
        from pf360.backend import StageContext, Tools, Emitter, CancelToken
        from pf360.workspace import Workspace
        ws = Workspace(Path(self.tmp) / "ws_ns", "out.ply")
        ctx = StageContext(
            manifest={
                "training": {
                    "engine": "nerfstudio", "method": "splatfacto", "iterations": 30000,
                    "downscaleFactor": 2, "shDegree": 3, "randomSeed": 42,
                    "useScaleRegularization": False, "cullAlphaThresh": None,
                    "densifyGradThresh": None, "cullScaleThresh": None,
                },
                "processing": {},
                "output": {"plyFileName": "out.ply", "directory": self.tmp},
            },
            tools=Tools(),
            workspace=ws,
            emitter=Emitter(),
            cancel=CancelToken(),
        )
        inputs = NerfstudioBackend().stage_inputs("training", ctx)
        self.assertNotIn("engine", inputs)
        self.assertEqual(inputs["method"], "splatfacto")


if __name__ == "__main__":
    unittest.main()

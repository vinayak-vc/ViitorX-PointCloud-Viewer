"""training_prep: seed-cloud thinning + misplaced-panorama filter (no Nerfstudio needed)."""
import json
import math
import random
import tempfile
import unittest
from pathlib import Path

from pf360 import manifest, training_prep as tp


def _mat(x, y, z):
    return [[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]]


class ThinTests(unittest.TestCase):
    def test_under_cap_keeps_everything(self):
        pts = [(float(i), 0.0, 0.0) for i in range(10)]
        self.assertEqual(tp.thin_indices(pts, 100), list(range(10)))
        self.assertEqual(tp.thin_indices(pts, 0), list(range(10)))          # 0 = no cap

    def test_cap_is_respected_and_coverage_kept(self):
        rnd = random.Random(1)
        dense = [(rnd.random() * 0.1, rnd.random() * 0.1, 0.0) for _ in range(20000)]   # one hot spot
        sparse = [(5 + rnd.random() * 5, 5 + rnd.random() * 5, 0.0) for _ in range(500)]
        keep = tp.thin_indices(dense + sparse, 2000)
        self.assertLessEqual(len(keep), 2000)
        self.assertGreater(len(keep), 1000)
        kept_sparse = sum(1 for i in keep if i >= len(dense))
        # Random sampling would keep ~2000 * 500/20500 = 49 of the sparse area; the grid keeps far more.
        self.assertGreater(kept_sparse, 300)


class PlyTests(unittest.TestCase):
    def test_ascii_and_binary_round_trip(self):
        d = Path(tempfile.mkdtemp())
        pts, cols = [(1.0, 2.0, 3.0), (-4.5, 0.25, 9.0)], [(1, 2, 3), (250, 0, 7)]
        tp.write_ply(d / "a.ply", pts, cols)
        self.assertEqual(tp.read_ply(d / "a.ply"), (pts, cols))
        import struct
        with open(d / "b.ply", "wb") as f:
            f.write(b"ply\nformat binary_little_endian 1.0\nelement vertex 2\nproperty float x\nproperty float y\n"
                    b"property float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
            for p, c in zip(pts, cols):
                f.write(struct.pack("<fffBBB", *p, *c))
        self.assertEqual(tp.read_ply(d / "b.ply"), (pts, cols))


def _workspace(n_pan=40, views=4, bad=(), spike=(), points=0):
    """processed/ + planar dir in the layout the camera stage leaves; panoramas walk along x."""
    root = Path(tempfile.mkdtemp())
    planar, processed = root / "planar", root / "processed"
    planar.mkdir()
    (processed / "images").mkdir(parents=True)
    names = sorted("frame_%05d_%d.jpg" % (k + 1, v) for k in range(n_pan) for v in range(views))
    frames = []
    for i, nm in enumerate(names):
        (planar / nm).write_bytes(b"")
        (processed / "images" / ("frame_%05d.jpg" % (i + 1))).write_bytes(b"")
        k, v = int(nm[6:11]) - 1, int(nm.rsplit("_", 1)[1][:-4])
        x, y = k * 0.1, 0.0
        if k in bad:
            x += v * 3.0                                  # views of this panorama scattered
        if k in spike:
            y += 5.0                                      # whole panorama thrown off the path
        frames.append({"file_path": "images/frame_%05d.jpg" % (i + 1), "transform_matrix": _mat(x, y, 0.0)})
    t = {"camera_model": "OPENCV", "frames": frames}
    if points:
        t["ply_file_path"] = "sparse_pc.ply"
        rnd = random.Random(3)
        tp.write_ply(processed / "sparse_pc.ply", [(rnd.random(), rnd.random(), rnd.random()) for _ in range(points)],
                     [(1, 2, 3)] * points)
    (processed / "transforms.json").write_text(json.dumps(t), encoding="utf-8")
    return processed, planar


class PrepareTests(unittest.TestCase):
    def test_drops_scattered_and_spiking_panoramas(self):
        processed, planar = _workspace(bad=(10, 11), spike=(25,))
        stats = tp.prepare(processed, planar, seed_cap=0)
        self.assertEqual(stats["droppedPanoramas"], 3)
        self.assertEqual(stats["droppedViews"], 12)
        t = json.loads((processed / "transforms.json").read_text(encoding="utf-8"))
        self.assertEqual(len(t["frames"]), 40 * 4 - 12)
        self.assertTrue((processed / tp.ALL_TRANSFORMS).is_file())

    def test_clean_walk_is_untouched_and_rerun_is_idempotent(self):
        processed, planar = _workspace()
        self.assertNotIn("droppedPanoramas", tp.prepare(processed, planar, seed_cap=0))
        processed, planar = _workspace(bad=(5,))
        a = tp.prepare(processed, planar, seed_cap=0)
        b = tp.prepare(processed, planar, seed_cap=0)       # resume: same answer, from transforms_all.json
        self.assertEqual(a, b)

    def test_mass_drop_is_refused(self):
        processed, planar = _workspace(bad=(5, 6, 7))
        old = tp.MAX_DROP_FRACTION
        tp.MAX_DROP_FRACTION = 0.05                         # 3 of 40 = 7.5% > 5%
        try:
            stats = tp.prepare(processed, planar, seed_cap=0)
        finally:
            tp.MAX_DROP_FRACTION = old
        self.assertNotIn("droppedPanoramas", stats)
        self.assertIn("would drop 3 of 40", stats["misplaced"]["skipped"])

    def test_half_scattered_walk_drops_nothing(self):
        # When most panoramas disagree, the "typical" panorama is no reference any more.
        processed, planar = _workspace(bad=tuple(range(0, 40, 2)))
        self.assertNotIn("droppedPanoramas", tp.prepare(processed, planar, seed_cap=0))

    def test_name_mismatch_skips_the_filter(self):
        processed, planar = _workspace(bad=(10,))
        next(planar.iterdir()).unlink()
        stats = tp.prepare(processed, planar, seed_cap=0)
        self.assertIn("do not match", stats["misplaced"]["skipped"])

    def test_seed_cloud_is_thinned_into_a_separate_file(self):
        processed, planar = _workspace(points=30000)
        stats = tp.prepare(processed, planar, seed_cap=10000)
        self.assertEqual(stats["seedPoints"], 30000)
        self.assertLessEqual(stats["seedPointsUsed"], 10000)
        t = json.loads((processed / "transforms.json").read_text(encoding="utf-8"))
        self.assertEqual(t["ply_file_path"], tp.TRAIN_PLY)
        self.assertEqual(len(tp.read_ply(processed / "sparse_pc.ply")[0]), 30000)   # original kept
        self.assertEqual(len(tp.read_ply(processed / tp.TRAIN_PLY)[0]), stats["seedPointsUsed"])


class ViewBudgetTests(unittest.TestCase):
    def test_budget_from_ram_and_resolution(self):
        t = {"w": 1088, "h": 1088, "frames": []}
        # The 24 Nov case: 127 GB, 1088^2 RGB = 3.55 MB per view, half of RAM -> ~19,200 views.
        self.assertEqual(tp.view_budget(t, 1, 127 * 2 ** 30), int(127 * 2 ** 30 * 0.5 // (1088 * 1088 * 3)))
        self.assertEqual(tp.view_budget(t, 2, 127 * 2 ** 30), int(127 * 2 ** 30 * 0.5 // (544 * 544 * 3)))
        self.assertEqual(tp.view_budget({"frames": []}, 1, 2 ** 30), 0)      # size unknown: no limit

    def test_whole_evenly_spaced_panoramas_are_kept(self):
        processed, planar = _workspace(n_pan=40, views=4, bad=(10,))
        stats = tp.prepare(processed, planar, seed_cap=0, max_views=60)
        self.assertEqual(stats["droppedPanoramas"], 1)
        self.assertEqual(stats["panoramasUsed"], 15)                       # 60 views / 4 per panorama
        t = json.loads((processed / "transforms.json").read_text(encoding="utf-8"))
        self.assertEqual(len(t["frames"]), 60)
        xs = sorted({round(f["transform_matrix"][0][3], 3) for f in t["frames"]})
        self.assertEqual(len(xs), 15)                                       # whole panoramas
        self.assertNotIn(1.0, xs)                                           # the dropped one stays out
        gaps = [b - a for a, b in zip(xs, xs[1:])]
        self.assertLessEqual(max(gaps), 0.41)                               # ~every 2.6th, +1 gap at the dropped one

    def test_auto_budget_uses_ram(self):
        processed, planar = _workspace(n_pan=40, views=4)
        t = json.loads((processed / "transforms.json").read_text(encoding="utf-8"))
        t.update(w=1000, h=1000)
        (processed / "transforms.json").write_text(json.dumps(t), encoding="utf-8")
        stats = tp.prepare(processed, planar, seed_cap=0, ram_bytes=80 * 3_000_000)   # 0.5 x 80 views
        self.assertEqual(stats["viewBudget"], 40)
        self.assertEqual(stats["trainingViews"], 40)


class ManifestTests(unittest.TestCase):
    def _m(self, **tr):
        d = tempfile.mkdtemp()
        v = Path(d) / "v.mp4"
        v.write_bytes(b"")
        raw = {"schemaVersion": 1, "jobId": "p1", "input": {"videoPath": str(v)},
               "processing": {"targetFrames": 4}, "training": tr, "output": {"directory": str(Path(d) / "ws")}}
        return manifest.resolve(raw, 10.0)

    def test_defaults_and_validation(self):
        tr = manifest.validate(self._m())["training"]
        self.assertEqual((tr["seedPointCap"], tr["dropMisplacedPanoramas"]), (500000, True))
        manifest.validate(self._m(seedPointCap=0))
        for bad in ({"seedPointCap": 5}, {"seedPointCap": True}, {"dropMisplacedPanoramas": "no"}):
            with self.assertRaises(manifest.ManifestError, msg=bad):
                manifest.validate(self._m(**bad))


class LongScheduleTests(unittest.TestCase):
    class _Em:
        def __init__(self):
            self.lines = []

        def log(self, msg, **kw):
            self.lines.append(msg)

        def warning(self, msg, **kw):
            self.lines.append("WARN " + msg)

    def _args(self, on, its, help_text="--pipeline.model.stop-split-at --optimizers.means.scheduler.max-steps",
              views=0):
        from types import SimpleNamespace
        from pf360.nerfstudio_backend import CliFlags, NerfstudioBackend
        ctx = SimpleNamespace(manifest={"training": {"longSchedule": on}}, emitter=self._Em())
        return NerfstudioBackend._long_schedule_args(ctx, CliFlags(help_text), its, views), ctx.emitter.lines

    def test_off_or_short_runs_keep_splatfacto_schedule(self):
        self.assertEqual(self._args(False, 150000)[0], [])
        self.assertEqual(self._args(True, 30000)[0], [])

    def test_long_run_stretches_split_and_lr_decay(self):
        args, _ = self._args(True, 150000)
        self.assertEqual(args, ["--pipeline.model.stop-split-at", "75000",
                                "--optimizers.means.scheduler.max-steps", "150000"])

    def test_many_views_stretch_the_opacity_reset(self):
        h = ("--pipeline.model.stop-split-at --optimizers.means.scheduler.max-steps "
             "--pipeline.model.reset-alpha-every")
        args, _ = self._args(True, 150000, h, views=19222)
        # 2 x 19,222 views = 385 refinements, capped so two cycles fit in the 75k split phase: 375.
        self.assertEqual(args[-2:], ["--pipeline.model.reset-alpha-every", "375"])
        args, _ = self._args(True, 400000, h, views=19222)
        self.assertEqual(args[-2:], ["--pipeline.model.reset-alpha-every", "385"])
        args, _ = self._args(True, 150000, h, views=800)              # small set: Splatfacto's 3000 steps
        self.assertNotIn("--pipeline.model.reset-alpha-every", args)

    def test_missing_flags_warn_instead_of_failing(self):
        args, lines = self._args(True, 150000, help_text="")
        self.assertEqual(args, [])
        self.assertTrue(lines[0].startswith("WARN"))

    def test_manifest_default_and_validation(self):
        tr = ManifestTests()._m()["training"]
        self.assertIs(tr["longSchedule"], False)
        with self.assertRaises(manifest.ManifestError):
            manifest.validate(ManifestTests()._m(longSchedule="yes"))


class TestLichtFeldNoRamBudget(unittest.TestCase):
    def _set_wh(self, processed: Path, w: int = 1000, h: int = 1000):
        tf = processed / "transforms.json"
        t = json.loads(tf.read_text(encoding="utf-8"))
        t["w"], t["h"] = w, h
        tf.write_text(json.dumps(t), encoding="utf-8")
        return t

    def test_no_ram_budget_keeps_all_good_views(self):
        processed, planar = _workspace(n_pan=40, views=4, bad=(10,))
        self._set_wh(processed)
        logs = []
        stats = tp.prepare(
            processed,
            planar,
            seed_cap=0,
            ram_bytes=120_000_000,
            apply_ram_budget=False,
            log=logs.append,
        )
        self.assertEqual(stats["trainingViews"], 156)
        self.assertEqual(stats["viewBudget"], "none (lichtfeld)")
        t = json.loads((processed / "transforms.json").read_text(encoding="utf-8"))
        self.assertEqual(len(t["frames"]), 156)
        self.assertTrue(any("no RAM view budget for LichtFeld" in line for line in logs))

    def test_no_ram_budget_still_respects_explicit_max_views(self):
        processed, planar = _workspace(n_pan=40, views=4, bad=(10,))
        self._set_wh(processed)
        stats = tp.prepare(
            processed,
            planar,
            seed_cap=0,
            max_views=20,
            ram_bytes=120_000_000,
            apply_ram_budget=False,
        )
        self.assertGreater(stats["trainingViews"], 0)
        self.assertLessEqual(stats["trainingViews"], 20)

    def test_ram_budget_enabled_caps_views_for_nerfstudio(self):
        processed, planar = _workspace(n_pan=40, views=4, bad=(10,))
        t = self._set_wh(processed)
        stats = tp.prepare(
            processed,
            planar,
            seed_cap=0,
            ram_bytes=120_000_000,
            apply_ram_budget=True,
        )
        budget = tp.view_budget(t, 1, 120_000_000)
        self.assertEqual(stats["viewBudget"], budget)
        self.assertLessEqual(stats["trainingViews"], budget)


if __name__ == "__main__":
    unittest.main()

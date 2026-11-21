"""Panorama rig completion + COLMAP model merging (ns_helpers/rig_complete.py), synthetic poses.

rig_complete.py runs inside the Nerfstudio environment and needs numpy; skipped without it."""
import json
import math
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
except ImportError:          # the worker itself is stdlib-only
    np = None

HELPERS = Path(__file__).resolve().parents[1] / "pf360" / "ns_helpers"
VIEWS = [[-180.0, 0.0], [-90.0, 0.0], [0.0, 0.0], [90.0, 0.0], [-180.0, 45.0], [0.0, 45.0],
         [-180.0, -45.0], [0.0, -45.0]]
PANOS = 40


@unittest.skipUnless(np is not None, "numpy not installed")
class RigMergeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(HELPERS))
        import rig_complete
        cls.rc = rig_complete

    def setUp(self):
        self.rng = np.random.default_rng(3)

    # ---------------------------------------------------------------- synthetic walk
    def truth(self):
        """{image_id: image} for a curving walk: PANOS panoramas x 8 views."""
        rc, imgs, iid = self.rc, {}, 1
        rots = [rc.view_rotation(y, p) for y, p in VIEWS]
        for k in range(PANOS):
            yaw = 0.03 * k
            centre = np.array([math.cos(yaw) * k * 0.5, math.sin(yaw) * k * 0.5, 0.02 * k])
            qa = rc.er_matrix([0.0, 0.0, 1.0], yaw)           # world -> panorama (a reflection, see rig_complete)
            q = np.diag([1.0, 1.0, -1.0]) @ qa
            for v in range(len(VIEWS)):
                rcw = rots[v].T @ q
                imgs[iid] = {"q": tuple(rc.rot_to_qvec(rcw)), "t": tuple(-rcw @ centre), "cam": 1,
                             "name": "frame_%05d_%d.jpg" % (k + 1, v), "npts": 0, "pts": b""}
                iid += 1
        return imgs

    def rand_sim3(self):
        q = self.rng.normal(size=4)
        return self.rc.qvec_to_rot(q / np.linalg.norm(q)), float(self.rng.uniform(0.3, 3.0)), self.rng.normal(size=3) * 4

    def piece(self, imgs, views, first, last, sim3=None):
        rc = self.rc
        r, s, t = sim3 or (np.eye(3), 1.0, np.zeros(3))
        out = {}
        for iid, im in imgs.items():
            head, v = rc.split_name(im["name"])
            if v in views and first <= int(head.split("_")[1]) <= last:
                out[iid] = rc.transform_image(im, r, s, t, 1)
        return out

    def compare(self, truth, out):
        """Max rotation (deg) / centre (fraction of walk length) error after one global alignment."""
        rc = self.rc
        gp, op = rc.model_panoramas(truth, VIEWS, 5), rc.model_panoramas(out, VIEWS, 5)
        r, s, t, _ = rc.estimate_sim3(gp, op, sorted(set(gp) & set(op)), 5)
        cent = lambda im: -rc.qvec_to_rot(im["q"]).T @ np.asarray(im["t"])
        ext = np.ptp(np.array([cent(im) for im in truth.values()]), axis=0).max()
        rot, pos = 0.0, 0.0
        for iid, im in out.items():
            a = rc.transform_image(im, r, s, t, 1)
            rot = max(rot, rc.angle_deg(rc.qvec_to_rot(a["q"]) @ rc.qvec_to_rot(truth[iid]["q"]).T))
            pos = max(pos, float(np.linalg.norm(cent(a) - cent(truth[iid]))) / ext)
        return rot, pos

    # ---------------------------------------------------------------- tests
    def test_direction_pieces_are_joined_and_completed(self):
        """COLMAP's typical real-footage split: forward / backward / side views of the SAME stretch."""
        rc, truth = self.rc, self.truth()
        models = [("0", self.piece(truth, {0, 1}, 1, 25), {}),
                  ("1", self.piece(truth, {2, 5}, 10, 40, self.rand_sim3()), {}),
                  ("2", self.piece(truth, {3, 7}, 20, 40, self.rand_sim3()), {}),
                  ("3", self.piece(truth, {4}, 30, 40, self.rand_sim3()), {})]
        imgs, _, st = rc.merge(models, VIEWS, 5.0)
        self.assertEqual(st["modelsMerged"], 4, st)
        self.assertEqual(st["unmergedModels"], [])
        name_to_id = {im["name"]: iid for iid, im in truth.items()}
        done, cst = rc.complete(imgs, VIEWS, name_to_id, 5.0)
        self.assertEqual(len(done), len(truth))
        self.assertEqual(cst["panoramasCompleted"], PANOS)
        rot, pos = self.compare(truth, done)
        self.assertLess(rot, 0.01)
        self.assertLess(pos, 1e-4)

    def test_disjoint_or_too_few_shared_panoramas_are_left_out(self):
        rc, truth = self.rc, self.truth()
        models = [("0", self.piece(truth, {0, 1, 2}, 1, 20), {}),
                  ("1", self.piece(truth, {3}, 18, 30, self.rand_sim3()), {}),     # shares 3 with 0
                  ("2", self.piece(truth, {5}, 28, 40, self.rand_sim3()), {})]     # shares 3 with 1 only
        _, _, st = rc.merge(models, VIEWS, 5.0)
        self.assertEqual(st["referenceModel"], "0")
        self.assertEqual(st["modelsMerged"], 3, st)                 # 2 joins through 1 (chained)
        models[1] = ("1", self.piece(truth, {3}, 19, 30, self.rand_sim3()), {})   # now shares only 2
        _, _, st = rc.merge(models, VIEWS, 5.0)
        self.assertEqual(st["modelsMerged"], 1, st)
        self.assertEqual(sorted(st["unmergedModels"]), ["1", "2"])

    def test_a_wrong_model_is_rejected(self):
        """Shared panoramas whose centres do not fit one similarity (a bad registration) must not join."""
        rc, truth = self.rc, self.truth()
        bad = self.piece(truth, {4, 5}, 1, 20, self.rand_sim3())
        for im in bad.values():                       # scramble the centres, keep rotations
            t = np.asarray(im["t"]) + self.rng.normal(size=3) * 10
            im["t"] = tuple(t)
        for iid in list(bad):                         # make the two views of a panorama agree again
            head, v = rc.split_name(bad[iid]["name"])
            if v == 5:
                twin = next(i for i, im in bad.items() if im["name"] == "%s_4.jpg" % head)
                c = -rc.qvec_to_rot(bad[twin]["q"]).T @ np.asarray(bad[twin]["t"])
                bad[iid]["t"] = tuple(-rc.qvec_to_rot(bad[iid]["q"]) @ c)
        models = [("0", self.piece(truth, {0, 1, 2}, 1, 20), {}), ("1", bad, {})]
        _, _, st = rc.merge(models, VIEWS, 5.0)
        self.assertEqual(st["unmergedModels"], ["1"])

    def test_cli_round_trip(self):
        """--sparse reads COLMAP binary models, writes one joined + completed model."""
        rc, truth = self.rc, self.truth()
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            for k, (views, a, b, sim) in enumerate([({0, 1}, 1, 25, None), ({2, 5}, 10, 40, self.rand_sim3())]):
                m = d / "sparse" / str(k)
                m.mkdir(parents=True)
                (m / "cameras.bin").write_bytes(b"cam")
                rc.write_images_bin(m / "images.bin", self.piece(truth, views, a, b, sim))
                rc.write_points3d_bin(m / "points3D.bin", [(np.array([1.0, 2.0, 3.0]), b"\x01\x02\x03", 0.5)])
            (d / "sparse" / "0_rig").mkdir()                     # not a mapper model: ignored
            con = sqlite3.connect(d / "db.db")
            con.execute("CREATE TABLE images (image_id INTEGER, name TEXT)")
            con.executemany("INSERT INTO images VALUES (?, ?)", [(i, im["name"]) for i, im in truth.items()])
            con.commit()
            con.close()
            self.assertEqual(rc.main(["--sparse", str(d / "sparse"), "--db", str(d / "db.db"),
                                      "--views", json.dumps(VIEWS), "--out", str(d / "out")]), 0)
            out = rc.read_images_bin(d / "out" / "images.bin")
            self.assertEqual(len(out), len(truth))
            self.assertEqual(len(rc.read_points3d_bin(d / "out" / "points3D.bin")), 2)
            self.assertEqual((d / "out" / "cameras.bin").read_bytes(), b"cam")
            rot, pos = self.compare(truth, out)
            self.assertLess(rot, 0.01)
            self.assertLess(pos, 1e-4)


if __name__ == "__main__":
    unittest.main()

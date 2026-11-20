import math
import os
import tempfile
import unittest
from unittest import mock

from pf360 import ply_validate
from tests import helpers


class PlyValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.p = os.path.join(self.tmp, "s.ply")

    def test_valid_file(self):
        helpers.write_ply(self.p, n=500)
        r = ply_validate.validate(self.p)
        self.assertTrue(r["ok"], r["errors"])
        self.assertEqual(r["gaussianCount"], 500)
        self.assertEqual(r["shDegree"], 3)

    def test_valid_file_without_numpy_is_sampled(self):
        helpers.write_ply(self.p, n=50)
        with mock.patch.dict("sys.modules", {"numpy": None}):
            r = ply_validate.validate(self.p)
        self.assertTrue(r["ok"], r["errors"])
        self.assertIn("sampled", r["numericCheck"])

    def test_missing_file_and_empty(self):
        self.assertFalse(ply_validate.validate(self.p)["ok"])
        open(self.p, "wb").close()
        self.assertIn("file is empty", ply_validate.validate(self.p)["errors"])

    def test_invalid_header(self):
        with open(self.p, "wb") as f:
            f.write(b"PLY?\nformat ascii\n")
        self.assertFalse(ply_validate.validate(self.p)["ok"])

    def test_missing_attributes(self):
        helpers.write_ply(self.p, props=["x", "y", "z", "opacity"])
        r = ply_validate.validate(self.p)
        self.assertFalse(r["ok"])
        self.assertTrue(any("missing Gaussian attributes" in e for e in r["errors"]))

    def test_ascii_and_uchar_rejected_for_splatreader(self):
        helpers.write_ply(self.p, fmt="ascii")
        self.assertFalse(ply_validate.validate(self.p)["ok"])
        helpers.write_ply(self.p, ptype="uchar", values=lambda i, props: [1] * len(props))
        self.assertTrue(any("unsupported property types" in e for e in ply_validate.validate(self.p)["errors"]))

    def test_zero_count(self):
        helpers.write_ply(self.p, n=0)
        self.assertIn("Gaussian count is zero", ply_validate.validate(self.p)["errors"])

    def test_nan_values(self):
        def vals(i, props):
            return [math.nan if (i == 7 and p == "opacity") else (1.0 if p == "rot_0" else float(i)) for p in props]
        helpers.write_ply(self.p, n=20, values=vals)
        for np_block in (False, True):
            ctx = mock.patch.dict("sys.modules", {"numpy": None}) if np_block else mock.patch.dict("sys.modules", {})
            with ctx:
                r = ply_validate.validate(self.p)
            self.assertFalse(r["ok"])
            self.assertTrue(any("NaN" in e for e in r["errors"]))

    def test_truncated(self):
        helpers.write_ply(self.p, n=10, truncate=8)
        self.assertTrue(any("truncated" in e for e in ply_validate.validate(self.p)["errors"]))

    def test_degenerate_positions(self):
        helpers.write_ply(self.p, n=10, values=lambda i, props: [1.0 if p == "rot_0" else 0.0 for p in props])
        self.assertTrue(any("degenerate" in e for e in ply_validate.validate(self.p)["errors"]))


if __name__ == "__main__":
    unittest.main()

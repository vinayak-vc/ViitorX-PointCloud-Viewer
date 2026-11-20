"""Masking settings: manifest defaults/validation, backend switches and cache keys.
No GPU, no model: the segmentation itself runs in the Nerfstudio env."""
import copy
import os
import tempfile
import unittest
from pathlib import Path

from pf360 import environment, manifest
from pf360.backend import StageContext
from pf360.environment import Tools
from pf360.nerfstudio_backend import NerfstudioBackend, _masking_on
from pf360.protocol import STAGES

RAW = {"schemaVersion": 1, "jobId": "m_001",
       "input": {"videoPath": "C:/x/clip.mp4"}, "training": {"qualityPreset": "preview"},
       "output": {"directory": "C:/x/out"}, "processing": {"targetFrames": 30}}


def resolved(masking=None):
    raw = copy.deepcopy(RAW)
    if masking is not None:
        raw["processing"]["masking"] = masking
    return manifest.resolve(raw, 30.0)


class ManifestMaskingTest(unittest.TestCase):
    def test_default_is_off(self):
        m = manifest.validate(resolved(), check_files=False)
        self.assertEqual(m["processing"]["masking"], {"enabled": False, "groups": ["people"], "dilatePx": 8})
        self.assertFalse(_masking_on(m))

    def test_valid_settings(self):
        m = manifest.validate(resolved({"enabled": True, "groups": ["people", "sky"], "dilatePx": 0}),
                              check_files=False)
        self.assertTrue(_masking_on(m))

    def test_rejects_bad_values(self):
        for bad, needle in (({"enabled": "yes"}, "enabled"),
                            ({"enabled": True, "groups": ["people", "trees"]}, "groups"),
                            ({"enabled": True, "groups": ["sky", "sky"]}, "groups"),
                            ({"enabled": True, "groups": []}, "must not be empty"),
                            ({"enabled": True, "groups": ["people"], "dilatePx": 65}, "dilatePx"),
                            ({"enabled": True, "groups": ["people"], "dilatePx": 2.5}, "dilatePx"),
                            ("on", "must be an object")):
            with self.assertRaises(manifest.ManifestError, msg=str(bad)) as cm:
                manifest.validate(resolved(bad), check_files=False)
            self.assertTrue(any(needle in p for p in cm.exception.problems), (bad, cm.exception.problems))

    def test_stop_after_masking_allowed(self):
        m = resolved()
        m["options"]["stopAfter"] = "masking"
        manifest.validate(m, check_files=False)

    def test_mask_model_dir_must_exist(self):
        m = resolved()
        m["environment"]["maskModelDir"] = "C:/definitely/not/here"
        with self.assertRaises(manifest.ManifestError):
            manifest.validate(m, check_files=True)


class BackendMaskingTest(unittest.TestCase):
    def ctx(self, m):
        return StageContext(manifest=m, tools=Tools(), workspace=None, emitter=None, cancel=None,
                            video_info={}, results={})

    def test_stage_order_and_protocol(self):
        be = NerfstudioBackend()
        self.assertEqual(list(be.stages()), ["panorama_projection", "masking", "camera_reconstruction",
                                             "training", "export"])
        self.assertEqual(STAGES.index("masking"), STAGES.index("panorama_projection") + 1)

    def test_enabled_switch(self):
        be = NerfstudioBackend()
        self.assertFalse(be.stage_enabled("masking", self.ctx(resolved())))
        on = resolved({"enabled": True, "groups": ["people"]})
        self.assertTrue(be.stage_enabled("masking", self.ctx(on)))
        self.assertTrue(be.stage_enabled("training", self.ctx(resolved())))

    def test_environment_gate(self):
        be = NerfstudioBackend()
        on = resolved({"enabled": True, "groups": ["people"]})
        probs = be.check_environment(Tools(), on)
        self.assertTrue(any("setup_masking.ps1" in p for p in probs))
        self.assertFalse(any("setup_masking" in p for p in be.check_environment(Tools(), resolved())))

    def test_cache_keys(self):
        be = NerfstudioBackend()
        a = be.stage_inputs("masking", self.ctx(resolved({"enabled": True, "groups": ["sky", "people"]})))
        b = be.stage_inputs("masking", self.ctx(resolved({"enabled": True, "groups": ["people", "sky"]})))
        self.assertEqual(a, b)                      # order-insensitive
        c = be.stage_inputs("masking", self.ctx(resolved({"enabled": True, "groups": ["people"], "dilatePx": 3})))
        self.assertNotEqual(a, c)
        off = be.stage_inputs("camera_reconstruction", self.ctx(resolved()))
        on = be.stage_inputs("camera_reconstruction", self.ctx(resolved({"enabled": True, "groups": ["people"]})))
        self.assertNotEqual(off, on)                # masks change the COLMAP result


class MaskModelDiscoveryTest(unittest.TestCase):
    def test_model_dir_needs_config_and_safetensors(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(environment._model_dir(d))
            Path(d, "config.json").write_text("{}")
            self.assertIsNone(environment._model_dir(d))
            Path(d, "pytorch_model.bin").write_bytes(b"x")       # pickle weights are not accepted
            self.assertIsNone(environment._model_dir(d))
            Path(d, "model.safetensors").write_bytes(b"x")
            self.assertEqual(environment._model_dir(d), d)
            old = os.environ.get("PF360_MASK_MODEL")
            os.environ["PF360_MASK_MODEL"] = d
            try:
                self.assertEqual(environment.resolve_tools({}).mask_model, d)
                self.assertEqual(environment.resolve_tools({"maskModelDir": d}).mask_model, d)
            finally:
                if old is None:
                    del os.environ["PF360_MASK_MODEL"]
                else:
                    os.environ["PF360_MASK_MODEL"] = old


if __name__ == "__main__":
    unittest.main()

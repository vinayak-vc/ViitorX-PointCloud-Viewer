"""Engine packs: worker discovery of the downloadable runtime/FFmpeg packs, and the
pack tooling's signing (scripts/pf360/engine_packs.py; skipped without `cryptography`)."""
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pf360 import environment

REPO = Path(__file__).resolve().parents[3]
PACKS_PY = REPO / "scripts" / "pf360" / "engine_packs.py"


def load_engine_packs():
    spec = importlib.util.spec_from_file_location("engine_packs", PACKS_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


try:
    import cryptography  # noqa: F401
    HAVE_CRYPTO = True
except ImportError:
    HAVE_CRYPTO = False


class PackEnvTest(unittest.TestCase):
    def test_self_managed_env_is_left_alone(self):
        with tempfile.TemporaryDirectory() as d:
            py = Path(d, "Scripts", "python.exe")
            py.parent.mkdir()
            py.write_bytes(b"")
            self.assertEqual(environment.pack_env(str(py)), {})
            self.assertEqual(environment.pack_env(None), {})

    def test_runtime_pack_is_isolated_and_uses_bundled_weights(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "python.exe").write_bytes(b"")
            Path(d, "PF360_PACK.json").write_text("{}")
            env = environment.pack_env(str(Path(d, "python.exe")))
            self.assertEqual(env, {"PYTHONNOUSERSITE": "1"})
            Path(d, "torch-home").mkdir()
            env = environment.pack_env(str(Path(d, "python.exe")))
            self.assertEqual(env["TORCH_HOME"], str(Path(d, "torch-home")))

    def test_pack_python_is_discovered_at_env_root(self):
        with tempfile.TemporaryDirectory() as d:
            env_dir = Path(d, "pf360-env")
            env_dir.mkdir()
            Path(env_dir, "python.exe").write_bytes(b"")
            self.assertEqual(environment._venv_python(env_dir), str(env_dir / "python.exe"))

    def test_ffmpeg_pack_preferred_over_path(self):
        with tempfile.TemporaryDirectory() as d:
            exe = Path(d, "ViitorX", "PointForge", "engines", "ffmpeg", "bin", "ffmpeg.exe")
            exe.parent.mkdir(parents=True)
            exe.write_bytes(b"")
            Path(exe.parent, "ffprobe.exe").write_bytes(b"")
            with mock.patch.dict(os.environ, {"LOCALAPPDATA": d}), \
                    mock.patch.object(environment.shutil, "which", return_value="C:/elsewhere/ffmpeg.exe"):
                os.environ.pop("PF360_FFMPEG", None)
                os.environ.pop("PF360_FFPROBE", None)
                t = environment.resolve_tools({})
                self.assertEqual(Path(t.ffmpeg), exe)
                # an explicit setting still wins
                t = environment.resolve_tools({"ffmpegExe": str(exe)})
                self.assertEqual(Path(t.ffmpeg), exe)


@unittest.skipUnless(HAVE_CRYPTO, "cryptography not installed (pf360 env has it)")
class SigningTest(unittest.TestCase):
    def test_sign_verify_roundtrip_and_tamper(self):
        ep = load_engine_packs()
        _, _, ec, _, _ = ep._crypto()
        key = ec.generate_private_key(ec.SECP256R1())
        pub = ep.public_hex(key.public_key())
        self.assertEqual(len(pub), 128)
        data = b'{"schema": 1}'
        sig = ep.sign_bytes(key, data)
        self.assertTrue(ep.verify_bytes(pub, data, sig))
        self.assertFalse(ep.verify_bytes(pub, data + b" ", sig))
        other = ep.public_hex(ec.generate_private_key(ec.SECP256R1()).public_key())
        self.assertFalse(ep.verify_bytes(other, data, sig))

    def test_catalog_shape(self):
        ep = load_engine_packs()
        packs = {pid: {"version": "1", "installDir": d, "postInstall": pi,
                       "files": [{"name": pid + ".zip", "kind": "zip", "size": 10, "sha256": "a" * 64,
                                  "installedBytes": 20, "fileCount": 3}]}
                 for pid, d, pi in (("pf360-runtime", "pf360-env", "python-relocate"),
                                    ("colmap", "colmap-conda", ""), ("ffmpeg", "ffmpeg", ""))}
        fake_mask = {"id": "mask-model", "version": "abc", "installDir": "models/m", "postInstall": "",
                     "files": [{"name": "config.json", "kind": "file", "size": 5, "sha256": "b" * 64,
                                "url": "https://huggingface.co/x/resolve/r/config.json"}]}
        with mock.patch.object(ep, "mask_model_pack", return_value=fake_mask):
            cat = ep.make_catalog(packs, "https://huggingface.co/org/repo/resolve/0123abc")
        self.assertEqual(cat["schema"], 1)
        self.assertGreater(cat["sequence"], 0)
        ids = [p["id"] for p in cat["packs"]]
        self.assertEqual(ids, ["pf360-runtime", "colmap", "ffmpeg", "mask-model"])
        rt = cat["packs"][0]
        self.assertEqual(rt["files"][0]["url"], "https://huggingface.co/org/repo/resolve/0123abc/pf360-runtime.zip")
        self.assertEqual(rt["installedBytes"], 20)
        self.assertEqual(rt["files"][0]["fileCount"], 3)
        self.assertTrue(rt["required"])
        self.assertFalse(cat["packs"][3]["required"])
        json.dumps(cat)

    def test_adding_optional_packs_keeps_the_published_ones(self):
        ep = load_engine_packs()
        live = {"schema": 1, "packs": [
            {"id": pid, "version": "old", "required": pid != "mask-model",
             "files": [{"name": pid + ".zip", "url": "https://huggingface.co/o/r/resolve/OLD/" + pid + ".zip"}]}
            for pid in ("pf360-runtime", "colmap", "ffmpeg", "mask-model")]}
        new = {"vocab-tree": {"version": "256K", "installDir": "models/colmap-vocab-tree", "postInstall": "",
                              "files": [{"name": "t.bin", "kind": "file", "size": 9, "sha256": "c" * 64}]}}
        cat = ep.make_catalog(new, "https://huggingface.co/o/r/resolve/NEW", keep=live)
        ids = [p["id"] for p in cat["packs"]]
        self.assertEqual(ids, ["pf360-runtime", "colmap", "ffmpeg", "mask-model", "vocab-tree"])   # glomap not built
        self.assertTrue(cat["packs"][1]["files"][0]["url"].endswith("/OLD/colmap.zip"))
        vt = cat["packs"][-1]
        self.assertFalse(vt["required"])
        self.assertEqual(vt["files"][0]["url"], "https://huggingface.co/o/r/resolve/NEW/t.bin")

    def test_lichtfeld_pack_includes_compliance_and_catalog_entry(self):
        import zipfile
        ep = load_engine_packs()
        with tempfile.TemporaryDirectory() as d:
            lfs_dir = Path(d, "dist-lichtfeld")
            out_dir = Path(d, "out")
            lfs_dir.mkdir()
            out_dir.mkdir()
            (lfs_dir / "LichtFeld-Studio.exe").write_bytes(b"MZ")
            (lfs_dir / "LICENSE").write_text("GPL-3.0", encoding="utf-8")
            (lfs_dir / "LICENSE-EXCEPTION.txt").write_text("NVIDIA Exception", encoding="utf-8")
            (lfs_dir / "THIRD-PARTY-LICENSES.md").write_text("# Notices", encoding="utf-8")
            rec = ep.build_lichtfeld(lfs_dir, out_dir, version="8c366f5")
            self.assertEqual(rec["id"], "lichtfeld")
            self.assertEqual(rec["installDir"], "lichtfeld")
            zip_path = out_dir / "lichtfeld.zip"
            self.assertTrue(zip_path.is_file())
            with zipfile.ZipFile(zip_path) as zf:
                names = set(zf.namelist())
            self.assertIn("lichtfeld/LichtFeld-Studio.exe", names)
            self.assertIn("lichtfeld/LICENSE", names)
            self.assertIn("lichtfeld/LICENSE-EXCEPTION.txt", names)
            self.assertIn("lichtfeld/THIRD-PARTY-LICENSES.md", names)
            self.assertIn("lichtfeld/SOURCE_OFFER.txt", names)
            live = {"schema": 1, "packs": [
                {"id": pid, "version": "old", "required": True,
                 "files": [{"name": pid + ".zip", "url": "https://huggingface.co/o/r/resolve/OLD/" + pid + ".zip"}]}
                for pid in ("pf360-runtime", "colmap", "ffmpeg")]}
            cat = ep.make_catalog({"lichtfeld": rec}, "https://huggingface.co/o/r/resolve/NEW", keep=live)
            lfs_entry = [p for p in cat["packs"] if p["id"] == "lichtfeld"][0]
            self.assertFalse(lfs_entry["required"])
            self.assertEqual(lfs_entry["installDir"], "lichtfeld")
            self.assertIn("GPL-3.0", lfs_entry["license"])


if __name__ == "__main__":
    unittest.main()

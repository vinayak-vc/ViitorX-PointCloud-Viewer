"""Build, sign and publish PointForge "engine packs" (downloadable add-ons).

PointForge's viewer ships small (~35 MB). The 360 Reconstruction engines
(Python + PyTorch + Nerfstudio, COLMAP, FFmpeg, the masking model; ~11 GB
installed) are downloaded on demand by the in-app installer
(src/viewer/EnginePacks.cpp) from a SIGNED catalog hosted on Hugging Face.

Developer workflow (run on the machine where the engines already work):

  1. One-time: create the signing key (keep the private key OFF the repo and
     backed up; anyone holding it can make every PointForge install run code):
       python engine_packs.py keygen --private-key %USERPROFILE%\\.pointforge\\engine-signing.pem
     This also writes the PUBLIC key into src/viewer/EnginePackKey.h -> rebuild.

  2. Build the packs from the installed engines (~5 GB of zips):
       python engine_packs.py build --out C:\\PF360Test\\engine-packs

  3. Upload packs + signed catalog to your Hugging Face repo (log in first with
     `huggingface-cli login`; the token never passes through this script):
       python engine_packs.py publish --repo your-org/pointforge-engines --packs-dir C:\\PF360Test\\engine-packs ^
           --private-key %USERPROFILE%\\.pointforge\\engine-signing.pem

  Other commands: `catalog` (write catalog.json for a base URL without
  uploading), `sign`, `verify`.

Run with the pf360 env's Python (needs `cryptography`; `publish` also needs
`huggingface_hub`), e.g. %LOCALAPPDATA%\\ViitorX\\PointForge\\engines\\pf360-env\\Scripts\\python.exe.

Catalog format (schema 1) is documented in docs/360-reconstruction.md
("Engine packs"). The signature is ECDSA P-256 / SHA-256 over the exact bytes
of catalog.json, stored base64 (raw r||s, 64 bytes) in catalog.json.sig.
"""
import argparse
import base64
import datetime as _dt
import getpass
import hashlib
import json
import os
import re
import sys
import time
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
COMPAT = REPO / "tools" / "pf360" / "compat.json"
KEY_HEADER = REPO / "src" / "viewer" / "EnginePackKey.h"
RELOCATE = Path(__file__).resolve().parent / "pack_relocate.py"

CATALOG_SCHEMA = 1
# Already-compressed payloads are stored, not deflated (saves minutes, costs ~nothing).
STORE_EXT = {".pth", ".pt", ".safetensors", ".zip", ".gz", ".whl", ".png", ".jpg", ".jpeg", ".mp4", ".bz2",
             ".xz", ".7z", ".zst", ".npz"}


def engines_root() -> Path:
    return Path(os.environ.get("LOCALAPPDATA", ".")) / "ViitorX" / "PointForge" / "engines"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def log(msg: str) -> None:
    print(msg, flush=True)


# ----------------------------------------------------------------------------- build
class ZipBuilder:
    def __init__(self, path: Path):
        self.path = path
        self.zf = zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True)
        self.raw = 0
        self.count = 0
        self.names = set()
        self._last = time.time()

    def add(self, src: Path, arc: str) -> None:
        arc = arc.replace("\\", "/")
        if arc in self.names:
            raise SystemExit("duplicate path in pack: " + arc)
        self.names.add(arc)
        comp = zipfile.ZIP_STORED if src.suffix.lower() in STORE_EXT else zipfile.ZIP_DEFLATED
        self.zf.write(src, arc, compress_type=comp)
        self.raw += src.stat().st_size
        self.count += 1
        if time.time() - self._last > 10:
            self._last = time.time()
            log("  ... %d files, %.2f GB read" % (self.count, self.raw / 1e9))

    def add_tree(self, src: Path, arc_prefix: str, skip_dir=lambda rel: False, skip_file=lambda rel: False) -> None:
        for dirpath, dirnames, filenames in os.walk(src):
            rel_dir = Path(dirpath).relative_to(src)
            keep = []
            for d in dirnames:
                rel = (rel_dir / d).as_posix()
                if d == "__pycache__" or skip_dir(rel):
                    continue
                keep.append(d)
            dirnames[:] = keep
            for fn in filenames:
                rel = (rel_dir / fn).as_posix()
                if fn.endswith((".pyc", ".pyo")) or skip_file(rel):
                    continue
                self.add(Path(dirpath) / fn, arc_prefix + "/" + rel)

    def close(self) -> dict:
        self.zf.close()
        return {"name": self.path.name, "size": self.path.stat().st_size, "sha256": sha256_file(self.path),
                "kind": "zip", "installedBytes": self.raw, "fileCount": self.count}


def venv_base(env: Path) -> Path:
    cfg = (env / "pyvenv.cfg").read_text(encoding="utf-8")
    m = re.search(r"^home\s*=\s*(.+)$", cfg, re.M)
    if not m:
        raise SystemExit("pyvenv.cfg has no 'home' line: " + str(env))
    return Path(m.group(1).strip())


def build_runtime(engines: Path, out: Path) -> dict:
    """pf360-env as a RELOCATABLE runtime: the venv's base CPython install + the
    venv's site-packages merged into one folder (a venv itself points at the
    packaging machine's Python and cannot be moved)."""
    env = engines / "pf360-env"
    base = venv_base(env)
    if not (base / "python.exe").is_file():
        raise SystemExit("base interpreter not found: %s" % base)
    ver = (env / "pyvenv.cfg").read_text(encoding="utf-8")
    pyver = re.search(r"^version\s*=\s*(.+)$", ver, re.M).group(1).strip()
    sp = env / "Lib" / "site-packages"
    dist = {p.name.split("-")[0].lower(): p.name for p in sp.glob("*.dist-info")}

    def dist_version(name: str) -> str:
        n = dist.get(name.lower(), "")
        m = re.match(r"[^-]+-(.+)\.dist-info$", n)
        return m.group(1) if m else "?"

    version = "py%s-torch%s-ns%s-gsplat%s" % (pyver, dist_version("torch"), dist_version("nerfstudio"),
                                              dist_version("gsplat"))
    alexnet = Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "alexnet-owt-7be5be79.pth"
    if not alexnet.is_file():
        raise SystemExit("LPIPS backbone weights missing (%s). Run one Splatfacto training once so torchvision "
                         "caches them, then rebuild." % alexnet)
    pth_abs = [p.name for p in sp.glob("*.pth") if re.search(r"[A-Za-z]:\\\\|[A-Za-z]:/", p.read_text(errors="ignore"))]
    if pth_abs:
        raise SystemExit("site-packages .pth files contain absolute paths (not relocatable): %s" % pth_abs)

    zb = ZipBuilder(out / "pf360-env.zip")
    root = "pf360-env"
    log("pf360-runtime: base CPython %s from %s" % (pyver, base))
    # Whitelist of the CPython install itself. Anything else at its root (etc/, share/, Scripts/ ...) is
    # data from packages installed into the BASE interpreter, which the venv never used.
    for item in sorted(base.iterdir()):
        if item.is_file() and not item.name.lower().endswith(".pdb"):
            zb.add(item, root + "/" + item.name)
    zb.add_tree(base / "DLLs", root + "/DLLs", skip_file=lambda rel: rel.lower().endswith(".pdb"))
    if (base / "tcl").is_dir():
        zb.add_tree(base / "tcl", root + "/tcl")
    lib_skip = {"site-packages", "test", "idlelib", "turtledemo", "ensurepip"}
    zb.add_tree(base / "Lib", root + "/Lib", skip_dir=lambda rel: rel in lib_skip)
    log("pf360-runtime: venv site-packages")
    zb.add_tree(sp, root + "/Lib/site-packages")
    # Files pip placed at the venv root (MSVC runtime DLLs shipped by wheels) and its data dirs.
    for item in env.iterdir():
        if item.name in ("Lib", "Scripts", "Include", "pyvenv.cfg") or item.name == "__pycache__":
            continue
        if item.is_dir():
            zb.add_tree(item, root + "/" + item.name)
        elif (root + "/" + item.name) not in zb.names:
            zb.add(item, root + "/" + item.name)
    zb.add(alexnet, root + "/torch-home/hub/checkpoints/" + alexnet.name)
    zb.add(RELOCATE, root + "/pf360_relocate.py")
    info = {"id": "pf360-runtime", "version": version, "installDir": "pf360-env", "builtFrom": "venv+base"}
    zb.zf.writestr(root + "/PF360_PACK.json", json.dumps(info, indent=1))
    rec = zb.close()
    log("pf360-runtime: %d files, %.2f GB -> %.2f GB" % (rec["fileCount"], rec["installedBytes"] / 1e9,
                                                        rec["size"] / 1e9))
    return {"id": "pf360-runtime", "version": version, "installDir": "pf360-env", "postInstall": "python-relocate",
            "files": [rec]}


def build_colmap(engines: Path, out: Path) -> dict:
    env = engines / "colmap-conda"
    exe = env / "Library" / "bin" / "colmap.exe"
    if not exe.is_file():
        raise SystemExit("COLMAP conda env not found: " + str(env))
    meta = sorted(env.glob("conda-meta/colmap-*.json"))
    version = json.loads(meta[-1].read_text())["version"] if meta else "?"
    skip_dirs = {"pkgs", "include", "libs", "Library/include", "Library/cmake", "Library/mkspecs", "Library/doc",
                 "Library/share/doc", "Library/qml", "Library/translations", "Library/phrasebooks"}

    def skip_file(rel: str) -> bool:
        low = rel.lower()
        # import / static libraries are build-time only
        return (low.startswith("library/lib/") and low.endswith((".lib", ".a"))) or low.endswith(".pdb")

    zb = ZipBuilder(out / "colmap-conda.zip")
    zb.add_tree(env, "colmap-conda", skip_dir=lambda rel: rel in skip_dirs, skip_file=skip_file)
    rec = zb.close()
    log("colmap: %d files, %.2f GB -> %.2f GB" % (rec["fileCount"], rec["installedBytes"] / 1e9, rec["size"] / 1e9))
    return {"id": "colmap", "version": version, "installDir": "colmap-conda", "postInstall": "", "files": [rec]}


def build_ffmpeg(ffmpeg_dir: Path, out: Path) -> dict:
    if not (ffmpeg_dir / "bin" / "ffmpeg.exe").is_file():
        raise SystemExit("ffmpeg folder must contain bin/ffmpeg.exe: " + str(ffmpeg_dir))
    m = re.search(r"ffmpeg-([0-9][^\\/-]*)", str(ffmpeg_dir))
    version = m.group(1) if m else "?"
    zb = ZipBuilder(out / "ffmpeg.zip")
    for f in sorted((ffmpeg_dir / "bin").iterdir()):
        if f.is_file() and f.name.lower() != "ffplay.exe":
            zb.add(f, "ffmpeg/bin/" + f.name)
    for f in ("LICENSE", "README.txt"):
        if (ffmpeg_dir / f).is_file():
            zb.add(ffmpeg_dir / f, "ffmpeg/" + f)
    rec = zb.close()
    log("ffmpeg: %d files -> %.2f GB" % (rec["fileCount"], rec["size"] / 1e9))
    return {"id": "ffmpeg", "version": version, "installDir": "ffmpeg", "postInstall": "", "files": [rec]}


def find_ffmpeg_dir() -> Path:
    import shutil
    exe = shutil.which("ffmpeg")
    if not exe:
        raise SystemExit("ffmpeg not on PATH; pass --ffmpeg-dir")
    return Path(exe).resolve().parent.parent


def cmd_build(a) -> None:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    engines = Path(a.engines) if a.engines else engines_root()
    only = set(a.only.split(",")) if a.only else {"pf360-runtime", "colmap", "ffmpeg"}
    manifest_path = out / "packs.json"
    packs = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    t0 = time.time()
    builders = [("ffmpeg", lambda: build_ffmpeg(Path(a.ffmpeg_dir) if a.ffmpeg_dir else find_ffmpeg_dir(), out)),
                ("colmap", lambda: build_colmap(engines, out)),
                ("pf360-runtime", lambda: build_runtime(engines, out))]
    for pid, fn in builders:
        if pid in only:
            packs[pid] = fn()
            manifest_path.write_text(json.dumps(packs, indent=1))   # keep finished packs if a later one fails
    log("built %s in %.0f s -> %s" % (", ".join(sorted(only)), time.time() - t0, manifest_path))


# ----------------------------------------------------------------------------- catalog
PACK_TEXT = {
    "pf360-runtime": ("Splatfacto training engine",
                      "Python 3.10, PyTorch (CUDA 11.8), Nerfstudio + gsplat. Trains the Gaussian splat on the GPU.",
                      "Nerfstudio Apache-2.0; PyTorch BSD-3; gsplat Apache-2.0; NVIDIA CUDA runtime libraries under "
                      "the NVIDIA CUDA EULA (redistributable components); Python PSF.", True),
    "colmap": ("COLMAP camera solver", "Finds camera positions in the extracted views (CUDA build, conda-forge).",
               "COLMAP BSD-3; bundled libraries under their own licences (see conda-meta).", True),
    "ffmpeg": ("FFmpeg", "Reads the 360 video and extracts frames.",
               "FFmpeg GPL-3.0 build (gyan.dev). Run as a separate program; the GPL source offer applies to these "
               "binaries.", True),
    "mask-model": ("Masking model (optional)",
                   "UperNet-ConvNeXt segmentation model. Hides people, vehicles, animals or sky during "
                   "reconstruction.", "MIT (openmmlab/upernet-convnext-base)", False),
}


def mask_model_pack() -> dict:
    mk = json.loads(COMPAT.read_text(encoding="utf-8"))["masking"]
    model_dir = engines_root() / "models" / mk["modelDirName"]
    files = []
    for name in mk["files"]:
        p = model_dir / name
        if not p.is_file():
            raise SystemExit("masking model file missing (run scripts/pf360/setup_masking.ps1 first): %s" % p)
        files.append({"name": name, "kind": "file", "size": p.stat().st_size, "sha256": sha256_file(p),
                      # straight from the model's own Hugging Face repo at the pinned revision
                      "url": "https://huggingface.co/%s/resolve/%s/%s" % (mk["model"], mk["revision"], name)})
    return {"id": "mask-model", "version": mk["revision"][:12], "installDir": "models/" + mk["modelDirName"],
            "postInstall": "", "files": files}


def make_catalog(packs: dict, base_url: str) -> dict:
    out = []
    for pid in ("pf360-runtime", "colmap", "ffmpeg", "mask-model"):
        p = mask_model_pack() if pid == "mask-model" else packs.get(pid)
        if not p:
            raise SystemExit("pack not built: " + pid)
        title, desc, lic, required = PACK_TEXT[pid]
        files = []
        installed = 0
        for f in p["files"]:
            url = f.get("url") or (base_url.rstrip("/") + "/" + f["name"])
            entry = {"name": f["name"], "kind": f["kind"], "size": f["size"], "sha256": f["sha256"], "url": url}
            if f.get("fileCount"):
                entry["fileCount"] = f["fileCount"]       # unpack progress only
            files.append(entry)
            installed += f.get("installedBytes", f["size"])
        out.append({"id": pid, "title": title, "description": desc, "version": p["version"], "group": "recon360",
                    "required": required, "installDir": p["installDir"], "postInstall": p["postInstall"],
                    "license": lic, "installedBytes": installed, "files": files})
    now = _dt.datetime.now(_dt.timezone.utc)
    return {"schema": CATALOG_SCHEMA, "name": "PointForge engines", "created": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            # monotonic: the app refuses a catalog older than the newest one it accepted (rollback protection)
            "sequence": int(now.timestamp()), "packs": out}


def cmd_catalog(a) -> None:
    packs = json.loads((Path(a.packs_dir) / "packs.json").read_text())
    cat = make_catalog(packs, a.base_url)
    Path(a.out).write_text(json.dumps(cat, indent=1), encoding="utf-8")
    log("wrote " + a.out)


# ----------------------------------------------------------------------------- keys / signing
def _crypto():
    try:
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature, encode_dss_signature
    except ImportError:
        raise SystemExit("needs the 'cryptography' package (present in the pf360 env)")
    return hashes, serialization, ec, decode_dss_signature, encode_dss_signature


def _passphrase(confirm: bool) -> bytes:
    env = os.environ.get("PF_ENGINE_KEY_PASSPHRASE")
    if env:
        return env.encode()
    p = getpass.getpass("Signing key passphrase: ")
    if confirm and getpass.getpass("Repeat passphrase: ") != p:
        raise SystemExit("passphrases differ")
    if len(p) < 12:
        raise SystemExit("use a passphrase of at least 12 characters")
    return p.encode()


def public_hex(pub) -> str:
    n = pub.public_numbers()
    return "%064x%064x" % (n.x, n.y)


def write_key_header(hexkey: str, path: Path) -> None:
    path.write_text(
        "#pragma once\n"
        "// PUBLIC half of the engine-pack catalog signing key (ECDSA P-256, X||Y hex).\n"
        "// Generated by scripts/pf360/engine_packs.py keygen. The private key never\n"
        "// enters the repository. Empty = this build cannot install engine packs.\n"
        "namespace pf {\n"
        "inline constexpr const char* kEnginePackPublicKeyHex =\n"
        "    \"%s\";\n"
        "}\n" % hexkey, encoding="utf-8")


def cmd_keygen(a) -> None:
    hashes, serialization, ec, _, _ = _crypto()
    path = Path(a.private_key)
    if path.exists():
        raise SystemExit("refusing to overwrite an existing key: %s" % path)
    key = ec.generate_private_key(ec.SECP256R1())
    enc = (serialization.NoEncryption() if a.no_passphrase
           else serialization.BestAvailableEncryption(_passphrase(confirm=True)))
    pem = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, enc)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(pem)
    hexkey = public_hex(key.public_key())
    log("private key: %s  (back it up; never commit it)" % path)
    log("public key : %s" % hexkey)
    if a.header:
        write_key_header(hexkey, Path(a.header))
        log("wrote %s - rebuild PointForge to embed it" % a.header)


def load_private(path: str, no_passphrase: bool):
    _, serialization, _, _, _ = _crypto()
    data = Path(path).read_bytes()
    pw = None
    if b"ENCRYPTED" in data:
        pw = _passphrase(confirm=False) if not no_passphrase else None
    return serialization.load_pem_private_key(data, password=pw)


def sign_bytes(key, data: bytes) -> str:
    hashes, _, ec, decode_dss_signature, _ = _crypto()
    r, s = decode_dss_signature(key.sign(data, ec.ECDSA(hashes.SHA256())))
    return base64.b64encode(r.to_bytes(32, "big") + s.to_bytes(32, "big")).decode()


def verify_bytes(pub_hex: str, data: bytes, sig_b64: str) -> bool:
    hashes, _, ec, _, encode_dss_signature = _crypto()
    from cryptography.exceptions import InvalidSignature
    raw = base64.b64decode(sig_b64.strip())
    if len(raw) != 64 or len(pub_hex) != 128:
        return False
    pub = ec.EllipticCurvePublicNumbers(int(pub_hex[:64], 16), int(pub_hex[64:], 16), ec.SECP256R1()).public_key()
    try:
        pub.verify(encode_dss_signature(int.from_bytes(raw[:32], "big"), int.from_bytes(raw[32:], "big")), data,
                   ec.ECDSA(hashes.SHA256()))
        return True
    except InvalidSignature:
        return False


def cmd_sign(a) -> None:
    key = load_private(a.private_key, a.no_passphrase)
    data = Path(a.catalog).read_bytes()
    sig = sign_bytes(key, data)
    Path(a.catalog + ".sig").write_text(sig + "\n", encoding="ascii")
    log("signed %s (public key %s)" % (a.catalog, public_hex(key.public_key())))


def cmd_verify(a) -> None:
    ok = verify_bytes(a.public_key_hex, Path(a.catalog).read_bytes(), Path(a.catalog + ".sig").read_text())
    log("signature " + ("OK" if ok else "INVALID"))
    sys.exit(0 if ok else 1)


# ----------------------------------------------------------------------------- publish
def cmd_publish(a) -> None:
    try:
        from huggingface_hub import HfApi
    except ImportError:
        raise SystemExit("needs huggingface_hub (present in the pf360 env)")
    api = HfApi()
    who = api.whoami()          # fails clearly when not logged in (huggingface-cli login)
    log("Hugging Face user: %s" % who.get("name"))
    packs_dir = Path(a.packs_dir)
    packs = json.loads((packs_dir / "packs.json").read_text())
    key = load_private(a.private_key, a.no_passphrase)
    api.create_repo(a.repo, repo_type=a.repo_type, private=False, exist_ok=True)
    names = [f["name"] for p in packs.values() for f in p["files"]]
    for n in names:
        if not (packs_dir / n).is_file():
            raise SystemExit("missing " + n)
    log("uploading %d pack files (resumable; large files go through Hugging Face's LFS/Xet storage)" % len(names))
    commit = api.upload_folder(repo_id=a.repo, repo_type=a.repo_type, folder_path=str(packs_dir),
                               allow_patterns=names, commit_message="engine packs")
    rev = commit.oid
    prefix = "https://huggingface.co/" + ("datasets/" if a.repo_type == "dataset" else "") + a.repo
    cat = make_catalog(packs, "%s/resolve/%s" % (prefix, rev))   # pack URLs pinned to this commit
    data = json.dumps(cat, indent=1).encode("utf-8")
    sig = sign_bytes(key, data)
    (packs_dir / "catalog.json").write_bytes(data)
    (packs_dir / "catalog.json.sig").write_text(sig + "\n", encoding="ascii")
    api.upload_file(path_or_fileobj=data, path_in_repo="catalog.json", repo_id=a.repo, repo_type=a.repo_type,
                    commit_message="catalog %d" % cat["sequence"])
    api.upload_file(path_or_fileobj=(sig + "\n").encode(), path_in_repo="catalog.json.sig", repo_id=a.repo,
                    repo_type=a.repo_type, commit_message="catalog signature %d" % cat["sequence"])
    log("catalog URL for PointForge: %s/resolve/main/catalog.json" % prefix)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="zip the installed engines into relocatable packs")
    b.add_argument("--out", required=True)
    b.add_argument("--engines", help="engines root (default %%LOCALAPPDATA%%\\ViitorX\\PointForge\\engines)")
    b.add_argument("--ffmpeg-dir", help="FFmpeg folder containing bin\\ffmpeg.exe (default: from PATH)")
    b.add_argument("--only", help="comma list: pf360-runtime,colmap,ffmpeg")
    k = sub.add_parser("keygen", help="create the catalog signing key")
    k.add_argument("--private-key", required=True)
    k.add_argument("--header", default=str(KEY_HEADER), help="C++ header to write the public key into")
    k.add_argument("--no-header", dest="header", action="store_const", const="",
                   help="only print the public key (test keys)")
    k.add_argument("--no-passphrase", action="store_true", help="unencrypted key (tests only)")
    c = sub.add_parser("catalog", help="write catalog.json for packs hosted under --base-url")
    c.add_argument("--packs-dir", required=True)
    c.add_argument("--base-url", required=True)
    c.add_argument("--out", required=True)
    s = sub.add_parser("sign", help="write <catalog>.sig")
    s.add_argument("--catalog", required=True)
    s.add_argument("--private-key", required=True)
    s.add_argument("--no-passphrase", action="store_true")
    v = sub.add_parser("verify")
    v.add_argument("--catalog", required=True)
    v.add_argument("--public-key-hex", required=True)
    p = sub.add_parser("publish", help="upload packs + signed catalog to Hugging Face")
    p.add_argument("--repo", required=True, help="e.g. your-org/pointforge-engines")
    p.add_argument("--repo-type", default="model", choices=["model", "dataset"])
    p.add_argument("--packs-dir", required=True)
    p.add_argument("--private-key", required=True)
    p.add_argument("--no-passphrase", action="store_true")
    a = ap.parse_args(argv)
    {"build": cmd_build, "keygen": cmd_keygen, "catalog": cmd_catalog, "sign": cmd_sign, "verify": cmd_verify,
     "publish": cmd_publish}[a.cmd](a)


if __name__ == "__main__":
    main()

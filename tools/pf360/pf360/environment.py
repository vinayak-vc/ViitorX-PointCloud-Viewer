"""Reconstruction environment discovery + `doctor` diagnostic.

Resolution order for every tool (first hit wins):
  1. explicit path in the manifest `environment` block / CLI flag
  2. environment variable (PF360_NS_PYTHON, PF360_COLMAP, PF360_FFMPEG, PF360_FFPROBE)
  3. the isolated env created by scripts/pf360/setup_environment.ps1
     (%LOCALAPPDATA%/ViitorX/PointForge/engines/pf360-env)
  4. PATH
  5. (COLMAP only) the PointForge photogrammetry engine install
     (%LOCALAPPDATA%/ViitorX/PointForge/engines/colmap/**/colmap.exe)

Nothing here installs anything or changes global state.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import procrun

EXE = ".exe" if os.name == "nt" else ""
NS_TOOLS = ("ns-process-data", "ns-train", "ns-export")


def engines_root() -> Path:
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / "ViitorX" / "PointForge" / "engines"


def default_env_dir() -> Path:
    return engines_root() / "pf360-env"


def _file(p: Optional[str]) -> Optional[str]:
    return p if p and os.path.isfile(p) else None


def _venv_python(env_dir: Path) -> Optional[str]:
    for c in (env_dir / "Scripts" / ("python" + EXE), env_dir / ("python" + EXE), env_dir / "bin" / "python"):
        if c.is_file():
            return str(c)
    return None


def ns_bin_dir(python_exe: str) -> Optional[Path]:
    """Folder holding ns-* console scripts for a given interpreter (venv or conda layout)."""
    py = Path(python_exe)
    for d in (py.parent, py.parent / "Scripts", py.parent / "bin"):
        if (d / ("ns-train" + EXE)).is_file():
            return d
    return None


def find_colmap_in_engines() -> Optional[str]:
    # colmap-conda: COLMAP 3.11.x from conda-forge (CLI options compatible with
    # Nerfstudio 1.1.5); colmap: the photogrammetry engine's official release.
    for sub in ("colmap-conda", "colmap"):
        root = engines_root() / sub
        if not root.is_dir():
            continue
        for p in root.rglob("colmap" + EXE):
            if p.is_file():
                return str(p)
    return None


_COMPILER_ENV: Optional[Dict[str, str]] = None


def compiler_env() -> Dict[str, str]:
    """Environment additions so gsplat can JIT-compile its CUDA kernels (Windows).

    gsplat 1.4 builds its extension with torch.utils.cpp_extension on first use,
    which needs MSVC `cl` on PATH and CUDA_HOME. PointForge is usually started
    from Explorer, not a developer prompt, so import vcvars64 for the child
    processes only. CUDA 11.8 nvcc rejects MSVC >= 19.40, so a 14.3x toolset is
    preferred. Returns {} when nothing is needed or nothing usable is found.
    Result is cached for the worker's lifetime."""
    global _COMPILER_ENV
    if _COMPILER_ENV is not None:
        return _COMPILER_ENV
    add: Dict[str, str] = {}
    if os.name == "nt":
        if not os.environ.get("CUDA_HOME"):
            cuda = os.environ.get("CUDA_PATH_V11_8") or os.environ.get("CUDA_PATH")
            if cuda and Path(cuda, "bin", "nvcc.exe").is_file():
                add["CUDA_HOME"] = cuda
        if not shutil.which("cl"):
            add.update(_vcvars_env())
    # torch.utils.cpp_extension builds with one job per CPU by default; each gsplat
    # nvcc/cicc job can take several GB, so a 16 GB machine thrashes for hours.
    # Budget ~4 GB of RAM per job unless the user set MAX_JOBS themselves.
    if not os.environ.get("MAX_JOBS"):
        ram_gb = total_ram_gb()
        cpus = os.cpu_count() or 2
        if ram_gb:
            add["MAX_JOBS"] = str(max(1, min(cpus, int(ram_gb // 4))))
    _COMPILER_ENV = add
    return add


def total_ram_gb() -> Optional[float]:
    """Installed physical memory in GB (Windows via GlobalMemoryStatusEx, POSIX via sysconf)."""
    try:
        if os.name == "nt":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            st = MEMORYSTATUSEX()
            st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
                return st.ullTotalPhys / (1024 ** 3)
            return None
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024 ** 3)
    except Exception:
        return None


def _vcvars_env() -> Dict[str, str]:
    pf86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    vswhere = Path(pf86, "Microsoft Visual Studio", "Installer", "vswhere.exe")
    roots: List[str] = []
    if vswhere.is_file():
        try:
            cp = procrun.capture([str(vswhere), "-all", "-products", "*", "-requires",
                                  "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                                  "-property", "installationPath"], timeout=30)
            roots = [l.strip() for l in _decode(cp.stdout).splitlines() if l.strip()]
        except Exception:
            roots = []
    for root in roots:
        vcvars = Path(root, "VC", "Auxiliary", "Build", "vcvars64.bat")
        if not vcvars.is_file():
            continue
        ver_arg = ""
        msvc = Path(root, "VC", "Tools", "MSVC")
        try:
            vers = sorted((tuple(int(x) for x in d.name.split(".")) for d in msvc.iterdir()
                           if d.is_dir() and re.match(r"^\d+\.\d+\.\d+$", d.name)), reverse=True)
        except OSError:
            vers = []
        ok = [v for v in vers if v[1] < 40]
        if ok:
            ver_arg = " -vcvars_ver=%d.%d" % (ok[0][0], ok[0][1])
        try:
            # cmd /c "<vcvars> [-vcvars_ver=..] >nul && set": capture the resulting env.
            # Passed as a raw command line: list2cmdline would backslash-escape the
            # inner quotes, which cmd.exe does not understand.
            import subprocess
            cp = subprocess.run('cmd.exe /d /s /c ""%s"%s >nul 2>&1 && set"' % (vcvars, ver_arg),
                                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                timeout=120, check=False,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except Exception:
            continue
        if cp.returncode != 0:
            continue
        env: Dict[str, str] = {}
        for line in _decode(cp.stdout).splitlines():
            k, sep, v = line.partition("=")
            if sep and k and not k.startswith("="):
                env[k] = v
        path = env.get("PATH") or env.get("Path")
        if path and any(Path(p, "cl.exe").is_file() for p in path.split(os.pathsep) if p):
            keys = ("PATH", "INCLUDE", "LIB", "LIBPATH", "VCToolsInstallDir", "VCINSTALLDIR",
                    "VSINSTALLDIR", "WindowsSdkDir", "WindowsSDKVersion", "UCRTVersion",
                    "UniversalCRTSdkDir", "VCToolsVersion", "VSCMD_ARG_TGT_ARCH", "Platform")
            out = {k: env[k] for k in keys if k in env}
            if "PATH" not in out and "Path" in env:
                out["PATH"] = env["Path"]
            return out
    return {}


def _decode(b) -> str:
    if b is None:
        return ""
    return b if isinstance(b, str) else b.decode("utf-8", "replace")


def colmap_cli_compatible(colmap_exe: str) -> Optional[bool]:
    """True if `feature_extractor` still accepts --SiftExtraction.use_gpu (what
    Nerfstudio 1.1.5's colmap_utils passes), False if not, None if unknown."""
    try:
        cp = procrun.capture([colmap_exe, "feature_extractor", "-h"], timeout=60,
                             env=procrun.child_env(colmap_path_dirs(colmap_exe)))
    except Exception:
        return None
    text = _decode(cp.stdout) + _decode(cp.stderr)
    if "SiftExtraction.use_gpu" in text:
        return True
    if "use_gpu" in text:
        return False
    return None


def colmap_path_dirs(colmap_exe: str) -> List[str]:
    """COLMAP Windows release needs its bin/ and lib/ dirs on PATH (what COLMAP.bat does)."""
    exe_dir = Path(colmap_exe).parent
    dirs = [str(exe_dir)]
    lib = exe_dir.parent / "lib"
    if lib.is_dir():
        dirs.append(str(lib))
    return dirs


@dataclass
class Tools:
    ns_python: Optional[str] = None
    ns_bin: Optional[str] = None
    colmap: Optional[str] = None
    ffmpeg: Optional[str] = None
    ffprobe: Optional[str] = None
    mask_model: Optional[str] = None   # local segmentation model dir (optional masking add-on)
    glomap: Optional[str] = None       # optional GLOMAP global mapper (engine pack "glomap")
    vocab_tree: Optional[str] = None   # optional COLMAP vocabulary tree (engine pack "vocab-tree")
    lichtfeld: Optional[str] = None    # optional LichtFeld-Studio training engine (engines/lichtfeld)

    def ns(self, tool: str) -> Optional[str]:
        if not self.ns_bin:
            return None
        return _file(str(Path(self.ns_bin) / (tool + EXE)))

    def path_prepend(self) -> List[str]:
        """PATH entries every Nerfstudio child needs (it shells out to `colmap` and `ffmpeg`)."""
        dirs: List[str] = []
        if self.ns_bin:
            dirs.append(self.ns_bin)
        if self.colmap:
            dirs += colmap_path_dirs(self.colmap)
        if self.ffmpeg:
            dirs.append(str(Path(self.ffmpeg).parent))
        return dirs

    def ns_env(self) -> Dict[str, str]:
        """Extra environment for processes run with the Nerfstudio interpreter."""
        return pack_env(self.ns_python)


def pack_env(ns_python: Optional[str]) -> Dict[str, str]:
    """When the Nerfstudio interpreter is PointForge's downloadable runtime pack
    (PF360_PACK.json beside python.exe), isolate it from the machine: no user
    site-packages from a Python the user installed themselves, and torchvision's
    LPIPS backbone weights come from the pack (otherwise Splatfacto downloads them
    from download.pytorch.org on first use). {} for a self-managed environment."""
    if not ns_python:
        return {}
    root = Path(ns_python).parent
    if root.name.lower() == "scripts":
        root = root.parent
    if not (root / "PF360_PACK.json").is_file():
        return {}
    add = {"PYTHONNOUSERSITE": "1"}
    if (root / "torch-home").is_dir():
        add["TORCH_HOME"] = str(root / "torch-home")
    return add


def pack_ffmpeg() -> Optional[str]:
    """ffmpeg.exe from the downloadable FFmpeg engine pack (engines/ffmpeg/bin)."""
    return _file(str(engines_root() / "ffmpeg" / "bin" / ("ffmpeg" + EXE)))


def resolve_tools(cfg: Optional[Dict[str, Any]] = None) -> Tools:
    cfg = cfg or {}
    t = Tools()
    t.ns_python = (_file(cfg.get("nerfstudioPython")) or _file(os.environ.get("PF360_NS_PYTHON"))
                   or _venv_python(default_env_dir()))
    if not t.ns_python:
        ns_train = shutil.which("ns-train")
        if ns_train:
            d = Path(ns_train).parent
            t.ns_python = _venv_python(d) or _venv_python(d.parent)
    if t.ns_python:
        d = ns_bin_dir(t.ns_python)
        t.ns_bin = str(d) if d else None
    t.colmap = (_file(cfg.get("colmapExe")) or _file(os.environ.get("PF360_COLMAP"))
                or _file(str(default_env_dir() / "colmap" / ("colmap" + EXE))) or shutil.which("colmap")
                or find_colmap_in_engines())
    t.ffmpeg = (_file(cfg.get("ffmpegExe")) or _file(os.environ.get("PF360_FFMPEG")) or pack_ffmpeg()
                or shutil.which("ffmpeg"))
    t.ffprobe = _file(cfg.get("ffprobeExe")) or _file(os.environ.get("PF360_FFPROBE")) or shutil.which("ffprobe")
    if not t.ffprobe and t.ffmpeg:
        t.ffprobe = _file(str(Path(t.ffmpeg).with_name("ffprobe" + EXE)))
    t.mask_model = (_model_dir(cfg.get("maskModelDir")) or _model_dir(os.environ.get("PF360_MASK_MODEL"))
                    or _model_dir(str(engines_root() / "models" / MASK_MODEL_DIRNAME)))
    t.glomap = (_file(cfg.get("glomapExe")) or _file(os.environ.get("PF360_GLOMAP"))
                or _file(str(engines_root() / GLOMAP_DIRNAME / "Library" / "bin" / ("glomap" + EXE)))
                or shutil.which("glomap"))
    t.vocab_tree = (_file(cfg.get("vocabTreePath")) or _file(os.environ.get("PF360_VOCAB_TREE"))
                    or _file(str(engines_root() / "models" / VOCAB_TREE_DIRNAME / VOCAB_TREE_FILE)))
    t.lichtfeld = (_file(cfg.get("lichtfeldExe")) or _file(os.environ.get("PF360_LICHTFELD"))
                   or _file(str(engines_root() / "lichtfeld" / ("LichtFeld-Studio" + EXE)))
                   or shutil.which("LichtFeld-Studio"))
    return t


# Optional engine packs (scripts/pf360/engine_packs.py): GLOMAP 1.2 (conda-forge, CUDA) and
# COLMAP's 256K-word vocabulary tree (the size COLMAP recommends for 10k-100k images; the FLANN
# format COLMAP 3.11 reads - 3.12+ switched to a different format).
GLOMAP_DIRNAME = "glomap-conda"
VOCAB_TREE_DIRNAME = "colmap-vocab-tree"
VOCAB_TREE_FILE = "vocab_tree_flickr100K_words256K.bin"


def glomap_path_dirs(glomap_exe: str) -> List[str]:
    """DLL directories of a conda GLOMAP (its own CUDA / Ceres libraries)."""
    return colmap_path_dirs(glomap_exe)


# Installed by scripts/pf360/setup_masking.ps1 (see compat.json "masking").
MASK_MODEL_DIRNAME = "upernet-convnext-base-ade20k"


def _model_dir(p: Optional[str]) -> Optional[str]:
    """A local segmentation model directory: config.json + safetensors weights."""
    if not p:
        return None
    d = Path(p)
    if (d / "config.json").is_file() and any(d.glob("*.safetensors")):
        return str(d)
    return None


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------

@dataclass
class Check:
    name: str
    status: str            # PASS | WARN | FAIL | SKIP
    detail: str = ""
    fix: str = ""


@dataclass
class Report:
    checks: List[Check] = field(default_factory=list)
    gpu: Dict[str, Any] = field(default_factory=dict)
    torch: Dict[str, Any] = field(default_factory=dict)
    methods: List[str] = field(default_factory=list)
    tools: Dict[str, Optional[str]] = field(default_factory=dict)
    nerfstudio_version: Optional[str] = None
    masking: bool = False   # optional masking add-on usable
    glomap: bool = False    # optional GLOMAP global mapper usable
    vocab_tree: bool = False  # optional vocabulary tree present (loop detection, vocab_tree matching)

    @property
    def ready(self) -> bool:
        return all(c.status != "FAIL" for c in self.checks)

    @property
    def preprocess_ready(self) -> bool:
        need = {"FFmpeg", "FFprobe", "COLMAP", "Nerfstudio", "PyTorch", "CUDA"}
        return all(c.status != "FAIL" for c in self.checks if c.name in need)

    def to_json(self) -> Dict[str, Any]:
        d = asdict(self)
        d["ready"] = self.ready
        d["overall"] = "READY" if self.ready else "NOT READY"
        return d


TORCH_PROBE = r"""
import json, sys
out = {"python": sys.version.split()[0]}
try:
    import torch
    out["torch"] = torch.__version__
    out["cudaBuild"] = torch.version.cuda
    out["cuda"] = bool(torch.cuda.is_available())
    if out["cuda"]:
        i = torch.cuda.current_device()
        p = torch.cuda.get_device_properties(i)
        out["device"] = p.name
        out["totalMB"] = int(p.total_memory // (1024 * 1024))
        try:
            free, total = torch.cuda.mem_get_info(i)
            out["freeMB"] = int(free // (1024 * 1024))
        except Exception:
            pass
except Exception as e:
    out["torchError"] = repr(e)
try:
    import importlib.metadata as md
    out["nerfstudio"] = md.version("nerfstudio")
except Exception:
    pass
for mod in ("gsplat", "tinycudann", "transformers"):
    try:
        m = __import__(mod)
        out[mod] = getattr(m, "__version__", "present")
    except Exception as e:
        out[mod + "Error"] = type(e).__name__
print("PF360JSON" + json.dumps(out))
"""


def query_nvidia_smi() -> Dict[str, Any]:
    smi = shutil.which("nvidia-smi")
    if not smi:
        return {}
    try:
        cp = procrun.capture([smi, "--query-gpu=name,memory.total,memory.free,driver_version,compute_cap",
                              "--format=csv,noheader,nounits"], timeout=20)
    except Exception:
        return {}
    if cp.returncode != 0:
        return {}
    gpus = []
    for line in cp.stdout.decode("utf-8", "replace").splitlines():
        parts = [s.strip() for s in line.split(",")]
        if len(parts) >= 4:
            try:
                g = {"name": parts[0], "totalMB": int(float(parts[1])), "freeMB": int(float(parts[2])),
                     "driver": parts[3]}
                if len(parts) >= 5:
                    g["compute_cap"] = parts[4]
                gpus.append(g)
            except ValueError:
                continue
    return {"gpus": gpus} if gpus else {}


def query_torch(ns_python: str) -> Dict[str, Any]:
    try:
        cp = procrun.capture([ns_python, "-c", TORCH_PROBE], timeout=180,
                             env=procrun.child_env(extra=pack_env(ns_python)))
    except Exception as e:
        return {"probeError": repr(e)}
    text = cp.stdout.decode("utf-8", "replace")
    for line in text.splitlines():
        if line.startswith("PF360JSON"):
            try:
                return json.loads(line[len("PF360JSON"):])
            except ValueError:
                break
    return {"probeError": (cp.stderr.decode("utf-8", "replace") or text)[-400:]}


def list_methods(tools: Tools) -> List[str]:
    """Splatfacto variants the installed Nerfstudio actually registers (parsed from ns-train --help)."""
    ns_train = tools.ns("ns-train")
    if not ns_train:
        return []
    try:
        cp = procrun.capture([ns_train, "--help"], timeout=180,
                             env=procrun.child_env(tools.path_prepend(), tools.ns_env()))
    except Exception:
        return []
    text = cp.stdout.decode("utf-8", "replace") + cp.stderr.decode("utf-8", "replace")
    found = []
    for name in ("splatfacto-big", "splatfacto-w", "splatfacto-mcmc", "splatfacto"):
        # crude token match on the subcommand list; "splatfacto" must not match only as a prefix
        for tok in text.replace("{", " ").replace("}", " ").replace(",", " ").split():
            if tok.strip("│|") == name:
                found.append(name)
                break
    return found


INSTALL_HINT = ("Use 'Install engines' in the 360 wizard (Environment > Engines) to download the engine packs, "
                "or run scripts\\pf360\\setup_environment.ps1 (developer setup), or point PointForge at an existing "
                "Nerfstudio Python in Edit > Preferences > 360 Reconstruction.")


def doctor(cfg: Optional[Dict[str, Any]] = None, deep: bool = True) -> Report:
    t = resolve_tools(cfg)
    r = Report(tools={"nerfstudioPython": t.ns_python, "nerfstudioBin": t.ns_bin, "colmap": t.colmap,
                      "ffmpeg": t.ffmpeg, "ffprobe": t.ffprobe, "maskModel": t.mask_model,
                      "glomap": t.glomap, "vocabTree": t.vocab_tree, "lichtfeld": t.lichtfeld})
    r.checks.append(Check("Worker Python", "PASS", "%s (%s)" % (sys.version.split()[0], sys.executable)))
    r.checks.append(Check("FFmpeg", "PASS" if t.ffmpeg else "FAIL", t.ffmpeg or "not found",
                          "" if t.ffmpeg else "winget install Gyan.FFmpeg  (or put ffmpeg.exe on PATH)"))
    r.checks.append(Check("FFprobe", "PASS" if t.ffprobe else "FAIL", t.ffprobe or "not found",
                          "" if t.ffprobe else "ffprobe ships with FFmpeg builds; put it on PATH"))
    if t.colmap:
        ok, version = False, None
        try:
            cp = procrun.capture([t.colmap, "help"], timeout=60, env=procrun.child_env(colmap_path_dirs(t.colmap)))
            ok = cp.returncode == 0
            m = re.search(r"COLMAP\s+(\d+\.\d+(?:\.\d+)?)", _decode(cp.stdout))
            version = m.group(1) if m else None
        except Exception:
            ok = False
        if not ok:
            r.checks.append(Check("COLMAP", "FAIL", t.colmap + " (does not start)",
                                  "Reinstall COLMAP; its bin/ and lib/ folders must stay together"))
        else:
            compat = colmap_cli_compatible(t.colmap) if deep else None
            label = "%s (%s)" % (version or "unknown version", t.colmap)
            if compat is False:
                r.checks.append(Check("COLMAP", "FAIL", label + " - no --SiftExtraction.use_gpu option",
                                      "Nerfstudio 1.1.5 passes --SiftExtraction.* / --SiftMatching.* options, which "
                                      "newer COLMAP releases renamed. Install COLMAP 3.11.x (e.g. conda-forge "
                                      "colmap=3.11.1) and select it in Preferences > 360 Reconstruction."))
            else:
                r.checks.append(Check("COLMAP", "PASS", label))
    else:
        r.checks.append(Check("COLMAP", "FAIL", "not found",
                              "Install COLMAP (https://github.com/colmap/colmap/releases) or use PointForge's "
                              "photogrammetry engine setup, which installs it automatically"))

    if not t.ns_python:
        r.checks.append(Check("Nerfstudio", "FAIL", "no Nerfstudio Python environment found", INSTALL_HINT))
        for name in ("PyTorch", "CUDA", "Training backend"):
            r.checks.append(Check(name, "SKIP", "needs the Nerfstudio environment"))
    else:
        missing = [x for x in NS_TOOLS if not t.ns(x)]
        tinfo = query_torch(t.ns_python) if deep else {}
        r.torch = tinfo
        r.nerfstudio_version = tinfo.get("nerfstudio")
        if missing:
            r.checks.append(Check("Nerfstudio", "FAIL", "missing %s next to %s" % (", ".join(missing), t.ns_python),
                                  INSTALL_HINT))
        else:
            r.checks.append(Check("Nerfstudio", "PASS", "%s (%s)" % (tinfo.get("nerfstudio", "?"), t.ns_bin)))
        if not deep:
            pass
        elif "torch" in tinfo:
            r.checks.append(Check("PyTorch", "PASS", "%s (CUDA build %s)" % (tinfo["torch"], tinfo.get("cudaBuild"))))
            if tinfo.get("cuda"):
                r.checks.append(Check("CUDA", "PASS", "%s, %s MB" % (tinfo.get("device"), tinfo.get("totalMB"))))
            else:
                r.checks.append(Check("CUDA", "FAIL", "torch.cuda.is_available() is False",
                                      "Install a CUDA build of PyTorch matching your NVIDIA driver; "
                                      "Gaussian training and Nerfstudio's 360 projection require CUDA"))
        else:
            r.checks.append(Check("PyTorch", "FAIL", tinfo.get("torchError") or tinfo.get("probeError", "not importable"),
                                  INSTALL_HINT))
            r.checks.append(Check("CUDA", "SKIP", "needs PyTorch"))
        if deep:
            if "gsplat" in tinfo:
                r.checks.append(Check("Training backend", "PASS", "gsplat %s" % tinfo["gsplat"]))
            elif "torch" in tinfo:
                r.checks.append(Check("Training backend", "FAIL", "gsplat not importable (%s)" % tinfo.get("gsplatError"),
                                      "pip install gsplat inside the Nerfstudio env (version pinned by nerfstudio)"))
            # tiny-cuda-nn is NOT needed by Splatfacto; reported for information only.
            r.checks.append(Check("tiny-cuda-nn", "PASS" if "tinycudann" in tinfo else "SKIP",
                                  "present" if "tinycudann" in tinfo else "not installed (not required by Splatfacto)"))
            r.methods = list_methods(t) if not missing else []
            if r.methods and "splatfacto" not in r.methods:
                r.checks.append(Check("Splatfacto method", "FAIL", "ns-train does not list splatfacto", INSTALL_HINT))
        # Optional add-on: never FAIL (jobs without masking do not need it).
        has_tf = "transformers" in tinfo if deep else True
        if t.mask_model and has_tf:
            r.masking = True
            r.checks.append(Check("Masking (optional)", "PASS", "%s%s" % (
                Path(t.mask_model).name, (", transformers %s" % tinfo["transformers"]) if tinfo.get("transformers") else "")))
        else:
            why = "segmentation model not installed" if not t.mask_model else "transformers not installed in the env"
            r.checks.append(Check("Masking (optional)", "SKIP", why + " - run scripts\\pf360\\setup_masking.ps1 "
                                  "(~0.5 GB) to enable people/vehicle/sky masking"))

    # Optional camera-solver add-ons: never FAIL (the default COLMAP path does not need them).
    if t.glomap:
        ok = False
        try:
            cp = procrun.capture([t.glomap, "mapper", "-h"], timeout=60,
                                 env=procrun.child_env(glomap_path_dirs(t.glomap)))
            ok = "--database_path" in _decode(cp.stdout) + _decode(cp.stderr)
        except Exception:
            ok = False
        r.glomap = ok
        r.checks.append(Check("GLOMAP (optional)", "PASS" if ok else "WARN",
                              t.glomap if ok else t.glomap + " (does not start)",
                              "" if ok else "Reinstall the GLOMAP engine pack."))
    else:
        r.checks.append(Check("GLOMAP (optional)", "SKIP", "not installed - download the GLOMAP engine pack for the "
                              "fast global camera solver"))
    r.vocab_tree = bool(t.vocab_tree)
    r.checks.append(Check("Vocabulary tree (optional)", "PASS" if t.vocab_tree else "SKIP",
                          t.vocab_tree or "not installed - download the vocabulary tree engine pack for loop "
                          "detection (fewer, larger camera models)"))
    if t.lichtfeld:
        r.checks.append(Check("LichtFeld-Studio (optional)", "PASS", t.lichtfeld))
    else:
        r.checks.append(Check("LichtFeld-Studio (optional)", "SKIP",
                              "not installed - optional fast training backend (engines/lichtfeld)"))

    smi = query_nvidia_smi()
    r.gpu = smi
    if smi.get("gpus"):
        g = smi["gpus"][0]
        r.checks.append(Check("GPU", "PASS", "%s, %d MB total, %d MB free, driver %s" %
                              (g["name"], g["totalMB"], g["freeMB"], g["driver"])))
    else:
        r.checks.append(Check("GPU", "WARN", "nvidia-smi not available - no NVIDIA GPU detected",
                              "Gaussian Splat training needs an NVIDIA CUDA GPU"))
    return r


def format_report(r: Report) -> str:
    lines = ["PointForge 360 Environment Diagnostic", ""]
    for c in r.checks:
        lines.append("%-22s %-5s %s" % (c.name, c.status, c.detail))
    if r.methods:
        lines.append("%-22s %-5s %s" % ("Methods", "", ", ".join(r.methods)))
    lines.append("")
    lines.append("%-22s %s" % ("Overall Status", "READY" if r.ready else "NOT READY"))
    fixes = [c for c in r.checks if c.status == "FAIL" and c.fix]
    if fixes:
        lines.append("")
        lines.append("How to fix:")
        for c in fixes:
            lines.append("  - %s: %s" % (c.name, c.fix))
    return "\n".join(lines)

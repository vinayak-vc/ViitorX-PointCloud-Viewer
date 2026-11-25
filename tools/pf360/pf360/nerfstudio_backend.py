"""Nerfstudio + COLMAP + Splatfacto backend.

Stages (all tools run as child processes of the worker):
  panorama_projection    Nerfstudio equirect_utils (via ns_helpers/planar.py)
  camera_reconstruction  ns-process-data images  (COLMAP features/matching/mapping)
  training               ns-train splatfacto | splatfacto-big
  export                 ns-export gaussian-splat

Flag spellings were taken from the Nerfstudio 1.1.5 source and are re-checked
against the installed CLI (`--help`) at run time, because tyro renders boolean
options differently across versions (`--flag True` vs `--flag/--no-flag`).
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from . import environment, lichtfeld_engine, procrun, training_prep
from .backend import ReconstructionBackend, StageContext, StageFailed
from .views import perspective_size, views_per_frame, yaw_pitch_pairs

IMG_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
TRAIN_STEP_RE = re.compile(r"^\s*(\d+)\s+\((\d+(?:\.\d+)?)%\)")
COLMAP_FILE_RE = re.compile(r"Processed file \[(\d+)/(\d+)\]")
COLMAP_MATCH_RE = re.compile(r"Matching (?:block|image) \[(\d+)/(\d+)")
COLMAP_REG_RE = re.compile(r"Registering image #\d+ \((\d+)\)")
HELPERS = Path(__file__).with_name("ns_helpers")
# Horizontal FOV of Nerfstudio's planar views (equirect_utils, 1.1.5): 8 -> 120, 14 -> 110.
PLANAR_FOV_DEG = {8: 120, 14: 110}
# COLMAP mapper: minimum triangulation angle of the initial pair (default 16).
INIT_MIN_TRI_ANGLE_DEG = 4
# COLMAP mapper: when to re-run a GLOBAL bundle adjustment. COLMAP does one once the model grows by 10%
# (ratio 1.1) OR by 500 images / 250k points (freq) - whichever comes first. Past ~5000 views the fixed
# 500 wins, and on a 2000-panorama clip (28000 views) each global BA took 7-20 min while registering
# those 500 views took ~20 s, so ~95% of the mapper's time was global BA.
#   thorough = COLMAP's defaults; balanced = the 10% ratio decides on big models (~3-4x fewer global
#   BAs, small models unchanged); fast = 20% growth between global BAs (most drift between them).
GLOBAL_REFINEMENT = {
    "thorough": [],
    "balanced": ["--Mapper.ba_global_images_freq", "2000", "--Mapper.ba_global_points_freq", "1000000"],
    "fast": ["--Mapper.ba_global_images_ratio", "1.2", "--Mapper.ba_global_points_ratio", "1.2",
             "--Mapper.ba_global_images_freq", "5000", "--Mapper.ba_global_points_freq", "5000000"],
}


def mapper_ba_args(pr: Dict[str, Any], cuda: bool = True) -> List[str]:
    """COLMAP `mapper` bundle-adjustment options chosen in the job (refinement interval, GPU)."""
    args = list(GLOBAL_REFINEMENT.get(pr.get("globalRefinement", "balanced"), GLOBAL_REFINEMENT["balanced"]))
    if pr.get("gpuBundleAdjustment") and cuda:
        args += ["--Mapper.ba_use_gpu", "1", "--Mapper.ba_gpu_index", "0"]
    return args
# Max `mapper --input_path` rounds that grow the joined, rig-completed model (camera stage).
EXTEND_ROUNDS = 3
# Training restore points: periodic checkpoint interval, and how long a pause may take to save one.
TRAIN_STEPS_PER_SAVE = 1000
LONG_SCHEDULE_MIN_ITS = 30000      # Splatfacto's own schedule is tuned for 30k steps
REFINE_EVERY = 100                 # Splatfacto refine_every (steps)
DEFAULT_RESET_ALPHA_EVERY = 30     # Splatfacto reset_alpha_every (refinements) = 3000 steps
TRAIN_PAUSE_GRACE_S = 150
# Written into colmap/sparse after `mapper` exits 0; a resumed camera stage then keeps the models.
MAPPER_DONE = "mapper_done.json"
COLMAP_INIT_RE = re.compile(r"Initializing with image pair")
# GLOMAP 1.2 mapper phases (glog lines with --log_to_stderr 1) -> camera-stage percent.
GLOMAP_PHASES = [
    ("preprocessing", 56.0, "GLOMAP: preparing the view graph"),
    ("view graph calibration", 58.0, "GLOMAP: checking the view graph"),
    ("relative pose estimation", 62.0, "GLOMAP: relative poses between matched views"),
    ("rotation averaging", 68.0, "GLOMAP: camera rotations (all at once)"),
    ("track establishment", 72.0, "GLOMAP: building point tracks"),
    ("global positioning", 76.0, "GLOMAP: camera positions (all at once)"),
    ("bundle adjustment", 82.0, "GLOMAP: refining cameras and points"),
    ("retriangulation", 86.0, "GLOMAP: re-triangulating points"),
    ("postprocessing", 89.0, "GLOMAP: writing the model"),
]
COLMAP_FIND_INIT_RE = re.compile(r"Finding good initial image pair")


def _fmt_dur(seconds: float) -> str:
    s = int(max(0.0, seconds))
    if s >= 3600:
        return "%dh %02dm" % (s // 3600, (s % 3600) // 60)
    return "%dm %02ds" % (s // 60, s % 60) if s >= 60 else "%ds" % s


class _Ticker:
    """Keeps a long, quiet child step visibly alive in the UI.

    COLMAP's mapper prints nothing while it bundle-adjusts a large model (10+ minutes on real
    footage), so the host saw a frozen message. `set` records the latest progress (and emits it);
    `seen` marks that the tool printed something. While the tool is quiet for `quiet_after`
    seconds the last progress is re-emitted every `interval` seconds with the silent time, and a
    log line is written every `log_every` seconds of silence. `tick` is the testable core."""

    def __init__(self, emitter, stage: str, *, interval: float = 5.0, quiet_after: float = 30.0,
                 log_every: float = 300.0, clock=None) -> None:
        import time as _time
        self.emitter, self.stage = emitter, stage
        self.interval, self.quiet_after, self.log_every = interval, quiet_after, log_every
        self.clock = clock or _time.monotonic
        self._lock = threading.Lock()
        self.pct: Optional[float] = None
        self.msg = ""
        self.last = self.clock()
        self._logged = 0
        self._stop = threading.Event()
        self._th: Optional[threading.Thread] = None

    def set(self, pct: Optional[float], msg: str) -> None:
        with self._lock:
            self.pct, self.msg, self.last, self._logged = pct, msg, self.clock(), 0
        self.emitter.progress(self.stage, pct, msg)

    def seen(self, _line: str = "") -> None:
        with self._lock:
            self.last, self._logged = self.clock(), 0

    def tick(self, now: Optional[float] = None) -> None:
        now = self.clock() if now is None else now
        with self._lock:
            quiet, pct, msg = now - self.last, self.pct, self.msg
            n = int(quiet // self.log_every) if self.log_every > 0 else 0
            log_now = n > self._logged
            if log_now:
                self._logged = n
        if quiet < self.quiet_after:
            return
        self.emitter.progress(self.stage, pct, "%s - still working, no output for %s" % (msg, _fmt_dur(quiet)))
        if log_now:
            self.emitter.log("%s: still working (no output from the tool for %s)" % (msg, _fmt_dur(quiet)),
                             stage=self.stage)

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            self.tick()

    def __enter__(self) -> "_Ticker":
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._th is not None:
            self._th.join()


def _say(ctx: StageContext, ticker: Optional[_Ticker], pct: Optional[float], msg: str) -> None:
    """Progress through the stage ticker when there is one (keeps the heartbeat message current)."""
    if ticker is not None:
        ticker.set(pct, msg)
    else:
        ctx.emitter.progress("camera_reconstruction", pct, msg)


class _MapperProgress:
    """COLMAP `mapper` output -> a percent that only grows + a readable message, over ALL models.

    "Registering image #id (n)" counts the CURRENT model only and restarts at every new model
    ("Initializing with image pair"), so the bar used to jump back each time COLMAP split the walk.
    Here views of finished models are added up (a view may sit in two models, so this is "views
    registered so far", capped at the image count). feed() returns (percent, message, log lines)
    for lines that change the state, else None.

    COLMAP throws away attempts with fewer than Mapper.min_model_size (10) images - on real footage
    most of its ~90 starts register 0-6 images - so only attempts reaching MIN_MODEL count as models
    (and get a log line); the rest are "attempts"."""

    MIN_MODEL = 10               # COLMAP Mapper.min_model_size default

    def __init__(self, total: int, lo: float, hi: float) -> None:
        self.total, self.lo, self.hi = max(1, total), lo, hi
        self.kept = 0            # finished models COLMAP keeps (>= MIN_MODEL views)
        self.attempts = 0        # initial pairs tried
        self.cur = 0             # views in the model being built
        self.done = 0            # views in finished, kept models
        self.best = lo

    @property
    def models(self) -> int:
        """Models kept so far, including the one being built once it is big enough."""
        return self.kept + (1 if self.cur >= self.MIN_MODEL else 0)

    def _pct(self) -> float:
        p = self.lo + (self.hi - self.lo) * min(1.0, (self.done + self.cur) / self.total)
        self.best = max(self.best, p)
        return self.best

    def _placed(self) -> str:
        return "%d of %d views registered so far" % (min(self.total, self.done + self.cur), self.total)

    def feed(self, line: str) -> Optional[tuple]:
        logs: List[str] = []
        if COLMAP_FIND_INIT_RE.search(line):
            self._close(logs)
            return (self._pct(), "looking for a starting image pair - %d model(s) so far, %d attempt(s) (%s)"
                    % (self.kept, self.attempts, self._placed()), logs)
        if COLMAP_INIT_RE.search(line):
            self._close(logs)
            self.attempts += 1
            if self.attempts == 1:
                logs.append("COLMAP mapper started (it may build several models if the walk splits)")
            return (self._pct(), "trying a starting image pair (attempt %d, %s)" % (self.attempts, self._placed()),
                    logs)
        m = COLMAP_REG_RE.search(line)
        if m:
            self.cur = int(m.group(1))
            return (self._pct(), "registering images - model %d: %d views (%s)"
                    % (self.kept + 1, self.cur, self._placed()), logs)
        if "bundle adjustment" in line.lower():
            return (self._pct(), "bundle adjustment - model %d, %d views (%s)"
                    % (self.kept + 1, self.cur, self._placed()), logs)
        return None

    def _close(self, logs: List[str]) -> None:
        """The model being built is finished (COLMAP looks for a new initial pair)."""
        n, self.cur = self.cur, 0
        if n >= self.MIN_MODEL:
            self.kept += 1
            self.done += n
            logs.append("COLMAP model %d finished with %d views (%s)" % (self.kept, n, self._placed()))

    def summary(self) -> str:
        return "COLMAP mapper finished: %d model(s) from %d attempt(s), %s" % (self.models, self.attempts,
                                                                              self._placed())


def _latest_valid_checkpoint(training_dir: Path) -> Optional[tuple]:
    """(step, path) of the newest COMPLETE Nerfstudio checkpoint, or None.

    torch.save writes a zip; a file cut short by a kill lacks the zip end-of-central-directory
    record, so it is skipped (and deleted) and the previous checkpoint is used instead."""
    if not training_dir.is_dir():
        return None
    found = []
    for p in training_dir.rglob("step-*.ckpt"):
        m = re.match(r"step-(\d+)\.ckpt$", p.name)
        if m:
            found.append((int(m.group(1)), p))
    for step, p in sorted(found, reverse=True):
        try:
            size = p.stat().st_size
            with open(p, "rb") as f:
                f.seek(max(0, size - 65536 - 22))
                tail = f.read()
        except OSError:
            continue
        if size > 0 and b"PK\x05\x06" in tail:
            return step, p
        try:
            p.unlink()                         # truncated: unusable, and would shadow older ones
        except OSError:
            pass
    return None


def planar_intrinsics(w: int, h: int, fov_deg: float) -> tuple:
    """(fx, fy, cx, cy) of a Nerfstudio planar view in COLMAP pixel convention.

    equirect2persp maps pixel index i to the ray offset (i - w/2) * 2*tan(fov/2)/w,
    so f = w / (2 tan(fov/2)); COLMAP puts pixel i's centre at i + 0.5, hence the
    principal point is w/2 + 0.5. hfov = h/w * fov in Nerfstudio, so fy = fx only
    for square views (always the case: size = sqrt(W*H/n) for both sides)."""
    fx = w / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    hfov = float(h) / w * fov_deg
    fy = h / (2.0 * math.tan(math.radians(hfov) / 2.0))
    return (fx, fy, w / 2.0 + 0.5, h / 2.0 + 0.5)


def _jpeg_size(path: Path) -> Optional[tuple]:
    """(width, height) from a JPEG's SOF marker, or PNG IHDR; None if unreadable."""
    try:
        with open(path, "rb") as f:
            head = f.read(24)
            if head[:8] == b"\x89PNG\r\n\x1a\n":
                return int.from_bytes(head[16:20], "big"), int.from_bytes(head[20:24], "big")
            if head[:2] != b"\xff\xd8":
                return None
            f.seek(2)
            while True:
                b = f.read(1)
                while b and b != b"\xff":
                    b = f.read(1)
                while b == b"\xff":
                    b = f.read(1)
                if not b:
                    return None
                marker = b[0]
                if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
                    continue
                seglen = int.from_bytes(f.read(2), "big")
                if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                    data = f.read(5)
                    return int.from_bytes(data[3:5], "big"), int.from_bytes(data[1:3], "big")
                f.seek(seglen - 2, 1)
    except OSError:
        return None


def _colmap_models(sparse_dir: Path) -> List[tuple]:
    """[(model_dir_name, registered_image_count)] for COLMAP binary models under sparse_dir."""
    out = []
    if not sparse_dir.is_dir():
        return out
    for d in sorted(p for p in sparse_dir.iterdir() if p.is_dir()):
        ib = d / "images.bin"
        try:
            with open(ib, "rb") as f:
                n = int.from_bytes(f.read(8), "little")
        except OSError:
            continue
        out.append((d.name, n))
    return out


def _masking_on(manifest: Dict[str, Any]) -> bool:
    mk = (manifest.get("processing") or {}).get("masking") or {}
    return bool(mk.get("enabled")) and bool(mk.get("groups"))


def _count_images(d: Path) -> int:
    if not d.is_dir():
        return 0
    return sum(1 for p in d.iterdir() if p.suffix.lower() in IMG_EXTS and p.is_file())


class CliFlags:
    """Builds option spellings that the installed tyro CLI actually accepts."""

    def __init__(self, help_text: str):
        self.help = help_text

    def has(self, flag: str) -> bool:
        return re.search(re.escape(flag) + r"(?![\w-])", self.help) is not None

    def boolean(self, dotted: str, value: bool) -> List[str]:
        # "--viewer.quit-on-train-completion" -> negative "--viewer.no-quit-on-train-completion"
        prefix, _, leaf = dotted[2:].rpartition(".")
        neg = "--" + (prefix + "." if prefix else "") + "no-" + leaf
        if self.has(neg):
            return [dotted] if value else [neg]
        return [dotted, "True" if value else "False"]


class NerfstudioBackend(ReconstructionBackend):
    name = "nerfstudio"

    def __init__(self) -> None:
        self._help_cache: Dict[str, str] = {}

    def stages(self) -> Sequence[str]:
        return ("panorama_projection", "masking", "camera_reconstruction", "training", "export")

    def stage_enabled(self, stage: str, ctx: StageContext) -> bool:
        if stage == "masking":
            return _masking_on(ctx.manifest)
        return True

    # ------------------------------------------------------------------ env
    def check_environment(self, tools, manifest) -> List[str]:
        p: List[str] = []
        if manifest.get("training", {}).get("engine") == "lichtfeld":
            if not tools.lichtfeld:
                p.append("LichtFeld-Studio executable not found (run `python -m pf360 doctor` or set PF360_LICHTFELD).")
            smi = environment.query_nvidia_smi()
            if smi.get("gpus"):
                g = smi["gpus"][0]
                driver = g.get("driver", "0")
                try:
                    drv_major = int(driver.split(".")[0])
                    if drv_major < 570:
                        p.append("LichtFeld-Studio requires NVIDIA driver >= 570 (detected driver %s)." % driver)
                except (ValueError, IndexError):
                    pass
                cap = g.get("compute_cap")
                if cap:
                    try:
                        if float(cap) < 7.5:
                            p.append("LichtFeld-Studio requires GPU compute capability >= 7.5 (detected %s)." % cap)
                    except ValueError:
                        pass
            else:
                p.append("LichtFeld-Studio requires an NVIDIA GPU with driver >= 570.")
        if _masking_on(manifest) and not tools.mask_model:
            p.append("Masking is enabled but the segmentation model is not installed "
                     "(run scripts\\pf360\\setup_masking.ps1, ~0.5 GB, or turn masking off).")
        if not tools.ns_python:
            p.append("No Nerfstudio Python environment found (run `python -m pf360 doctor`).")
        if manifest.get("training", {}).get("engine") != "lichtfeld":
            for t in ("ns-train", "ns-export"):
                if tools.ns_python and not tools.ns(t):
                    p.append("%s not found next to %s" % (t, tools.ns_python))
        if tools.ns_python and not tools.ns("ns-process-data"):
            p.append("ns-process-data not found next to %s" % tools.ns_python)
        if not tools.colmap:
            p.append("COLMAP not found.")
        if not tools.ffmpeg:
            p.append("FFmpeg not found.")
        return p

    def _help(self, ctx: StageContext, key: str, args: List[str]) -> CliFlags:
        if key not in self._help_cache:
            try:
                cp = procrun.capture(args, timeout=300, env=self._env(ctx))
                self._help_cache[key] = cp.stdout.decode("utf-8", "replace") + cp.stderr.decode("utf-8", "replace")
            except Exception:
                self._help_cache[key] = ""
        return CliFlags(self._help_cache[key])

    def _env(self, ctx: StageContext) -> Dict[str, str]:
        # MSVC + CUDA_HOME so gsplat can JIT-compile its kernels if they are not
        # cached yet (see environment.compiler_env). The vcvars PATH replaces the
        # base PATH, then the tool dirs are prepended as usual.
        extra = dict(environment.compiler_env())
        extra.update(ctx.tools.ns_env())       # downloadable runtime pack: isolation + bundled weights
        env = procrun.child_env((), extra)
        dirs = [d for d in ctx.tools.path_prepend() if d]
        if dirs:
            env["PATH"] = os.pathsep.join(dirs + [env.get("PATH", "")])
        return env

    # --------------------------------------------------------------- caching
    def stage_inputs(self, stage: str, ctx: StageContext) -> Dict[str, Any]:
        pr, tr = ctx.manifest["processing"], ctx.manifest["training"]
        if stage == "panorama_projection":
            return {"views": pr["imagesPerEquirect"],
                    "crop": [pr["cropTop"], pr["cropBottom"], pr["cropLeft"], pr["cropRight"]]}
        if stage == "masking":
            mk = pr.get("masking") or {}
            if not _masking_on(ctx.manifest):
                return {"enabled": False}
            return {"enabled": True, "groups": sorted(mk.get("groups", [])), "dilatePx": mk.get("dilatePx", 8),
                    "model": environment.MASK_MODEL_DIRNAME, "helper": "mask-v1"}
        if stage == "camera_reconstruction":
            inputs = {"matching": pr["matchingMethod"], "colmap": ctx.tools.colmap, "numDownscales": 3,
                      "sfm": "pf360-fixed-pinhole-rig-v2", "masks": _masking_on(ctx.manifest)}
            # Only non-default solver choices enter the fingerprint, so jobs from before these
            # options still resume. (Refinement interval + GPU BA change speed, not the inputs.)
            if pr.get("cameraSolver", "colmap") != "colmap":
                inputs["solver"] = pr["cameraSolver"]
            if pr.get("loopDetection"):
                inputs["loopDetection"] = True
            return inputs
        if stage == "training":
            if tr.get("engine") == "lichtfeld":
                inputs = {
                    "engine": "lichtfeld",
                    "lfsStrategy": tr.get("lfsStrategy", "mrnf"),
                    "maxGaussians": tr.get("maxGaussians", 2_000_000),
                    "iterations": tr.get("iterations"),
                    "downscaleFactor": tr.get("downscaleFactor"),
                    "shDegree": tr.get("shDegree"),
                }
                if tr.get("seedPointCap", training_prep.DEFAULT_SEED_POINT_CAP) != training_prep.DEFAULT_SEED_POINT_CAP:
                    inputs["seedPointCap"] = tr["seedPointCap"]
                if tr.get("dropMisplacedPanoramas", True) is not True:
                    inputs["dropMisplacedPanoramas"] = False
                if tr.get("maxTrainingViews", 0):
                    inputs["maxTrainingViews"] = tr["maxTrainingViews"]
                return inputs
            keys = ("method", "iterations", "downscaleFactor", "shDegree", "randomSeed", "useScaleRegularization",
                    "cullAlphaThresh", "densifyGradThresh", "cullScaleThresh")
            inputs = {k: tr.get(k) for k in keys}
            # Clean-up options enter the fingerprint only when changed from the defaults, so
            # finished jobs from before these options do not retrain on resume.
            if tr.get("seedPointCap", training_prep.DEFAULT_SEED_POINT_CAP) != training_prep.DEFAULT_SEED_POINT_CAP:
                inputs["seedPointCap"] = tr["seedPointCap"]
            if tr.get("dropMisplacedPanoramas", True) is not True:
                inputs["dropMisplacedPanoramas"] = False
            if tr.get("maxTrainingViews", 0):
                inputs["maxTrainingViews"] = tr["maxTrainingViews"]
            if tr.get("longSchedule"):
                inputs["longSchedule"] = 2          # v2: opacity reset scaled to the view count
            return inputs
        if stage == "export":
            return {"plyFileName": ctx.manifest["output"].get("plyFileName")}
        return {}

    def stage_outputs_exist(self, stage: str, ctx: StageContext) -> bool:
        ws = ctx.workspace
        tr = ctx.manifest.get("training", {})
        if stage == "panorama_projection":
            return _count_images(ws.planar_dir) > 0
        if stage == "masking":
            n = _count_images(ws.planar_dir)
            return n > 0 and ws.masks_dir.is_dir() and sum(1 for _ in ws.masks_dir.glob("*.png")) == n
        if stage == "camera_reconstruction":
            return (ws.processed_dir / "transforms.json").is_file()
        if stage == "training":
            if tr.get("engine") == "lichtfeld":
                target = ws.training_dir / "lfs" / ("splat_%d.ply" % int(tr.get("iterations", 30000)))
                return target.is_file()
            return self._find_config(ctx) is not None
        if stage == "export":
            return ws.export_ply.is_file()
        return False

    # ---------------------------------------------------------------- stages
    def run_stage(self, stage: str, ctx: StageContext) -> Dict[str, Any]:
        return getattr(self, "_" + stage)(ctx)

    def _panorama_projection(self, ctx: StageContext) -> Dict[str, Any]:
        ws, pr = ctx.workspace, ctx.manifest["processing"]
        crop = (pr["cropTop"], pr["cropBottom"], pr["cropLeft"], pr["cropRight"])
        n_frames = _count_images(ws.frames_dir)
        if n_frames == 0:
            raise StageFailed("No extracted frames to project.")
        vpf = views_per_frame(pr["imagesPerEquirect"], crop)
        expected = n_frames * vpf
        if not ctx.results.get("_resume_panorama_projection"):
            ws.reset_dir(ws.planar_dir)          # else: keep finished panoramas (restore point)
        views = yaw_pitch_pairs(pr["imagesPerEquirect"], crop)
        stop = threading.Event()

        def poll() -> None:
            while not stop.wait(1.0):
                done = _count_images(ws.planar_dir)
                ctx.emitter.progress("panorama_projection", 100.0 * done / expected,
                                     "%d / %d perspective images" % (done, expected))

        th = threading.Thread(target=poll, daemon=True)
        th.start()
        result: Dict[str, Any] = {}

        def on_line(l: str) -> None:
            if l.startswith("PF360PLANARPROG"):        # per-panorama progress, not the result line
                return
            if l.startswith("PF360PLANAR"):
                result.update(json.loads(l[len("PF360PLANAR"):]))
        try:
            with ws.open_log("panorama.log") as log:
                rc = procrun.run([ctx.tools.ns_python, str(HELPERS / "planar.py"), str(ws.frames_dir),
                                  str(pr["imagesPerEquirect"])] + [repr(float(c)) for c in crop] +
                                 [json.dumps(views)],
                                 cancel=ctx.cancel, on_line=on_line, log=log, env=self._env(ctx))
        finally:
            stop.set()
            th.join()
        got = _count_images(ws.planar_dir)
        if rc != 0:
            raise StageFailed("Nerfstudio panorama projection failed (exit %d) - see logs/panorama.log. "
                              "It requires a CUDA-capable PyTorch." % rc)
        if got != expected:
            ctx.emitter.warning("Expected %d perspective images, found %d." % (expected, got), stage="panorama_projection")
        if got == 0:
            raise StageFailed("Panorama projection produced no images.")
        return {"frames": n_frames, "viewsPerFrame": vpf, "perspectiveImages": got,
                "perspectiveSize": result.get("size")}

    def _masking(self, ctx: StageContext) -> Dict[str, Any]:
        """Semantic masks (people / vehicles / animals / sky) for every planar view,
        computed in the Nerfstudio env by ns_helpers/mask.py. Used by COLMAP
        (features ignored) and Splatfacto (pixels excluded from the loss)."""
        ws, mk = ctx.workspace, ctx.manifest["processing"]["masking"]
        if not ctx.tools.mask_model:
            raise StageFailed("Masking model not installed - run scripts\\pf360\\setup_masking.ps1 or turn masking off.")
        total = _count_images(ws.planar_dir)
        if total == 0:
            raise StageFailed("No perspective images to mask.")
        if not ctx.results.get("_resume_masking"):
            ws.reset_dir(ws.masks_dir)            # else: keep finished masks (restore point)
        stats: Dict[str, Any] = {}

        def on_line(l: str) -> None:
            if l.startswith("PF360MASKDONE"):
                stats.update(json.loads(l[len("PF360MASKDONE"):]))
            elif l.startswith("PF360MASK"):
                try:
                    d = json.loads(l[len("PF360MASK"):])
                    ctx.emitter.progress("masking", 100.0 * d["done"] / max(1, d["total"]),
                                         "%d / %d views" % (d["done"], d["total"]))
                except (ValueError, KeyError):
                    pass
        args = [ctx.tools.ns_python, str(HELPERS / "mask.py"), "--model", ctx.tools.mask_model,
                "--images", str(ws.planar_dir), "--out", str(ws.masks_dir),
                "--groups", ",".join(mk["groups"]), "--dilate", str(int(mk.get("dilatePx", 8)))]
        ctx.emitter.progress("masking", None, "loading segmentation model")
        with ws.open_log("masking.log") as log:
            rc = procrun.run(args, cancel=ctx.cancel, on_line=on_line, log=log, env=self._env(ctx))
        got = sum(1 for _ in ws.masks_dir.glob("*.png")) if ws.masks_dir.is_dir() else 0
        if rc != 0 or got != total:
            raise StageFailed("Masking failed (exit %d, %d of %d masks) - see logs/masking.log" % (rc, got, total))
        mean = float(stats.get("meanMaskedFraction") or 0.0)
        if mean > 0.4:
            ctx.emitter.warning("On average %.0f%% of each view is masked; COLMAP and training have little left to "
                                "work with. Consider fewer mask categories." % (100 * mean), stage="masking")
        heavy = int(stats.get("imagesMaskedOver60Percent") or 0)
        if heavy:
            ctx.emitter.warning("%d view(s) are more than 60%% masked." % heavy, stage="masking")
        out = {"enabled": True, "groups": list(mk["groups"]), "dilatePx": int(mk.get("dilatePx", 8)), "masks": got}
        out.update({k: stats[k] for k in ("meanMaskedFraction", "maxMaskedFraction", "imagesMaskedOver60Percent",
                                          "device") if k in stats})
        return out

    def _camera_reconstruction(self, ctx: StageContext) -> Dict[str, Any]:
        """COLMAP with KNOWN intrinsics, then ns-process-data --skip-colmap.

        Nerfstudio's own COLMAP call starts from COLMAP's EXIF-less focal guess
        (1.2 x max(w, h)), about 4x too long for the 110-120 degree planar views,
        and refines it in bundle adjustment. On real runs that leaves almost no
        image registered. The planar views are exact pinhole projections
        (equirect_utils.equirect2persp: f = w / (2 tan(fov/2)), principal point
        at the image centre), so the worker runs COLMAP itself with a fixed
        PINHOLE camera and hands the largest model to ns-process-data."""
        ws, pr = ctx.workspace, ctx.manifest["processing"]
        tool = ctx.tools.ns("ns-process-data")
        colmap = ctx.tools.colmap
        if not colmap:
            raise StageFailed("COLMAP was not found - see the environment diagnostic.")
        total = _count_images(ws.planar_dir)
        if total == 0:
            raise StageFailed("No perspective images to reconstruct.")
        size = _jpeg_size(next(p for p in sorted(ws.planar_dir.iterdir()) if p.suffix.lower() in IMG_EXTS))
        if not size:
            raise StageFailed("Could not read the perspective image size.")
        w, h = size
        fov = PLANAR_FOV_DEG.get(int(pr["imagesPerEquirect"]))
        if fov is None:
            raise StageFailed("Unsupported views per frame: %s" % pr["imagesPerEquirect"])
        intr = planar_intrinsics(w, h, fov)
        vpf = views_per_frame(pr["imagesPerEquirect"],
                              (pr["cropTop"], pr["cropBottom"], pr["cropLeft"], pr["cropRight"]))

        colmap_dir = ws.processed_dir / "colmap"
        sparse_dir = colmap_dir / "sparse"
        db = colmap_dir / "database.db"
        # Restore point: COLMAP's database holds the extracted features and finished matches;
        # feature_extractor / *_matcher skip images and pairs already in it, so an interrupted
        # run continues. The SQLite journal files move with it (a hot journal = recovery data).
        # A FINISHED mapper (marker written after it exits 0) is kept as well: on real footage it takes
        # hours, while everything after it (joining, completion, extension) is quick to redo.
        kept, kept_models = [], []
        if ctx.results.get("_resume_camera_reconstruction") and db.is_file():
            hold = ws.root / ".colmap-db-hold"
            ws.reset_dir(hold)
            for f in colmap_dir.glob("database.db*"):
                os.replace(f, hold / f.name)
                kept.append(f.name)
            if (sparse_dir / MAPPER_DONE).is_file() and _colmap_models(sparse_dir):
                (hold / "sparse").mkdir()
                for d in [sparse_dir / name for name, _ in _colmap_models(sparse_dir)] + [sparse_dir / MAPPER_DONE]:
                    os.replace(d, hold / "sparse" / d.name)
                    kept_models.append(d.name)
        ws.reset_dir(ws.processed_dir)
        sparse_dir.mkdir(parents=True, exist_ok=True)
        if kept:
            for name in kept:
                os.replace(ws.root / ".colmap-db-hold" / name, colmap_dir / name)
            for name in kept_models:
                os.replace(ws.root / ".colmap-db-hold" / "sparse" / name, sparse_dir / name)
            shutil.rmtree(ws.root / ".colmap-db-hold", ignore_errors=True)
            ctx.emitter.log("keeping COLMAP features + matches from the interrupted run", stage="camera_reconstruction")
        reuse_mapper = bool(kept_models)
        if reuse_mapper:
            ctx.emitter.log("reusing COLMAP's finished mapper output (%d models) from the interrupted run"
                            % (len(kept_models) - 1), stage="camera_reconstruction")
        cuda = bool((ctx.results.get("_env") or {}).get("cuda", True))
        gpu = "1" if cuda else "0"
        env = self._env(ctx)
        phase = {"name": "feature extraction", "pct": 0.0}
        ticker = _Ticker(ctx.emitter, "camera_reconstruction")
        mapper = _MapperProgress(total, 55.0, 90.0)

        def on_line(l: str) -> None:
            ticker.seen()
            m = COLMAP_FILE_RE.search(l)
            if m:
                phase.update(name="feature extraction", pct=25.0 * int(m.group(1)) / max(1, int(m.group(2))))
                ticker.set(phase["pct"], "feature extraction - %s / %s images" % (m.group(1), m.group(2)))
                return
            m2 = COLMAP_MATCH_RE.search(l)
            if m2:
                phase.update(name="feature matching", pct=25.0 + 30.0 * int(m2.group(1)) / max(1, int(m2.group(2))))
                ticker.set(phase["pct"], "feature matching - %s / %s images" % (m2.group(1), m2.group(2)))
                return
            got = mapper.feed(l)
            if got:
                pct, msg, logs = got
                for x in logs:
                    ctx.emitter.log(x, stage="camera_reconstruction")
                phase.update(name="registering images", pct=pct)
                ticker.set(pct, msg)

        method = pr["matchingMethod"]
        matcher = [colmap, "%s_matcher" % method, "--database_path", str(db), "--SiftMatching.use_gpu", gpu]
        if method == "sequential":
            # Images are ordered panorama-major (frame_N_0..V-1): cover ~5 panoramas so a view
            # direction is matched against itself a few steps along the walk (3 was too short on
            # real footage: COLMAP kept starting tiny models that never joined).
            matcher += ["--SequentialMatching.overlap", str(max(10, 5 * vpf))]
            if pr.get("loopDetection"):
                # Every 10th image is also matched against its 50 most similar images anywhere in the
                # video (vocabulary-tree retrieval): revisited places and pieces the sequential chain
                # broke between get matches, so COLMAP builds fewer, larger models.
                if ctx.tools.vocab_tree:
                    # 512 features per image for the retrieval index: uncapped, indexing took 2.6 s
                    # per image (~20 h for 28,000 views); capped 0.05 s. Matching itself still uses
                    # all features.
                    matcher += ["--SequentialMatching.loop_detection", "1",
                                "--SequentialMatching.loop_detection_max_num_features", "512",
                                "--SequentialMatching.vocab_tree_path", ctx.tools.vocab_tree]
                else:
                    ctx.emitter.warning("Loop detection was requested but the vocabulary tree engine pack is not "
                                        "installed - matching without it.", stage="camera_reconstruction")
        elif method == "vocab_tree":
            matcher += ["--VocabTreeMatching.vocab_tree_path", self._vocab_tree(ctx, env)]
        masks_on = _masking_on(ctx.manifest)
        if masks_on and sum(1 for _ in ws.masks_dir.glob("*.png")) != total:
            raise StageFailed("Masking is enabled but the masks are incomplete - re-run the job.")
        extract = [colmap, "feature_extractor", "--database_path", str(db), "--image_path", str(ws.planar_dir),
                   "--ImageReader.single_camera", "1", "--ImageReader.camera_model", "PINHOLE",
                   "--ImageReader.camera_params", ",".join("%.6f" % v for v in intr),
                   "--SiftExtraction.use_gpu", gpu]
        if masks_on:
            # COLMAP: mask file "<image name>.png", features in 0-valued pixels are dropped.
            extract += ["--ImageReader.mask_path", str(ws.masks_dir)]
        colmap_mapper = (
            "registering images",
             [colmap, "mapper", "--database_path", str(db), "--image_path", str(ws.planar_dir),
              "--output_path", str(sparse_dir),
              "--Mapper.ba_refine_focal_length", "0", "--Mapper.ba_refine_principal_point", "0",
              "--Mapper.ba_refine_extra_params", "0", "--Mapper.ba_global_function_tolerance=1e-6",
              # Walking video: consecutive panoramas are ~0.5 m apart, far below what COLMAP's
              # default 16 deg initial triangulation angle wants ("No good initial image pair").
              "--Mapper.init_min_tri_angle", str(INIT_MIN_TRI_ANGLE_DEG)] + mapper_ba_args(pr, cuda))
        use_glomap = pr.get("cameraSolver") == "glomap"
        if use_glomap and not ctx.tools.glomap:
            ctx.emitter.warning("GLOMAP was chosen but its engine pack is not installed - using COLMAP's "
                                "incremental mapper.", stage="camera_reconstruction")
            use_glomap = False
        steps = [("feature extraction", extract), ("feature matching", matcher)]
        if not use_glomap:
            steps.append(colmap_mapper)
        with ws.open_log("colmap.log") as log, ticker:
            log.write("pf360: PINHOLE %dx%d fov=%d -> fx=fy=%.3f cx=%.3f cy=%.3f (fixed)\n"
                      % (w, h, fov, intr[0], intr[2], intr[3]))
            for name, args in ([] if reuse_mapper else steps):
                ticker.set(phase["pct"], name)
                rc = procrun.run(args, cancel=ctx.cancel, on_line=on_line, log=log, env=env, cwd=str(ws.root))
                if rc != 0:
                    raise StageFailed("COLMAP %s failed (exit %d) - see logs/colmap.log" % (name, rc))
            if use_glomap and not reuse_mapper:
                if not self._glomap(ctx, log, sparse_dir, db, cuda, ticker):
                    ctx.emitter.warning("GLOMAP did not produce a camera model - falling back to COLMAP's "
                                        "incremental mapper (see logs/colmap.log).", stage="camera_reconstruction")
                    for d in sparse_dir.iterdir():
                        if d.is_dir():
                            shutil.rmtree(d, ignore_errors=True)
                    ticker.set(55.0, colmap_mapper[0])
                    rc = procrun.run(colmap_mapper[1], cancel=ctx.cancel, on_line=on_line, log=log, env=env,
                                     cwd=str(ws.root))
                    if rc != 0:
                        raise StageFailed("COLMAP mapper failed (exit %d) - see logs/colmap.log" % rc)
                    use_glomap = False
            models = _colmap_models(sparse_dir)
            if reuse_mapper:
                log.write("pf360: reusing the finished mapper output (%d models)\n" % len(models))
            else:
                if not use_glomap:
                    ctx.emitter.log(mapper.summary(), stage="camera_reconstruction")
                if models:
                    with open(sparse_dir / MAPPER_DONE, "w", encoding="utf-8") as f:
                        json.dump({"models": len(models), "views": sum(n for _, n in models)}, f)
            if not models:
                raise StageFailed("COLMAP could not reconstruct any cameras - the footage probably lacks "
                                  "texture/overlap or camera movement.",
                                  {"generatedImages": total, "registeredImages": 0})
            best = max(models, key=lambda kv: kv[1])
            rig = self._rig_complete(ctx, env, log, sparse_dir, db, pr, ticker=ticker)
            if rig:
                rig["dir"] = self._triangulate(ctx, env, log, colmap, sparse_dir, rig["dir"], db, ticker=ticker)
                rig = self._extend(ctx, env, log, colmap, sparse_dir, db, pr, rig, ticker=ticker)
            model_rel = "colmap/sparse/%s" % (rig["dir"] if rig else best[0])
            ticker.set(92.0, "writing transforms.json (ns-process-data)")
            flags = self._help(ctx, "process", [tool, "images", "--help"])
            args = [tool, "images", "--data", str(ws.planar_dir), "--output-dir", str(ws.processed_dir),
                    "--camera-type", "perspective", "--num-downscales", "3",
                    "--colmap-model-path", model_rel]
            args += flags.boolean("--skip-colmap", True)
            args += flags.boolean("--verbose", True)
            rc = procrun.run(args, cancel=ctx.cancel, on_line=ticker.seen, log=log, env=env, cwd=str(ws.root))
            if rc != 0:
                raise StageFailed("ns-process-data failed (exit %d) - see logs/colmap.log" % rc)
            # A fresh transforms.json: the training clean-up must start again from it.
            for stale in (training_prep.ALL_TRANSFORMS, training_prep.TRAIN_PLY):
                try:
                    (ws.processed_dir / stale).unlink()
                except FileNotFoundError:
                    pass
            if masks_on:
                # Splatfacto: per-frame mask_path + masks_N matching images_N sizes.
                ticker.set(96.0, "attaching masks for training")
                rc = procrun.run([ctx.tools.ns_python, str(HELPERS / "mask.py"), "--attach", str(ws.processed_dir),
                                  "--masks", str(ws.masks_dir), "--planar", str(ws.planar_dir)],
                                 cancel=ctx.cancel, on_line=ticker.seen, log=log, env=env)
                if rc != 0:
                    raise StageFailed("Attaching masks to the training data failed (exit %d) - see logs/colmap.log" % rc)
        data = self._validate_reconstruction(ctx, total, models, rig["dir"] if rig else best[0], rig)
        data["trainingMasks"] = masks_on
        return data

    def _glomap(self, ctx: StageContext, log, sparse_dir: Path, db: Path, cuda: bool,
                ticker: Optional["_Ticker"] = None) -> bool:
        """GLOMAP global mapper on COLMAP's database (features + matches). Writes sparse/0 like COLMAP.
        It adds (empty) rigs/frames tables to the database; COLMAP 3.11 ignores them (verified:
        point_triangulator on that database + GLOMAP's model). Measured on 60 panoramas / 840 views
        of a real walk: GLOMAP 160 s, one model with 810 views, all 60 panoramas rig-consistent,
        path jerk 0.16; COLMAP mapper 340 s, 13 models, largest 323 views.
        Intrinsics stay fixed (our PINHOLE). Returns False when it failed or produced nothing (the
        caller then falls back to COLMAP's incremental mapper). Cancellation propagates."""
        glomap = ctx.tools.glomap
        gpu = "1" if cuda else "0"
        msg = {"text": "GLOMAP: starting"}
        _say(ctx, ticker, 55.0, msg["text"])
        ctx.emitter.log("solving all cameras at once with GLOMAP", stage="camera_reconstruction")

        def on_line(l: str) -> None:
            if ticker is not None:
                ticker.seen()
            low = l.lower()
            for key, pct, text in GLOMAP_PHASES:
                if key in low and ("running" in low or "-----" in l or low.strip().startswith(key)):
                    if text != msg["text"]:
                        msg["text"] = text
                        _say(ctx, ticker, pct, text)
                    return

        env = procrun.child_env(environment.glomap_path_dirs(glomap))
        t0 = time.monotonic()
        rc = procrun.run([glomap, "mapper", "--database_path", str(db), "--image_path", str(ctx.workspace.planar_dir),
                          "--output_path", str(sparse_dir), "--log_to_stderr", "1",
                          "--BundleAdjustment.optimize_intrinsics", "0",
                          "--BundleAdjustment.optimize_principal_point", "0",
                          # GLOMAP 1.2's own retriangulation aborts on a COLMAP 3.11 database ("Check
                          # failed: existing_frame.DataIds() == frame.DataIds()"); COLMAP's
                          # point_triangulator re-triangulates the joined model afterwards anyway.
                          "--skip_retriangulation", "1",
                          "--GlobalPositioning.use_gpu", gpu, "--BundleAdjustment.use_gpu", gpu],
                         cancel=ctx.cancel, on_line=on_line, log=log, env=env, cwd=str(ctx.workspace.root))
        models = _colmap_models(sparse_dir)
        log.write("pf360: GLOMAP exit %d after %s, %d model(s)\n" % (rc, _fmt_dur(time.monotonic() - t0), len(models)))
        if rc == 0 and models:
            ctx.emitter.log("GLOMAP finished in %s: %s" % (
                _fmt_dur(time.monotonic() - t0), ", ".join("model %s with %d views" % m for m in models)),
                stage="camera_reconstruction")
            return True
        return False

    def _extend(self, ctx: StageContext, env: Dict[str, str], log, colmap: str, sparse_dir: Path, db: Path,
                pr: Dict[str, Any], rig: Dict[str, Any], ticker: Optional[_Ticker] = None) -> Dict[str, Any]:
        """Grows the joined model where COLMAP's own chains broke.

        After completion the joined model has ALL view directions posed and triangulated, so views of
        the next panoramas - whatever direction they face - find 3D points to register against (COLMAP
        alone only extended the directions it happened to start with). `mapper --input_path` continues
        the model with the existing (rig-consistent) cameras fixed; the result is completed + joined
        with COLMAP's remaining models again. Repeats while it grows."""
        total = len({p.stem.rpartition("_")[0] for p in ctx.workspace.planar_dir.iterdir()
                     if p.suffix.lower() in IMG_EXTS})
        colmap_views = rig.get("registeredViews")
        rounds = 0
        for k in range(1, EXTEND_ROUNDS + 1):
            have = int(rig.get("panoramasCompleted") or 0)
            if have >= total:
                break
            ext = sparse_dir / ("extend%d" % k)
            shutil.rmtree(ext, ignore_errors=True)
            ext.mkdir(parents=True)
            msg = "extending the model (%d / %d panoramas, round %d of up to %d)" % (have, total, k, EXTEND_ROUNDS)
            _say(ctx, ticker, 91.0, msg)
            ctx.emitter.log(msg, stage="camera_reconstruction")

            def on_line(l: str, msg=msg) -> None:
                if ticker is not None:
                    ticker.seen()
                m = COLMAP_REG_RE.search(l)
                if m:
                    _say(ctx, ticker, 91.0, "%s - %s views in the model" % (msg, m.group(1)))
                elif "bundle adjustment" in l.lower():
                    _say(ctx, ticker, 91.0, "%s - bundle adjustment" % msg)

            rc = procrun.run([colmap, "mapper", "--database_path", str(db), "--image_path", str(ctx.workspace.planar_dir),
                              "--input_path", str(sparse_dir / rig["dir"]), "--output_path", str(ext),
                              "--Mapper.fix_existing_images", "1", "--Mapper.multiple_models", "0",
                              "--Mapper.init_min_tri_angle", str(INIT_MIN_TRI_ANGLE_DEG),
                              "--Mapper.ba_refine_focal_length", "0", "--Mapper.ba_refine_principal_point", "0",
                              "--Mapper.ba_refine_extra_params", "0"]
                             + mapper_ba_args(pr, bool((ctx.results.get("_env") or {}).get("cuda", True))),
                             cancel=ctx.cancel, on_line=on_line, log=log, env=env, cwd=str(ctx.workspace.root))
            model = ext if (ext / "images.bin").is_file() else ext / "0"
            if rc != 0 or not (model / "images.bin").is_file():
                ctx.emitter.warning("Extending the joined model failed (round %d, see logs/colmap.log); keeping "
                                    "what was joined so far." % k, stage="camera_reconstruction")
                break
            new = self._rig_complete(ctx, env, log, sparse_dir, db, pr, out_name="extend%d_rig" % k, base=model,
                                     ticker=ticker)
            if not new:
                break
            new["dir"] = self._triangulate(ctx, env, log, colmap, sparse_dir, new["dir"], db, ticker=ticker)
            got = int(new.get("panoramasCompleted") or 0)
            ctx.emitter.log("extension round %d: %d -> %d of %d panoramas" % (k, have, got, total),
                            stage="camera_reconstruction")
            if got <= have:
                break
            rig, rounds = new, k
            if got - have < max(2, total // 100):          # barely growing: not worth another mapper run
                break
        rig["registeredViews"] = colmap_views          # COLMAP's own placements (before completion)
        rig["extensionRounds"] = rounds
        return rig

    def _rig_complete(self, ctx: StageContext, env: Dict[str, str], log, sparse_dir: Path, db: Path,
                      pr: Dict[str, Any], out_name: str = "joined_rig",
                      base: Optional[Path] = None, ticker: Optional[_Ticker] = None) -> Optional[Dict[str, Any]]:
        """Joins COLMAP's models through shared panoramas, then poses every view of each registered
        panorama from its registered views (ns_helpers/rig_complete.py --sparse; `base` = a model that
        is always the reference). Returns the helper's stats + "dir", or None (then the largest plain
        COLMAP model is used)."""
        views = yaw_pitch_pairs(pr["imagesPerEquirect"], (pr["cropTop"], pr["cropBottom"], pr["cropLeft"],
                                                          pr["cropRight"]))
        _say(ctx, ticker, 90.5, "joining models + completing panoramas")
        lines: List[str] = []

        def on_line(l: str) -> None:
            lines.append(l)
            if ticker is not None:
                ticker.seen()

        args = [ctx.tools.ns_python, str(HELPERS / "rig_complete.py"), "--sparse", str(sparse_dir),
                "--db", str(db), "--views", json.dumps(views), "--out", str(sparse_dir / out_name)]
        if base is not None:
            args += ["--base", str(base)]
        rc = procrun.run(args, cancel=ctx.cancel, on_line=on_line, log=log, env=env)
        stats = next((json.loads(l[len("PF360RIG"):]) for l in lines if l.startswith("PF360RIG")), None)
        if rc != 0 or not stats:
            ctx.emitter.warning("Panorama completion failed (see logs/colmap.log); using COLMAP's views only.",
                                stage="camera_reconstruction")
            return None
        stats["dir"] = out_name
        if stats.get("brokenModels"):
            ctx.emitter.log("left out %d COLMAP model(s) whose camera path zig-zags (built from the downward "
                            "views only, poses not trustworthy): %s" % (len(stats["brokenModels"]),
                                                                       ", ".join(stats["brokenModels"])),
                            stage="camera_reconstruction")
        if stats.get("modelsTotal", 0) > 1:
            ctx.emitter.log("joined %d of %d COLMAP models through shared panoramas"
                            % (stats.get("modelsMerged", 0), stats["modelsTotal"]), stage="camera_reconstruction")
        return stats

    def _triangulate(self, ctx: StageContext, env: Dict[str, str], log, colmap: str, sparse_dir: Path,
                     model: str, db: Path, ticker: Optional[_Ticker] = None) -> str:
        """COLMAP point_triangulator on the joined model: camera poses stay fixed, the sparse points are
        rebuilt from ALL matches (so completed views and the seams between joined models get points).
        Returns the model to use; on failure the joined model (its points have no tracks)."""
        out_name = model + "_tri"
        out = sparse_dir / out_name
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True)
        _say(ctx, ticker, 91.0, "triangulating points (poses fixed)")
        rc = procrun.run([colmap, "point_triangulator", "--database_path", str(db),
                          "--image_path", str(ctx.workspace.planar_dir), "--input_path", str(sparse_dir / model),
                          "--output_path", str(out), "--clear_points", "1", "--refine_intrinsics", "0",
                          "--Mapper.ba_refine_focal_length", "0", "--Mapper.ba_refine_principal_point", "0",
                          "--Mapper.ba_refine_extra_params", "0"],
                         cancel=ctx.cancel, on_line=ticker.seen if ticker is not None else None, log=log, env=env,
                         cwd=str(ctx.workspace.root))
        if rc != 0 or not (out / "points3D.bin").is_file():
            ctx.emitter.warning("Re-triangulating the joined model failed (see logs/colmap.log); using the "
                                "points of COLMAP's models instead.", stage="camera_reconstruction")
            return model
        return out_name

    def _vocab_tree(self, ctx: StageContext, env: Dict[str, str]) -> str:
        """The vocabulary-tree engine pack; else Nerfstudio's cached tree (downloaded on first use)."""
        if ctx.tools.vocab_tree:
            return ctx.tools.vocab_tree
        out: List[str] = []
        rc = procrun.run([ctx.tools.ns_python, "-c",
                          "from nerfstudio.process_data.colmap_utils import get_vocab_tree; "
                          "print('PF360VOCAB' + str(get_vocab_tree()))"],
                         cancel=ctx.cancel, on_line=lambda l: out.append(l), env=env)
        path = next((l[len("PF360VOCAB"):] for l in out if l.startswith("PF360VOCAB")), "")
        if rc != 0 or not path or not Path(path).is_file():
            raise StageFailed("The COLMAP vocabulary tree is unavailable (download failed?). Use sequential or "
                              "exhaustive matching instead.")
        return path

    def _validate_reconstruction(self, ctx: StageContext, total: int,
                                 models: Optional[List] = None, used: Optional[str] = None,
                                 rig: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Camera registration checks. Exit code 0 alone is not success."""
        ws = ctx.workspace
        tf = ws.processed_dir / "transforms.json"
        if not tf.is_file():
            raise StageFailed("COLMAP did not register enough images to produce transforms.json - the footage "
                              "probably lacks texture/overlap or camera movement.")
        with open(tf, "r", encoding="utf-8") as f:
            t = json.load(f)
        frames = t.get("frames") or []
        reg = len(frames)
        if models is None:
            models = _colmap_models(ws.processed_dir / "colmap" / "sparse")
        centers = []
        for fr in frames:
            m = fr.get("transform_matrix")
            if m and len(m) >= 3:
                centers.append((float(m[0][3]), float(m[1][3]), float(m[2][3])))
        data: Dict[str, Any] = {"generatedImages": total, "registeredImages": reg,
                                "unregisteredImages": max(0, total - reg),
                                "registrationRatio": round(reg / total, 4) if total else 0.0,
                                "colmapModels": len(models),
                                "sparsePointCloud": str(ws.processed_dir / "sparse_pc.ply")
                                if (ws.processed_dir / "sparse_pc.ply").is_file() else None}
        if centers:
            lo = [min(c[i] for c in centers) for i in range(3)]
            hi = [max(c[i] for c in centers) for i in range(3)]
            extent = max(h - l for l, h in zip(lo, hi))
            data["cameraExtent"] = extent
            clusters = _cluster_count(centers, 0.02 * extent) if extent > 0 else 1
            data["distinctCameraPositions"] = clusters
        panoramas = len({p.stem.rpartition("_")[0] for p in ws.planar_dir.iterdir() if p.suffix.lower() in IMG_EXTS})
        data["panoramas"] = panoramas
        if rig:
            # COLMAP placed `registeredViews`; completion posed the rest of those panoramas.
            data["colmapRegisteredViews"] = rig.get("registeredViews")
            data["completedViews"] = rig.get("addedViews")
            data["panoramasRegistered"] = rig.get("panoramasCompleted")
            data["colmapModelsJoined"] = rig.get("modelsMerged")
            if rig.get("panoramasInconsistent"):
                ctx.emitter.warning("%d panorama(s) had inconsistent view poses and were not completed."
                                    % rig["panoramasInconsistent"], stage="camera_reconstruction")
        ratio = data["registrationRatio"]
        if reg < 8 or ratio < 0.3:
            raise StageFailed(
                "Only %d of %d perspective images could be placed by COLMAP (%.0f%%, %s of %d panoramas). The "
                "reconstruction is not usable. Try: crop the bottom (removes the person / stick that moves with the "
                "camera), mask People, use 8 views per frame, fewer frames, or walk more slowly with less turning."
                % (reg, total, 100 * ratio, data.get("panoramasRegistered", "?"), panoramas), data)
        if ratio < 0.7:
            ctx.emitter.warning("Only %.0f%% of perspective images were placed; expect gaps." % (100 * ratio),
                                stage="camera_reconstruction")
        if used is not None:
            data["colmapModelUsed"] = used
        if len(models) > 1:
            if rig and rig.get("modelsMerged"):
                placed = int(rig.get("panoramasCompleted") or 0)
                if placed < panoramas:
                    ctx.emitter.warning("COLMAP split the walk into %d pieces; after joining and extending, %d of "
                                        "%d panoramas have cameras - the rest of the walk is missing from the splat."
                                        % (len(models), placed, panoramas), stage="camera_reconstruction")
                for why in (rig.get("unmergedWhy") or [])[:8]:      # diagnostics for logs/worker.log
                    ctx.emitter.log("piece %s not joined: %s" % (why.get("model"), json.dumps(why)),
                                    stage="camera_reconstruction")
            else:
                ctx.emitter.warning("COLMAP split the scene into %d disconnected models; only the largest (%d "
                                    "images) is used. Capture with more overlap or try exhaustive matching."
                                    % (len(models), max(n for _, n in models)), stage="camera_reconstruction")
        if data.get("cameraExtent", 0.0) <= 1e-9 or data.get("distinctCameraPositions", 0) < 3:
            raise StageFailed("The registered cameras barely move (camera rotated in place). Gaussian Splats need "
                              "the camera to travel through the scene to recover depth.", data)
        if data["sparsePointCloud"] is None:
            ctx.emitter.warning("No sparse point cloud (sparse_pc.ply) was written; Splatfacto will start from "
                                "random points.", stage="camera_reconstruction")
        return data

    @staticmethod
    def _long_schedule_args(ctx: StageContext, flags, total_its: int, views: int = 0) -> List[str]:
        """training.longSchedule: Splatfacto splits/clones Gaussians only until step 15000 and decays
        the position learning rate over 30000 steps, whatever --max-num-iterations says. On a long
        run the rest of the steps only polish what exists. Stretch both over the run (split for
        the first half, like the 15k-of-30k default), so more steps also mean more Gaussians.

        With many training views the opacity reset (every 30 x 100 = 3000 steps) also has to stretch:
        a reset fades every Gaussian, and only the ones seen again before the next refinement get
        their opacity back - the rest are pruned. Seen on a 19,222-view run: ~120k of ~390k Gaussians
        pruned after every reset, so the count saw-toothed and never grew past the seed cloud. The
        reset then comes every ~2 passes over the views (as in the original 3DGS, which resets every
        3000 steps on a few hundred images), capped so at least two cycles fit before splitting stops."""
        if not ctx.manifest["training"].get("longSchedule") or total_its <= LONG_SCHEDULE_MIN_ITS:
            return []
        args: List[str] = []
        if flags.has("--pipeline.model.stop-split-at"):
            args += ["--pipeline.model.stop-split-at", str(total_its // 2)]
        if flags.has("--optimizers.means.scheduler.max-steps"):
            args += ["--optimizers.means.scheduler.max-steps", str(total_its)]
        # reset-alpha-every counts refinements (refine_every = 100 steps, Splatfacto's default).
        resets = min(-(-2 * views // REFINE_EVERY), total_its // 2 // 2 // REFINE_EVERY)
        if resets > DEFAULT_RESET_ALPHA_EVERY and flags.has("--pipeline.model.reset-alpha-every"):
            args += ["--pipeline.model.reset-alpha-every", str(resets)]
        if args:
            ctx.emitter.log("long schedule: Gaussians split until step %d, position learning rate decays over "
                            "%d steps, opacity reset every %d steps" % (
                                total_its // 2, total_its, max(resets, DEFAULT_RESET_ALPHA_EVERY) * REFINE_EVERY),
                            stage="training")
        else:
            ctx.emitter.warning("This Nerfstudio version has no stop-split-at / scheduler flags; the long "
                                "schedule option is ignored.", stage="training")
        return args

    def _training(self, ctx: StageContext) -> Dict[str, Any]:
        if ctx.manifest.get("training", {}).get("engine") == "lichtfeld":
            return lichtfeld_engine.train_lfs(ctx)
        ws, tr = ctx.workspace, ctx.manifest["training"]
        tool = ctx.tools.ns("ns-train")
        method = tr["method"]
        methods = (ctx.results.get("_env") or {}).get("methods") or []
        if methods and method not in methods:
            fb = tr.get("fallbackMethod")
            if fb and fb in methods:
                ctx.emitter.warning("%s is not available in this Nerfstudio install; using %s instead." % (method, fb),
                                    stage="training")
                method = fb
            else:
                raise StageFailed("Training method %s is not available (installed: %s)." % (method, ", ".join(methods)))
        flags = self._help(ctx, "train-" + method, [tool, method, "--help"])
        total_its = int(tr["iterations"])
        # Restore point: continue from the newest COMPLETE checkpoint of the interrupted run.
        ckpt = _latest_valid_checkpoint(ws.training_dir) if ctx.results.get("_resume_training") else None
        start = 0
        if ckpt:
            start = ckpt[0] + 1
            if start >= total_its:            # finished, but paused before the stage was recorded
                cfg = self._find_config(ctx)
                if cfg:
                    return {"method": method, "iterations": total_its, "lastLoggedStep": ckpt[0],
                            "config": str(cfg), "resumedFromStep": ckpt[0]}
            ctx.emitter.log("resuming training from checkpoint step %d / %d" % (ckpt[0], total_its), stage="training")
        else:
            ws.reset_dir(ws.training_dir)
        prep = self._prepare_training_data(ctx)
        # Nerfstudio trains max_num_iterations MORE steps after a loaded checkpoint, so pass the remainder.
        args = [tool, method, "--data", str(ws.processed_dir), "--output-dir", str(ws.training_dir),
                "--experiment-name", "pf360", "--timestamp", "run",
                "--max-num-iterations", str(total_its - start),
                "--steps-per-save", str(TRAIN_STEPS_PER_SAVE),
                "--machine.seed", str(int(tr["randomSeed"])),
                "--vis", "tensorboard"]
        if ckpt:
            args += ["--load-checkpoint", str(ckpt[1])]
        # Splatfacto's config scores EVERY held-out image every 1000 steps. Nothing in pf360 reads
        # those scores, and with many images it dominates the stage: 1,205 eval images took ~415 s
        # per pass, 39 passes = ~4.5 h of a 4.8 h run whose 40k steps took ~10 min. An interval
        # past the last step disables it (Nerfstudio's step_check never fires at step 0).
        if flags.has("--steps-per-eval-all-images"):
            args += ["--steps-per-eval-all-images", str(total_its + 1)]
        args += flags.boolean("--viewer.quit-on-train-completion", True)
        args += ["--pipeline.model.sh-degree", str(int(tr["shDegree"]))]
        args += flags.boolean("--pipeline.model.use-scale-regularization", bool(tr.get("useScaleRegularization")))
        for key, flag in (("cullAlphaThresh", "--pipeline.model.cull-alpha-thresh"),
                          ("densifyGradThresh", "--pipeline.model.densify-grad-thresh"),
                          ("cullScaleThresh", "--pipeline.model.cull-scale-thresh")):
            if tr.get(key) is not None:
                args += [flag, repr(float(tr[key]))]
        args += self._long_schedule_args(ctx, flags, total_its, int(prep.get("trainingViews") or 0))
        args += ["nerfstudio-data", "--downscale-factor", str(int(tr["downscaleFactor"]))]
        last = {"pct": -1.0, "step": 0}
        tail: List[str] = []

        def on_line(l: str) -> None:
            tail.append(l)
            del tail[:-40]
            m = TRAIN_STEP_RE.match(l)
            if m:
                # Absolute step / total (Nerfstudio's own % covers only the remaining steps after a resume).
                last["step"] = int(m.group(1))
                pct = 100.0 * last["step"] / max(1, total_its)
                if pct - last["pct"] >= 0.25:
                    last["pct"] = pct
                    ctx.emitter.progress("training", pct, "step %s / %d" % (m.group(1), int(tr["iterations"])))
            elif "out of memory" in l.lower():
                ctx.emitter.warning("CUDA out of memory: lower the training resolution (downscale factor), lower the "
                                    "seed point cap, use fewer frames, choose standard Splatfacto, or close other "
                                    "GPU applications.",
                                    stage="training")
        # Run through ns_helpers/train_resumable.py (= ns-train + graceful pause). On pause/cancel
        # the trainer is asked to save a checkpoint and stop; it is killed only if that takes
        # longer than TRAIN_PAUSE_GRACE_S.
        pause_file = ws.root / ".pause-training"
        args = [ctx.tools.ns_python, str(HELPERS / "train_resumable.py"), str(pause_file)] + args[1:]
        own = procrun.CancelToken()
        finished = threading.Event()

        def watch_pause() -> None:
            while not finished.is_set():
                if ctx.cancel.wait(0.5):
                    if finished.is_set():
                        return
                    try:
                        pause_file.write_text("pause", encoding="utf-8")
                    except OSError:
                        own.cancel()
                        return
                    ctx.emitter.progress("training", None, "pausing - saving a checkpoint")
                    if not finished.wait(TRAIN_PAUSE_GRACE_S):
                        own.cancel()
                    return

        watcher = threading.Thread(target=watch_pause, daemon=True)
        watcher.start()
        try:
            with ws.open_log("training.log") as log:
                rc = procrun.run(args, cancel=own, on_line=on_line, log=log, env=self._env(ctx), cwd=str(ws.root))
        finally:
            finished.set()
            watcher.join(timeout=2)
            try:
                pause_file.unlink()
            except OSError:
                pass
        if ctx.cancel.cancelled:
            raise procrun.Cancelled()          # paused: the checkpoint is the restore point
        cfg = self._find_config(ctx)
        if rc != 0 or cfg is None:
            hint = " | ".join(tail[-3:])
            raise StageFailed("ns-train failed (exit %d) - see logs/training.log. Last output: %s" % (rc, hint[-400:]))
        out = {"method": method, "iterations": total_its, "lastLoggedStep": last["step"], "config": str(cfg)}
        if ckpt:
            out["resumedFromStep"] = ckpt[0]
        out.update({k: prep[k] for k in ("seedPoints", "seedPointsUsed", "droppedPanoramas", "droppedViews",
                                         "trainingViews", "viewBudget", "panoramasUsed") if k in prep})
        return out

    def _prepare_training_data(self, ctx: StageContext) -> Dict[str, Any]:
        """Seed-cloud cap + misplaced-panorama filter (training_prep). Deterministic, so it gives
        the same data again when training resumes from a checkpoint."""
        tr = ctx.manifest["training"]
        cap = int(tr.get("seedPointCap", training_prep.DEFAULT_SEED_POINT_CAP))
        drop = bool(tr.get("dropMisplacedPanoramas", True))
        ctx.emitter.progress("training", None, "preparing training data")
        try:
            stats = training_prep.prepare(ctx.workspace.processed_dir, ctx.workspace.planar_dir, seed_cap=cap,
                                          drop_misplaced=drop, max_views=int(tr.get("maxTrainingViews", 0)),
                                          downscale=int(tr["downscaleFactor"]), seed=int(tr["randomSeed"]),
                                          log=lambda m: ctx.emitter.log(m, stage="training"))
        except (OSError, ValueError) as e:
            # Training can still run on the camera stage's data, as before this step existed.
            ctx.emitter.warning("Training-data clean-up skipped (%s); training on the full camera set." % e,
                                stage="training")
            return {}
        if stats.get("viewBudget"):
            ctx.emitter.warning("Training uses %d of the views: all of them would not fit in RAM (Splatfacto keeps "
                                "every image decoded in memory)." % stats["trainingViews"], stage="training")
        if stats.get("droppedPanoramas"):
            ctx.emitter.warning("%d panorama(s) had cameras placed inconsistently and were left out of training."
                                % stats["droppedPanoramas"], stage="training")
        return stats

    def _find_config(self, ctx: StageContext) -> Optional[Path]:
        root = ctx.workspace.training_dir
        if not root.is_dir():
            return None
        cands = sorted(root.rglob("config.yml"), key=lambda p: p.stat().st_mtime, reverse=True)
        for c in cands:
            if (c.parent / "nerfstudio_models").is_dir() and any((c.parent / "nerfstudio_models").glob("*.ckpt")):
                return c
        return None

    def _export(self, ctx: StageContext) -> Dict[str, Any]:
        if ctx.manifest.get("training", {}).get("engine") == "lichtfeld":
            return lichtfeld_engine.export_lfs(ctx)
        ws = ctx.workspace
        cfg = self._find_config(ctx)
        if cfg is None:
            raise StageFailed("No trained model checkpoint found to export.")
        tool = ctx.tools.ns("ns-export")
        ws.export_dir.mkdir(parents=True, exist_ok=True)
        target = ws.export_ply
        before = {p: p.stat().st_mtime for p in ws.export_dir.glob("*.ply")}
        args = [tool, "gaussian-splat", "--load-config", str(cfg), "--output-dir", str(ws.export_dir),
                "--output-filename", target.name]
        ctx.emitter.progress("export", None, "exporting Gaussian Splat PLY")
        with ws.open_log("export.log") as log:
            rc = procrun.run(args, cancel=ctx.cancel, log=log, env=self._env(ctx), cwd=str(ws.root))
        if rc != 0:
            raise StageFailed("ns-export failed (exit %d) - see logs/export.log" % rc)
        # Resolve the produced file dynamically: the requested name, else the newest new/changed .ply.
        if not target.is_file():
            changed = [p for p in ws.export_dir.glob("*.ply") if before.get(p) != p.stat().st_mtime]
            if not changed:
                raise StageFailed("ns-export reported success but wrote no .ply in %s" % ws.export_dir)
            newest = max(changed, key=lambda p: p.stat().st_mtime)
            newest.replace(target)
        # Write sidecar for Nerfstudio output
        sidecar = {"engine": "nerfstudio", "frame": "ns_dataparser"}
        try:
            with open(str(target) + ".pf360.json", "w", encoding="utf-8") as sf:
                json.dump(sidecar, sf, indent=2)
        except OSError:
            pass
        return {"path": str(target), "fileSizeBytes": target.stat().st_size}

    def result_ply(self, ctx: StageContext) -> Optional[Path]:
        return ctx.workspace.export_ply if ctx.workspace.export_ply.is_file() else None


def _cluster_count(points: List[tuple], eps: float) -> int:
    """Greedy count of positions further than eps apart (perspective views of one frame collapse)."""
    reps: List[tuple] = []
    for p in points:
        for r in reps:
            if math.dist(p, r) <= eps:
                break
        else:
            reps.append(p)
            if len(reps) > 2000:
                break
    return len(reps)


def estimate(width: int, height: int, frames: int, images_per_equirect: int, crop: Sequence[float]) -> Dict[str, Any]:
    vpf = views_per_frame(images_per_equirect, crop)
    return {"viewsPerFrame": vpf, "perspectiveImages": frames * vpf,
            "perspectiveSize": perspective_size(width, height, images_per_equirect)}

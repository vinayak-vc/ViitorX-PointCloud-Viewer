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
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from . import environment, procrun
from .backend import ReconstructionBackend, StageContext, StageFailed
from .views import perspective_size, views_per_frame

IMG_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
TRAIN_STEP_RE = re.compile(r"^\s*(\d+)\s+\((\d+(?:\.\d+)?)%\)")
COLMAP_FILE_RE = re.compile(r"Processed file \[(\d+)/(\d+)\]")
COLMAP_MATCH_RE = re.compile(r"Matching (?:block|image) \[(\d+)/(\d+)")
COLMAP_REG_RE = re.compile(r"Registering image #\d+ \((\d+)\)")
HELPERS = Path(__file__).with_name("ns_helpers")
# Horizontal FOV of Nerfstudio's planar views (equirect_utils, 1.1.5): 8 -> 120, 14 -> 110.
PLANAR_FOV_DEG = {8: 120, 14: 110}


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
        if _masking_on(manifest) and not tools.mask_model:
            p.append("Masking is enabled but the segmentation model is not installed "
                     "(run scripts\\pf360\\setup_masking.ps1, ~0.5 GB, or turn masking off).")
        if not tools.ns_python:
            p.append("No Nerfstudio Python environment found (run `python -m pf360 doctor`).")
        for t in ("ns-process-data", "ns-train", "ns-export"):
            if tools.ns_python and not tools.ns(t):
                p.append("%s not found next to %s" % (t, tools.ns_python))
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
            return {"matching": pr["matchingMethod"], "colmap": ctx.tools.colmap, "numDownscales": 3,
                    "sfm": "pf360-fixed-pinhole-v1", "masks": _masking_on(ctx.manifest)}
        if stage == "training":
            keys = ("method", "iterations", "downscaleFactor", "shDegree", "randomSeed", "useScaleRegularization",
                    "cullAlphaThresh", "densifyGradThresh", "cullScaleThresh")
            return {k: tr.get(k) for k in keys}
        if stage == "export":
            return {"plyFileName": ctx.manifest["output"].get("plyFileName")}
        return {}

    def stage_outputs_exist(self, stage: str, ctx: StageContext) -> bool:
        ws = ctx.workspace
        if stage == "panorama_projection":
            return _count_images(ws.planar_dir) > 0
        if stage == "masking":
            n = _count_images(ws.planar_dir)
            return n > 0 and ws.masks_dir.is_dir() and sum(1 for _ in ws.masks_dir.glob("*.png")) == n
        if stage == "camera_reconstruction":
            return (ws.processed_dir / "transforms.json").is_file()
        if stage == "training":
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
        ws.reset_dir(ws.planar_dir)
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
            if l.startswith("PF360PLANAR"):
                result.update(json.loads(l[len("PF360PLANAR"):]))
        try:
            with ws.open_log("panorama.log") as log:
                rc = procrun.run([ctx.tools.ns_python, str(HELPERS / "planar.py"), str(ws.frames_dir),
                                  str(pr["imagesPerEquirect"])] + [repr(float(c)) for c in crop],
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
        ws.reset_dir(ws.masks_dir)
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

        ws.reset_dir(ws.processed_dir)
        colmap_dir = ws.processed_dir / "colmap"
        sparse_dir = colmap_dir / "sparse"
        sparse_dir.mkdir(parents=True, exist_ok=True)
        db = colmap_dir / "database.db"
        cuda = bool((ctx.results.get("_env") or {}).get("cuda", True))
        gpu = "1" if cuda else "0"
        env = self._env(ctx)
        phase = {"name": "feature extraction", "pct": 0.0}

        def on_line(l: str) -> None:
            m = COLMAP_FILE_RE.search(l)
            if m:
                phase.update(name="feature extraction", pct=25.0 * int(m.group(1)) / max(1, int(m.group(2))))
            m2 = COLMAP_MATCH_RE.search(l)
            if m2:
                phase.update(name="feature matching", pct=25.0 + 30.0 * int(m2.group(1)) / max(1, int(m2.group(2))))
            m3 = COLMAP_REG_RE.search(l)
            if m3:
                phase.update(name="registering images",
                             pct=55.0 + 35.0 * min(1.0, int(m3.group(1)) / max(1, total)))
            if m or m2 or m3:
                ctx.emitter.progress("camera_reconstruction", phase["pct"], phase["name"])
            elif "bundle adjustment" in l.lower():
                ctx.emitter.progress("camera_reconstruction", None, "bundle adjustment")

        method = pr["matchingMethod"]
        matcher = [colmap, "%s_matcher" % method, "--database_path", str(db), "--SiftMatching.use_gpu", gpu]
        if method == "sequential":
            # Images are ordered panorama-major (frame_N_0..V-1): cover ~3 panoramas.
            matcher += ["--SequentialMatching.overlap", str(max(10, 3 * vpf))]
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
        steps = [
            ("feature extraction", extract),
            ("feature matching", matcher),
            ("registering images",
             [colmap, "mapper", "--database_path", str(db), "--image_path", str(ws.planar_dir),
              "--output_path", str(sparse_dir),
              "--Mapper.ba_refine_focal_length", "0", "--Mapper.ba_refine_principal_point", "0",
              "--Mapper.ba_refine_extra_params", "0", "--Mapper.ba_global_function_tolerance=1e-6"]),
        ]
        with ws.open_log("colmap.log") as log:
            log.write("pf360: PINHOLE %dx%d fov=%d -> fx=fy=%.3f cx=%.3f cy=%.3f (fixed)\n"
                      % (w, h, fov, intr[0], intr[2], intr[3]))
            for name, args in steps:
                ctx.emitter.progress("camera_reconstruction", phase["pct"], name)
                rc = procrun.run(args, cancel=ctx.cancel, on_line=on_line, log=log, env=env, cwd=str(ws.root))
                if rc != 0:
                    raise StageFailed("COLMAP %s failed (exit %d) - see logs/colmap.log" % (name, rc))
            models = _colmap_models(sparse_dir)
            if not models:
                raise StageFailed("COLMAP could not reconstruct any cameras - the footage probably lacks "
                                  "texture/overlap or camera movement.",
                                  {"generatedImages": total, "registeredImages": 0})
            best = max(models, key=lambda kv: kv[1])
            ctx.emitter.progress("camera_reconstruction", 92.0, "writing transforms.json")
            flags = self._help(ctx, "process", [tool, "images", "--help"])
            args = [tool, "images", "--data", str(ws.planar_dir), "--output-dir", str(ws.processed_dir),
                    "--camera-type", "perspective", "--num-downscales", "3",
                    "--colmap-model-path", "colmap/sparse/%s" % best[0]]
            args += flags.boolean("--skip-colmap", True)
            args += flags.boolean("--verbose", True)
            rc = procrun.run(args, cancel=ctx.cancel, log=log, env=env, cwd=str(ws.root))
            if rc != 0:
                raise StageFailed("ns-process-data failed (exit %d) - see logs/colmap.log" % rc)
            if masks_on:
                # Splatfacto: per-frame mask_path + masks_N matching images_N sizes.
                ctx.emitter.progress("camera_reconstruction", 96.0, "attaching masks for training")
                rc = procrun.run([ctx.tools.ns_python, str(HELPERS / "mask.py"), "--attach", str(ws.processed_dir),
                                  "--masks", str(ws.masks_dir), "--planar", str(ws.planar_dir)],
                                 cancel=ctx.cancel, log=log, env=env)
                if rc != 0:
                    raise StageFailed("Attaching masks to the training data failed (exit %d) - see logs/colmap.log" % rc)
        data = self._validate_reconstruction(ctx, total, models, best[0])
        data["trainingMasks"] = masks_on
        return data

    def _vocab_tree(self, ctx: StageContext, env: Dict[str, str]) -> str:
        """Nerfstudio's cached vocabulary tree (downloaded by Nerfstudio on first use)."""
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
                                 models: Optional[List] = None, used: Optional[str] = None) -> Dict[str, Any]:
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
        ratio = data["registrationRatio"]
        if reg < 8 or ratio < 0.3:
            raise StageFailed("Only %d of %d perspective images were registered by COLMAP (%.0f%%). The "
                              "reconstruction is not usable." % (reg, total, 100 * ratio), data)
        if ratio < 0.7:
            ctx.emitter.warning("Only %.0f%% of perspective images were registered; expect gaps." % (100 * ratio),
                                stage="camera_reconstruction")
        if used is not None:
            data["colmapModelUsed"] = used
        if len(models) > 1:
            ctx.emitter.warning("COLMAP split the scene into %d disconnected models; only the largest (%d images) "
                                "is used. Capture with more overlap or try exhaustive matching."
                                % (len(models), max(n for _, n in models)), stage="camera_reconstruction")
        if data.get("cameraExtent", 0.0) <= 1e-9 or data.get("distinctCameraPositions", 0) < 3:
            raise StageFailed("The registered cameras barely move (camera rotated in place). Gaussian Splats need "
                              "the camera to travel through the scene to recover depth.", data)
        if data["sparsePointCloud"] is None:
            ctx.emitter.warning("No sparse point cloud (sparse_pc.ply) was written; Splatfacto will start from "
                                "random points.", stage="camera_reconstruction")
        return data

    def _training(self, ctx: StageContext) -> Dict[str, Any]:
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
        ws.reset_dir(ws.training_dir)
        args = [tool, method, "--data", str(ws.processed_dir), "--output-dir", str(ws.training_dir),
                "--experiment-name", "pf360", "--timestamp", "run",
                "--max-num-iterations", str(int(tr["iterations"])),
                "--machine.seed", str(int(tr["randomSeed"])),
                "--vis", "tensorboard"]
        args += flags.boolean("--viewer.quit-on-train-completion", True)
        args += ["--pipeline.model.sh-degree", str(int(tr["shDegree"]))]
        args += flags.boolean("--pipeline.model.use-scale-regularization", bool(tr.get("useScaleRegularization")))
        for key, flag in (("cullAlphaThresh", "--pipeline.model.cull-alpha-thresh"),
                          ("densifyGradThresh", "--pipeline.model.densify-grad-thresh"),
                          ("cullScaleThresh", "--pipeline.model.cull-scale-thresh")):
            if tr.get(key) is not None:
                args += [flag, repr(float(tr[key]))]
        args += ["nerfstudio-data", "--downscale-factor", str(int(tr["downscaleFactor"]))]
        last = {"pct": -1.0, "step": 0}
        tail: List[str] = []

        def on_line(l: str) -> None:
            tail.append(l)
            del tail[:-40]
            m = TRAIN_STEP_RE.match(l)
            if m:
                pct = float(m.group(2))
                last["step"] = int(m.group(1))
                if pct - last["pct"] >= 0.25:
                    last["pct"] = pct
                    ctx.emitter.progress("training", pct, "step %s / %d" % (m.group(1), int(tr["iterations"])))
            elif "out of memory" in l.lower():
                ctx.emitter.warning("CUDA out of memory: lower the training resolution (downscale factor), use fewer "
                                    "frames, choose standard Splatfacto, or close other GPU applications.",
                                    stage="training")
        with ws.open_log("training.log") as log:
            rc = procrun.run(args, cancel=ctx.cancel, on_line=on_line, log=log, env=self._env(ctx), cwd=str(ws.root))
        cfg = self._find_config(ctx)
        if rc != 0 or cfg is None:
            hint = " | ".join(tail[-3:])
            raise StageFailed("ns-train failed (exit %d) - see logs/training.log. Last output: %s" % (rc, hint[-400:]))
        return {"method": method, "iterations": int(tr["iterations"]), "lastLoggedStep": last["step"],
                "config": str(cfg)}

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

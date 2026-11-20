"""Job manifest: loading, preset resolution and validation.

The manifest is written by the host (PointForge) or by hand, and is the ONLY
input of `python -m pf360 run`. It never carries commands: executables are
resolved by `environment.py` from explicit paths (validated to be existing
files) or from PATH, and arguments are built from typed, range-checked fields.
"""
from __future__ import annotations

import copy
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import MANIFEST_SCHEMA_VERSION
from .views import views_per_frame

PRESETS_PATH = Path(__file__).with_name("presets.json")
SUPPORTED_CONTAINERS = (".mp4", ".mov", ".mkv")
JOB_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,80}$")
PLY_NAME_RE = re.compile(r"^[^<>:\"/\\|?*\x00-\x1f]{1,120}\.ply$", re.IGNORECASE)
STOP_AFTER = ("frame_extraction", "panorama_projection", "camera_reconstruction", "training", "export")


class ManifestError(ValueError):
    """Raised with a list of human-readable problems."""

    def __init__(self, problems: List[str]):
        super().__init__("; ".join(problems))
        self.problems = problems


def load_presets() -> Dict[str, Any]:
    with open(PRESETS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def frames_for_window(fps_target: float, window_seconds: float, lo: int, hi: int) -> int:
    """Target frame count for a preset: framesPerSecond * window, clamped."""
    if window_seconds <= 0:
        return lo
    return int(max(lo, min(hi, round(fps_target * window_seconds))))


def _num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


def _in(problems: List[str], name: str, v: Any, lo: float, hi: float, integer: bool = False) -> None:
    if not _num(v) or (integer and float(v) != int(v)):
        problems.append("%s must be a%s number" % (name, "n integer" if integer else ""))
    elif not (lo <= float(v) <= hi):
        problems.append("%s must be in [%s, %s] (got %s)" % (name, lo, hi, v))


def resolve(raw: Dict[str, Any], video_duration: Optional[float] = None) -> Dict[str, Any]:
    """Fills defaults from the chosen quality preset. Explicit fields win.

    `training.qualityPreset` = preview|balanced|high selects defaults;
    "custom" takes defaults from "balanced" and expects explicit overrides.
    """
    m = copy.deepcopy(raw)
    presets = load_presets()["presets"]
    tr = m.setdefault("training", {})
    pr = m.setdefault("processing", {})
    m.setdefault("input", {})
    m.setdefault("output", {})
    m.setdefault("environment", {})
    m.setdefault("options", {})
    preset_name = tr.get("qualityPreset", "balanced")
    base = presets.get(preset_name if preset_name in presets else "balanced", presets["balanced"])

    tr.setdefault("method", base["method"])
    tr.setdefault("iterations", base["iterations"])
    tr.setdefault("downscaleFactor", base["downscaleFactor"])
    tr.setdefault("shDegree", base["shDegree"])
    tr.setdefault("useScaleRegularization", base["useScaleRegularization"])
    tr.setdefault("randomSeed", 42)
    if "fallbackMethod" in base:
        tr.setdefault("fallbackMethod", base["fallbackMethod"])
    pr.setdefault("imagesPerEquirect", base["imagesPerEquirect"])
    pr.setdefault("matchingMethod", base["matchingMethod"])
    for k in ("cropTop", "cropBottom", "cropLeft", "cropRight"):
        pr.setdefault(k, 0.0)
    if "targetFrames" not in pr and video_duration:
        start = float(m["input"].get("startTimeSeconds") or 0.0)
        end = m["input"].get("endTimeSeconds")
        end = float(end) if end is not None else float(video_duration)
        pr["targetFrames"] = frames_for_window(base["framesPerSecond"], end - start, base["minFrames"], base["maxFrames"])
    m["output"].setdefault("format", "ply")
    m["output"].setdefault("keepIntermediate", True)
    m["environment"].setdefault("backend", "native")
    m["options"].setdefault("allowCache", True)
    return m


def validate(m: Dict[str, Any], check_files: bool = True) -> Dict[str, Any]:
    """Validates a resolved manifest; raises ManifestError listing every problem."""
    problems: List[str] = []
    limits = load_presets()["limits"]

    if m.get("schemaVersion") != MANIFEST_SCHEMA_VERSION:
        problems.append("schemaVersion must be %d" % MANIFEST_SCHEMA_VERSION)
    if m.get("jobType", "360_video_to_gaussian_splat") != "360_video_to_gaussian_splat":
        problems.append("jobType must be 360_video_to_gaussian_splat")
    job_id = m.get("jobId")
    if not isinstance(job_id, str) or not JOB_ID_RE.match(job_id):
        problems.append("jobId must be 1-80 chars of [A-Za-z0-9_.-]")

    inp = m.get("input") or {}
    video = inp.get("videoPath")
    if not isinstance(video, str) or not video:
        problems.append("input.videoPath is required")
    else:
        if Path(video).suffix.lower() not in SUPPORTED_CONTAINERS:
            problems.append("input.videoPath must be one of %s" % ", ".join(SUPPORTED_CONTAINERS))
        if check_files and not os.path.isfile(video):
            problems.append("input.videoPath does not exist: %s" % video)
    if inp.get("projection", "equirectangular") != "equirectangular":
        problems.append("input.projection: only 'equirectangular' (stitched) video is supported in this version")
    start = inp.get("startTimeSeconds", 0) or 0
    end = inp.get("endTimeSeconds")
    if not _num(start) or float(start) < 0:
        problems.append("input.startTimeSeconds must be >= 0")
    if end is not None and (not _num(end) or float(end) <= float(start if _num(start) else 0)):
        problems.append("input.endTimeSeconds must be > startTimeSeconds (or null for end of video)")

    pr = m.get("processing") or {}
    lo, hi = limits["targetFrames"]
    _in(problems, "processing.targetFrames", pr.get("targetFrames"), lo, hi, integer=True)
    if pr.get("imagesPerEquirect") not in limits["imagesPerEquirect"]:
        problems.append("processing.imagesPerEquirect must be 8 or 14")
    clo, chi = limits["crop"]
    for k in ("cropTop", "cropBottom", "cropLeft", "cropRight"):
        _in(problems, "processing." + k, pr.get(k, 0.0), clo, chi)
    if pr.get("matchingMethod") not in limits["matchingMethod"]:
        problems.append("processing.matchingMethod must be one of %s" % limits["matchingMethod"])
    if pr.get("imagesPerEquirect") in (8, 14) and all(_num(pr.get(k, 0.0)) for k in ("cropTop", "cropBottom", "cropLeft", "cropRight")):
        try:
            n = views_per_frame(pr["imagesPerEquirect"], (pr.get("cropTop", 0.0), pr.get("cropBottom", 0.0),
                                                          pr.get("cropLeft", 0.0), pr.get("cropRight", 0.0)))
            if n == 0:
                problems.append("crop settings remove every perspective view")
        except ValueError as e:
            problems.append(str(e))

    tr = m.get("training") or {}
    if tr.get("method") not in limits["method"]:
        problems.append("training.method must be one of %s" % limits["method"])
    _in(problems, "training.iterations", tr.get("iterations"), *limits["iterations"], integer=True)
    if tr.get("downscaleFactor") not in limits["downscaleFactor"]:
        problems.append("training.downscaleFactor must be one of %s" % limits["downscaleFactor"])
    _in(problems, "training.shDegree", tr.get("shDegree"), *limits["shDegree"], integer=True)
    _in(problems, "training.randomSeed", tr.get("randomSeed"), 0, 2 ** 31 - 1, integer=True)
    for k in ("cullAlphaThresh", "densifyGradThresh", "cullScaleThresh"):
        if tr.get(k) is not None:
            _in(problems, "training." + k, tr.get(k), *limits[k])
    if not isinstance(tr.get("useScaleRegularization", False), bool):
        problems.append("training.useScaleRegularization must be true/false")

    out = m.get("output") or {}
    d = out.get("directory")
    if not isinstance(d, str) or not d:
        problems.append("output.directory is required")
    elif check_files:
        p = Path(d)
        parent = p if p.exists() else p.parent
        if p.exists() and not p.is_dir():
            problems.append("output.directory exists and is not a directory")
        elif not parent.exists():
            problems.append("output.directory parent does not exist: %s" % parent)
    name = out.get("plyFileName")
    if name is not None and (not isinstance(name, str) or not PLY_NAME_RE.match(name)):
        problems.append("output.plyFileName must be a plain file name ending in .ply")
    if out.get("format", "ply") != "ply":
        problems.append("output.format must be 'ply'")

    env = m.get("environment") or {}
    if env.get("backend", "native") != "native":
        problems.append("environment.backend: only 'native' is implemented in this version")
    for k in ("nerfstudioPython", "colmapExe", "ffmpegExe", "ffprobeExe"):
        v = env.get(k)
        if v not in (None, "") and (not isinstance(v, str) or (check_files and not os.path.isfile(v))):
            problems.append("environment.%s is not an existing file: %s" % (k, v))

    opts = m.get("options") or {}
    sa = opts.get("stopAfter")
    if sa is not None and sa not in STOP_AFTER:
        problems.append("options.stopAfter must be one of %s" % (STOP_AFTER,))

    if problems:
        raise ManifestError(problems)
    return m


def load(path: str, video_duration: Optional[float] = None, check_files: bool = True) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ManifestError(["manifest root must be a JSON object"])
    return validate(resolve(raw, video_duration), check_files=check_files)

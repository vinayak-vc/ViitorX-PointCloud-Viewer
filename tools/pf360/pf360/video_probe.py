"""Video inspection: ffprobe metadata, projection classification, quality screen.

Projection classification never relies on aspect ratio alone:
  * Spherical metadata (Google spatial-media / ffprobe "Spherical Mapping"
    side data) is authoritative.
  * Known raw-camera containers (.insv, .360) and multi-video-stream files are
    rejected as unstitched.
  * Without metadata, a 2:1 frame is only *possibly* equirectangular: a sampled
    frame is checked for the black corners typical of side-by-side dual
    fisheye, and the result is marked `requiresConfirmation`.
"""
from __future__ import annotations

import json
import math
import os
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from . import procrun

RAW_360_EXTS = {".insv": "Insta360 raw (.insv)", ".360": "GoPro MAX raw (.360, equi-angular cubemap)"}


def _fps(rate: Optional[str]) -> Optional[float]:
    if not rate or rate in ("0/0", "0"):
        return None
    try:
        v = float(Fraction(rate))
        return v if v > 0 else None
    except (ValueError, ZeroDivisionError):
        return None


def run_ffprobe(ffprobe: str, video: str) -> Dict[str, Any]:
    cp = procrun.capture([ffprobe, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", video],
                         timeout=120)
    if cp.returncode != 0:
        raise RuntimeError("ffprobe could not read the file: " + cp.stderr.decode("utf-8", "replace").strip()[-300:])
    return json.loads(cp.stdout.decode("utf-8", "replace") or "{}")


def parse_probe(data: Dict[str, Any], path: str) -> Dict[str, Any]:
    """Turns raw ffprobe JSON into the metadata block shown to the user."""
    streams = data.get("streams") or []
    fmt = data.get("format") or {}
    videos = [s for s in streams if s.get("codec_type") == "video" and not (s.get("disposition") or {}).get("attached_pic")]
    audio = [s for s in streams if s.get("codec_type") == "audio"]
    info: Dict[str, Any] = {
        "path": path,
        "fileName": os.path.basename(path),
        "fileSizeBytes": int(fmt.get("size") or 0) or (os.path.getsize(path) if os.path.isfile(path) else 0),
        "container": fmt.get("format_name"),
        "videoStreams": len(videos),
        "hasAudio": bool(audio),
        "readable": bool(videos),
    }
    if not videos:
        return info
    v = videos[0]
    w, h = int(v.get("width") or 0), int(v.get("height") or 0)
    # Honour display rotation (phones / some cameras store rotated frames).
    rot = 0
    for sd in v.get("side_data_list") or []:
        if "rotation" in sd:
            try:
                rot = int(sd["rotation"])
            except (TypeError, ValueError):
                pass
    if abs(rot) in (90, 270):
        w, h = h, w
    fps = _fps(v.get("avg_frame_rate")) or _fps(v.get("r_frame_rate"))
    dur = None
    for src in (v.get("duration"), fmt.get("duration")):
        try:
            if src is not None and float(src) > 0:
                dur = float(src)
                break
        except ValueError:
            pass
    nb = v.get("nb_frames")
    frames = int(nb) if nb and str(nb).isdigit() and int(nb) > 0 else (int(round(dur * fps)) if dur and fps else None)
    spherical = None
    for sd in v.get("side_data_list") or []:
        if "spherical" in str(sd.get("side_data_type", "")).lower():
            spherical = {k: sd[k] for k in sd if k != "side_data_type"}
    tags = {str(k).lower(): v2 for k, v2 in (v.get("tags") or {}).items()}
    info.update({
        "width": w, "height": h, "rotation": rot,
        "fps": round(fps, 3) if fps else None,
        "durationSeconds": round(dur, 3) if dur else None,
        "frameCount": frames,
        "codec": v.get("codec_name"),
        "codecLongName": v.get("codec_long_name"),
        "pixelFormat": v.get("pix_fmt"),
        "aspectRatio": round(w / h, 4) if w and h else None,
        "sphericalMetadata": spherical,
        "stereoMode": tags.get("stereo_mode"),
    })
    return info


def classify_projection(info: Dict[str, Any], corner_stats: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """Returns {projection, confidence, requiresConfirmation, supported, reasons[]}."""
    reasons: List[str] = []
    ext = Path(info.get("path", "")).suffix.lower()
    if ext in RAW_360_EXTS:
        return {"projection": "raw_camera", "confidence": "container", "requiresConfirmation": False,
                "supported": False,
                "reasons": ["%s files are unstitched camera originals. Export a stitched equirectangular MP4 "
                            "from the camera vendor's software first." % RAW_360_EXTS[ext]]}
    if info.get("videoStreams", 0) > 1:
        return {"projection": "multi_stream", "confidence": "container", "requiresConfirmation": False,
                "supported": False,
                "reasons": ["The file has %d video streams (typical of raw dual-lens 360 recordings). Export a "
                            "stitched equirectangular video first." % info["videoStreams"]]}
    sph = info.get("sphericalMetadata")
    if sph:
        proj = str(sph.get("projection", "")).lower()
        if proj == "equirectangular":
            stereo = str(info.get("stereoMode") or sph.get("stereo_mode") or "").lower()
            if stereo and stereo not in ("mono", "monoscopic", "none", "2d"):
                return {"projection": "stereo_equirectangular", "confidence": "metadata", "requiresConfirmation": False,
                        "supported": False,
                        "reasons": ["Stereoscopic (%s) 360 video is not supported; export a monoscopic version." % stereo]}
            return {"projection": "equirectangular", "confidence": "metadata", "requiresConfirmation": False,
                    "supported": True, "reasons": ["Spherical metadata declares an equirectangular projection."]}
        if proj:
            return {"projection": proj, "confidence": "metadata", "requiresConfirmation": False, "supported": False,
                    "reasons": ["Spherical metadata declares a '%s' projection; only equirectangular is supported. "
                                "Re-export as equirectangular." % proj]}
    ar = info.get("aspectRatio")
    if not ar:
        return {"projection": "unknown", "confidence": "none", "requiresConfirmation": True, "supported": False,
                "reasons": ["Could not read frame dimensions."]}
    if abs(ar - 2.0) > 0.02:
        return {"projection": "not_equirectangular", "confidence": "heuristic", "requiresConfirmation": False,
                "supported": False,
                "reasons": ["Frame aspect ratio is %.3f:1. Stitched equirectangular video is 2:1." % ar]}
    if corner_stats:
        dark = corner_stats.get("cornerMean", 255.0)
        mid = corner_stats.get("centerMean", 0.0)
        seam = corner_stats.get("seamMean", 255.0)
        if dark < 12.0 and mid > 3.0 * max(dark, 1.0) and seam < 20.0:
            return {"projection": "dual_fisheye", "confidence": "heuristic", "requiresConfirmation": False,
                    "supported": False,
                    "reasons": ["The frame shows black corners and a dark centre seam typical of side-by-side dual "
                                "fisheye footage. Stitch it to equirectangular in the camera software first."]}
    reasons.append("2:1 frame without spherical metadata - probably equirectangular, but this cannot be "
                   "proven from the file. Please confirm the projection.")
    return {"projection": "likely_equirectangular", "confidence": "heuristic", "requiresConfirmation": True,
            "supported": True, "reasons": reasons}


# ---------------------------------------------------------------------------
# frame sampling helpers (raw grayscale frames via ffmpeg pipe; no numpy)
# ---------------------------------------------------------------------------
SAMPLE_W, SAMPLE_H = 256, 128


def sample_gray_frames(ffmpeg: str, video: str, times: Sequence[float], w: int = SAMPLE_W, h: int = SAMPLE_H,
                       timeout: float = 60.0) -> List[Optional[bytes]]:
    out: List[Optional[bytes]] = []
    for t in times:
        try:
            cp = procrun.capture([ffmpeg, "-v", "error", "-ss", "%.3f" % max(0.0, t), "-i", video, "-frames:v", "1",
                                  "-vf", "scale=%d:%d,format=gray" % (w, h), "-f", "rawvideo", "-"], timeout=timeout)
            data = cp.stdout
            out.append(data if cp.returncode == 0 and len(data) == w * h else None)
        except Exception:
            out.append(None)
    return out


def corner_stats(frame: bytes, w: int = SAMPLE_W, h: int = SAMPLE_H) -> Dict[str, float]:
    """Mean luma of the 4 corners, the two half-frame centres, and the vertical centre seam."""
    def mean(x0: int, y0: int, x1: int, y1: int) -> float:
        s = n = 0
        for y in range(y0, y1):
            row = y * w
            for x in range(x0, x1):
                s += frame[row + x]
                n += 1
        return s / n if n else 0.0
    cw, ch = w // 16, h // 8
    corners = [mean(0, 0, cw, ch), mean(w - cw, 0, w, ch), mean(0, h - ch, cw, h), mean(w - cw, h - ch, w, h)]
    centers = [mean(w // 4 - cw, h // 2 - ch, w // 4 + cw, h // 2 + ch),
               mean(3 * w // 4 - cw, h // 2 - ch, 3 * w // 4 + cw, h // 2 + ch)]
    seam = (mean(w // 2 - 2, 0, w // 2 + 2, ch) + mean(w // 2 - 2, h - ch, w // 2 + 2, h)) / 2
    return {"cornerMean": sum(corners) / 4, "centerMean": sum(centers) / 2, "seamMean": seam}


def frame_metrics(frame: bytes, w: int = SAMPLE_W, h: int = SAMPLE_H) -> Dict[str, float]:
    """Mean luma + Laplacian variance (sharpness) on a downscaled frame."""
    total = 0
    for b in frame:
        total += b
    mean_l = total / len(frame)
    s = s2 = 0.0
    n = 0
    for y in range(1, h - 1):
        r = y * w
        for x in range(1, w - 1):
            i = r + x
            lap = 4 * frame[i] - frame[i - 1] - frame[i + 1] - frame[i - w] - frame[i + w]
            s += lap
            s2 += lap * lap
            n += 1
    var = s2 / n - (s / n) ** 2 if n else 0.0
    return {"mean": mean_l, "sharpness": var}


def mean_abs_diff(a: bytes, b: bytes) -> float:
    return sum(abs(x - y) for x, y in zip(a, b)) / max(1, len(a))


def quality_screen(ffmpeg: str, video: str, start: float, end: float, samples: int = 24) -> Dict[str, Any]:
    """Basic screening on evenly spread sampled frames. Results are warnings, not verdicts."""
    span = max(0.0, end - start)
    n = max(2, min(samples, int(span) + 1 if span > 0 else 2))
    times = [start + span * (k + 0.5) / n for k in range(n)]
    frames = sample_gray_frames(ffmpeg, video, times)
    good = [(t, f) for t, f in zip(times, frames) if f]
    res: Dict[str, Any] = {"sampled": n, "decoded": len(good), "warnings": []}
    if not good:
        res["warnings"].append("No frame could be decoded from the selected time range.")
        return res
    metrics = [frame_metrics(f) for _, f in good]
    black = [t for (t, _), m in zip(good, metrics) if m["mean"] < 16.0]
    sharp = sorted(m["sharpness"] for m in metrics if m["mean"] >= 16.0)
    median = sharp[len(sharp) // 2] if sharp else 0.0
    blurry = [t for (t, _), m in zip(good, metrics) if m["mean"] >= 16.0 and median > 0 and m["sharpness"] < 0.35 * median]
    dups = 0
    for (_, a), (_, b) in zip(good, good[1:]):
        if mean_abs_diff(a, b) < 1.0:
            dups += 1
    res.update({"blackFrames": len(black), "blurryFrames": len(blurry), "nearDuplicatePairs": dups,
                "medianSharpness": round(median, 1)})
    if len(good) < n:
        res["warnings"].append("%d of %d sampled frames could not be decoded - the file may be damaged." % (n - len(good), n))
    if black:
        res["warnings"].append("%d of %d sampled frames are almost black (e.g. at %.1fs)." % (len(black), len(good), black[0]))
    if blurry:
        res["warnings"].append("%d of %d sampled frames are much blurrier than the rest (motion blur?)." % (len(blurry), len(good)))
    if dups >= max(2, (len(good) - 1) // 3):
        res["warnings"].append("Many sampled frames are near-identical - the camera may be standing still. "
                               "Gaussian Splats need the camera to move through the scene.")
    if median and median < 15.0:
        res["warnings"].append("The footage looks very soft overall (low detail), which limits reconstruction quality.")
    if good:
        res["cornerStats"] = corner_stats(good[len(good) // 2][1])
    return res


def probe(ffprobe: str, ffmpeg: Optional[str], video: str, quality: bool = False,
          start: float = 0.0, end: Optional[float] = None) -> Dict[str, Any]:
    if not os.path.isfile(video):
        raise FileNotFoundError(video)
    info = parse_probe(run_ffprobe(ffprobe, video), video)
    warnings: List[str] = []
    errors: List[str] = []
    if not info.get("readable"):
        errors.append("No readable video stream found.")
        info.update({"classification": classify_projection(info), "warnings": warnings, "errors": errors})
        return info
    cs = None
    q = None
    dur = info.get("durationSeconds") or 0.0
    if ffmpeg:
        if quality:
            q = quality_screen(ffmpeg, video, start, end if end is not None else dur)
            cs = q.pop("cornerStats", None)
            warnings += q.get("warnings", [])
        else:
            fr = sample_gray_frames(ffmpeg, video, [dur * 0.5 if dur else 0.0])
            cs = corner_stats(fr[0]) if fr and fr[0] else None
    info["classification"] = classify_projection(info, cs)
    if q is not None:
        info["quality"] = q
    w, h = info.get("width", 0), info.get("height", 0)
    if w and w < 2048:
        warnings.append("Low resolution (%dx%d). Equirectangular video below ~4K gives soft perspective views." % (w, h))
    if dur and dur < 5.0:
        warnings.append("Very short clip (%.1fs). A few seconds rarely contain enough camera movement." % dur)
    if info.get("frameCount") is not None and info["frameCount"] < 30:
        warnings.append("Only %d frames in the file - too few for a reliable reconstruction." % info["frameCount"])
    if not info.get("fps"):
        warnings.append("Frame rate could not be determined.")
    info["warnings"] = warnings
    info["errors"] = errors
    return info

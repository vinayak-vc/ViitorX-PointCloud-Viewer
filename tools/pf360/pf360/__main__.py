"""CLI: python -m pf360 {doctor,probe,estimate,presets,validate-ply,run}"""
from __future__ import annotations

import argparse
import json
import sys

from . import __version__, environment, manifest, ply_validate, video_probe
from .nerfstudio_backend import estimate


def _print_json(obj) -> None:
    sys.stdout.write(json.dumps(obj, indent=None, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(prog="pf360", description="PointForge 360 video -> Gaussian Splat worker")
    ap.add_argument("--version", action="version", version="pf360 " + __version__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("doctor", help="environment diagnostic")
    d.add_argument("--json", action="store_true")
    d.add_argument("--quick", action="store_true", help="skip the (slow) PyTorch/Nerfstudio import probe")
    d.add_argument("--ns-python")
    d.add_argument("--colmap")
    d.add_argument("--ffmpeg")
    d.add_argument("--ffprobe")

    p = sub.add_parser("probe", help="inspect a video file")
    p.add_argument("video")
    p.add_argument("--json", action="store_true")
    p.add_argument("--quality", action="store_true", help="also run the sampled quality screen")
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--end", type=float)
    p.add_argument("--ffmpeg")
    p.add_argument("--ffprobe")

    e = sub.add_parser("estimate", help="perspective image count for a configuration")
    e.add_argument("--width", type=int, required=True)
    e.add_argument("--height", type=int, required=True)
    e.add_argument("--frames", type=int, required=True)
    e.add_argument("--views", type=int, choices=(8, 14), required=True)
    e.add_argument("--crop", type=float, nargs=4, default=(0.0, 0.0, 0.0, 0.0), metavar=("TOP", "BOTTOM", "LEFT", "RIGHT"))

    fr = sub.add_parser("frame", help="write one downscaled RGB24 frame (raw) for previews")
    fr.add_argument("video")
    fr.add_argument("--time", type=float, default=0.0)
    fr.add_argument("--width", type=int, default=768)
    fr.add_argument("--out", required=True)
    fr.add_argument("--ffmpeg")

    mp = sub.add_parser("mask-preview", help="mask overlay for one raw RGB24 image (needs the masking add-on)")
    mp.add_argument("--in", dest="inp", required=True, help="raw RGB24 file")
    mp.add_argument("--size", type=int, nargs=2, required=True, metavar=("W", "H"))
    mp.add_argument("--out", required=True, help="raw RGB24 overlay output (same size)")
    mp.add_argument("--groups", default="people")
    mp.add_argument("--dilate", type=int, default=8)

    sub.add_parser("presets", help="print the quality presets JSON")

    v = sub.add_parser("validate-ply", help="validate a 3DGS PLY")
    v.add_argument("ply")
    v.add_argument("--json", action="store_true")

    r = sub.add_parser("run", help="run a reconstruction job (JSONL events on stdout)")
    r.add_argument("--manifest", required=True)
    r.add_argument("--stdin-control", action="store_true", help="accept 'cancel' lines on stdin (host mode)")

    a = ap.parse_args(argv)

    if a.cmd == "doctor":
        cfg = {"nerfstudioPython": a.ns_python, "colmapExe": a.colmap, "ffmpegExe": a.ffmpeg, "ffprobeExe": a.ffprobe}
        rep = environment.doctor(cfg, deep=not a.quick)
        if a.json:
            _print_json(rep.to_json())
        else:
            print(environment.format_report(rep))
        return 0 if rep.ready else 1

    if a.cmd == "probe":
        t = environment.resolve_tools({"ffmpegExe": a.ffmpeg, "ffprobeExe": a.ffprobe})
        if not t.ffprobe:
            msg = {"ok": False, "errors": ["FFprobe not found - install FFmpeg (winget install Gyan.FFmpeg)."]}
            _print_json(msg) if a.json else print(msg["errors"][0])
            return 2
        try:
            info = video_probe.probe(t.ffprobe, t.ffmpeg, a.video, quality=a.quality, start=a.start, end=a.end)
            info["ok"] = not info.get("errors")
        except Exception as ex:  # noqa: BLE001
            info = {"ok": False, "errors": [str(ex)]}
        if a.json:
            _print_json(info)
        else:
            print(json.dumps(info, indent=2, ensure_ascii=False))
        return 0 if info.get("ok") else 2

    if a.cmd == "estimate":
        try:
            _print_json(estimate(a.width, a.height, a.frames, a.views, a.crop))
            return 0
        except ValueError as ex:
            _print_json({"error": str(ex)})
            return 2

    if a.cmd == "frame":
        t = environment.resolve_tools({"ffmpegExe": a.ffmpeg})
        w = max(64, min(4096, a.width - a.width % 2))
        h = w // 2
        if not t.ffmpeg:
            _print_json({"ok": False, "error": "FFmpeg not found"})
            return 2
        from . import procrun
        cp = procrun.capture([t.ffmpeg, "-v", "error", "-ss", "%.3f" % max(0.0, a.time), "-i", a.video,
                              "-frames:v", "1", "-vf", "scale=%d:%d" % (w, h), "-pix_fmt", "rgb24", "-f", "rawvideo",
                              "-"], timeout=120)
        if cp.returncode != 0 or len(cp.stdout) != w * h * 3:
            _print_json({"ok": False, "error": cp.stderr.decode("utf-8", "replace")[-300:] or "no frame decoded"})
            return 2
        with open(a.out, "wb") as f:
            f.write(cp.stdout)
        _print_json({"ok": True, "width": w, "height": h, "path": a.out})
        return 0

    if a.cmd == "mask-preview":
        from pathlib import Path
        from . import procrun
        t = environment.resolve_tools({})
        if not (t.ns_python and t.mask_model):
            _print_json({"ok": False, "error": "Masking add-on not installed (wizard: Environment > Engines > "
                                               "Masking model)"})
            return 2
        groups = [g for g in a.groups.split(",") if g]
        if not groups or any(g not in manifest.load_presets()["limits"]["maskGroups"] for g in groups):
            _print_json({"ok": False, "error": "invalid mask groups: %s" % a.groups})
            return 2
        helper = Path(__file__).with_name("ns_helpers") / "mask.py"
        cp = procrun.capture([t.ns_python, str(helper), "--model", t.mask_model, "--image", a.inp,
                              "--raw-size", str(a.size[0]), str(a.size[1]), "--overlay", a.out,
                              "--groups", ",".join(groups), "--dilate", str(max(0, min(64, a.dilate)))],
                             timeout=600, env=procrun.child_env(extra=t.ns_env()))
        text = cp.stdout.decode("utf-8", "replace")
        done = [l for l in text.splitlines() if l.startswith("PF360MASKDONE")]
        if cp.returncode != 0 or not done:
            _print_json({"ok": False, "error": (cp.stderr.decode("utf-8", "replace")[-400:] or "mask preview failed")})
            return 2
        res = json.loads(done[-1][len("PF360MASKDONE"):])
        res["ok"] = True
        _print_json(res)
        return 0

    if a.cmd == "presets":
        _print_json(manifest.load_presets())
        return 0

    if a.cmd == "validate-ply":
        rep = ply_validate.validate(a.ply)
        if a.json:
            _print_json(rep)
        else:
            print(json.dumps(rep, indent=2))
        return 0 if rep.get("ok") else 2

    if a.cmd == "run":
        from .worker import run_manifest
        return run_manifest(a.manifest, stdin_control=a.stdin_control)
    return 2


if __name__ == "__main__":
    sys.exit(main())

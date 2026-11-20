"""Per-job workspace layout, stage cache fingerprints and the final report.

    <output.directory>/
        manifest.json               resolved manifest actually used
        state.json                  per-stage fingerprints + results (cache)
        source_reference.json       source video identity (never a copy)
        frames/                     extracted equirectangular frames
        frames/planar_projections/  Nerfstudio perspective views
        processed/                  ns-process-data output (images*, colmap/, transforms.json, sparse_pc.ply)
        training/                   ns-train output (config.yml, checkpoints)
        export/<name>.ply           final Gaussian Splat
        logs/                       worker.log + one log per external tool
        reconstruction_report.json / .txt

Cache reuse: a stage is skipped only when its fingerprint - a SHA-256 over the
source video identity (size, mtime, hash of first+last 8 MB) and every setting
that affects it, chained with the upstream stage's fingerprint - matches the
recorded one AND its outputs still exist. Directories are only ever cleared
inside a workspace that carries our state.json marker.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

STATE_MARKER = "pf360-workspace"


def video_identity(path: str, chunk: int = 8 << 20) -> Dict[str, Any]:
    st = os.stat(path)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(chunk))
        if st.st_size > 2 * chunk:
            f.seek(st.st_size - chunk)
            h.update(f.read(chunk))
    return {"path": os.path.abspath(path), "size": st.st_size, "mtimeNs": st.st_mtime_ns, "headTailSha256": h.hexdigest()}


def fingerprint(parent: Optional[str], inputs: Dict[str, Any]) -> str:
    h = hashlib.sha256()
    h.update((parent or "").encode())
    h.update(json.dumps(inputs, sort_keys=True, default=str).encode())
    return h.hexdigest()


class Workspace:
    def __init__(self, root: str, ply_name: str):
        self.root = Path(root)
        self.frames_dir = self.root / "frames"
        self.planar_dir = self.frames_dir / "planar_projections"
        self.processed_dir = self.root / "processed"
        self.training_dir = self.root / "training"
        self.export_dir = self.root / "export"
        self.logs_dir = self.root / "logs"
        self.export_ply = self.export_dir / ply_name
        self.state_path = self.root / "state.json"
        self.state: Dict[str, Any] = {}

    def init(self) -> None:
        if self.root.exists() and any(self.root.iterdir()) and not self.state_path.is_file():
            raise RuntimeError("Output directory %s is not empty and is not a pf360 workspace; choose an empty or "
                               "new folder so nothing is overwritten." % self.root)
        for d in (self.root, self.logs_dir):
            d.mkdir(parents=True, exist_ok=True)
        if self.state_path.is_file():
            with open(self.state_path, "r", encoding="utf-8") as f:
                self.state = json.load(f)
        if self.state.get("marker") not in (None, STATE_MARKER):
            raise RuntimeError("state.json in %s was not written by pf360" % self.root)
        self.state["marker"] = STATE_MARKER
        self.state.setdefault("stages", {})
        self.save()

    def save(self) -> None:
        tmp = self.state_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2)
        os.replace(tmp, self.state_path)

    def reset_dir(self, d: Path) -> None:
        """Clears a stage directory. Only allowed inside this (marked) workspace."""
        d = d.resolve()
        if self.root.resolve() not in d.parents:
            raise RuntimeError("refusing to clear %s outside the workspace" % d)
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    def cached(self, stage: str, fp: str) -> Optional[Dict[str, Any]]:
        s = self.state["stages"].get(stage)
        if s and s.get("fingerprint") == fp and s.get("status") == "completed":
            return s.get("data") or {}
        return None

    def record(self, stage: str, fp: str, status: str, data: Dict[str, Any], seconds: float) -> None:
        self.state["stages"][stage] = {"fingerprint": fp, "status": status, "data": data,
                                       "seconds": round(seconds, 2), "finishedAt": time.time()}
        self.save()

    def invalidate_from(self, stage_order, stage: str) -> None:
        hit = False
        for s in stage_order:
            hit = hit or s == stage
            if hit:
                self.state["stages"].pop(s, None)
        self.save()

    @contextlib.contextmanager
    def open_log(self, name: str) -> Iterator[Any]:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        f = open(self.logs_dir / name, "a", encoding="utf-8", errors="replace")
        try:
            f.write("\n===== %s =====\n" % time.strftime("%Y-%m-%d %H:%M:%S"))
            yield f
        finally:
            f.close()

    def free_bytes(self) -> int:
        try:
            return shutil.disk_usage(self.root).free
        except OSError:
            return -1


def write_report(ws: Workspace, report: Dict[str, Any]) -> None:
    with open(ws.root / "reconstruction_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    lines = ["PointForge 360 Reconstruction Report", "=" * 36, ""]

    def val(v: Any) -> str:
        return "unavailable" if v is None else str(v)
    lines.append("Job:      %s" % report.get("jobId"))
    lines.append("Status:   %s" % report.get("status"))
    if report.get("error"):
        lines.append("Error:    %s" % report["error"])
    inp = report.get("input", {})
    lines += ["", "Input", "  File:        %s" % val(inp.get("filename")),
              "  Resolution:  %s" % val(inp.get("resolution")),
              "  Duration:    %s s" % val(inp.get("durationSeconds"))]
    pr = report.get("processing", {})
    lines += ["", "Processing"] + ["  %-24s %s" % (k + ":", val(v)) for k, v in pr.items()]
    tr = report.get("training", {})
    lines += ["", "Training"] + ["  %-24s %s" % (k + ":", val(v)) for k, v in tr.items()]
    out = report.get("output", {})
    lines += ["", "Output"] + ["  %-24s %s" % (k + ":", val(v)) for k, v in out.items()]
    tm = report.get("timings", {})
    lines += ["", "Timings (seconds)"] + ["  %-24s %s" % (k + ":", val(v)) for k, v in tm.items()]
    if report.get("warnings"):
        lines += ["", "Warnings"] + ["  - " + w for w in report["warnings"]]
    with open(ws.root / "reconstruction_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

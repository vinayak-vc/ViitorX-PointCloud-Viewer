"""Structured worker -> host event protocol (JSON Lines on stdout).

Every event is one JSON object on one line, carrying:
    v      protocol version (int)
    type   event type (see EVENT_TYPES)
    jobId  job id from the manifest ("" before a manifest is loaded)
    ts     unix time, seconds (float)
plus type-specific fields:
    stage    stage id (see STAGES)                      stage_*/progress/log
    percent  0..100 float, or null when unknown         progress
    message  human readable text                        most events
    level    "debug"|"info"|"warning"|"error"           log
    data     free-form object with structured results   stage_completed/job_*

stdout carries ONLY these events. Human diagnostics go to stderr and to the
job's logs/worker.log, so a host can parse stdout line by line safely.
"""
from __future__ import annotations

import json
import sys
import threading
import time
from typing import IO, Any, Dict, Optional

from . import PROTOCOL_VERSION

EVENT_TYPES = (
    "worker_ready",
    "job_started",
    "stage_started",
    "progress",
    "log",
    "warning",
    "error",
    "stage_completed",
    "stage_skipped",
    "cancellation_requested",
    "job_completed",
    "job_failed",
    "job_cancelled",
)

# Ordered pipeline stages. The host maps these to its own status labels.
STAGES = (
    "video_validation",
    "frame_extraction",
    "panorama_projection",
    "camera_reconstruction",
    "training",
    "export",
    "validation",
)


class Emitter:
    """Thread-safe JSONL writer. One instance per worker process."""

    def __init__(self, stream: Optional[IO[str]] = None, job_id: str = "", log_file: Optional[IO[str]] = None):
        self._stream = stream if stream is not None else sys.stdout
        self._lock = threading.Lock()
        self.job_id = job_id
        self.log_file = log_file

    def emit(self, type_: str, **fields: Any) -> Dict[str, Any]:
        if type_ not in EVENT_TYPES:
            raise ValueError("unknown event type: %s" % type_)
        ev: Dict[str, Any] = {"v": PROTOCOL_VERSION, "type": type_, "jobId": self.job_id, "ts": round(time.time(), 3)}
        for k, v in fields.items():
            if v is not None or k == "percent":
                ev[k] = v
        line = json.dumps(ev, ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            self._stream.write(line + "\n")
            self._stream.flush()
            if self.log_file is not None:
                try:
                    self.log_file.write(line + "\n")
                    self.log_file.flush()
                except (OSError, ValueError):
                    pass
        return ev

    # convenience wrappers --------------------------------------------------
    def log(self, message: str, level: str = "info", stage: Optional[str] = None) -> None:
        self.emit("log", level=level, message=message, stage=stage)

    def warning(self, message: str, stage: Optional[str] = None, code: Optional[str] = None) -> None:
        self.emit("warning", message=message, stage=stage, code=code)

    def progress(self, stage: str, percent: Optional[float], message: Optional[str] = None) -> None:
        if percent is not None:
            percent = max(0.0, min(100.0, round(float(percent), 2)))
        self.emit("progress", stage=stage, percent=percent, message=message)


def parse_event(line: str) -> Dict[str, Any]:
    """Parses and validates one protocol line (used by tests and by other hosts)."""
    ev = json.loads(line)
    if not isinstance(ev, dict):
        raise ValueError("event is not an object")
    if ev.get("v") != PROTOCOL_VERSION:
        raise ValueError("unsupported protocol version: %r" % ev.get("v"))
    if ev.get("type") not in EVENT_TYPES:
        raise ValueError("unknown event type: %r" % ev.get("type"))
    if "jobId" not in ev:
        raise ValueError("event without jobId")
    if ev["type"] == "progress":
        p = ev.get("percent")
        if p is not None and not (0.0 <= float(p) <= 100.0):
            raise ValueError("percent out of range")
    return ev

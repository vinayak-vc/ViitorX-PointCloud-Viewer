"""Job orchestration: manifest -> stages -> validated 3DGS PLY.

Exit codes: 0 completed, 2 failed (user-facing reason in job_failed),
3 cancelled, 4 invalid manifest, 1 unexpected internal error.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import __version__, environment, manifest as manifest_mod, ply_validate, procrun, video_probe
from .backend import StageContext, StageFailed
from .nerfstudio_backend import NerfstudioBackend
from .protocol import Emitter
from .workspace import Workspace, fingerprint, video_identity, write_report

EXIT_OK, EXIT_INTERNAL, EXIT_FAILED, EXIT_CANCELLED, EXIT_MANIFEST = 0, 1, 2, 3, 4
PKG_PARENT = str(Path(__file__).resolve().parent.parent)


def _stderr(msg: str) -> None:
    try:
        sys.stderr.write(msg.rstrip() + "\n")
        sys.stderr.flush()
    except (OSError, ValueError):
        pass


def start_stdin_listener(cancel: procrun.CancelToken, emitter: Emitter) -> None:
    """Host control channel: a line 'cancel' on stdin requests cancellation."""
    def loop() -> None:
        try:
            for line in sys.stdin:
                if line.strip().lower() == "cancel":
                    emitter.emit("cancellation_requested", message="cancel requested by host")
                    cancel.cancel()
                    return
        except (OSError, ValueError):
            pass
    threading.Thread(target=loop, daemon=True, name="pf360-stdin").start()


class Job:
    def __init__(self, raw: Dict[str, Any], emitter: Emitter, cancel: procrun.CancelToken):
        self.raw = raw
        self.em = emitter
        self.cancel = cancel
        self.backend = NerfstudioBackend()
        self.m: Dict[str, Any] = {}
        self.warnings: List[str] = []
        self.timings: Dict[str, float] = {}
        self.results: Dict[str, Dict[str, Any]] = {}
        self.ws: Optional[Workspace] = None
        self.info: Dict[str, Any] = {}

    # ----------------------------------------------------------------------
    def warn(self, msg: str, stage: Optional[str] = None) -> None:
        self.warnings.append(msg)
        self.em.warning(msg, stage=stage)

    def run(self) -> int:
        job_id = str(self.raw.get("jobId", ""))
        self.em.job_id = job_id
        self.em.emit("job_started", message="pf360 %s" % __version__)
        status, err, code = "completed", None, EXIT_OK
        try:
            self._run_stages()
        except procrun.Cancelled:
            status, code = "cancelled", EXIT_CANCELLED
        except manifest_mod.ManifestError as e:
            status, err, code = "failed", "Invalid job settings: " + "; ".join(e.problems), EXIT_MANIFEST
        except StageFailed as e:
            status, err, code = "failed", str(e), EXIT_FAILED
        except Exception as e:  # noqa: BLE001 - report anything, never hang the host
            status, err, code = "failed", "Internal error: %s" % e, EXIT_INTERNAL
            _stderr(traceback.format_exc())
        report = self._report(status, err)
        if self.ws is not None:
            try:
                write_report(self.ws, report)
            except OSError as e:
                _stderr("could not write report: %s" % e)
        if status == "completed":
            self.em.emit("job_completed", outputPath=report["output"].get("path"), data=report,
                         message="Reconstruction complete" if report["output"].get("path") else "Stopped as requested")
        elif status == "cancelled":
            self.em.emit("job_cancelled", message="Cancelled - intermediate data kept in the workspace", data=report)
        else:
            self.em.emit("job_failed", message=err, data=report)
        if self.em.log_file is not None:
            self.em.log_file.close()
            self.em.log_file = None
        return code

    # ----------------------------------------------------------------------
    def _stage(self, stage: str, fn, fp: Optional[str] = None) -> Dict[str, Any]:
        """Runs (or reuses) one stage with events, timing and cache bookkeeping."""
        if self.cancel.cancelled:
            raise procrun.Cancelled()
        allow = bool(self.m.get("options", {}).get("allowCache", True)) if self.m else False
        if fp and self.ws is not None and allow:
            hit = self.ws.cached(stage, fp)
            if hit is not None and self._outputs_exist(stage):
                self.em.emit("stage_skipped", stage=stage, message="reusing cached result (settings unchanged)", data=hit)
                self.results[stage] = hit
                return hit
        # Restore point: an interrupted run of this stage with the same settings is continued
        # (stages keep their partial output); "running" is recorded first so a crash or power
        # loss also counts as interrupted.
        resume = bool(fp and self.ws is not None and allow and self.ws.resumable(stage, fp))
        self.results["_resume_" + stage] = resume
        if fp and self.ws is not None:
            self.ws.record(stage, fp, "running", {}, 0.0)
        if resume:
            self.em.emit("stage_started", stage=stage, data={"resume": True})
            self.em.log("continuing %s from its restore point" % stage, stage=stage)
        else:
            self.em.emit("stage_started", stage=stage)
        t0 = time.time()
        try:
            data = fn()
        except procrun.Cancelled:
            if fp and self.ws:
                self.ws.record(stage, fp, "cancelled", {}, time.time() - t0)
            raise
        except StageFailed as e:
            if fp and self.ws:
                self.ws.record(stage, fp, "failed", e.data, time.time() - t0)
            self.results[stage] = e.data
            self.em.emit("error", stage=stage, message=str(e))
            raise
        dt = time.time() - t0
        self.timings[stage] = round(dt, 2)
        self.results[stage] = data
        if fp and self.ws:
            self.ws.record(stage, fp, "completed", data, dt)
        self.em.emit("stage_completed", stage=stage, data=data)
        return data

    def _outputs_exist(self, stage: str) -> bool:
        if stage == "frame_extraction":
            return self.ws is not None and any(self.ws.frames_dir.glob("frame_*.jpg"))
        return self.backend.stage_outputs_exist(stage, self._ctx())

    def _ctx(self) -> StageContext:
        return StageContext(manifest=self.m, tools=self.tools, workspace=self.ws, emitter=self.em,
                            cancel=self.cancel, video_info=self.info, results=self.results)

    def _stop_after(self, stage: str) -> bool:
        return self.m.get("options", {}).get("stopAfter") == stage

    # ----------------------------------------------------------------------
    def _run_stages(self) -> None:
        self.tools = environment.resolve_tools(self.raw.get("environment") or {})
        self._stage("video_validation", self._validate_input)
        ws = self.ws
        assert ws is not None
        vid = video_identity(self.m["input"]["videoPath"])
        with open(ws.root / "source_reference.json", "w", encoding="utf-8") as f:
            json.dump(vid, f, indent=2)
        inp, pr = self.m["input"], self.m["processing"]
        fp = fingerprint(None, {"video": vid, "start": inp.get("startTimeSeconds", 0), "end": inp.get("endTimeSeconds"),
                                "targetFrames": pr["targetFrames"]})
        self._stage("frame_extraction", self._extract_frames, fp)
        if self._stop_after("frame_extraction"):
            return
        problems = self.backend.check_environment(self.tools, self.m)
        if problems:
            raise StageFailed("Reconstruction environment incomplete: " + " ".join(problems) +
                              " Run `python -m pf360 doctor` for details.")
        for stage in self.backend.stages():
            fp = fingerprint(fp, self.backend.stage_inputs(stage, self._ctx()))
            if not self.backend.stage_enabled(stage, self._ctx()):
                self.results[stage] = {"enabled": False}
                self.em.emit("stage_skipped", stage=stage, message="not enabled for this job", data={"enabled": False})
                continue
            self._stage(stage, lambda s=stage: self.backend.run_stage(s, self._ctx()), fp)
            if self._stop_after(stage):
                return
        self._stage("validation", self._validate_output)

    # ------------------------------------------------------------ stage: input
    def _validate_input(self) -> Dict[str, Any]:
        st = "video_validation"
        raw_in = self.raw.get("input") or {}
        video = raw_in.get("videoPath")
        if not isinstance(video, str) or not os.path.isfile(video):
            raise StageFailed("Source video not found: %s" % video)
        if not self.tools.ffprobe:
            raise StageFailed("FFprobe not found - install FFmpeg (winget install Gyan.FFmpeg).")
        self.em.progress(st, 5, "reading metadata")
        start = float(raw_in.get("startTimeSeconds") or 0.0)
        end = raw_in.get("endTimeSeconds")
        info = video_probe.probe(self.tools.ffprobe, self.tools.ffmpeg, video, quality=True, start=start,
                                 end=float(end) if end is not None else None)
        self.info = info
        self.em.progress(st, 70, "checking projection")
        if info.get("errors"):
            raise StageFailed(" ".join(info["errors"]), info)
        cls = info["classification"]
        if not cls["supported"]:
            raise StageFailed(" ".join(cls["reasons"]), info)
        if cls["requiresConfirmation"] and not raw_in.get("projectionConfirmed"):
            raise StageFailed("The projection could not be verified from the file (no spherical metadata). Confirm "
                              "that this is a stitched equirectangular video (input.projectionConfirmed = true).", info)
        for w in info.get("warnings", []):
            self.warn(w, st)
        dur = info.get("durationSeconds")
        if not dur:
            raise StageFailed("Video duration could not be determined.", info)
        self.m = manifest_mod.validate(manifest_mod.resolve(self.raw, dur))
        if start >= dur:
            raise StageFailed("Start time %.1fs is beyond the end of the video (%.1fs)." % (start, dur), info)
        end_eff = min(float(end), dur) if end is not None else dur
        window = end_eff - start
        avail = int(window * (info.get("fps") or 0)) if info.get("fps") else None
        tf = self.m["processing"]["targetFrames"]
        if avail is not None and tf > avail:
            self.warn("Requested %d frames but the selected range only contains ~%d; all of them will be used." %
                      (tf, avail), st)
        ws = Workspace(self.m["output"]["directory"], self._ply_name())
        try:
            ws.init()
        except RuntimeError as e:
            raise StageFailed(str(e), info)
        self.ws = ws
        try:
            self.em.log_file = open(ws.logs_dir / "worker.log", "a", encoding="utf-8")
        except OSError:
            self.em.log_file = None
        with open(self.ws.root / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(self.m, f, indent=2)
        self.em.log("workspace: %s" % self.ws.root, stage=st)
        self._check_gpu_and_disk(info)
        self.em.progress(st, 100, "ready")
        out = {k: info.get(k) for k in ("fileName", "width", "height", "fps", "durationSeconds", "frameCount", "codec",
                                        "pixelFormat", "hasAudio", "fileSizeBytes")}
        out.update(projection=cls["projection"], projectionConfidence=cls["confidence"], quality=info.get("quality"))
        return out

    def _ply_name(self) -> str:
        name = self.m["output"].get("plyFileName")
        if name:
            return name
        stem = Path(self.m["input"]["videoPath"]).stem
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)[:80] or "scene"
        return safe + "_splat.ply"

    def _check_gpu_and_disk(self, info: Dict[str, Any]) -> None:
        st = "video_validation"
        stop = self.m["options"].get("stopAfter")
        needs_cuda = stop != "frame_extraction"
        env: Dict[str, Any] = {"cuda": None, "methods": []}
        if needs_cuda and self.tools.ns_python:
            self.em.progress(st, 80, "checking GPU / PyTorch")
            t = environment.query_torch(self.tools.ns_python)
            env.update(cuda=bool(t.get("cuda")), device=t.get("device"), totalMB=t.get("totalMB"),
                       freeMB=t.get("freeMB"), torch=t.get("torch"), nerfstudio=t.get("nerfstudio"))
            env["methods"] = environment.list_methods(self.tools)
            if not env["cuda"]:
                raise StageFailed("CUDA is not available to the Nerfstudio PyTorch (%s). Gaussian training and "
                                  "Nerfstudio's 360 projection need an NVIDIA CUDA GPU; CPU training is not "
                                  "attempted. You can still run frame extraction only." %
                                  (t.get("torchError") or t.get("probeError") or "torch.cuda.is_available() is False"))
            free = env.get("freeMB")
            tr = self.m["training"]
            if free is not None:
                need = 6000 if tr["method"] == "splatfacto-big" else 4000
                need = int(need * {1: 1.6, 2: 1.0, 4: 0.6, 8: 0.5}[tr["downscaleFactor"]])
                if free < need:
                    self.warn("Only %d MB of GPU memory is free; this configuration may need roughly %d MB or more "
                              "(rough guide, not a guarantee). Consider a higher downscale factor, fewer frames, "
                              "standard Splatfacto, or closing other GPU applications." % (free, need), st)
        self.results["_env"] = env
        # Disk estimate (JPEG ~0.35 B/px): frames + perspective views + downscaled copies + checkpoints.
        w, h = info.get("width") or 0, info.get("height") or 0
        pr = self.m["processing"]
        est = pr["targetFrames"] * w * h * 0.35 * (1 + 1.4 * 1.33 * (0 if stop == "frame_extraction" else 1))
        est += 0 if stop == "frame_extraction" else 1.5e9
        free_b = self.ws.free_bytes() if self.ws else -1
        if free_b >= 0:
            if free_b < est:
                raise StageFailed("Not enough disk space in the output folder: ~%.1f GB estimated, %.1f GB free." %
                                  (est / 1e9, free_b / 1e9))
            if free_b < est * 2:
                self.warn("Disk space is tight: ~%.1f GB estimated, %.1f GB free." % (est / 1e9, free_b / 1e9), st)

    # ----------------------------------------------------------- stage: frames
    def _extract_frames(self) -> Dict[str, Any]:
        st = "frame_extraction"
        ws = self.ws
        assert ws is not None
        if not self.tools.ffmpeg:
            raise StageFailed("FFmpeg not found.")
        ws.reset_dir(ws.frames_dir)
        inp = self.m["input"]
        dur = float(self.info["durationSeconds"])
        start = float(inp.get("startTimeSeconds") or 0.0)
        end = min(float(inp["endTimeSeconds"]), dur) if inp.get("endTimeSeconds") is not None else dur
        window = max(0.001, end - start)
        n = int(self.m["processing"]["targetFrames"])
        rate = n / window
        args = [self.tools.ffmpeg, "-nostdin", "-hide_banner", "-y", "-ss", "%.3f" % start, "-t", "%.3f" % window,
                "-i", inp["videoPath"], "-vf", "fps=%.6f" % rate, "-frames:v", str(n), "-q:v", "2",
                "-progress", "pipe:1", "-nostats", str(ws.frames_dir / "frame_%05d.jpg")]

        def on_line(l: str) -> None:
            if l.startswith("frame="):
                try:
                    k = int(l.split("=", 1)[1])
                    self.em.progress(st, 100.0 * k / n, "%d / %d frames" % (k, n))
                except ValueError:
                    pass
        with ws.open_log("ffmpeg.log") as log:
            rc = procrun.run(args, cancel=self.cancel, on_line=on_line, log=log)
        got = sorted(ws.frames_dir.glob("frame_*.jpg"))
        if rc != 0 or not got:
            raise StageFailed("FFmpeg frame extraction failed (exit %d) - see logs/ffmpeg.log" % rc)
        if len(got) < 0.9 * n:
            self.warn("Extracted %d frames (requested %d)." % (len(got), n), st)
        if len(got) < 10:
            self.warn("Only %d frames - reconstruction quality will be very limited." % len(got), st)
        return {"framesSelected": len(got), "framesRequested": n, "startSeconds": start, "endSeconds": end,
                "samplingRateFps": round(rate, 4)}

    # --------------------------------------------------------- stage: validate
    def _validate_output(self) -> Dict[str, Any]:
        st = "validation"
        ply = self.backend.result_ply(self._ctx())
        if ply is None:
            raise StageFailed("No exported PLY to validate.")
        self.em.progress(st, None, "checking PLY header and values")
        rep = None
        if self.tools.ns_python:     # numpy lives there -> full numeric scan
            try:
                env = procrun.child_env(extra={"PYTHONPATH": PKG_PARENT})
                cp = procrun.capture([self.tools.ns_python, "-m", "pf360", "validate-ply", str(ply), "--json"],
                                     timeout=1800, env=env)
                rep = json.loads(cp.stdout.decode("utf-8", "replace"))
            except Exception as e:  # noqa: BLE001
                _stderr("validate-ply in Nerfstudio env failed (%s); falling back to sampled check" % e)
        if rep is None:
            rep = ply_validate.validate(str(ply))
        for w in rep.get("warnings", []):
            self.warn(w, st)
        if not rep.get("ok"):
            raise StageFailed("Exported PLY failed validation: " + "; ".join(rep.get("errors", [])), rep)
        return rep

    # ------------------------------------------------------------------ report
    def _report(self, status: str, err: Optional[str]) -> Dict[str, Any]:
        r = self.results
        vv, fe = r.get("video_validation", {}), r.get("frame_extraction", {})
        pp, cr = r.get("panorama_projection", {}), r.get("camera_reconstruction", {})
        tr, ex, va = r.get("training", {}), r.get("export", {}), r.get("validation", {})
        mt = self.m.get("training", {}) if self.m else {}
        t = self.timings
        def s(*keys: str) -> Optional[float]:
            vals = [t[k] for k in keys if k in t]
            return round(sum(vals), 2) if vals else None
        return {
            "jobId": self.em.job_id, "status": status, "error": err, "workerVersion": __version__,
            "workspace": str(self.ws.root) if self.ws else None,
            "input": {"filename": vv.get("fileName") or self.info.get("fileName"),
                      "durationSeconds": vv.get("durationSeconds"),
                      "resolution": [vv["width"], vv["height"]] if vv.get("width") else None,
                      "fps": vv.get("fps"), "codec": vv.get("codec"), "projection": vv.get("projection")},
            "processing": {"framesSelected": fe.get("framesSelected"), "viewsPerFrame": pp.get("viewsPerFrame"),
                           "generatedImages": pp.get("perspectiveImages"),
                           "perspectiveSize": pp.get("perspectiveSize"),
                           "registeredImages": cr.get("registeredImages"),
                           "registrationRatio": cr.get("registrationRatio"),
                           "distinctCameraPositions": cr.get("distinctCameraPositions"),
                           "sparsePointCloud": cr.get("sparsePointCloud"),
                           "masking": r.get("masking") or {"enabled": False}},
            "training": {"engine": tr.get("engine") or mt.get("engine", "nerfstudio"),
                         "method": tr.get("method") or mt.get("method"), "iterations": tr.get("iterations"),
                         "strategy": tr.get("strategy") or mt.get("lfsStrategy"),
                         "qualityPreset": mt.get("qualityPreset"), "downscaleFactor": mt.get("downscaleFactor"),
                         "gpu": (r.get("_env") or {}).get("device"),
                         "lfsCommit": tr.get("lfsCommit"), "lfsExeSha256": ex.get("lfsExeSha256")},
            "output": {"format": "3dgs-ply", "path": ex.get("path") if va.get("ok") else None,
                       "gaussianCount": va.get("gaussianCount"), "fileSizeBytes": va.get("fileSizeBytes"),
                       "shDegree": va.get("shDegree"), "numericCheck": va.get("numericCheck"),
                       "upAxis": "z_up" if va.get("ok") else None},
            "timings": {"validationSeconds": s("video_validation"),
                        "preprocessingSeconds": s("frame_extraction", "panorama_projection", "masking"),
                        "reconstructionSeconds": s("camera_reconstruction"), "trainingSeconds": s("training"),
                        "exportSeconds": s("export", "validation")},
            "warnings": self.warnings,
        }


def run_manifest(path: str, stdin_control: bool = False) -> int:
    em = Emitter()
    cancel = procrun.CancelToken()
    em.emit("worker_ready", message="pf360 %s, Python %s" % (__version__, sys.version.split()[0]))
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            raise ValueError("manifest root must be an object")
    except (OSError, ValueError) as e:
        em.emit("job_failed", message="Cannot read manifest: %s" % e)
        return EXIT_MANIFEST
    if stdin_control:
        start_stdin_listener(cancel, em)
    job = Job(raw, em, cancel)
    try:
        return job.run()
    except KeyboardInterrupt:
        cancel.cancel()
        em.emit("job_cancelled", message="interrupted")
        return EXIT_CANCELLED
    finally:
        if em.log_file is not None:
            em.log_file.close()

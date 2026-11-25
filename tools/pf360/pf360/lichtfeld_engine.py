"""LichtFeld-Studio training and export engine.

Executes LichtFeld-Studio.exe (GPL-3.0) as an isolated child subprocess via CLI,
parses progress telemetry, handles graceful cancel/pause via CTRL_BREAK_EVENT,
and normalises the exported PLY frame back to transforms.json coordinates (Z-up).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from . import procrun, training_prep
from .backend import StageContext, StageFailed

LFS_PROGRESS_RE = re.compile(r"(\d+)/(\d+)\s*\|\s*Loss:\s*([\d.]+)\s*\|\s*Splats:\s*(\d+)")
LFS_COMMIT = "8c366f5"
LFS_VALID_DOWNSCALES = (1, 2, 4, 8)
TRAIN_PAUSE_GRACE_S = 150

# Per-coefficient real-SH basis polynomial signs under (x, y, z) -> (x, -y, -z):
# (-1)^(deg_y + deg_z):
# Band 1 (deg 1: y, z, x): -, -, +
# Band 2 (deg 2: xy, yz, 3z^2-r^2, xz, x^2-y^2): -, +, +, -, +
# Band 3 (deg 3): -, +, -, -, +, -, +
SH_SIGNS_15 = [-1, -1, 1,  -1, 1, 1, -1, 1,  -1, 1, -1, -1, 1, -1, 1]


def is_lfs_job(manifest: Dict[str, Any]) -> bool:
    """Returns True if the manifest specifies LichtFeld training engine."""
    return manifest.get("training", {}).get("engine") == "lichtfeld"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def build_cmd(exe: str, processed_dir: Path, out_dir: Path, tr: Dict[str, Any],
              resume: bool = False, project_path: Optional[Path] = None,
              cropbox: Optional[str] = None) -> List[str]:
    """Constructs headless CLI invocation for LichtFeld-Studio."""
    total_its = int(tr.get("iterations", 30000))
    if resume:
        if not project_path or not project_path.is_file():
            raise StageFailed("Cannot resume LichtFeld training: project checkpoint missing: %s" % project_path)
        return [exe, "--headless", "--resume", str(project_path), "-o", str(out_dir),
                "--iter", str(total_its), "--export", "ply"]

    if (processed_dir / "transforms_train.json").is_file():
        raise StageFailed("transforms_train.json found in %s; LichtFeld would prefer it over transforms.json."
                          % processed_dir)

    downscale = int(tr.get("downscaleFactor", 1))
    if downscale not in LFS_VALID_DOWNSCALES:
        raise StageFailed("LichtFeld downscale factor must be 1, 2, 4, or 8 (got %s)." % downscale)

    strategy = tr.get("lfsStrategy", "mrnf")
    if strategy not in ("mrnf", "mcmc"):
        raise StageFailed("LichtFeld strategy must be 'mrnf' or 'mcmc' (got %s)." % strategy)

    max_cap = int(tr.get("maxGaussians", 2_000_000))
    sh_deg = int(tr.get("shDegree", 3))

    cmd = [
        exe, "--headless",
        "-d", str(processed_dir),
        "-o", str(out_dir),
        "--iter", str(total_its),
        "--strategy", strategy,
        "--max-cap", str(max_cap),
        "--sh-degree", str(sh_deg),
        "-r", str(downscale),
        "--export", "ply"
    ]
    if cropbox:
        cmd.extend(["--cropbox", cropbox, "--cropbox-loss-weight", "0", "--crop"])
    return cmd


def rotate_ply_zup(src_path: Path, dst_path: Path, chunk_size: int = 100_000) -> None:
    """Post-rotates an LFS exported PLY by diag(1, -1, -1) (180 deg about X) into transforms.json frame.

    Transforms:
      Positions: (x, y, z) -> (x, -y, -z)
      Quaternions (w-first): q' = (0, 1, 0, 0) (x) (w, x, y, z) = (-x, w, -z, y), normalized.
      SH coefficients: (-1)^(deg_y + deg_z) sign flips per channel.
    Streams vertex data in chunks to keep memory bounded regardless of cloud size.
    """
    with open(src_path, "rb") as src_f:
        header = b""
        while b"end_header\n" not in header:
            line = src_f.readline()
            if not line:
                break
            header += line

        lines = header.decode("utf-8", "replace").splitlines()
        props: List[str] = []
        n_vertices = 0
        for l in lines:
            if l.startswith("element vertex"):
                n_vertices = int(l.split()[-1])
            elif l.startswith("property"):
                parts = l.split()
                if len(parts) >= 3 and parts[1] == "float":
                    props.append(parts[-1])
                else:
                    raise StageFailed("Unsupported non-float property in PLY: %s" % l)

        stride = len(props) * 4
        dtype = [(p, "<f4") for p in props]

        rest_props = [p for p in props if p.startswith("f_rest_")]
        k = len(rest_props) // 3
        neg_props: List[str] = []
        if k > 0:
            for c in range(3):
                for i in range(min(k, len(SH_SIGNS_15))):
                    if SH_SIGNS_15[i] == -1:
                        neg_props.append("f_rest_%d" % (c * k + i))

        with open(dst_path, "wb") as dst_f:
            dst_f.write(header)
            rem = n_vertices
            while rem > 0:
                cur = min(chunk_size, rem)
                buf = src_f.read(cur * stride)
                if len(buf) < cur * stride:
                    raise StageFailed("Unexpected end of file while streaming PLY data (%d read vs %d expected)"
                                      % (len(buf), cur * stride))
                arr = np.frombuffer(buf, dtype=dtype).copy()

                # Positions
                arr["y"] = -arr["y"]
                arr["z"] = -arr["z"]

                # Quaternions
                w = arr["rot_0"].copy()
                x = arr["rot_1"].copy()
                y = arr["rot_2"].copy()
                z = arr["rot_3"].copy()
                arr["rot_0"] = -x
                arr["rot_1"] = w
                arr["rot_2"] = -z
                arr["rot_3"] = y
                norm = np.sqrt(arr["rot_0"] ** 2 + arr["rot_1"] ** 2 + arr["rot_2"] ** 2 + arr["rot_3"] ** 2)
                norm = np.where(norm > 0, norm, 1.0)
                arr["rot_0"] /= norm
                arr["rot_1"] /= norm
                arr["rot_2"] /= norm
                arr["rot_3"] /= norm

                # Spherical harmonics
                for p in neg_props:
                    arr[p] = -arr[p]

                dst_f.write(arr.tobytes())
                rem -= cur


def train_lfs(ctx: StageContext) -> Dict[str, Any]:
    """Runs LichtFeld-Studio training."""
    ws = ctx.workspace
    tr = ctx.manifest["training"]
    exe = ctx.tools.lichtfeld
    if not exe or not os.path.isfile(exe):
        raise StageFailed("LichtFeld-Studio executable not found.")

    total_its = int(tr.get("iterations", 30000))
    lfs_out = ws.training_dir / "lfs"
    project_file = lfs_out / "project.licht"
    is_resuming = bool(ctx.results.get("_resume_training")) and project_file.is_file()

    if not is_resuming:
        ws.reset_dir(lfs_out)
        lfs_out.mkdir(parents=True, exist_ok=True)
    else:
        ctx.emitter.log("resuming LichtFeld training from checkpoint %s" % project_file, stage="training")

    # Data preparation (seed cloud cap, misplaced panoramas, view budget)
    cap = int(tr.get("seedPointCap", training_prep.DEFAULT_SEED_POINT_CAP))
    drop = bool(tr.get("dropMisplacedPanoramas", True))
    ctx.emitter.progress("training", None, "preparing training data")
    try:
        prep = training_prep.prepare(ws.processed_dir, ws.planar_dir, seed_cap=cap,
                                     drop_misplaced=drop, max_views=int(tr.get("maxTrainingViews", 0)),
                                     downscale=int(tr.get("downscaleFactor", 1)), seed=int(tr.get("randomSeed", 42)),
                                     log=lambda m: ctx.emitter.log(m, stage="training"))
    except (OSError, ValueError) as e:
        ctx.emitter.warning("Training-data clean-up skipped (%s); training on full camera set." % e, stage="training")
        prep = {}

    if (ws.processed_dir / "transforms_train.json").is_file():
        raise StageFailed("transforms_train.json exists in %s; LichtFeld would prefer it over transforms.json."
                          % ws.processed_dir)

    ignored: List[str] = []
    for k in ("randomSeed", "useScaleRegularization", "cullAlphaThresh", "densifyGradThresh", "cullScaleThresh", "longSchedule"):
        if tr.get(k) is not None and tr.get(k) is not False:
            ignored.append(k)
    if ignored:
        ctx.emitter.log("LichtFeld engine ignores Nerfstudio-specific parameters: %s" % ", ".join(ignored),
                        stage="training")

    num_tiles = int(tr.get("tiles", 1))
    if num_tiles > 1:
        return train_lfs_tiles(ctx, num_tiles, prep)

    args = build_cmd(exe, ws.processed_dir, lfs_out, tr, resume=is_resuming, project_path=project_file)

    last = {"pct": -1.0, "step": 0, "splats": 0, "loss": 0.0}
    tail: List[str] = []

    def on_line(l: str) -> None:
        tail.append(l)
        del tail[:-40]
        m = LFS_PROGRESS_RE.search(l)
        if m:
            step = int(m.group(1))
            total = int(m.group(2))
            loss = float(m.group(3))
            splats = int(m.group(4))
            last["step"] = step
            last["splats"] = splats
            last["loss"] = loss
            pct = 100.0 * step / max(1, total_its)
            if pct - last["pct"] >= 0.25:
                last["pct"] = pct
                ctx.emitter.progress("training", pct, "step %d / %d | Loss: %.4f | Splats: %s"
                                     % (step, total_its, loss, f"{splats:,}"))
        elif "out of memory" in l.lower() or "cuda error: out of memory" in l.lower():
            ctx.emitter.warning("CUDA out of memory in LichtFeld-Studio. Lower downscale factor or max Gaussians.",
                                stage="training")

    kwargs: Dict[str, Any] = {}
    if os.name == "nt":
        # Must be CREATE_NEW_PROCESS_GROUP alone (shares console so GenerateConsoleCtrlEvent works,
        # but in its own process group so CTRL_BREAK targets the child without interrupting parent).
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True

    finished = threading.Event()
    with ws.open_log("training.log") as log:
        if log is not None:
            log.write("\n$ " + subprocess.list2cmdline(args) + "\n")
            log.flush()

        p = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             cwd=str(ws.root), **kwargs)

        def watch_cancel() -> None:
            while not finished.is_set():
                if ctx.cancel.wait(0.5):
                    if finished.is_set():
                        return
                    ctx.emitter.progress("training", None, "pausing - saving checkpoint (CTRL_BREAK)")
                    try:
                        if os.name == "nt":
                            os.kill(p.pid, signal.CTRL_BREAK_EVENT)
                        else:
                            os.kill(p.pid, signal.SIGINT)
                    except OSError:
                        pass
                    if not finished.wait(TRAIN_PAUSE_GRACE_S):
                        procrun.kill_tree(p)
                    return

        watcher = threading.Thread(target=watch_cancel, daemon=True)
        watcher.start()

        try:
            assert p.stdout is not None
            buf = b""
            while True:
                chunk = p.stdout.read1(65536) if hasattr(p.stdout, "read1") else p.stdout.read(4096)
                if not chunk:
                    break
                buf += chunk
                while True:
                    idx = min([i for i in (buf.find(b"\n"), buf.find(b"\r")) if i >= 0], default=-1)
                    if idx < 0:
                        break
                    raw, buf = buf[:idx], buf[idx + 1:]
                    procrun._deliver(raw, on_line, log)
                if len(buf) > 1 << 20:
                    procrun._deliver(buf, on_line, log)
                    buf = b""
            if buf:
                procrun._deliver(buf, on_line, log)
            rc = p.wait()
        finally:
            finished.set()
            watcher.join(timeout=2)
            if p.poll() is None:
                procrun.kill_tree(p)
            if p.stdout is not None:
                p.stdout.close()

    if ctx.cancel.cancelled:
        raise procrun.Cancelled()

    if rc != 0:
        hint = " | ".join(tail[-3:])
        raise StageFailed("LichtFeld-Studio failed (exit %d) - see logs/training.log. Last output: %s"
                          % (rc, hint[-400:]))

    out: Dict[str, Any] = {
        "engine": "lichtfeld",
        "strategy": tr.get("lfsStrategy", "mrnf"),
        "iterations": total_its,
        "lastLoggedStep": last["step"],
        "splats": last.get("splats", 0),
        "loss": last.get("loss", 0.0),
        "lfsCommit": LFS_COMMIT,
        "lfsExe": exe,
    }
    if is_resuming:
        out["resumedFromCheckpoint"] = str(project_file)
    out.update({k: prep[k] for k in ("seedPoints", "seedPointsUsed", "droppedPanoramas", "droppedViews",
                                     "trainingViews", "viewBudget", "panoramasUsed") if k in prep})
    return out


def train_lfs_tiles(ctx: StageContext, num_tiles: int, prep: Dict[str, Any]) -> Dict[str, Any]:
    """Runs multi-tile LichtFeld training sequentially for VRAM safety, then merges."""
    from . import tiles

    ws = ctx.workspace
    tr = ctx.manifest["training"]
    exe = ctx.tools.lichtfeld
    if not exe or not os.path.isfile(exe):
        raise StageFailed("LichtFeld-Studio executable not found.")

    total_its = int(tr.get("iterations", 30000))
    lfs_out = ws.training_dir / "lfs"
    overlap_fraction = float(tr.get("tileOverlap", 0.15))

    # Tiled runs always use MCMC: MRNF gradient densification saturates at a small
    # fraction of the per-tile cap inside a cropbox, leaving the merged model under-dense.
    tile_tr = dict(tr)
    tile_tr["lfsStrategy"] = "mcmc"

    tile_specs = tiles.plan_tiles_from_transforms(ws.processed_dir / "transforms.json", num_tiles,
                                                  overlap_fraction=overlap_fraction)
    ctx.emitter.log(
        f"Planned {num_tiles} spatial tiles along camera trajectory "
        f"(overlap {overlap_fraction:.0%} of walk extent, strategy: mcmc forced)",
        stage="training"
    )

    max_cap = int(tr.get("maxGaussians", 2_000_000))
    cap_per_tile = int((max_cap / num_tiles) * (1.0 + overlap_fraction * num_tiles))

    tile_plys: List[Path] = []
    tile_times: List[float] = []

    for tile_idx, spec in enumerate(tile_specs):
        tile_out = lfs_out / f"tile_{tile_idx}"
        target_ply = tile_out / f"splat_{total_its}.ply"
        tile_plys.append(target_ply)

        project_file = tile_out / "project.licht"
        is_resuming = bool(ctx.results.get("_resume_training")) and project_file.is_file()

        if target_ply.is_file():
            ctx.emitter.log(f"Tile {tile_idx + 1}/{num_tiles} already completed ({target_ply.name} exists); skipping.",
                            stage="training")
            tile_times.append(0.0)
            continue

        if not is_resuming:
            ws.reset_dir(tile_out)
            tile_out.mkdir(parents=True, exist_ok=True)
        else:
            ctx.emitter.log(f"resuming Tile {tile_idx + 1}/{num_tiles} from checkpoint {project_file}", stage="training")

        tile_tr["maxGaussians"] = cap_per_tile
        args = build_cmd(exe, ws.processed_dir, tile_out, tile_tr,
                         resume=is_resuming, project_path=project_file,
                         cropbox=spec.crop_box_lfs_str)

        last = {"pct": -1.0, "step": 0, "splats": 0, "loss": 0.0}
        tail: List[str] = []

        def on_line(l: str) -> None:
            tail.append(l)
            del tail[:-40]
            m = LFS_PROGRESS_RE.search(l)
            if m:
                step = int(m.group(1))
                loss = float(m.group(3))
                splats = int(m.group(4))
                last["step"] = step
                last["splats"] = splats
                last["loss"] = loss
                overall_step = tile_idx * total_its + step
                pct = 100.0 * overall_step / max(1, num_tiles * total_its)
                if pct - last["pct"] >= 0.25:
                    last["pct"] = pct
                    ctx.emitter.progress("training", pct,
                                         f"[tile {tile_idx + 1}/{num_tiles}] step {step} / {total_its} | Loss: {loss:.4f} | Splats: {splats:,}")
            elif "out of memory" in l.lower() or "cuda error: out of memory" in l.lower():
                ctx.emitter.warning("CUDA out of memory in LichtFeld-Studio tile training.", stage="training")

        kwargs: Dict[str, Any] = {}
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True

        finished = threading.Event()
        t0 = time.perf_counter()
        with ws.open_log(f"training_tile_{tile_idx}.log") as log:
            if log is not None:
                log.write("\n$ " + subprocess.list2cmdline(args) + "\n")
                log.flush()

            p = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 cwd=str(ws.root), **kwargs)

            def watch_cancel() -> None:
                while not finished.is_set():
                    if ctx.cancel.wait(0.5):
                        if finished.is_set():
                            return
                        ctx.emitter.progress("training", None, f"pausing tile {tile_idx + 1} - saving checkpoint (CTRL_BREAK)")
                        try:
                            if os.name == "nt":
                                os.kill(p.pid, signal.CTRL_BREAK_EVENT)
                            else:
                                os.kill(p.pid, signal.SIGINT)
                        except OSError:
                            pass
                        if not finished.wait(TRAIN_PAUSE_GRACE_S):
                            procrun.kill_tree(p)
                        return

            watcher = threading.Thread(target=watch_cancel, daemon=True)
            watcher.start()

            try:
                assert p.stdout is not None
                buf = b""
                while True:
                    chunk = p.stdout.read1(65536) if hasattr(p.stdout, "read1") else p.stdout.read(4096)
                    if not chunk:
                        break
                    buf += chunk
                    while True:
                        idx_sep = min([i for i in (buf.find(b"\n"), buf.find(b"\r")) if i >= 0], default=-1)
                        if idx_sep < 0:
                            break
                        raw, buf = buf[:idx_sep], buf[idx_sep + 1:]
                        procrun._deliver(raw, on_line, log)
                    if len(buf) > 1 << 20:
                        procrun._deliver(buf, on_line, log)
                        buf = b""
                if buf:
                    procrun._deliver(buf, on_line, log)
                rc = p.wait()
            finally:
                finished.set()
                watcher.join(timeout=2)
                if p.poll() is None:
                    procrun.kill_tree(p)
                if p.stdout is not None:
                    p.stdout.close()

        dur = time.perf_counter() - t0
        tile_times.append(dur)

        if ctx.cancel.cancelled:
            raise procrun.Cancelled()

        if rc != 0 or not target_ply.is_file():
            hint = " | ".join(tail[-3:])
            raise StageFailed(f"LichtFeld-Studio tile {tile_idx + 1} failed (exit {rc}) - see logs/training_tile_{tile_idx}.log. Last output: {hint[-400:]}")

    # Merge per-tile PLYs
    merged_lfs_ply = lfs_out / f"splat_{total_its}.ply"
    ctx.emitter.progress("training", 100.0, f"merging {num_tiles} tile PLYs with core-box deduplication")
    total_kept = tiles.merge_tile_plys(tile_plys, tile_specs, merged_lfs_ply)
    ctx.emitter.log(f"Merged {num_tiles} spatial tiles: {total_kept:,} splats (core-box deduplicated)", stage="training")

    out: Dict[str, Any] = {
        "engine": "lichtfeld",
        "strategy": "mcmc",  # tiled runs always use mcmc (forced)
        "strategyForced": "mcmc",
        "iterations": total_its,
        "splats": total_kept,
        "tiles": num_tiles,
        "tileOverlapFraction": overlap_fraction,
        "tileTimes": tile_times,
        "lfsCommit": LFS_COMMIT,
        "lfsExe": exe,
    }
    out.update({k: prep[k] for k in ("seedPoints", "seedPointsUsed", "droppedPanoramas", "droppedViews",
                                     "trainingViews", "viewBudget", "panoramasUsed") if k in prep})
    return out


def export_lfs(ctx: StageContext) -> Dict[str, Any]:
    """Finds LFS exported PLY, normalises frame via post-rotation, and writes sidecar."""
    ws = ctx.workspace
    tr = ctx.manifest["training"]
    lfs_out = ws.training_dir / "lfs"

    # Find the exported PLY (must be the completed iteration target, not an interrupted partial)
    target_name = "splat_%d.ply" % int(tr.get("iterations", 30000))
    cand = lfs_out / target_name
    if not cand.is_file():
        raise StageFailed("Expected exported PLY %s missing in %s after LichtFeld training." % (target_name, lfs_out))

    ws.export_dir.mkdir(parents=True, exist_ok=True)
    dst = ws.export_ply

    ctx.emitter.progress("export", None, "normalising PLY orientation (Z-up transforms frame)")
    rotate_ply_zup(cand, dst)

    # Write sidecar <ply>.pf360.json
    strategy = tr.get("lfsStrategy", "mrnf")
    sidecar = {
        "engine": "lichtfeld",
        "strategy": strategy,
        "frame": "transforms",
        "upAxis": "z_up",
        "lfsCommit": LFS_COMMIT,
    }
    if int(tr.get("tiles", 1)) > 1:
        sidecar["tiles"] = int(tr["tiles"])
        sidecar["tileOverlapFraction"] = float(tr.get("tileOverlap", 0.15))

    sidecar_path = Path(str(dst) + ".pf360.json")
    with open(sidecar_path, "w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=2)

    exe = ctx.tools.lichtfeld
    sha = file_sha256(Path(exe)) if exe and os.path.isfile(exe) else ""

    res: Dict[str, Any] = {
        "path": str(dst),
        "fileSizeBytes": dst.stat().st_size,
        "engine": "lichtfeld",
        "strategy": strategy,
        "upAxis": "z_up",
        "lfsCommit": LFS_COMMIT,
        "lfsExeSha256": sha,
    }
    if int(tr.get("tiles", 1)) > 1:
        res["tiles"] = int(tr["tiles"])
    return res

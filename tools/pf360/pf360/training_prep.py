"""Training-data clean-up between the camera stage and Splatfacto (stdlib only).

Two problems seen on a 2,000-panorama GLOMAP solve (24 Nov 2026):

1. Seed cloud too large. A complete camera solve triangulated 3,357,891 points; Splatfacto starts
   one Gaussian per point, and splatfacto-big at full resolution ran out of 24 GB of VRAM within
   minutes. The earlier (partial) solve had 108,004 points and trained fine. -> thin the seed
   cloud on a voxel grid to at most `cap` points; densification restores detail where it is needed.

2. Misplaced panoramas. All 14 views of a panorama were captured from one spot, so their camera
   centres must coincide. In 56 panoramas they were scattered up to 28 scene units apart (the
   global solver placed the views independently and wrongly), and the path jumped out and back.
   Training on them smears that part of the scene and inflates the scene box. -> drop panoramas
   whose views disagree or whose position spikes away from its neighbours.

3. Image cache larger than RAM. Splatfacto decodes and keeps EVERY training and eval image in
   RAM (1088 x 1088 uint8 = 3.5 MB each). 27,202 views = ~97 GB; on a 127 GB PC caching the eval
   images raised MemoryError. -> keep evenly spaced whole panoramas within a RAM budget (consecutive
   panoramas at 10 fps are a few cm apart, so the views overlap heavily).

Nothing here touches the camera solve: the original transforms.json is kept as
transforms_all.json and every run starts from it, so the step is idempotent and safe to repeat
on resume.
"""
import json
import math
import os
import random
import re
import statistics
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

ALL_TRANSFORMS = "transforms_all.json"
TRAIN_PLY = "sparse_pc_train.ply"
DEFAULT_SEED_POINT_CAP = 500_000
MAX_DROP_FRACTION = 0.25          # dropping more than this means the test, not the cameras, is wrong
RAM_FRACTION = 0.5                # share of physical RAM Splatfacto's decoded image cache may use
SPIKE_WINDOW = 8                 # neighbours on each side for the local-median position test
SPIKE_FACTOR = 40.0               # x median step between panoramas
RIG_FACTOR = 2.0                  # views of one panorama may disagree by this x median step

Point = Tuple[float, float, float]
_FRAME_RE = re.compile(r"frame_(\d+)\.")

_PLY_TYPES = {"char": "b", "int8": "b", "uchar": "B", "uint8": "B", "short": "h", "int16": "h",
              "ushort": "H", "uint16": "H", "int": "i", "int32": "i", "uint": "I", "uint32": "I",
              "float": "f", "float32": "f", "double": "d", "float64": "d"}


# ------------------------------------------------------------------ PLY I/O
def read_ply(path: Path) -> Tuple[List[Point], Optional[List[Tuple[int, int, int]]]]:
    """Vertices (x, y, z) and optional (r, g, b) of an ASCII or binary_little_endian PLY
    whose first element is `vertex` (what ns-process-data writes)."""
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            raise ValueError("%s is not a PLY file" % path)
        fmt, count, props, in_vertex = None, 0, [], False
        while True:
            line = f.readline()
            if not line:
                raise ValueError("%s: PLY header has no end_header" % path)
            w = line.decode("ascii", "replace").split()
            if not w:
                continue
            if w[0] == "format":
                fmt = w[1]
            elif w[0] == "element":
                in_vertex = w[1] == "vertex"    # other elements (if any) follow the vertices
                if in_vertex:
                    count = int(w[2])
            elif w[0] == "property" and in_vertex:
                if w[1] == "list":
                    raise ValueError("%s: list properties on vertices are not supported" % path)
                props.append((w[2], w[1]))
            elif w[0] == "end_header":
                break
        names = [n for n, _ in props]
        try:
            ix = [names.index(a) for a in ("x", "y", "z")]
        except ValueError:
            raise ValueError("%s: vertices have no x/y/z" % path)
        ic = [names.index(a) for a in ("red", "green", "blue")] if all(
            a in names for a in ("red", "green", "blue")) else None
        pts: List[Point] = []
        cols: Optional[List[Tuple[int, int, int]]] = [] if ic else None
        if fmt == "ascii":
            for _ in range(count):
                v = f.readline().split()
                pts.append((float(v[ix[0]]), float(v[ix[1]]), float(v[ix[2]])))
                if ic:
                    cols.append((int(float(v[ic[0]])), int(float(v[ic[1]])), int(float(v[ic[2]]))))
        elif fmt == "binary_little_endian":
            rec = struct.Struct("<" + "".join(_PLY_TYPES[t] for _, t in props))
            data = f.read(rec.size * count)
            if len(data) < rec.size * count:
                raise ValueError("%s: truncated vertex data" % path)
            for v in rec.iter_unpack(data):
                pts.append((float(v[ix[0]]), float(v[ix[1]]), float(v[ix[2]])))
                if ic:
                    cols.append((int(v[ic[0]]), int(v[ic[1]]), int(v[ic[2]])))
        else:
            raise ValueError("%s: unsupported PLY format %s" % (path, fmt))
    return pts, cols


def write_ply(path: Path, pts: Sequence[Point], cols: Optional[Sequence[Tuple[int, int, int]]]) -> None:
    """ASCII PLY in the same layout ns-process-data writes (x y z [red green blue])."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="ascii", newline="\n") as f:
        f.write("ply\nformat ascii 1.0\nelement vertex %d\n" % len(pts))
        f.write("property float x\nproperty float y\nproperty float z\n")
        if cols is not None:
            f.write("property uint8 red\nproperty uint8 green\nproperty uint8 blue\n")
        f.write("end_header\n")
        for i, p in enumerate(pts):
            if cols is not None:
                c = cols[i]
                f.write("%.6f %.6f %.6f %d %d %d\n" % (p[0], p[1], p[2], c[0], c[1], c[2]))
            else:
                f.write("%.6f %.6f %.6f\n" % p)
    tmp.replace(path)


# ------------------------------------------------------------ seed thinning
def _voxel_keep(pts: Sequence[Point], size: float) -> List[int]:
    """Index of the first point in every occupied voxel of edge `size`."""
    seen: Set[Tuple[int, int, int]] = set()
    keep: List[int] = []
    inv = 1.0 / size
    for i, (x, y, z) in enumerate(pts):
        k = (math.floor(x * inv), math.floor(y * inv), math.floor(z * inv))
        if k not in seen:
            seen.add(k)
            keep.append(i)
    return keep


def thin_indices(pts: Sequence[Point], cap: int, seed: int = 42) -> List[int]:
    """Indices of at most `cap` points, spread evenly in space (voxel grid; the finest voxel size
    that fits under the cap, found by bisection on a log scale). Plain random sampling would keep
    the dense, well-seen areas dense and starve the sparse ones; a voxel grid keeps coverage."""
    n = len(pts)
    if cap <= 0 or n <= cap:
        return list(range(n))
    lo_b = [min(p[a] for p in pts) for a in range(3)]
    hi_b = [max(p[a] for p in pts) for a in range(3)]
    diag = math.dist(lo_b, hi_b) or 1.0
    small, big = diag * 1e-7, diag          # too fine (count > cap) ... coarse enough (count <= cap)
    best = _voxel_keep(pts, big)
    for _ in range(24):
        mid = math.sqrt(small * big)
        keep = _voxel_keep(pts, mid)
        if len(keep) > cap:
            small = mid
        else:
            big, best = mid, keep
        if big / small < 1.03:
            break
    if len(best) > cap:                      # only if every point shares one voxel at `diag`
        best = sorted(random.Random(seed).sample(best, cap))
    return best


# --------------------------------------------------- misplaced panoramas
def panorama_centres(frames: Sequence[Dict[str, Any]], names: Sequence[str]
                     ) -> Dict[str, List[Tuple[int, Point]]]:
    """frame index -> (panorama key, camera centre). ns-process-data copies the planar views in
    sorted order and renames them frame_00001.jpg..., so view i is names[i-1]; its panorama is the
    name without the trailing _<view>."""
    out: Dict[str, List[Tuple[int, Point]]] = {}
    for fi, fr in enumerate(frames):
        m = _FRAME_RE.search(str(fr.get("file_path", "")))
        mat = fr.get("transform_matrix")
        if not m or not mat:
            continue
        i = int(m.group(1)) - 1
        if not 0 <= i < len(names):
            continue
        key = Path(names[i]).stem.rpartition("_")[0] or Path(names[i]).stem
        out.setdefault(key, []).append((fi, (float(mat[0][3]), float(mat[1][3]), float(mat[2][3]))))
    return out


def find_misplaced(pans: Dict[str, List[Tuple[int, Point]]]) -> Tuple[List[str], Dict[str, Any]]:
    """Panoramas to drop + diagnostics. Two tests, both relative to the walk's own median step
    between consecutive panoramas, so they are independent of the solve's arbitrary scale:
    - rig: a panorama's view centres must coincide (they share one capture position);
    - spike: its centre must lie near the median of its neighbours' centres."""
    keys = sorted(pans)
    info: Dict[str, Any] = {"panoramas": len(keys)}
    if len(keys) < 5:
        info["skipped"] = "too few panoramas"
        return [], info
    centre = {k: tuple(statistics.fmean(p[a] for _, p in pans[k]) for a in range(3)) for k in keys}
    spread = {k: max(math.dist(p, centre[k]) for _, p in pans[k]) for k in keys}
    steps = [math.dist(centre[a], centre[b]) for a, b in zip(keys, keys[1:])]
    step = statistics.median(steps)
    multi = [spread[k] for k in keys if len(pans[k]) > 1]
    med_spread = statistics.median(multi) if multi else 0.0
    info.update(medianStep=step, medianRigSpread=med_spread)
    if step <= 0:
        info["skipped"] = "camera does not move"
        return [], info
    if med_spread > step:
        # Views of a typical panorama disagree more than consecutive panoramas: either the
        # view->panorama mapping is not what we assume or the rig model does not hold. Do nothing.
        info["skipped"] = "views of a panorama do not share a position"
        return [], info
    rig = {k for k in keys if len(pans[k]) > 1 and spread[k] > RIG_FACTOR * step}
    spikes = set()
    for i, k in enumerate(keys):
        nb = [centre[keys[j]] for j in range(max(0, i - SPIKE_WINDOW), min(len(keys), i + SPIKE_WINDOW + 1))
              if j != i]
        med = tuple(statistics.median(p[a] for p in nb) for a in range(3))
        if math.dist(centre[k], med) > SPIKE_FACTOR * step:
            spikes.add(k)
    bad = sorted(rig | spikes)
    info.update(rigInconsistent=len(rig), positionSpikes=len(spikes))
    if len(bad) > MAX_DROP_FRACTION * len(keys):
        info["skipped"] = "would drop %d of %d panoramas" % (len(bad), len(keys))
        return [], info
    return bad, info


# ------------------------------------------------------------------ driver
def _planar_names(planar_dir: Path) -> List[str]:
    exts = {".jpg", ".jpeg", ".png"}
    return sorted(p.name for p in planar_dir.iterdir() if p.suffix.lower() in exts) if planar_dir.is_dir() else []


def total_ram_bytes() -> int:
    """Physical memory of this machine (0 if unknown)."""
    try:
        if hasattr(os, "sysconf") and "SC_PHYS_PAGES" in os.sysconf_names:
            return int(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
        import ctypes

        class _MemStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        st_ = _MemStatus()
        st_.dwLength = ctypes.sizeof(_MemStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st_)):
            return int(st_.ullTotalPhys)
    except Exception:
        pass
    return 0


def view_budget(t: Dict[str, Any], downscale: int, ram_bytes: int, fraction: float = RAM_FRACTION) -> int:
    """How many views Splatfacto can cache: it keeps EVERY training and eval image decoded in RAM
    (uint8 RGB at the training resolution). 0 = unknown / no limit."""
    frames = t.get("frames") or []
    w = t.get("w") or (frames[0].get("w") if frames else None)
    h = t.get("h") or (frames[0].get("h") if frames else None)
    if not w or not h or ram_bytes <= 0:
        return 0
    per_view = (int(w) // max(1, downscale)) * (int(h) // max(1, downscale)) * 3
    return max(1, int(ram_bytes * fraction // per_view))


def subsample_panoramas(frames: Sequence[Dict[str, Any]], pans: Dict[str, List[Tuple[int, Point]]],
                        max_views: int) -> Tuple[List[Dict[str, Any]], int]:
    """Evenly spaced whole panoramas (all their views) so that at most `max_views` views remain.
    Returns the kept frames and how many panoramas were kept."""
    by_frame = {fi: k for k, v in pans.items() for fi, _ in v}
    keys = sorted({by_frame[i] for i in range(len(frames)) if i in by_frame})
    if not keys:
        return list(frames), 0
    per = max(1, round(len(by_frame) / len(keys)))
    n = max(1, min(len(keys), max_views // per))
    chosen = {keys[int(i * len(keys) / n)] for i in range(n)}
    kept = [fr for i, fr in enumerate(frames) if by_frame.get(i) in chosen]
    return kept, len(chosen)


def prepare(processed_dir: Path, planar_dir: Path, *, seed_cap: int = DEFAULT_SEED_POINT_CAP,
            drop_misplaced: bool = True, max_views: int = 0, apply_ram_budget: bool = True,
            downscale: int = 1, ram_bytes: Optional[int] = None,
            seed: int = 42, log=None) -> Dict[str, Any]:
    """Writes processed/transforms.json (training view) from transforms_all.json (the camera
    stage's output, saved on first call) and, when thinning, sparse_pc_train.ply. Returns stats."""
    say = log or (lambda _m: None)
    tf, all_tf = processed_dir / "transforms.json", processed_dir / ALL_TRANSFORMS
    if not all_tf.is_file():
        if not tf.is_file():
            return {"skipped": "no transforms.json"}
        tf.replace(all_tf)
    with open(all_tf, "r", encoding="utf-8") as f:
        t = json.load(f)
    frames = list(t.get("frames") or [])
    stats: Dict[str, Any] = {"views": len(frames)}

    pans: Optional[Dict[str, List[Tuple[int, Point]]]] = None
    names = _planar_names(planar_dir)
    images_dir = processed_dir / "images"
    copied = sum(1 for p in images_dir.iterdir() if p.is_file()) if images_dir.is_dir() else len(names)
    if names and copied == len(names):
        pans = panorama_centres(frames, names)
    mismatch = "view names do not match the training images (%d vs %d)" % (len(names), copied)

    if drop_misplaced:
        if pans is not None:
            bad, info = find_misplaced(pans)
            stats["misplaced"] = info
            if bad:
                drop = {fi for k in bad for fi, _ in pans[k]}
                keep_idx = [i for i in range(len(frames)) if i not in drop]
                # Re-index the panorama map to the surviving frames.
                remap = {old: new for new, old in enumerate(keep_idx)}
                pans = {k: [(remap[fi], p) for fi, p in v if fi in remap] for k, v in pans.items() if k not in bad}
                frames = [frames[i] for i in keep_idx]
                stats["droppedPanoramas"] = len(bad)
                stats["droppedViews"] = len(drop)
                stats["droppedPanoramaNames"] = bad[:200]
                say("dropped %d misplaced panorama(s) (%d views): %s%s"
                    % (len(bad), len(drop), ", ".join(bad[:12]), " ..." if len(bad) > 12 else ""))
            elif info.get("skipped"):
                say("misplaced-panorama check skipped: %s" % info["skipped"])
        else:
            stats["misplaced"] = {"skipped": mismatch}

    if max_views <= 0:                           # auto: what fits in RAM (Nerfstudio only)
        if apply_ram_budget:
            max_views = view_budget(t, downscale, total_ram_bytes() if ram_bytes is None else ram_bytes)
        else:
            stats["viewBudget"] = "none (lichtfeld)"
            say("training on %d of %d views (no RAM view budget for LichtFeld)" % (len(frames), stats["views"]))
    if max_views > 0 and len(frames) > max_views:
        before = len(frames)
        if pans is not None:
            frames, kept_pans = subsample_panoramas(frames, pans, max_views)
            stats["panoramasUsed"] = kept_pans
        else:                                   # no panorama map: evenly spaced single views
            frames = [frames[int(i * before / max_views)] for i in range(max_views)]
        stats["viewBudget"] = max_views
        say("training on %d of %d views (evenly spaced%s) so every image fits in RAM (budget %d views)"
            % (len(frames), before, ", %d panoramas" % stats["panoramasUsed"] if "panoramasUsed" in stats else "",
               max_views))

    out = dict(t)
    out["frames"] = frames
    ply_rel = t.get("ply_file_path")
    if ply_rel and seed_cap > 0 and (processed_dir / ply_rel).is_file():
        pts, cols = read_ply(processed_dir / ply_rel)
        stats["seedPoints"] = len(pts)
        if len(pts) > seed_cap:
            keep = thin_indices(pts, seed_cap, seed)
            write_ply(processed_dir / TRAIN_PLY, [pts[i] for i in keep],
                      [cols[i] for i in keep] if cols is not None else None)
            out["ply_file_path"] = TRAIN_PLY
            stats["seedPointsUsed"] = len(keep)
            say("seed point cloud thinned %d -> %d points (cap %d)" % (len(pts), len(keep), seed_cap))
        else:
            stats["seedPointsUsed"] = len(pts)
    tmp = tf.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=4)
    tmp.replace(tf)
    stats["trainingViews"] = len(frames)
    return stats

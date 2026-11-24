"""Runs INSIDE the Nerfstudio Python environment (needs numpy).

Panorama rig completion for COLMAP models of 360-video planar views.

All V perspective views of one panorama (frame_NNNNN_0 .. _{V-1}) were rendered
by Nerfstudio's equirect2persp from ONE camera centre with KNOWN relative
rotations (yaw/pitch per view). COLMAP 3.11 has no rig constraint, so on real
footage it typically registers only some view directions of each panorama
(e.g. only the views that see textured surfaces) and drops the rest. Here every
panorama with at least one registered view gets ALL its views posed:

    M_i  = rotation camera(OpenCV: x right, y down, z forward) -> panorama frame
    R_cw = M_i^T Q    (Q: panorama frame -> world, shared by the panorama's views)

Q is estimated from each registered view (Q_i = M_i R_cw_i) and averaged; the
centre is the mean of the registered views' centres. Panoramas whose views
disagree by more than --max-deg are left untouched (a sign of a bad
registration). Registered views keep their bundle-adjusted poses.

Model merging (--sparse): on real walking footage COLMAP also splits the walk
into many models that cover the SAME stretch of time - e.g. one holds the
forward-facing views, another the backward-facing ones, because those views
share no features. A panorama registered in two models ties them together:
its rig rotation Q gives the models' relative rotation, and the panorama centres
give scale + offset. Models are joined greedily (most shared panoramas first)
into the largest one; models that share too few panoramas are left out. The
joined model is then completed as above. Output images carry no 2D-3D links and
points3D are the transformed points of all joined models without tracks; run
`colmap point_triangulator` (poses fixed) on it for proper tracks.

usage:
  python rig_complete.py --sparse <sparse> [--base <model>] --db <database.db> --views '[[yaw,pitch],...]' --out <dir>
  python rig_complete.py --model <sparse/K> --db <database.db> --views '[[yaw,pitch],...]' --out <dir>
  python rig_complete.py --model <sparse/K> --views '[[...]]' --verify
prints: PF360RIG{json stats}
"""
import argparse
import json
import math
import shutil
import sqlite3
import struct
import sys
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- rotations
def er_matrix(axis, theta):
    """Nerfstudio's euler_rodriguez_rotation_matrix (equirect_utils.py), numpy copy."""
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / math.sqrt(float(axis @ axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


# OpenCV camera ray (x right, y down, z forward) -> equirect2persp's local ray
# (x forward, y right, z up).
P_CV_TO_LOCAL = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


def view_rotation(yaw_deg, pitch_deg):
    """Camera(OpenCV) -> panorama-frame rotation of one planar view (equirect2persp)."""
    r1 = er_matrix([0.0, 0.0, 1.0], math.radians(yaw_deg))
    r2 = er_matrix(r1 @ np.array([0.0, 1.0, 0.0]), math.radians(-pitch_deg))
    return r2 @ r1 @ P_CV_TO_LOCAL


def project_orthogonal(m, det_sign):
    """Nearest orthogonal matrix with determinant `det_sign` (chordal mean of a sum of samples).

    equirect2persp's local frame (x forward, y right, z up) is LEFT-handed, so the
    panorama matrices Q = M_i R_cw are reflections (det -1); camera rotations
    R_cw = M_i^T Q are proper again."""
    u, _, vt = np.linalg.svd(m)
    r = u @ vt
    if np.sign(np.linalg.det(r)) != det_sign:
        u[:, -1] *= -1
        r = u @ vt
    return r


def mean_frame(samples):
    return project_orthogonal(sum(samples), np.sign(np.linalg.det(samples[0])))


def angle_deg(r):
    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(r) - 1.0) / 2.0))))


def qvec_to_rot(q):
    w, x, y, z = q
    return np.array([[1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
                     [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
                     [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]])


def rot_to_qvec(r):
    """COLMAP read_write_model.rotmat2qvec (w, x, y, z)."""
    rxx, ryx, rzx, rxy, ryy, rzy, rxz, ryz, rzz = np.asarray(r, dtype=np.float64).flat
    k = np.array([[rxx - ryy - rzz, 0, 0, 0],
                  [ryx + rxy, ryy - rxx - rzz, 0, 0],
                  [rzx + rxz, rzy + ryz, rzz - rxx - ryy, 0],
                  [ryz - rzy, rzx - rxz, rxy - ryx, rxx + ryy + rzz]]) / 3.0
    vals, vecs = np.linalg.eigh(k)
    q = vecs[[3, 0, 1, 2], np.argmax(vals)]
    return q if q[0] >= 0 else -q


# --------------------------------------------------------------------------- COLMAP binary I/O
def read_images_bin(path):
    images = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            iid, qw, qx, qy, qz, tx, ty, tz, cam = struct.unpack("<idddddddi", f.read(64))
            name = b""
            while True:
                ch = f.read(1)
                if ch == b"\x00":
                    break
                name += ch
            npts = struct.unpack("<Q", f.read(8))[0]
            pts = f.read(24 * npts)
            images[iid] = {"q": (qw, qx, qy, qz), "t": (tx, ty, tz), "cam": cam, "name": name.decode("utf-8"),
                           "npts": npts, "pts": pts}
    return images


def write_images_bin(path, images):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(images)))
        for iid in sorted(images):
            im = images[iid]
            f.write(struct.pack("<idddddddi", iid, *im["q"], *im["t"], im["cam"]))
            f.write(im["name"].encode("utf-8") + b"\x00")
            f.write(struct.pack("<Q", im["npts"]))
            f.write(im["pts"])


def read_points3d_bin(path):
    """{id: (xyz ndarray, rgb bytes, error)} - tracks are dropped."""
    pts = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            pid, x, y, z, r, g, b, err, tl = struct.unpack("<QdddBBBdQ", f.read(51))
            f.seek(8 * tl, 1)
            pts[pid] = (np.array([x, y, z]), bytes((r, g, b)), err)
    return pts


def write_points3d_bin(path, pts):
    """pts: iterable of (xyz, rgb bytes, error); written with empty tracks."""
    pts = list(pts)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(pts)))
        for i, (xyz, rgb, err) in enumerate(pts, 1):
            f.write(struct.pack("<Qddd", i, *(float(v) for v in xyz)) + rgb + struct.pack("<dQ", err, 0))


def split_name(name):
    """frame_00012_7.jpg -> ("frame_00012", 7); None if not a planar view name."""
    stem = name.rsplit(".", 1)[0]
    head, _, idx = stem.rpartition("_")
    return (head, int(idx)) if head and idx.isdigit() else None


# --------------------------------------------------------------------------- core
def panorama_estimates(images, views):
    """{panorama: [(view, Q_i, C_i)]} for registered views."""
    rots = [view_rotation(y, p) for y, p in views]
    out = {}
    for im in images.values():
        k = split_name(im["name"])
        if not k or k[1] >= len(views):
            continue
        rcw = qvec_to_rot(im["q"])
        c = -rcw.T @ np.asarray(im["t"])
        out.setdefault(k[0], []).append((k[1], rots[k[1]] @ rcw, c))
    return out, rots


def verify(images, views):
    """Spread of the per-view panorama rotations on a model: ~0 deg if the convention is right."""
    est, _ = panorama_estimates(images, views)
    worst, spreads = 0.0, []
    for lst in est.values():
        if len(lst) < 2:
            continue
        q = mean_frame([e[1] for e in lst])
        s = max(angle_deg(q.T @ e[1]) for e in lst)
        spreads.append(s)
        worst = max(worst, s)
    return {"panoramas": len(spreads), "medianSpreadDeg": float(np.median(spreads)) if spreads else None,
            "maxSpreadDeg": worst}


def complete(images, views, name_to_id, max_deg):
    est, rots = panorama_estimates(images, views)
    registered_views = len(images)
    added, skipped, used = 0, 0, 0
    out = dict(images)
    cam = next(iter(images.values()))["cam"] if images else 1
    for pano, lst in est.items():
        q = mean_frame([e[1] for e in lst])
        if max(angle_deg(q.T @ e[1]) for e in lst) > max_deg:
            skipped += 1
            continue
        used += 1
        c = np.mean([e[2] for e in lst], axis=0)
        have = {e[0] for e in lst}
        for v in range(len(views)):
            if v in have:
                continue
            name = "%s_%d.jpg" % (pano, v)
            iid = name_to_id.get(name)
            if iid is None or iid in out:
                continue
            rcw = rots[v].T @ q
            out[iid] = {"q": tuple(rot_to_qvec(rcw)), "t": tuple(-rcw @ c), "cam": cam, "name": name,
                        "npts": 0, "pts": b""}
            added += 1
    return out, {"registeredViews": registered_views, "addedViews": added, "completedViews": len(out),
                 "panoramasRegistered": len(est), "panoramasCompleted": used, "panoramasInconsistent": skipped}


# --------------------------------------------------------------------------- model merging
MIN_SHARED = 3          # panoramas two models must share to be joined
# Camera-path check. A walking 360 video moves the camera smoothly, so consecutive panorama
# centres lie on a smooth curve. On real footage COLMAP also builds large models from only the
# two downward views of each panorama (ground left/right or front/back, pitch -45): with fixed
# intrinsics the flat ground lets it converge to wrong poses, and the path zig-zags (jerk 0.75-1.9
# on the 2026-11-23 run, vs <= 0.3 for every model with 4+ view directions). Such a model was the
# largest and became the join reference, so every correct piece was rejected against it.
MAX_PATH_JERK = 0.5
MIN_PATH_TRIPLES = 8    # fewer equally spaced triples: too short to judge, kept


def pano_index(name):
    """frame_00012 -> 12 (None if the name has no trailing number)."""
    tail = name.rsplit("_", 1)[-1]
    return int(tail) if tail.isdigit() else None


def path_jerk(panos):
    """Median roughness of a model's camera path: |c+ - 2c + c-| / (|c+ - c-| / 2) over triples of
    panoramas with equal frame gaps (<= 3). ~0 for a smooth walk, ~1+ for a broken model.
    Returns (jerk or None, triples used)."""
    pts = sorted((pano_index(p), np.asarray(qc[1], dtype=np.float64)) for p, qc in panos.items()
                 if pano_index(p) is not None)
    tri = []
    for (i0, c0), (i1, c1), (i2, c2) in zip(pts, pts[1:], pts[2:]):
        g = i1 - i0
        if g != i2 - i1 or g > 3:
            continue
        tri.append((np.linalg.norm(c2 - c0) / (2.0 * g), np.linalg.norm(c2 - 2.0 * c1 + c0) / g))
    if not tri:
        return None, 0
    # Where the walker stands still the step is ~0 and the ratio is noise / noise: skip those.
    moving = float(np.median([t[0] for t in tri]))
    r = [acc / half for half, acc in tri if half > 0.2 * moving and half > 0]
    if len(r) < MIN_PATH_TRIPLES:
        return None, len(r)
    return float(np.median(r)), len(r)


def model_panoramas(images, views, max_deg):
    """{panorama: (Q, centre)} for the panoramas whose registered views agree."""
    est, _ = panorama_estimates(images, views)
    out = {}
    for pano, lst in est.items():
        q = mean_frame([e[1] for e in lst])
        if max(angle_deg(q.T @ e[1]) for e in lst) <= max_deg:
            out[pano] = (q, np.mean([e[2] for e in lst], axis=0))
    return out


def estimate_sim3(ref, mov, shared, max_deg, why=None):
    """Similarity x_ref = s * R @ x_mov + t from panoramas present in both models.

    R comes from the panorama rotations (Q_ref = Q_mov R^T, so every shared panorama
    gives a full estimate even when the walk is a straight line); s and t from the
    centres. Returns (R, s, t, info) or None when the models do not agree; `why` (a dict)
    then says which check failed, for the log."""
    why = {} if why is None else why
    rs = [ref[p][0].T @ mov[p][0] for p in shared]
    keep = list(range(len(rs)))
    r = None
    for _ in range(3):
        if len(keep) < MIN_SHARED:
            why.update(reason="rotation", rotInliers=len(keep))
            return None
        r = project_orthogonal(sum(rs[i] for i in keep), 1.0)
        keep = [i for i in range(len(rs)) if angle_deg(r.T @ rs[i]) <= max_deg]
    why["rotErrMedDeg"] = round(float(np.median([angle_deg(r.T @ x) for x in rs])), 2)
    if len(keep) < max(MIN_SHARED, len(rs) // 2):
        why.update(reason="rotation", rotInliers=len(keep))
        return None
    a = np.array([r @ mov[shared[i]][1] for i in keep])
    b = np.array([ref[shared[i]][1] for i in keep])
    sel = np.linspace(0, len(a) - 1, min(len(a), 200)).astype(int)     # bound the pair count
    ia, ib = np.triu_indices(len(sel), 1)
    da = np.linalg.norm(a[sel][ia] - a[sel][ib], axis=1)
    db = np.linalg.norm(b[sel][ia] - b[sel][ib], axis=1)
    if not len(da) or da.max() <= 0 or (da > 0.25 * da.max()).sum() < 2:
        why.update(reason="no-baseline")
        return None
    far = da > 0.25 * da.max()
    s = float(np.median(db[far] / da[far]))
    t = np.median(b - s * a, axis=0)
    res = np.linalg.norm(b - (s * a + t), axis=1)
    spread = float(np.sqrt(np.mean(np.sum((b - b.mean(axis=0)) ** 2, axis=1))))
    if spread <= 0 or s <= 0:
        why.update(reason="no-baseline")
        return None
    inl = res <= max(3.0 * float(np.median(res)), 1e-3 * spread)
    if inl.sum() < MIN_SHARED:
        why.update(reason="centres", centreInliers=int(inl.sum()))
        return None
    ai, bi = a[inl], b[inl]
    am, bm = ai.mean(axis=0), bi.mean(axis=0)
    den = float(np.sum((ai - am) ** 2))
    if den > 0:
        s = float(np.sum((ai - am) * (bi - bm)) / den)
    if s <= 0:
        why.update(reason="centres")
        return None
    t = bm - s * am
    rms = float(np.sqrt(np.mean(np.sum((bi - (s * ai + t)) ** 2, axis=1))))
    why.update(rmsRel=round(rms / spread, 3), scale=round(s, 4))
    if rms > 0.15 * spread:            # centres disagree: not the same stretch of the walk
        why.update(reason="centres")
        return None
    return r, s, t, {"shared": len(shared), "inliers": int(inl.sum()), "scale": s, "rmsRel": rms / spread}


def transform_image(im, r, s, t, cam):
    rcw = qvec_to_rot(im["q"])
    c = -rcw.T @ np.asarray(im["t"])
    rcw2 = rcw @ r.T
    c2 = s * (r @ c) + t
    return {"q": tuple(rot_to_qvec(rcw2)), "t": tuple(-rcw2 @ c2), "cam": cam, "name": im["name"],
            "npts": 0, "pts": b""}


def merge(models, views, max_deg, base_first=False):
    """models: [(name, images, points)]. Joins every model that can be tied to the reference
    (the one with most panoramas, or models[0] with base_first) through shared panoramas.
    Returns (images, points, stats)."""
    ests = [model_panoramas(m[1], views, max_deg) for m in models]
    jerk = [path_jerk(e)[0] for e in ests]
    broken = {i for i, j in enumerate(jerk) if j is not None and j > MAX_PATH_JERK}
    if base_first:
        broken.discard(0)              # the extended model was grown from a checked reference
    if len(broken) == len(models):     # nothing passes: keep the old behaviour rather than nothing
        broken = set()
    order = sorted((i for i in range(len(models)) if i not in broken),
                   key=lambda i: (len(ests[i]), len(models[i][1])), reverse=True)
    if base_first and models:
        order = [0] + [i for i in order if i != 0]
    if not order:
        return {}, [], {"modelsTotal": 0, "modelsMerged": 0, "joins": []}
    ref = order[0]
    cam = next(iter(models[ref][1].values()))["cam"]
    eye = (np.eye(3), 1.0, np.zeros(3))
    imgs = {iid: transform_image(im, *eye, cam) for iid, im in models[ref][1].items()}
    pts = [(xyz, rgb, err) for xyz, rgb, err in models[ref][2].values()]
    panos = dict(ests[ref])
    pending = set(order[1:])
    joins = []
    while pending:
        best = None
        for i in pending:
            shared = sorted(p for p in ests[i] if p in panos)
            if len(shared) < MIN_SHARED:
                continue
            fit = estimate_sim3(panos, ests[i], shared, max_deg)
            if fit and (best is None or fit[3]["inliers"] > best[1][3]["inliers"]):
                best = (i, fit)
        if best is None:
            break
        i, (r, s, t, info) = best
        pending.discard(i)
        new_panos = 0
        for iid, im in models[i][1].items():
            k = split_name(im["name"])
            if iid in imgs or (k and k[0] in panos):      # keep the panorama as already joined
                continue
            imgs[iid] = transform_image(im, r, s, t, cam)
        for p, (q, c) in ests[i].items():
            if p not in panos:
                panos[p] = (q @ r.T, s * (r @ c) + t)
                new_panos += 1
        pts.extend((s * (r @ xyz) + t, rgb, err) for xyz, rgb, err in models[i][2].values())
        info.update(model=models[i][0], newPanoramas=new_panos)
        joins.append(info)
    # Why the rest stayed out (largest first), for the log.
    left = []
    for i in sorted(pending | broken, key=lambda i: len(models[i][1]), reverse=True)[:25]:
        shared = sorted(p for p in ests[i] if p in panos)
        why = {"model": models[i][0], "views": len(models[i][1]), "panoramas": len(ests[i]),
               "newPanoramas": len([p for p in ests[i] if p not in panos]), "shared": len(shared)}
        if i in broken:
            why.update(reason="broken-path", pathJerk=round(jerk[i], 2))
        elif len(shared) < MIN_SHARED:
            why["reason"] = "too-few-shared"
        else:
            estimate_sim3(panos, ests[i], shared, max_deg, why)
        left.append(why)
    return imgs, pts, {"modelsTotal": len(models), "modelsMerged": 1 + len(joins),
                       "referenceModel": models[ref][0], "joins": joins,
                       "unmergedModels": sorted(models[i][0] for i in pending | broken), "unmergedWhy": left,
                       "brokenModels": sorted(models[i][0] for i in broken),
                       "panoramasJoined": len(panos)}


def colmap_model_dirs(sparse):
    """COLMAP mapper output models (numeric sub-directories with images.bin)."""
    return sorted((d for d in Path(sparse).iterdir() if d.is_dir() and d.name.isdigit()
                   and (d / "images.bin").is_file()), key=lambda d: int(d.name))


def merge_main(a, views):
    dirs = colmap_model_dirs(a.sparse)
    if a.base:
        dirs = [Path(a.base)] + dirs
    if not dirs:
        print("pf360: no COLMAP models in %s" % a.sparse, file=sys.stderr)
        return 2
    models = [(d.name, read_images_bin(d / "images.bin"), read_points3d_bin(d / "points3D.bin")) for d in dirs]
    imgs, pts, mstats = merge(models, views, a.max_deg, base_first=bool(a.base))
    con = sqlite3.connect("file:%s?mode=ro" % Path(a.db).as_posix(), uri=True)
    name_to_id = {n: i for i, n in con.execute("SELECT image_id, name FROM images")}
    con.close()
    done, stats = complete(imgs, views, name_to_id, a.max_deg)
    stats["registeredViews"] = sum(len(m[1]) for m in models)         # COLMAP's placements, all models
    stats["joinedViews"] = len(imgs)
    stats.update(mstats)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ref = dirs[0] if a.base else next(d for d in dirs if d.name == mstats["referenceModel"])
    shutil.copyfile(ref / "cameras.bin", out / "cameras.bin")
    write_points3d_bin(out / "points3D.bin", pts)
    write_images_bin(out / "images.bin", done)
    print("PF360RIG" + json.dumps(stats), flush=True)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--sparse", help="directory of COLMAP models to merge (mapper --output_path)")
    ap.add_argument("--base", help="with --sparse: a model that is always the reference (an extended model)")
    ap.add_argument("--views", required=True, help="JSON [[yaw, pitch], ...] in view-index order")
    ap.add_argument("--db")
    ap.add_argument("--out")
    ap.add_argument("--max-deg", type=float, default=5.0)
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args(argv)
    views = json.loads(a.views)
    if a.sparse:
        return merge_main(a, views)
    if not a.model:
        ap.error("--model or --sparse is required")
    model = Path(a.model)
    images = read_images_bin(model / "images.bin")
    if a.verify:
        print("PF360RIG" + json.dumps(verify(images, views)), flush=True)
        return 0
    con = sqlite3.connect("file:%s?mode=ro" % Path(a.db).as_posix(), uri=True)
    name_to_id = {n: i for i, n in con.execute("SELECT image_id, name FROM images")}
    con.close()
    done, stats = complete(images, views, name_to_id, a.max_deg)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in ("cameras.bin", "points3D.bin"):
        shutil.copyfile(model / f, out / f)
    write_images_bin(out / "images.bin", done)
    print("PF360RIG" + json.dumps(stats), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

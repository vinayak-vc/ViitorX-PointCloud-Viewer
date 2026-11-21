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

usage:
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


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--views", required=True, help="JSON [[yaw, pitch], ...] in view-index order")
    ap.add_argument("--db")
    ap.add_argument("--out")
    ap.add_argument("--max-deg", type=float, default=5.0)
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args(argv)
    views = json.loads(a.views)
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

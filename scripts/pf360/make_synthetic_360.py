"""Render a SYNTHETIC stitched equirectangular 360 walk-through video for
end-to-end testing of the pf360 pipeline (FFmpeg -> 360 projection -> COLMAP ->
Splatfacto -> PLY) without real footage.

Scene: a textured box room (8 x 8 x 3 m, +Z up) with a few textured pillars for
parallax. The camera walks a closed loop at eye height while the panorama keeps a
fixed world orientation (like a stabilised stitched 360 camera). Textures are
multi-octave value noise in world units so SIFT finds plenty of features.

This only proves the pipeline mechanics; it is NOT a substitute for a real capture
when judging reconstruction quality.

Requirements: torch (CUDA if available, CPU otherwise) - run it with the pf360
Nerfstudio environment's python - and ffmpeg on PATH (or --ffmpeg).

  python make_synthetic_360.py out.mp4 --width 2880 --seconds 30 --fps 10
"""
import argparse
import math
import shutil
import subprocess
import sys

import torch
import torch.nn.functional as F

ROOM_MIN = (-4.0, -4.0, 0.0)
ROOM_MAX = (4.0, 4.0, 3.0)
# Axis-aligned pillars / blocks (min, max) for parallax.
BOXES = [
    ((-2.6, 1.2, 0.0), (-2.0, 1.8, 2.2)),
    ((1.8, 1.9, 0.0), (2.6, 2.5, 1.1)),
    ((0.6, -2.8, 0.0), (1.1, -2.3, 3.0)),
    ((-1.5, -2.2, 0.0), (-0.7, -1.4, 0.8)),
]


def make_textures(n, size, device, seed=7):
    """n RGB textures (n,3,size,size), multi-octave value noise in [0,1]."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    tex = torch.zeros(n, 3, size, size)
    amp, total = 1.0, 0.0
    for octave in (8, 16, 32, 64, 128, 256, 512):
        base = torch.rand(n, 3, octave, octave, generator=g)
        # Periodic bilinear upsampling (indices wrap mod octave) -> the texture
        # tiles seamlessly, so world-coordinate wrapping leaves no visible seam.
        x = torch.arange(size) * octave / size
        i0 = x.floor().long()
        f = (x - i0)
        i0 %= octave
        i1 = (i0 + 1) % octave
        rows = base[:, :, i0] * (1 - f)[None, None, :, None] + base[:, :, i1] * f[None, None, :, None]
        up = rows[:, :, :, i0] * (1 - f) + rows[:, :, :, i1] * f
        tex += amp * up
        total += amp
        amp *= 0.65
    tex /= total
    tex = (tex - tex.amin(dim=(2, 3), keepdim=True)) / (
        tex.amax(dim=(2, 3), keepdim=True) - tex.amin(dim=(2, 3), keepdim=True) + 1e-6)
    return tex.to(device)


def ray_dirs(width, height, device):
    """World-space unit directions for an equirect image, +Z up.
    Column -> yaw in [-pi, pi), row 0 -> pitch +pi/2 (zenith)."""
    xs = (torch.arange(width, device=device) + 0.5) / width
    ys = (torch.arange(height, device=device) + 0.5) / height
    yaw = (xs - 0.5) * 2 * math.pi
    pitch = (0.5 - ys) * math.pi
    pitch, yaw = torch.meshgrid(pitch, yaw, indexing="ij")
    cp = torch.cos(pitch)
    return torch.stack([cp * torch.sin(yaw), cp * torch.cos(yaw), torch.sin(pitch)], -1)


def slab(o, d, bmin, bmax, inside):
    """Ray/AABB: returns (t, axis, sign) of the hit. inside=True -> exit hit."""
    inv = 1.0 / torch.where(d.abs() < 1e-9, torch.full_like(d, 1e-9), d)
    t0 = (bmin - o) * inv
    t1 = (bmax - o) * inv
    tmin = torch.minimum(t0, t1)
    tmax = torch.maximum(t0, t1)
    if inside:
        t, axis = tmax.min(-1)
        return t, axis
    tn, axis = tmin.max(-1)
    tf = tmax.min(-1).values
    miss = (tn > tf) | (tn <= 1e-4)
    tn = torch.where(miss, torch.full_like(tn, float("inf")), tn)
    return tn, axis


def render(o, dirs, tex, texel_m=0.004):
    dev = dirs.device
    o_t = torch.tensor(o, device=dev)
    t, axis = slab(o_t, dirs, torch.tensor(ROOM_MIN, device=dev), torch.tensor(ROOM_MAX, device=dev), True)
    sign = torch.gather(dirs, -1, axis[..., None])[..., 0] > 0
    tex_id = axis * 2 + sign.long()  # 6 room faces
    for i, (bmin, bmax) in enumerate(BOXES):
        tb, ab = slab(o_t, dirs, torch.tensor(bmin, device=dev), torch.tensor(bmax, device=dev), False)
        closer = tb < t
        t = torch.where(closer, tb, t)
        axis = torch.where(closer, ab, axis)
        tex_id = torch.where(closer, torch.full_like(tex_id, 6 + i), tex_id)
    p = o_t + dirs * t[..., None]
    # in-plane coordinates of the hit face (drop the hit axis)
    u = torch.where(axis == 0, p[..., 1], p[..., 0])
    v = torch.where(axis == 2, p[..., 1], p[..., 2])
    size = tex.shape[-1]
    period = size * texel_m
    gu = (torch.remainder(u, period) / period) * 2 - 1
    gv = (torch.remainder(v, period) / period) * 2 - 1
    h, w = dirs.shape[:2]
    out = torch.zeros(3, h, w, device=dev)
    grid = torch.stack([gu, gv], -1)[None]
    for k in range(tex.shape[0]):
        m = tex_id == k
        if not m.any():
            continue
        s = F.grid_sample(tex[k:k + 1], grid, mode="bilinear", padding_mode="border", align_corners=False)[0]
        out = torch.where(m[None], s, out)
    # mild distance shading so depth reads in the image
    shade = (1.0 / (1.0 + 0.04 * t)).clamp(0.5, 1.0)
    return (out * shade[None]).clamp(0, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--width", type=int, default=2880)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--radius", type=float, default=1.3, help="walk loop radius (m)")
    ap.add_argument("--ffmpeg", default=shutil.which("ffmpeg") or "ffmpeg")
    a = ap.parse_args()
    width = a.width - a.width % 2
    height = width // 2
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tex = make_textures(6 + len(BOXES), 1024, dev)
    dirs = ray_dirs(width, height, dev)
    n = int(round(a.seconds * a.fps))
    cmd = [a.ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{width}x{height}", "-r", str(a.fps), "-i", "-",
           "-c:v", "libx264", "-crf", "16", "-preset", "medium", "-pix_fmt", "yuv420p", a.out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for i in range(n):
            ang = 2 * math.pi * i / n
            # loop with a gentle figure-eight wobble and height bob -> real parallax
            o = (a.radius * math.cos(ang), a.radius * math.sin(ang) + 0.3 * math.sin(2 * ang),
                 1.6 + 0.05 * math.sin(3 * ang))
            img = render(o, dirs, tex)
            proc.stdin.write((img.permute(1, 2, 0) * 255).byte().cpu().numpy().tobytes())
            if i % max(1, n // 10) == 0:
                print(f"frame {i + 1}/{n}", file=sys.stderr, flush=True)
    finally:
        proc.stdin.close()
        rc = proc.wait()
    if rc != 0:
        sys.exit(f"ffmpeg failed ({rc})")
    print(f"wrote {a.out}: {width}x{height}, {n} frames @ {a.fps} fps (synthetic equirectangular, +Z up)")


if __name__ == "__main__":
    main()

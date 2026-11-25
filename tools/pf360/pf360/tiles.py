"""Spatial tiling support for LichtFeld-Studio training (BT-12).

Plans overlapping spatial tiles along the camera walk trajectory, converts
cropboxes into the LichtFeld frame (COLMAP RDF with Y/Z inverted), executes
per-tile training sequentially with VRAM safety and resume support, and merges
per-tile PLY exports by deduplicating splats against each tile's core box before
a single final Z-up frame normalisation.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .backend import StageFailed


@dataclass
class TileSpec:
    """Specification of a spatial tile."""
    index: int
    core_box: Tuple[float, float, float, float, float, float]       # [x0, y0, z0, x1, y1, z1] in transforms frame
    crop_box: Tuple[float, float, float, float, float, float]       # [x0, y0, z0, x1, y1, z1] in transforms frame
    core_box_lfs: Tuple[float, float, float, float, float, float]   # in LFS frame
    crop_box_lfs: Tuple[float, float, float, float, float, float]   # in LFS frame
    walk_axis: int                                                 # 0=X, 1=Y, 2=Z in transforms frame

    @property
    def crop_box_lfs_str(self) -> str:
        """Formatted string for LichtFeld-Studio --cropbox argument."""
        b = self.crop_box_lfs
        return f"{b[0]:.4f},{b[1]:.4f},{b[2]:.4f},{b[3]:.4f},{b[4]:.4f},{b[5]:.4f}"


def box_to_lfs(box: Sequence[float]) -> Tuple[float, float, float, float, float, float]:
    """Converts a bounding box [x0, y0, z0, x1, y1, z1] from transforms frame to LFS frame.

    Under (x, y, z) -> (x, -y, -z):
      x_lfs = x       => [x0, x1]
      y_lfs = -y      => [-y1, -y0]
      z_lfs = -z      => [-z1, -z0]
    Minima remain strictly less than maxima.
    """
    x0, y0, z0, x1, y1, z1 = [float(v) for v in box]
    lx0, lx1 = min(x0, x1), max(x0, x1)
    ly0, ly1 = min(-y0, -y1), max(-y0, -y1)
    lz0, lz1 = min(-z0, -z1), max(-z0, -z1)
    return (lx0, ly0, lz0, lx1, ly1, lz1)


AUTO_TILE_VIEW_THRESHOLD = 20000        # camera frames (BT-20: 16,814 views fit in 1 run on 24 GB)


def should_auto_tile(walk_extent: float, num_views: int) -> bool:
    """Returns True if view count exceeds the threshold for auto-tiling.

    NOTE: walk_extent is not used because transforms.json coordinates are
    COLMAP-scaled and arbitrary -- the same scene can have any numeric extent
    depending on reconstruction scale.  Only view count is a reliable signal.
    """
    return num_views >= AUTO_TILE_VIEW_THRESHOLD


def plan_tiles_from_cameras(camera_centers: np.ndarray, num_tiles: int = 2,
                            overlap: Optional[float] = None, overlap_fraction: float = 0.15,
                            scene_margin: float = 2000.0) -> List[TileSpec]:
    """Plans 1D spatial subdivision along the principal walk axis.

    Args:
        camera_centers: (N, 3) array of camera positions in transforms coordinates.
        num_tiles: number of spatial tiles (>= 1).
        overlap: optional absolute overlap margin in transforms units. If None, computed as
            overlap_fraction * walk_extent.
        overlap_fraction: overlap margin as a fraction of total walk extent (default 0.15 / 15%).
        scene_margin: margin added to non-walk dimensions to fully encompass scene extent.
    """
    if len(camera_centers) == 0:
        raise StageFailed("Cannot plan tiles: camera positions array is empty.")
    if num_tiles < 1:
        raise StageFailed("num_tiles must be >= 1 (got %d)." % num_tiles)

    min_xyz = camera_centers.min(axis=0)
    max_xyz = camera_centers.max(axis=0)
    extent = max_xyz - min_xyz

    # Walk axis is the dimension with the largest camera spread
    walk_axis = int(np.argmax(extent))

    # Fallback if trajectory has zero extent (e.g. single camera)
    if extent[walk_axis] < 1e-4:
        extent[walk_axis] = 1.0

    walk_extent = float(extent[walk_axis])
    eff_overlap = overlap if overlap is not None else (walk_extent * overlap_fraction)

    w_min = float(min_xyz[walk_axis])
    w_max = float(max_xyz[walk_axis])
    step = (w_max - w_min) / float(num_tiles)
    splits = [w_min + i * step for i in range(num_tiles + 1)]

    # Compute bounding boxes for non-walk axes
    bounds_min = [float(min_xyz[i] - scene_margin) for i in range(3)]
    bounds_max = [float(max_xyz[i] + scene_margin) for i in range(3)]

    tiles: List[TileSpec] = []
    for i in range(num_tiles):
        # Core box along walk axis:
        # First tile extends to wide boundary; last tile extends to wide boundary
        core_w0 = bounds_min[walk_axis] if i == 0 else splits[i]
        core_w1 = bounds_max[walk_axis] if i == num_tiles - 1 else splits[i + 1]

        # Crop box with overlap along walk axis:
        crop_w0 = bounds_min[walk_axis] if i == 0 else splits[i] - eff_overlap
        crop_w1 = bounds_max[walk_axis] if i == num_tiles - 1 else splits[i + 1] + eff_overlap

        # Assemble full 3D boxes in transforms frame
        c_box = list(bounds_min) + list(bounds_max)
        c_box[walk_axis] = core_w0
        c_box[walk_axis + 3] = core_w1

        cr_box = list(bounds_min) + list(bounds_max)
        cr_box[walk_axis] = crop_w0
        cr_box[walk_axis + 3] = crop_w1

        c_box_tuple = tuple(c_box)
        cr_box_tuple = tuple(cr_box)

        c_lfs = box_to_lfs(c_box_tuple)
        cr_lfs = box_to_lfs(cr_box_tuple)

        tiles.append(TileSpec(
            index=i,
            core_box=c_box_tuple,
            crop_box=cr_box_tuple,
            core_box_lfs=c_lfs,
            crop_box_lfs=cr_lfs,
            walk_axis=walk_axis,
        ))

    return tiles


def plan_tiles_from_transforms(transforms_path: Path, num_tiles: int = 2,
                               overlap: Optional[float] = None, overlap_fraction: float = 0.15,
                               scene_margin: float = 2000.0) -> List[TileSpec]:
    """Reads camera centers from transforms.json (or transforms_train.json) and plans spatial tiles."""
    if not transforms_path.is_file():
        raise StageFailed("transforms file not found: %s" % transforms_path)
    with open(transforms_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    frames = data.get("frames", [])
    if not frames:
        raise StageFailed("No frames found in transforms file: %s" % transforms_path)

    centers: List[List[float]] = []
    for f in frames:
        mat = f.get("transform_matrix")
        if mat and len(mat) >= 3 and len(mat[0]) >= 4:
            centers.append([float(mat[0][3]), float(mat[1][3]), float(mat[2][3])])

    if not centers:
        raise StageFailed("Could not extract camera centers from %s" % transforms_path)

    return plan_tiles_from_cameras(np.array(centers, dtype=np.float64), num_tiles,
                                  overlap=overlap, overlap_fraction=overlap_fraction,
                                  scene_margin=scene_margin)


def merge_tile_plys(tile_ply_paths: Sequence[Path], tile_specs: Sequence[TileSpec],
                    out_merged_ply: Path, chunk_size: int = 100_000) -> int:
    """Merges per-tile PLYs (in LFS frame) by deduplicating against core boxes.

    Reads each tile PLY, checks coordinates against that tile's transforms-frame core box
    via (x, y, z) = (x_lfs, -y_lfs, -z_lfs), keeping points in [w0, w1) for earlier tiles
    and [w0, w1] for the final tile. Writes a single consolidated PLY file in LFS frame.
    All properties and vertex schema are preserved verbatim.
    """
    if len(tile_ply_paths) != len(tile_specs):
        raise StageFailed("Mismatch between number of tile PLYs (%d) and specs (%d)."
                          % (len(tile_ply_paths), len(tile_specs)))

    for p in tile_ply_paths:
        if not p.is_file():
            raise StageFailed("Tile PLY not found: %s" % p)

    # Read schema from the first PLY
    with open(tile_ply_paths[0], "rb") as f0:
        header0 = b""
        while b"end_header\n" not in header0:
            line = f0.readline()
            if not line:
                break
            header0 += line

    lines = header0.decode("utf-8", "replace").splitlines()
    props: List[str] = []
    for l in lines:
        if l.startswith("property"):
            parts = l.split()
            if len(parts) >= 3 and parts[1] == "float":
                props.append(parts[-1])
            else:
                raise StageFailed("Unsupported non-float property in PLY: %s" % l)

    stride = len(props) * 4
    dtype = [(p, "<f4") for p in props]

    temp_data_path = out_merged_ply.with_suffix(".tmp.bin")
    total_kept = 0

    try:
        with open(temp_data_path, "wb") as tmp_out:
            for tile_idx, (ply_path, spec) in enumerate(zip(tile_ply_paths, tile_specs)):
                # Core box in transforms frame: [x0, y0, z0, x1, y1, z1]
                c_box = spec.core_box
                cx0, cy0, cz0, cx1, cy1, cz1 = c_box
                walk_axis = spec.walk_axis

                with open(ply_path, "rb") as pf:
                    # Skip header
                    while b"end_header\n" not in pf.readline():
                        pass

                    # Stream vertices in chunks
                    while True:
                        buf = pf.read(chunk_size * stride)
                        if not buf:
                            break
                        n_chunk = len(buf) // stride
                        arr = np.frombuffer(buf[:n_chunk * stride], dtype=dtype)

                        # Convert from LFS frame to transforms frame to test against transforms core box
                        # x = x_lfs, y = -y_lfs, z = -z_lfs
                        tx = arr["x"]
                        ty = -arr["y"]
                        tz = -arr["z"]

                        # Check bounding box inclusion in transforms coordinates
                        mask = (tx >= cx0) & (tx <= cx1) & (tz >= cz0) & (tz <= cz1)
                        if walk_axis == 1:
                            if tile_idx == len(tile_specs) - 1:
                                mask &= (ty >= cy0) & (ty <= cy1)
                            else:
                                mask &= (ty >= cy0) & (ty < cy1)
                        elif walk_axis == 0:
                            if tile_idx == len(tile_specs) - 1:
                                mask &= (tx >= cx0) & (tx <= cx1)
                            else:
                                mask &= (tx >= cx0) & (tx < cx1)
                        else:
                            if tile_idx == len(tile_specs) - 1:
                                mask &= (tz >= cz0) & (tz <= cz1)
                            else:
                                mask &= (tz >= cz0) & (tz < cz1)

                        kept = arr[mask]
                        if len(kept) > 0:
                            tmp_out.write(kept.tobytes())
                            total_kept += len(kept)

        # Assemble final PLY with updated vertex count in header
        new_header_lines: List[str] = []
        for l in lines:
            if l.startswith("element vertex"):
                new_header_lines.append(f"element vertex {total_kept}")
            else:
                new_header_lines.append(l)
        new_header = ("\n".join(new_header_lines) + "\n").encode("utf-8")

        out_merged_ply.parent.mkdir(parents=True, exist_ok=True)
        with open(out_merged_ply, "wb") as final_f:
            final_f.write(new_header)
            with open(temp_data_path, "rb") as tmp_in:
                while True:
                    chunk = tmp_in.read(1 << 20)
                    if not chunk:
                        break
                    final_f.write(chunk)
    finally:
        if temp_data_path.exists():
            temp_data_path.unlink()

    return total_kept


def train_tiles(
    exe: str,
    data_dir: Path,
    output_dir: Path,
    num_tiles: int = 2,
    iterations: int = 30000,
    strategy: str = "mcmc",
    max_cap_per_tile: Optional[int] = None,
    total_max_cap: int = 1660000,
    downscale: int = 1,
    sh_degree: int = 3,
    overlap: Optional[float] = None,
    overlap_fraction: float = 0.15,
    scene_margin: float = 2000.0,
    output_basename: Optional[str] = None,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Executes sequential multi-tile LichtFeld training, merges PLYs, and normalises to Z-up."""
    from .lichtfeld_engine import LFS_COMMIT, file_sha256, rotate_ply_zup

    def log(msg: str) -> None:
        if log_fn:
            log_fn(msg)
        else:
            print(f"[tiles] {msg}", flush=True)

    if not os.path.isfile(exe):
        raise StageFailed(f"LichtFeld-Studio executable not found: {exe}")

    transforms_path = data_dir / "transforms_train.json"
    if not transforms_path.is_file():
        transforms_path = data_dir / "transforms.json"
    if not transforms_path.is_file():
        raise StageFailed(f"Neither transforms_train.json nor transforms.json found in {data_dir}")

    tile_specs = plan_tiles_from_transforms(transforms_path, num_tiles,
                                           overlap=overlap, overlap_fraction=overlap_fraction,
                                           scene_margin=scene_margin)
    log(f"Planned {num_tiles} spatial tiles along camera walk (axis {tile_specs[0].walk_axis}, overlap fraction {overlap_fraction:.1%})")

    cap = max_cap_per_tile if max_cap_per_tile is not None else int((total_max_cap / num_tiles) * (1.0 + overlap_fraction * num_tiles))
    output_dir.mkdir(parents=True, exist_ok=True)

    tile_plys: List[Path] = []
    tile_times: List[float] = []

    for idx, spec in enumerate(tile_specs):
        t_out = output_dir / f"tile_{idx}"
        target_ply = t_out / f"splat_{iterations}.ply"
        tile_plys.append(target_ply)

        if target_ply.is_file():
            log(f"Tile {idx + 1}/{num_tiles} already completed ({target_ply.name} exists); skipping training.")
            tile_times.append(0.0)
            continue

        t_out.mkdir(parents=True, exist_ok=True)
        cmd = [
            exe, "--headless",
            "-d", str(data_dir),
            "-o", str(t_out),
            "--iter", str(iterations),
            "--strategy", strategy,
            "--max-cap", str(cap),
            "--sh-degree", str(sh_degree),
            "-r", str(downscale),
            "--cropbox", spec.crop_box_lfs_str,
            "--cropbox-loss-weight", "0",
            "--crop",
            "--export", "ply"
        ]

        log(f"Starting Tile {idx + 1}/{num_tiles}: cap={cap:,}, cropbox={spec.crop_box_lfs_str}")
        t0 = time.perf_counter()

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                cwd=str(output_dir), text=True, bufsize=1)
        last_line = ""
        assert proc.stdout is not None
        import re
        step_re = re.compile(r"(\d+)/(\d+)\s*\|\s*Loss:\s*([\d\.]+)\s*\|\s*Splats:\s*(\d+)")
        for line in proc.stdout:
            last_line = line.strip()
            m = step_re.search(last_line)
            if m:
                step, total, loss, splats = int(m.group(1)), int(m.group(2)), m.group(3), int(m.group(4))
                if step % 5000 == 0 or step == total:
                    log(f"[tile {idx + 1}] step {step}/{total} | Loss: {loss} | Splats: {splats:,}")
            elif "Training completed" in line or "Exported final splat" in line:
                log(f"[tile {idx + 1}] {last_line}")

        rc = proc.wait()
        dur = time.perf_counter() - t0
        tile_times.append(dur)

        if rc != 0 or not target_ply.is_file():
            raise StageFailed(f"Tile {idx + 1}/{num_tiles} failed with exit code {rc}. Last output: {last_line}")
        log(f"Tile {idx + 1}/{num_tiles} completed in {dur:.1f}s -> {target_ply.name}")

    # Merge per-tile PLYs
    merged_lfs_ply = output_dir / f"splat_{iterations}_lfs_merged.ply"
    log(f"Merging {num_tiles} tile PLYs with core-box deduplication...")
    t_merge0 = time.perf_counter()
    total_splats = merge_tile_plys(tile_plys, tile_specs, merged_lfs_ply)
    merge_time = time.perf_counter() - t_merge0
    log(f"Merged into {merged_lfs_ply.name} ({total_splats:,} splats kept) in {merge_time:.2f}s")

    # Frame normalisation: rotate once into transforms frame (Z-up)
    if output_basename:
        final_zup_name = output_basename if output_basename.endswith(".ply") else output_basename + ".ply"
    else:
        final_zup_name = f"splat_{iterations}_{num_tiles}tiles_{strategy}_zup.ply"
    final_zup_ply = output_dir / final_zup_name
    log(f"Normalising orientation to Z-up transforms frame -> {final_zup_ply.name}...")
    rotate_ply_zup(merged_lfs_ply, final_zup_ply)

    # Write sidecar
    sidecar = {
        "engine": "lichtfeld",
        "strategy": strategy,
        "frame": "transforms",
        "upAxis": "z_up",
        "tiles": num_tiles,
        "tileOverlap": overlap,
        "totalSplats": total_splats,
        "lfsCommit": LFS_COMMIT,
    }
    sidecar_path = Path(str(final_zup_ply) + ".pf360.json")
    with open(sidecar_path, "w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=2)

    return {
        "numTiles": num_tiles,
        "strategy": strategy,
        "iterations": iterations,
        "totalSplats": total_splats,
        "tileTimes": tile_times,
        "mergeTime": merge_time,
        "mergedLfsPly": str(merged_lfs_ply),
        "finalZUpPly": str(final_zup_ply),
        "sidecar": str(sidecar_path),
        "exeSha256": file_sha256(Path(exe)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="LichtFeld-Studio multi-tile training runner")
    parser.add_argument("--exe", default=r"C:\lfs\LichtFeld-Studio\build-windows-release\LichtFeld-Studio.exe")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tiles", default=2, type=int)
    parser.add_argument("--iter", default=30000, type=int)
    parser.add_argument("--strategy", default="mcmc", choices=["mrnf", "mcmc"])
    parser.add_argument("--max-cap", default=1660000, type=int)
    parser.add_argument("--max-cap-per-tile", default=None, type=int)
    parser.add_argument("--downscale", default=1, type=int)
    parser.add_argument("--overlap-fraction", default=0.15, type=float)
    args = parser.parse_args()

    res = train_tiles(
        exe=args.exe,
        data_dir=args.data,
        output_dir=args.output,
        num_tiles=args.tiles,
        iterations=args.iter,
        strategy=args.strategy,
        max_cap_per_tile=args.max_cap_per_tile,
        total_max_cap=args.max_cap,
        downscale=args.downscale,
        overlap_fraction=args.overlap_fraction,
    )
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

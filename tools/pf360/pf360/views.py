"""Perspective-view geometry for equirectangular frames.

This is a dependency-free port of the view-selection logic in Nerfstudio 1.1.5
`nerfstudio/process_data/equirect_utils.py`
(`generate_planar_projections_from_equirectangular`, `_crop_bound_arr_vertical`,
`compute_resolution_from_equirect`). It is used ONLY to tell the user, before
anything runs, how many perspective images a configuration will produce and
at what size. The actual projection is always performed by Nerfstudio itself.

Crop semantics (Nerfstudio): crop = (top, bottom, left, right), each in [0,1].
  top/bottom  remove elevation rings (fraction of the 180 deg vertical span)
  left/right  narrow the yaw span (fraction of the 360 deg horizontal span)
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple


_UNSUPPORTED_COMBO = ("this top + bottom crop combination removes a whole view ring, which Nerfstudio's "
                      "equirectangular projection cannot handle (it crashes); reduce the top or bottom crop")


def _crop_bottom(bound_arr: List[Optional[float]], fov: float, crop: float) -> List[Optional[float]]:
    degrees_chopped = 180 * crop
    new_bottom_start = 90 - degrees_chopped - fov / 2
    for i in reversed(range(len(bound_arr))):
        el = bound_arr[i]
        if el is None:
            continue
        if el > new_bottom_start + fov / 2:
            bound_arr[i] = None
        elif el > new_bottom_start:
            diff = el - new_bottom_start
            bound_arr[i] = new_bottom_start
            for j in range(i - 1, -1, -1):
                if bound_arr[j] is not None:
                    bound_arr[j] -= diff / (2 ** (i - j))
            break
    return bound_arr


def _crop_top(bound_arr: List[Optional[float]], fov: float, crop: float) -> List[Optional[float]]:
    degrees_chopped = 180 * crop
    new_top_start = -90 + degrees_chopped + fov / 2
    for i in range(len(bound_arr)):
        el = bound_arr[i]
        if el is None:
            # Nerfstudio 1.1.5 compares None < float here and crashes with a
            # TypeError; reject the combination up front instead.
            raise ValueError(_UNSUPPORTED_COMBO)
        if el < new_top_start - fov / 2:
            bound_arr[i] = None
        elif el < new_top_start:
            diff = new_top_start - el
            bound_arr[i] = new_top_start
            for j in range(i + 1, len(bound_arr)):
                if bound_arr[j] is None:      # Nerfstudio: None += float -> TypeError
                    raise ValueError(_UNSUPPORTED_COMBO)
                bound_arr[j] += diff / (2 ** (j - i))
            break
    return bound_arr


def _arange_count(start: float, stop: float, step: float) -> int:
    # numpy.arange element count
    return max(0, int(math.ceil((stop - start) / step)))


def yaw_pitch_pairs(images_per_equirect: int, crop: Sequence[float] = (0.0, 0.0, 0.0, 0.0)) -> List[Tuple[float, float]]:
    """Returns the (yaw, pitch) view directions Nerfstudio generates per frame."""
    if images_per_equirect not in (8, 14):
        raise ValueError("images_per_equirect must be 8 or 14")
    top, bottom, left, right = (float(c) for c in crop)
    left_bound, right_bound = -180.0, 180.0
    if right > 0:
        left_bound = -180 + 360 * right
    if left > 0:
        right_bound = 180 - 360 * left
    if images_per_equirect == 8:
        fov, steps = 120, (180, 90, 180)       # (pitch -45, 0, +45) yaw steps
    else:
        fov, steps = 110, (90, 60, 90)
    bound: List[Optional[float]] = [-45.0, 0.0, 45.0]
    if bottom > 0:
        bound = _crop_bottom(bound, fov, bottom)
    if top > 0:
        bound = _crop_top(bound, fov, top)
    pairs: List[Tuple[float, float]] = []
    # Same emission order as Nerfstudio: middle ring, then +45 ring, then -45 ring.
    for ring in (1, 2, 0):
        pitch = bound[ring]
        if pitch is None:
            continue
        n = _arange_count(left_bound, right_bound, steps[ring])
        for k in range(n):
            pairs.append((left_bound + k * steps[ring], pitch))
    return pairs


def views_per_frame(images_per_equirect: int, crop: Sequence[float] = (0.0, 0.0, 0.0, 0.0)) -> int:
    return len(yaw_pitch_pairs(images_per_equirect, crop))


def perspective_size(width: int, height: int, images_per_equirect: int) -> int:
    """Square perspective image edge: num_images * res^2 = W * H (Nerfstudio heuristic)."""
    if width <= 0 or height <= 0:
        return 0
    return int(math.sqrt((width * height) / float(images_per_equirect)))

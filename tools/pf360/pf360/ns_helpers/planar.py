"""Runs INSIDE the Nerfstudio Python environment (needs torch + nerfstudio).

Calls Nerfstudio's own equirectangular -> perspective projection on a folder of
equirectangular frames that pf360 extracted. Using the library functions
directly (rather than `ns-process-data images --camera-type equirectangular`)
avoids that command path re-applying `crop_factor` as a pixel crop to the
already-generated perspective images, so the crop behaves exactly like the
documented `ns-process-data video --camera-type equirectangular` workflow.

usage: python planar.py <frames_dir> <images_per_equirect> <top> <bottom> <left> <right>
prints one line: PF360PLANAR{"dir": ..., "size": [w, h]}
"""
import json
import sys
from pathlib import Path


def main() -> int:
    frames = Path(sys.argv[1])
    n = int(sys.argv[2])
    crop = tuple(float(v) for v in sys.argv[3:7])
    from nerfstudio.process_data import equirect_utils  # noqa: deferred heavy import

    size = equirect_utils.compute_resolution_from_equirect(frames, n)
    out = equirect_utils.generate_planar_projections_from_equirectangular(frames, size, n, crop_factor=crop)
    print("PF360PLANAR" + json.dumps({"dir": str(out), "size": list(size)}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

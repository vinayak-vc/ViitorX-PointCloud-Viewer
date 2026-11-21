"""Runs INSIDE the Nerfstudio Python environment (needs torch + nerfstudio).

Equirectangular -> perspective views with Nerfstudio's own projection
(equirect_utils.equirect2persp, same resolution rule, same file names
`<frame>_<k>.jpg`, same view order), but RESUMABLE:

* a panorama whose views all exist is skipped, so an interrupted run continues
  where it stopped (restore point) instead of starting over;
* every view is written to a temp name and renamed, so an interruption never
  leaves a truncated JPEG behind.

Using the library functions directly (rather than `ns-process-data images
--camera-type equirectangular`) also avoids that command path re-applying
`crop_factor` as a pixel crop to the already-generated views.

usage: python planar.py <frames_dir> <images_per_equirect> <top> <bottom> <left> <right> '<views json>'
       views json = [[yaw, pitch], ...] in Nerfstudio's order (pf360.views.yaw_pitch_pairs)
prints PF360PLANARPROG<done>/<total> per panorama and finally
       PF360PLANAR{"dir": ..., "size": [w, h], "skipped": n}
"""
import json
import os
import sys
from pathlib import Path

EXTS = (".jpg", ".png", ".jpeg", ".tiff", ".tif")
FOV = {8: 120, 14: 110}


def main() -> int:
    frames = Path(sys.argv[1])
    n = int(sys.argv[2])
    views = json.loads(sys.argv[7])
    import cv2  # noqa: deferred heavy imports
    import numpy as np
    import torch
    from nerfstudio.process_data import equirect_utils

    size = equirect_utils.compute_resolution_from_equirect(frames, n)
    fov = FOV[n]
    out = frames / "planar_projections"
    out.mkdir(exist_ok=True)
    for t in out.glob("*.tmp.jpg"):          # left by an interrupted run
        t.unlink()
    names = sorted(f for f in os.listdir(frames) if f.lower().endswith(EXTS))
    device = torch.device("cuda")
    skipped = 0
    for i, name in enumerate(names):
        stem = name[:-4]                     # same naming as Nerfstudio (i[:-4])
        targets = [out / ("%s_%d.jpg" % (stem, k)) for k in range(len(views))]
        if all(t.is_file() and t.stat().st_size > 0 for t in targets):
            skipped += 1
            print("PF360PLANARPROG%d/%d" % (i + 1, len(names)), flush=True)
            continue
        im = np.array(cv2.imread(str(frames / name)))
        im = torch.tensor(im, dtype=torch.float32, device=device)
        im = torch.permute(im, (2, 0, 1)).unsqueeze(0) / 255.0
        for k, (u_deg, v_deg) in enumerate(views):
            pers = equirect_utils.equirect2persp(im, fov, u_deg, v_deg, size[1], size[0]) * 255.0
            img = pers.squeeze().permute(1, 2, 0).type(torch.uint8).to("cpu").numpy()
            tmp = out / ("%s_%d.tmp.jpg" % (stem, k))
            if not cv2.imwrite(str(tmp), img):
                print("cannot write %s" % tmp, file=sys.stderr)
                return 3
            os.replace(tmp, targets[k])
        print("PF360PLANARPROG%d/%d" % (i + 1, len(names)), flush=True)
    print("PF360PLANAR" + json.dumps({"dir": str(out), "size": list(size), "skipped": skipped}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

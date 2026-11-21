"""Automatic masks for pf360 (runs INSIDE the Nerfstudio environment: torch,
PIL, transformers from scripts/pf360/setup_masking.ps1).

Semantic segmentation (UperNet + ConvNeXt, ADE20K, MIT licence) of every
perspective view; pixels of the selected class groups are masked out.

Output convention (shared by COLMAP and Nerfstudio): one 8-bit grayscale PNG
per image, 255 = keep, 0 = ignore. File name = "<image file name>.png", which is
what COLMAP's --ImageReader.mask_path expects.

  python -m pf360.ns_helpers.mask --images DIR --out DIR --model DIR --groups people,vehicles [--dilate 8]
  python -m pf360.ns_helpers.mask --self-test --model DIR
  python -m pf360.ns_helpers.mask --image IMG --overlay OUT.png --model DIR --groups people,sky
  python -m pf360.ns_helpers.mask --attach PROCESSED --masks DIR --planar DIR   (after ns-process-data)

stdout lines: PF360MASK{"done":i,"total":n} progress, PF360MASKDONE{stats} at the end.
Never downloads anything (local_files_only).
"""
import argparse
import json
import os
import sys
from pathlib import Path

# ADE20K label names (as in the model's config.id2label) per user-facing group.
# "people" also covers what people ride (bicycle / minibike), like the wizard says.
GROUPS = {
    "people": ("person", "bicycle", "minibike"),
    "vehicles": ("car", "bus", "truck", "van", "minibike", "bicycle", "boat", "airplane", "ship"),
    "animals": ("animal",),
    "sky": ("sky",),
}
IMG_EXTS = (".jpg", ".jpeg", ".png")
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def _emit(tag, obj):
    sys.stdout.write(tag + json.dumps(obj) + "\n")
    sys.stdout.flush()


def class_ids(id2label, groups):
    """ADE20K class indices for the requested groups (matched on the first
    comma-separated synonym of each label, case-insensitive)."""
    want = set()
    for g in groups:
        if g not in GROUPS:
            raise ValueError("unknown mask group %r (valid: %s)" % (g, ", ".join(GROUPS)))
        want.update(GROUPS[g])
    ids = sorted(int(i) for i, name in id2label.items() if str(name).split(",")[0].strip().lower() in want)
    return ids


def load_model(model_dir, device):
    import torch
    from transformers import UperNetForSemanticSegmentation
    model = UperNetForSemanticSegmentation.from_pretrained(model_dir, local_files_only=True, use_safetensors=True)
    model.eval().to(device)
    if device == "cuda":
        model.half()
    return model


def segment(model, imgs, device, max_edge):
    """imgs: list of HxWx3 uint8 numpy arrays of the SAME size -> HxW int64 label maps (torch, on device)."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    h, w = imgs[0].shape[:2]
    x = torch.from_numpy(np.stack(imgs)).to(device).permute(0, 3, 1, 2).float() / 255.0
    s = min(1.0, float(max_edge) / max(h, w))
    # The backbone downsamples by 32; keep the input a multiple of 32.
    th, tw = max(32, int(round(h * s / 32)) * 32), max(32, int(round(w * s / 32)) * 32)
    if (th, tw) != (h, w):
        x = F.interpolate(x, size=(th, tw), mode="bilinear", align_corners=False)
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    x = (x - mean) / std
    if device == "cuda":
        x = x.half()
    with torch.no_grad():
        logits = model(pixel_values=x).logits.float()
    logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
    return logits.argmax(1)


def masks_from_labels(labels, ids, dilate):
    """bool keep-mask (True = keep) from label maps; masked regions grown by `dilate` px."""
    import torch
    import torch.nn.functional as F
    drop = torch.zeros_like(labels, dtype=torch.bool)
    for i in ids:
        drop |= labels == i
    if dilate > 0:
        k = 2 * int(dilate) + 1
        drop = F.max_pool2d(drop.float().unsqueeze(1), kernel_size=k, stride=1, padding=int(dilate)).squeeze(1) > 0.5
    return ~drop


def run(args):
    import numpy as np
    import torch
    from PIL import Image
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.model, device)
    groups = [g for g in args.groups.split(",") if g]
    ids = class_ids(model.config.id2label, groups)
    if not ids:
        raise SystemExit("no ADE20K classes matched groups %s" % groups)
    src, out = Path(args.images), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for t in out.glob("*.tmp"):               # left by an interrupted run
        t.unlink()
    files = sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXTS and p.is_file())
    total = len(files)
    _emit("PF360MASK", {"done": 0, "total": total, "device": device, "classIds": ids})
    fractions, heavy = [], 0
    # Restore point: masks finished by an interrupted run are kept (written atomically, so an
    # existing file is complete); they still count in the statistics.
    todo = []
    for p in files:
        m = out / (p.name + ".png")
        if m.is_file() and m.stat().st_size > 0:
            frac = 1.0 - float(np.asarray(Image.open(m).convert("L")).mean()) / 255.0
            fractions.append(frac)
            heavy += frac > 0.6
        else:
            todo.append(p)
    done_before = total - len(todo)
    files = todo
    i = 0
    while i < len(files):
        batch = files[i:i + args.batch]
        imgs = [np.asarray(Image.open(p).convert("RGB")) for p in batch]
        # Group equal sizes (always equal for one pf360 job, but stay safe).
        if len({im.shape for im in imgs}) != 1:
            batch, imgs = batch[:1], imgs[:1]
        keep = masks_from_labels(segment(model, imgs, device, args.max_edge), ids, args.dilate)
        for p, k in zip(batch, keep.cpu().numpy()):
            frac = 1.0 - float(k.mean())
            fractions.append(frac)
            heavy += frac > 0.6
            tmp = out / (p.name + ".png.tmp")
            Image.fromarray((k * 255).astype(np.uint8), mode="L").save(tmp, format="PNG")
            os.replace(tmp, out / (p.name + ".png"))
        i += len(batch)
        _emit("PF360MASK", {"done": done_before + i, "total": total})
    stats = {"images": total, "resumedFrom": done_before, "groups": groups, "classIds": ids, "dilatePx": args.dilate, "device": device,
             "meanMaskedFraction": round(float(np.mean(fractions)), 4) if fractions else 0.0,
             "maxMaskedFraction": round(float(np.max(fractions)), 4) if fractions else 0.0,
             "imagesMaskedOver60Percent": int(heavy)}
    _emit("PF360MASKDONE", stats)


def overlay(args):
    """Single-image preview: writes the image with masked pixels tinted red."""
    import numpy as np
    import torch
    from PIL import Image
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.model, device)
    ids = class_ids(model.config.id2label, [g for g in args.groups.split(",") if g])
    if args.raw_size:
        # Raw RGB24 in/out (used by the PointForge wizard preview).
        w, h = args.raw_size
        im = np.fromfile(args.image, dtype=np.uint8)
        if im.size != w * h * 3:
            raise SystemExit("raw image size mismatch (%d bytes for %dx%d)" % (im.size, w, h))
        im = im.reshape(h, w, 3)
    else:
        im = np.asarray(Image.open(args.image).convert("RGB"))
    keep = masks_from_labels(segment(model, [im], device, args.max_edge), ids, args.dilate)[0].cpu().numpy()
    vis = im.astype(np.float32)
    vis[~keep] = vis[~keep] * 0.35 + np.array([255, 40, 40]) * 0.65
    vis = vis.clip(0, 255).astype(np.uint8)
    if args.raw_size:
        vis.tofile(args.overlay)
    else:
        Image.fromarray(vis).save(args.overlay)
    _emit("PF360MASKDONE", {"maskedFraction": round(1.0 - float(keep.mean()), 4), "overlay": args.overlay})


def attach(args):
    """After `ns-process-data --skip-colmap`: give Nerfstudio per-frame masks.

    ns-process-data copies the planar views to images/frame_00001.jpg... in sorted
    source order (process_data_utils.copy_images). For each frame in
    transforms.json this writes masks/<frame>.png plus masks_2/4/8 at EXACTLY the
    size of the matching images_N file (Splatfacto asserts equal shapes), and sets
    frame["mask_path"]. The name mapping is verified by comparing pixels."""
    import numpy as np
    from PIL import Image
    processed, masks, planar = Path(args.attach), Path(args.masks), Path(args.planar)
    src = sorted(p for p in planar.iterdir() if p.suffix.lower() in IMG_EXTS and p.is_file())
    tf_path = processed / "transforms.json"
    tf = json.loads(tf_path.read_text(encoding="utf-8"))
    by_name = {"frame_%05d%s" % (i + 1, p.suffix.lower()): p for i, p in enumerate(src)}
    frames = tf.get("frames") or []
    checked = 0
    for k, fr in enumerate(frames):
        img_rel = Path(fr["file_path"])
        orig = by_name.get(img_rel.name.lower()) or by_name.get(img_rel.name)
        if orig is None:
            raise SystemExit("no source view for %s" % img_rel.name)
        m = Image.open(masks / (orig.name + ".png")).convert("L")
        if k in (0, len(frames) - 1, len(frames) // 2):      # mapping sanity check
            a = np.asarray(Image.open(processed / img_rel).convert("L").resize((64, 64)), np.float32)
            b = np.asarray(Image.open(orig).convert("L").resize((64, 64)), np.float32)
            if float(np.abs(a - b).mean()) > 6.0:
                raise SystemExit("image/mask name mapping mismatch at %s (%s)" % (img_rel.name, orig.name))
            checked += 1
        stem = img_rel.stem + ".png"
        full = processed / img_rel
        for folder, img_dir in [("masks", full.parent)] + [("masks_%d" % f, processed / ("images_%d" % f))
                                                            for f in (2, 4, 8)]:
            img_file = img_dir / img_rel.name
            if not img_file.is_file():
                continue
            size = Image.open(img_file).size
            out_dir = processed / folder
            out_dir.mkdir(exist_ok=True)
            (m if m.size == size else m.resize(size, Image.NEAREST)).save(out_dir / stem)
        fr["mask_path"] = "masks/" + stem
        if k % 50 == 0:
            _emit("PF360MASK", {"done": k + 1, "total": len(frames), "phase": "attach"})
    tf_path.write_text(json.dumps(tf, indent=4), encoding="utf-8")
    _emit("PF360MASKDONE", {"frames": len(frames), "mappingChecks": checked})


def self_test(args):
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.model, device)
    import numpy as np
    labels = segment(model, [np.zeros((256, 256, 3), np.uint8)], device, 512)
    found = {g: len(class_ids(model.config.id2label, [g])) for g in GROUPS}
    _emit("PF360MASKDONE", {"ok": tuple(labels.shape) == (1, 256, 256) and all(found.values()),
                            "device": device, "classesPerGroup": found, "numLabels": len(model.config.id2label)})
    if not all(found.values()):
        raise SystemExit("model labels do not cover every mask group: %s" % found)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="local model directory (config.json + model.safetensors)")
    ap.add_argument("--attach", help="processed dataset dir (transforms.json) to attach masks to")
    ap.add_argument("--masks", help="mask dir written by --images/--out")
    ap.add_argument("--planar", help="planar view dir the masks were computed from")
    ap.add_argument("--images")
    ap.add_argument("--out")
    ap.add_argument("--groups", default="people")
    ap.add_argument("--dilate", type=int, default=8)
    ap.add_argument("--max-edge", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--image")
    ap.add_argument("--overlay")
    ap.add_argument("--raw-size", type=int, nargs=2, metavar=("W", "H"),
                    help="--image/--overlay are raw RGB24 files of this size")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.attach:
        if not (a.masks and a.planar):
            ap.error("--attach needs --masks and --planar")
        attach(a)
        return
    if not a.model:
        ap.error("--model is required")
    if a.self_test:
        self_test(a)
    elif a.image and a.overlay:
        overlay(a)
    elif a.images and a.out:
        run(a)
    else:
        ap.error("need --images/--out, --image/--overlay or --self-test")


if __name__ == "__main__":
    main()

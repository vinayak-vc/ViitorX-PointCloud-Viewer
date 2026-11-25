# LichtFeld-Studio Integration Specification

Source: BT-9 Integration Design (Claude, 2026-11-25)
Status: APPROVED for Phase 1 Integration

## 0. Principles
- LichtFeld-Studio (LFS) runs ONLY as a separate process. No LFS code, headers or libs in pfcore/pfview/pf360.
- Nerfstudio stays as a working fallback engine. Nothing in the NS path changes behaviour.
- Everything is proven on the 10 s clip first. Full-video run only after Claude's written OK.

## 1. pf360: training engine switch (BT-10)
- Manifest: `training.engine` = `"nerfstudio"` (default for old manifests) | `"lichtfeld"`.
  New keys (LFS only): `lfsStrategy` = `"mrnf"` (default) | `"mcmc"`; `maxGaussians` (default 2,000,000 → `--max-cap`).
  `manifest.py`: validate, default-fill, and keep old manifests valid.
- Keep ONE backend class (camera stages are shared). In `NerfstudioBackend`, `_training`/`_export` dispatch on
  engine to `_training_lfs`/`_export_lfs` (new module `lichtfeld_engine.py`, no NS imports).
- `stage_inputs("training")`: add `engine`, `lfsStrategy`, `maxGaussians` ONLY when engine == lichtfeld
  (old NS fingerprints unchanged → finished jobs don't retrain).
- Data: run `training_prep.prepare()` exactly as for NS (seed cap, misplaced-panorama drop, view budget). LFS then
  reads `processed/transforms.json` + `ply_file_path` directly. Assert no `transforms_train.json` exists there
  (LFS would silently prefer it).
- Command: `LichtFeld-Studio.exe --headless -d <processed> -o <ws>/training/lfs --iter N --strategy S
  --max-cap C --sh-degree D -r R --export ply`. `downscaleFactor` → `-r` only for 1/2/4/8; anything else = StageFailed
  with a clear message. `randomSeed`, `useScaleRegularization`, cull/densify thresholds: no LFS equivalent →
  ignored for LFS, and the report says so. Do not pass `--eval`.
- Progress: parse `(\d+)/(\d+)\s*\|\s*Loss:\s*([\d.]+)\s*\|\s*Splats:\s*(\d+)` → existing JSONL progress events
  (percent, "step N / M", splats). Keep the last 40 lines for error messages, as `_training` does.
- Cancel/pause: start with `CREATE_NEW_PROCESS_GROUP`; on cancel or `.pause-training` send `CTRL_BREAK_EVENT`,
  wait up to TRAIN_PAUSE_GRACE_S, then hard kill (job object). Resume (`_resume_training`): if
  `training/lfs/project.licht` exists → `--resume <project.licht> -o <same> --iter N`. Exit code != 0 → StageFailed.
- `stage_outputs_exist("training")` for LFS: `training/lfs/splat_<N>.ply` exists.
- Tool discovery: `Tools.lichtfeld` = settings path, else `<app>/engines/lichtfeld/LichtFeld-Studio.exe`.
  `check_environment`: exe present; driver ≥ 570; GPU compute capability ≥ 7.5. Missing → engine unavailable,
  wizard falls back to Nerfstudio with a warning.

## 2. Frame normalisation (decided, from the BT-18 finding)
LFS writes its PLY in its COLMAP frame = our transforms.json frame rotated 180° about X
(mean camera up = −Z, i.e. Z-down). The viewer must not need a special case. So `_export_lfs`:
- Rotate the PLY back into the transforms.json frame (Z-up), in pf360 (numpy, streaming in chunks):
  position (x,y,z) → (x,−y,−z); quaternion q' = (0,1,0,0) ⊗ q (w-first, normalised); scales, opacity, f_dc
  unchanged; f_rest: per coefficient sign (−1)^(deg_y + deg_z) of its real-SH basis polynomial
  (band1 y,z,x → −,−,+; band2 xy,yz,3z²−1,xz,x²−y² → −,+,+,−,+; band3 → −,+,−,−,+,−,+), applied per colour channel
  in 3DGS f_rest ordering (channel-major).
- PROOF REQUIRED (test): ab_score of the rotated PLY with `--frame raw` must equal the original with
  `--frame colmap` (PSNR within 0.01 dB). This also checks the SH signs.
- Write sidecar `<ply>.pf360.json`: `{"engine":"lichtfeld","strategy":...,"frame":"transforms","upAxis":"z_up"}`.
  NS exports get `{"engine":"nerfstudio","frame":"ns_dataparser"}`.
- Viewer `recon360CameraPath`: if sidecar frame == "transforms" → use transforms.json frames directly
  (P = p, F = −col2, up = col1; no dataparser T, no scale). Else the current NS path. Spawn, keep-level,
  near-clip then work unchanged.
- Report: `upAxis z_up`, `gaussianCount`, engine/strategy, LFS commit 8c366f5, exe SHA256.

## 3. Wizard (BT-11)
- Training section: "Training engine" combo: LichtFeld (recommended) / Nerfstudio. Impact text:
  "LichtFeld: about 3-5x faster training and sharper results in our test (10 s clip: +1.2 dB PSNR, LPIPS 0.217 vs
  0.270). NVIDIA driver 570+ needed. Nerfstudio: previous engine, kept as fallback."
- LichtFeld options: Strategy (Quality = MRNF, Fast = MCMC; impact text from the A/B), Max Gaussians
  (file size ≈ 236 bytes × count). Hide NS-only options when LFS is selected.
- Disabled with a tooltip when the exe or driver is missing.

## 4. Tile training for large scenes (BT-12) — after BT-10/11 pass on the 10 s clip
- When the camera-path extent exceeds a threshold, split along the walk into overlapping tiles in the transforms
  frame; convert each box to the LFS frame (y,z → −y,−z, swap min/max) before passing `--cropbox`.
- Per tile: `--cropbox … --cropbox-loss-weight 0 --crop`, own output dir, resumable; then merge (concatenate
  PLYs, drop splats outside each tile's core box to avoid double coverage), then frame-normalise once.
- Prove on the 10 s clip with 2 tiles (score vs single run); full video only with Claude's OK.

## 5. "Edit in LichtFeld" button (BT-13)
- Viewer File menu + Properties button when a splat is loaded: launches `LichtFeld-Studio.exe -v <ply>` detached
  (CreateProcessW, own process group, no wait). Disabled with tooltip if the exe is missing.
- Watch the PLY's mtime; when the user saves over it (or saves a new PLY next to it), offer "Reload edited splat".
- Edited files saved from LFS are in whatever frame was loaded (our normalised Z-up) — check with one edit
  round-trip (delete a region, save, reload, orientation unchanged).

## 6. Packaging + GPL (BT-14) — needs Owner legal OK before any release
- Rebuild LFS with ONE toolset for vcpkg and app (pin MSVC 14.44 in an overlay triplet); remove the
  compat_msvc_stl.c lib injection. Set explicit CUDA archs (e.g. 75;80;86;89;90;120 + PTX), not "native".
- Ship in `engines/lichtfeld/` (separate folder, separate exe). Include LICENSE, LICENSE-EXCEPTION.txt,
  THIRD_PARTY_LICENSES.md, a written source offer, the exact commit, BUILD_NOTES.md and our overlay patches.
- Optional download pack instead of bundling (installer size ~ exe 189 MB + 86 DLLs).

## 7. Tests (BT-15)
- Unit: PLY frame rotation incl. SH signs (synthetic PLY + the scorer-equivalence test above); progress parser;
  `-r` mapping errors; fingerprint unchanged for old NS manifests; sidecar read in the viewer.
- Integration on the 10 s clip: pf360 run with engine=lichtfeld end-to-end (time, score, pfview render via
  `--export-video`, GLB/FBX reload); cancel + resume through pf360; Nerfstudio engine still passes.
- Existing 145 pf360 tests must stay green.

## Order
BT-18 (finish) → BT-10 + frame normalisation → BT-11 → BT-15 (10 s clip) → BT-13 → BT-12 → BT-14.

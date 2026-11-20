# 360° Reconstruction Studio — 360 video → 3D Gaussian Splat

Status (2026-11-20): host integration, worker, protocol, UI and tests are
implemented and tested **without** a GPU training run. The full
video → COLMAP → Splatfacto → PLY chain has **not** been executed yet on this
machine, because Nerfstudio is not installed (see *Verification status*).

## 1. What it does

File ▸ **Gaussian Splat from 360 Video…** (Ctrl+R, also Tools menu, command
palette, or drop an `.mp4/.mov/.mkv` on the window) opens a 6-step wizard:

| Step | What happens |
|---|---|
| 1 Video | ffprobe metadata, projection classification, sampled quality screen, preview frame |
| 2 Processing | time range, frame sampling, 8/14 perspective views, crop with live overlay, COLMAP matching |
| 3 Quality | Preview / Balanced / High Quality / Custom, detected GPU + VRAM, environment diagnostic |
| 4 Output | project folder, PLY name, auto-load; refuses to overwrite foreign folders / existing PLYs |
| 5 Run | real per-stage progress (indeterminate when a tool reports none), log, Cancel, Run in background |
| 6 Done | Gaussian count, PLY size, method, iterations, time; Open in PointForge / folder / report / logs / sparse cloud; Retry |

The job is a `ConvertJob::Kind::Recon360` in the existing `JobQueue`, so it
shows in the Jobs panel and status-bar pill, and runs one-at-a-time with other
heavy jobs. On success the PLY is loaded on the **existing Gaussian-splat path**
(`loadOctree` → `SplatReader` → `SplatRenderer`); it is never converted to an
octree.

Scope (v1): **static** scenes from **stitched, monoscopic equirectangular**
video. Dual-fisheye, raw camera files (`.insv`, `.360`), multi-stream files,
stereo 360 and non-equirectangular spherical metadata are rejected with an
explanation. Moving objects cause ghosts/floaters (no 4D reconstruction).

## 2. Architecture

```
PointForge (C++)                                   pf360 worker (Python, stdlib only)
------------------------------------------------   -----------------------------------------------
Recon360Panel (wizard, ImGui)                      __main__  doctor | probe | frame | estimate |
  └ async: recon360ProbeVideo / RunDoctor /                  presets | validate-ply | run
           ExtractFrame  -> python -m pf360 ...    worker.Job  stage orchestration, cache, report
JobQueue (Jobs.h) Kind::Recon360                   video_probe  ffprobe + projection + quality
  └ recon360Run (Recon360.cpp)                     backend.ReconstructionBackend  (extension point)
      CreateProcessW + Job Object (kill-on-close)    └ NerfstudioBackend
      stdout JSONL -> Recon360Live (stages, log)        panorama_projection  ns_helpers/planar.py
      stderr -> Console                                 camera_reconstruction ns-process-data images
      stdin "cancel" -> graceful stop, 20 s grace       training             ns-train splatfacto[-big]
      then TerminateJobObject                           export               ns-export gaussian-splat
main.cpp: menu/shortcut/drop/toasts/prefs,         ply_validate  SplatReader-compatible checks
  Z-up detection for pf360 outputs                 procrun  list-args, tree kill, streaming logs
```

* The worker is independently executable (`python -m pf360 run --manifest job.json`)
  and has no dependency on PointForge; any other app can drive it.
* The worker itself only needs CPython ≥ 3.8. FFmpeg, COLMAP and the Nerfstudio
  CLI (with CUDA PyTorch) are external executables discovered at run time.
* Equirect → perspective uses Nerfstudio's own `equirect_utils` functions
  (called from `ns_helpers/planar.py` inside the Nerfstudio Python). This is
  equivalent to `ns-process-data video --camera-type equirectangular` but lets
  pf360 control frame extraction (time range, count) and avoids the `images`
  path re-applying the crop as a pixel crop.
* The 8 or 14 views of one frame are rotations about one camera centre; COLMAP
  solves them (shared intrinsics, Nerfstudio's `use_single_camera_mode`), and
  the reconstruction check counts *distinct* camera positions, not images.

## 3. Pipeline stages (protocol stage ids)

| id | tool | success criteria (not just exit code 0) |
|---|---|---|
| `video_validation` | ffprobe/ffmpeg | readable stream, supported projection (or user-confirmed), duration, GPU/CUDA + disk check |
| `frame_extraction` | ffmpeg `-ss/-t -vf fps=N/window -frames:v N` | ≥1 frame; warns < 90 % or < 10 |
| `panorama_projection` | Nerfstudio `equirect_utils` | expected perspective image count |
| `camera_reconstruction` | `ns-process-data images` (COLMAP) | `transforms.json`; ≥ 30 % and ≥ 8 images registered; > 2 distinct camera positions (rotation-only capture fails); warns < 70 %, split models, missing sparse cloud |
| `training` | `ns-train splatfacto[-big]` | `config.yml` + checkpoint exist |
| `export` | `ns-export gaussian-splat` | PLY found (requested name, else newest new `.ply`) |
| `validation` | `pf360 validate-ply` (numpy full scan in the NS env, else sampled) | binary LE, `vertex` first, float/double only, all 3DGS attributes, count > 0, no NaN/Inf, non-degenerate bounds |

Progress: frame extraction (ffmpeg `frame=`), projection (output file count),
COLMAP (`Processed file [i/n]`, `Matching … [i/n]`, `Registering image #k (n)`),
training (`<step> (<pct>%)` table rows). Anything else is reported with
`percent: null` and drawn as an indeterminate bar — never a fabricated value.

## 4. Job manifest (schema v1)

Written by PointForge to `%LOCALAPPDATA%\ViitorX\PointForge\pf360-jobs\<jobId>.json`;
the resolved copy is stored as `<workspace>/manifest.json`.

```json
{
  "schemaVersion": 1, "jobId": "pf360_20261120_141500", "jobType": "360_video_to_gaussian_splat",
  "input": {"videoPath": "D:/Capture/room.mp4", "projection": "equirectangular",
            "projectionConfirmed": true, "startTimeSeconds": 0, "endTimeSeconds": null},
  "processing": {"targetFrames": 90, "imagesPerEquirect": 14, "cropTop": 0, "cropBottom": 0.2,
                 "cropLeft": 0, "cropRight": 0, "matchingMethod": "sequential"},
  "training": {"qualityPreset": "high", "method": "splatfacto-big", "fallbackMethod": "splatfacto",
               "iterations": 40000, "downscaleFactor": 1, "shDegree": 3, "randomSeed": 42,
               "useScaleRegularization": true,
               "cullAlphaThresh": null, "densifyGradThresh": null, "cullScaleThresh": null},
  "output": {"directory": "D:/PointForgeProjects/room", "plyFileName": "room_splat.ply", "format": "ply"},
  "environment": {"backend": "native", "nerfstudioPython": null, "colmapExe": null,
                  "ffmpegExe": null, "ffprobeExe": null},
  "options": {"allowCache": true, "stopAfter": null}
}
```

* Omitted fields come from the preset (`tools/pf360/pf360/presets.json`, the
  single source of truth also read by the wizard). `targetFrames` defaults to
  `framesPerSecond × window`, clamped.
* Every field is type/range checked (`manifest.validate`); all problems are
  reported at once. Paths in `environment` must be existing files; the manifest
  can never inject commands or arguments.
* `stopAfter`: `frame_extraction | panorama_projection | camera_reconstruction | training | export`.
* Crop semantics are Nerfstudio's: top/bottom remove elevation rings, left/right
  narrow the yaw span. Top+bottom combos that make Nerfstudio 1.1.5 crash are
  rejected up front (port verified identical on 1,800 combinations).
* Not exposed because Nerfstudio 1.1.x has no such setting: a maximum Gaussian count.

## 5. Event protocol (JSON Lines on stdout, v1)

One object per line: `{"v":1,"type":…,"jobId":…,"ts":…, …}`. stdout carries only
events; diagnostics go to stderr and `logs/worker.log`.

| type | extra fields |
|---|---|
| `worker_ready` | message |
| `job_started` | message |
| `stage_started` / `stage_completed` / `stage_skipped` | stage, data (completed/skipped) |
| `progress` | stage, percent (0–100 or **null** = unknown), message |
| `log` | level, message, stage? |
| `warning` / `error` | message, stage?, code? |
| `cancellation_requested` | message |
| `job_completed` | outputPath (null for partial runs), data = report |
| `job_failed` / `job_cancelled` | message, data = report |

Host control: write `cancel\n` to the worker's stdin (`--stdin-control`).
Exit codes: 0 done, 2 failed, 3 cancelled, 4 invalid manifest, 1 internal.

## 6. Workspace

```
<project>/manifest.json  state.json  source_reference.json
         frames/frame_00001.jpg …  frames/planar_projections/*.jpg
         processed/  (images, images_2/4/8, colmap/, transforms.json, sparse_pc.ply)
         training/pf360/<method>/run/  (config.yml, nerfstudio_models/*.ckpt)
         export/<name>.ply
         logs/worker.log ffmpeg.log panorama.log colmap.log training.log export.log
         reconstruction_report.json  reconstruction_report.txt
```

* The source video is never copied or modified (`source_reference.json` stores
  size, mtime and a SHA-256 of its first + last 8 MB).
* **Cache / retry**: each stage records a fingerprint chained from the video
  identity and every setting that affects it. A stage is reused only if the
  fingerprint matches *and* its outputs exist. Changing e.g. iterations re-runs
  training + export only. The wizard's *Retry* reuses this automatically.
* The worker only ever clears stage folders inside a folder carrying its own
  `state.json`; a non-empty foreign folder is refused. Nothing is deleted on
  failure or cancel. Delete the project folder yourself when finished.
* `reconstruction_report.json` reports unavailable metrics as `null`, never
  invented values; the `.txt` shows them as "unavailable".

## 7. Coordinate system

Nerfstudio's dataparser orients the scene so the average camera up is **+Z**
(`orientation_method="up"`) and `ns-export` writes model-space means. PointForge
loads such files as Z-up: `recon360IsZUpOutput()` recognises a pf360 output by
the workspace report beside it and overrides the global *Splat up axis*
preference (default Y-down, the convention of most downloaded 3DGS files) for
that file only. Quaternions are `rot_0..3 = w,x,y,z`, opacity is a logit and
scales are logs — exactly what `SplatReader::loadPlyGaussians` expects.
**Unverified until a real run** (see below).

## 8. Environment setup

1. FFmpeg: `winget install Gyan.FFmpeg` (or set the path in Preferences).
2. COLMAP **3.11.x**. Nerfstudio 1.1.5 passes `--SiftExtraction.use_gpu` /
   `--SiftMatching.use_gpu`, which newer COLMAP releases renamed; the doctor checks
   `colmap feature_extractor -h` and FAILs an incompatible build. Known-good:
   conda-forge `colmap=3.11.1=cuda_126*` into an isolated prefix that the worker
   discovers automatically:
   `conda create -y --prefix %LOCALAPPDATA%\ViitorX\PointForge\engines\colmap-conda -c conda-forge --override-channels "colmap=3.11.1=cuda_126*"`
   (the photogrammetry engine's 4.1.0 release under `engines\colmap` is only used
   as a fallback). Or set the path in Edit ▸ Preferences ▸ 360 Reconstruction.
3. Nerfstudio: `powershell -ExecutionPolicy Bypass -File scripts\pf360\setup_environment.ps1`
   — creates an isolated venv under `%LOCALAPPDATA%\ViitorX\PointForge\engines\pf360-env`,
   asks before downloading (~8 GB), never touches global Python, drivers or CUDA.
   Needs the **CUDA Toolkit 11.8** and Visual Studio C++ tools with an
   **MSVC 14.3x** toolset (CUDA 11.8's nvcc rejects MSVC ≥ 19.40). The script and the
   worker import a VS developer environment themselves (`scripts\pf360\msvc_env.ps1`,
   `environment.compiler_env()`), preferring a 14.3x toolset. It also pins
   `setuptools<70` (torch 2.1 imports `pkg_resources.packaging`).
   Versions: `tools/pf360/compat.json`.
4. Check: `scripts\pf360\doctor.ps1` or the wizard's *Full diagnostic*.
5. First training run compiles gsplat's CUDA kernels (cached in
   `%LOCALAPPDATA%\torch_extensions`); expect tens of minutes once.

Only the `native` execution backend is implemented. WSL2/Docker are not offered
in the UI; `environment.backend` is the extension point for them.

## 9. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| "No Nerfstudio Python environment found" | run the setup script or set *Nerfstudio Python* in Preferences |
| CUDA FAIL in doctor | CPU-only PyTorch or driver too old; install the CUDA wheel from compat.json. Training is never attempted on CPU. |
| Training fails immediately on Windows | gsplat JIT needs MSVC 14.3x build tools + CUDA toolkit 11.8; see `logs/training.log` |
| `No module named 'pkg_resources'` | setuptools ≥ 70 in the env: `<env>\Scripts\python -m pip install "setuptools<70"` |
| pip fails building `fpsample` (`nmake` / `CMAKE_CXX_COMPILER not set`) | no C++ compiler visible; re-run the setup script (it imports vcvars) |
| COLMAP FAIL "no --SiftExtraction.use_gpu option" | COLMAP too new for Nerfstudio 1.1.5; install 3.11.x (see §8) |
| COLMAP release download resets | some networks block `release-assets.githubusercontent.com`; use the conda-forge route in §8 |
| "registered cameras barely move" | the camera rotated in place; recapture walking through the scene |
| Low registration % | too little overlap/texture; try 14 views, more frames, or `exhaustive` matching |
| CUDA out of memory | raise downscale factor, fewer frames, standard Splatfacto, close GPU apps |
| Upside-down result | file not recognised as pf360 output (report moved?) — use Properties ▸ Up axis |

## 10. Tests

* Python: `cd tools\pf360 && python -m unittest discover -s tests -t .` — 68 tests
  (manifest, presets, Nerfstudio view-count parity, ffprobe parsing + projection
  classification, quality metrics, protocol, PLY validation incl. NaN/truncation/
  types, process-tree cancellation, Unicode/space paths, CLI flag construction
  for old/new tyro, COLMAP registration checks, export resolution, full worker
  orchestration with a fake backend incl. cache reuse/failure/cancel). Uses real
  ffmpeg/ffprobe when present (skips otherwise); no GPU.
* C++: `pf360hosttest` (built with `PF_BUILD_TESTS`) — 46 checks: view port,
  presets, manifest UTF-8 round trip, real worker launch + JSONL parsing, probe,
  frames-only job + cache reuse, failure propagation, cancel (worker exit 3,
  no surviving `ffmpeg.exe`), Z-up detection.
* Integration media needed for the GPU chain (not in the repo): a 20–60 s,
  ≥ 4K stitched equirectangular MP4 walked slowly through a static, textured
  room, camera at head height. Run it through the wizard with the *Preview*
  preset first, then *High Quality*.

## 11. Verification status

| Item | Status |
|---|---|
| Worker CLI, manifest, protocol, cache, report, validation | tested (68 Python tests) |
| Host launch / JSONL / cancel / Job Object | tested (pf360hosttest 46/46) |
| Static release build (VS 18, MSVC 14.51) incl. `pftest` PASS | done |
| Viewer starts and exits cleanly with the new code | done |
| Wizard UI interaction | **not exercised** by an automated run (native SDL window) |
| Nerfstudio projection → COLMAP → Splatfacto → export → load | **done 2026-11-21** on a *synthetic* 30 s 2880×1440 clip (`scripts/pf360/make_synthetic_360.py`), Preview preset, RTX 3060: 240/240 views registered, 30 distinct positions, 7000 its in 3.2 min, 259,599 Gaussians, validated PLY, loaded in PointForge (`--export-glb` smoke). Versions: `compat.json` |
| Z-up orientation of a real export | **verified** on that run: camera centres vary only in X/Y, mean camera-up = +Z in export space |
| Real-world 360 footage | **not run yet** (people/sky/moving objects untested; no masking) |

### Findings from the first real run (fixed)

* Nerfstudio's own COLMAP call registered **2 of 240** views: without EXIF it starts
  from COLMAP's focal guess (1.2 × 720 = 864 px) while the 120° planar views have
  f ≈ 208 px. The worker now runs COLMAP itself with a fixed `PINHOLE` camera
  (f = w / (2·tan(fov/2)), cx = w/2 + 0.5 — `planar_intrinsics()`), no intrinsics
  refinement, sequential overlap = 3 panoramas, picks the **largest** model (Nerfstudio
  only reads `sparse/0`) and calls `ns-process-data --skip-colmap --colmap-model-path`.
  Result: 240/240.
* gsplat's JIT build fails on Windows + CUDA 11.8 (`small` macro from `rpcndr.h`
  vs. CUB) and needs ~4 GB RAM per nvcc job; setup installs the prebuilt
  `gsplat==1.4.0+pt21cu118` wheel instead, and the worker caps `MAX_JOBS` by RAM.
* `dataparser_transforms.json` maps from the *original COLMAP* frame
  (`transform @ applied_transform`), not from `transforms.json` — use
  `T @ inv(applied_transform)` when relating the two.

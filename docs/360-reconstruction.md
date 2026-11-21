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

**Batch mode** (step 1: *Batch: all videos in a folder*): pick a root folder; every
`.mp4/.mov/.mkv` under it (optionally including subfolders, up to 1000, never inside
`PointForge360` or an existing workspace — `recon360FindVideos`) is probed in the
background and listed with length, resolution and a 360 check. Unusable files are
unticked; files without spherical metadata need one confirmation for all. Steps 2–3
apply to every selected video, which is processed over its **full length** (with
*Follow quality preset* each video gets its own frame count from the preset rate;
otherwise one fixed count); the preview uses the first video as the example. Step 4
picks one output folder (default `<root>\PointForge360`) and shows the plan per video:
workspace = file name (folder prefix / number on clashes), `new`, `resume` (existing
workspace, finished stages reused) or `skip` (already has its PLY, or a foreign folder) —
`recon360WorkspaceState`. *Start N reconstructions* queues one `Recon360` job each
(ids `pf360_<time>_<index>`, not auto-opened); they run one after another, a failure
does not stop the rest, and the batch table shows per-video state with Open / Folder,
*Cancel remaining*, *Retry failed / cancelled*, *Run in background*. Re-running the
same folder later skips finished videos, so an interrupted batch simply continues.

**Pause / resume (restore points).** *Pause* (run page, batch *Pause all*) asks the
worker to stop after saving a restore point; the job then shows as paused, and
*Resume* continues it. The wizard's first page lists **Unfinished reconstructions**
(`recon360ListUnfinished`: saved manifests whose workspace has no completed final
stage — paused, failed, or interrupted by closing PointForge / a crash / power loss)
with *Resume* / *Resume all*, so nothing depends on the app staying open. What is kept:

| Stage | Restore point |
|---|---|
| finished stages | reused via the fingerprint cache (as before) |
| panorama projection | panoramas whose views all exist are skipped; views are written to a temp name + renamed, so no truncated JPEG survives |
| masking | finished masks kept (atomic writes), still counted in the statistics |
| camera reconstruction | `database.db` (+ SQLite journal) kept: COLMAP skips extracted images and matched pairs; only the mapper (minutes) re-runs |
| training | checkpoint every 1000 steps; **Pause** makes the trainer save one at the current step first (`ns_helpers/train_resumable.py` sets Nerfstudio's `stop_training`, exit 75; killed only after 150 s). Resume: newest complete checkpoint (`--load-checkpoint`, truncated zips skipped) with `--max-num-iterations` = the remaining steps (Nerfstudio would otherwise train that many *more*) |
| frame extraction, export | short; re-run |

A stage counts as interrupted when `state.json` shows it `running` (crash), `cancelled`
(paused) or `failed` **with the same fingerprint**; changed settings start it fresh.
On app exit PointForge does not wait for a new checkpoint — the last periodic one
(≤ 1000 steps) is the restore point. Host grace for a pause: 180 s.

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
                                                        masking (optional)   ns_helpers/mask.py
      stderr -> Console                                 camera_reconstruction colmap (fixed PINHOLE) + ns-process-data --skip-colmap
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
| `masking` (optional; `stage_skipped` with `data.enabled=false` when off) | `ns_helpers/mask.py` in the NS env: UperNet-ConvNeXt-Base ADE20K semantic segmentation per perspective view | one mask per view; warns when > 40 % masked on average or views > 60 % masked |
| `camera_reconstruction` | COLMAP with fixed PINHOLE intrinsics (+ `--ImageReader.mask_path`), then `ns-process-data images --skip-colmap` (+ masks attached per frame) | `transforms.json`; ≥ 30 % and ≥ 8 images registered; > 2 distinct camera positions (rotation-only capture fails); warns < 70 %, split models, missing sparse cloud |
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
                 "cropLeft": 0, "cropRight": 0, "matchingMethod": "sequential",
                 "masking": {"enabled": true, "groups": ["people", "vehicles"], "dilatePx": 8}},
  "training": {"qualityPreset": "high", "method": "splatfacto-big", "fallbackMethod": "splatfacto",
               "iterations": 40000, "downscaleFactor": 1, "shDegree": 3, "randomSeed": 42,
               "useScaleRegularization": true,
               "cullAlphaThresh": null, "densifyGradThresh": null, "cullScaleThresh": null},
  "output": {"directory": "D:/PointForgeProjects/room", "plyFileName": "room_splat.ply", "format": "ply"},
  "environment": {"backend": "native", "nerfstudioPython": null, "colmapExe": null,
                  "ffmpegExe": null, "ffprobeExe": null, "maskModelDir": null},
  "options": {"allowCache": true, "stopAfter": null}
}
```

* Omitted fields come from the preset (`tools/pf360/pf360/presets.json`, the
  single source of truth also read by the wizard). `targetFrames` defaults to
  `framesPerSecond × window`, clamped.
* Every field is type/range checked (`manifest.validate`); all problems are
  reported at once. Paths in `environment` must be existing files; the manifest
  can never inject commands or arguments.
* `stopAfter`: `frame_extraction | panorama_projection | masking | camera_reconstruction | training | export`.
* `processing.masking` (default off): `groups` ⊆ `people` (incl. bicycle/minibike riders),
  `vehicles` (car, bus, truck, van, minibike, bicycle, boat, airplane, ship — parked ones too),
  `animals`, `sky`; `dilatePx` 0–64 grows the masked regions. Masks are 8-bit PNGs, 255 keep /
  0 ignore, named `<view>.jpg.png` (COLMAP `--ImageReader.mask_path`); for training they are
  copied to `processed/masks*/frame_NNNNN.png` at the exact size of each `images_N` file and
  referenced as `mask_path` in `transforms.json` (Splatfacto multiplies both rendered and
  ground-truth images by the mask). The image-name mapping is verified by pixel comparison.
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
         frames/frame_00001.jpg …  frames/planar_projections/*.jpg  frames/masks/*.png (if masking)
         processed/  (images, images_2/4/8, masks, masks_2/4/8, colmap/, transforms.json, sparse_pc.ply)
         training/pf360/<method>/run/  (config.yml, nerfstudio_models/*.ckpt)
         export/<name>.ply
         logs/worker.log ffmpeg.log panorama.log masking.log colmap.log training.log export.log
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

### 8.1 End users: engine packs (in-app download)

The viewer ships without the engines (~35 MB instead of ~11 GB). The wizard's
*Environment ▸ Engines (download)* section (opened automatically while the engines
are missing) installs them as **engine packs** from a signed catalog on Hugging Face:

| Pack id | Installs into `%LOCALAPPDATA%\ViitorX\PointForge\engines\` | Download | Required |
|---|---|---|---|
| `pf360-runtime` | `pf360-env\` — relocatable CPython 3.10 + PyTorch 2.1.2 cu118 + Nerfstudio 1.1.5 + prebuilt gsplat + transformers, LPIPS/AlexNet weights in `torch-home\` | ~3.2 GB | yes |
| `colmap` | `colmap-conda\` — COLMAP 3.11.1 CUDA (conda-forge), build-time files stripped | ~1.9 GB | yes |
| `ffmpeg` | `ffmpeg\bin\` — FFmpeg 9.0.2 shared build | ~0.1 GB | yes |
| `mask-model` | `models\upernet-convnext-base-ade20k\` — fetched straight from `openmmlab/upernet-convnext-base` at the pinned revision | ~0.5 GB | optional |

The pf360 worker itself is compiled into the exe (`tools/embed_worker.cmake` →
`EmbeddedWorker.h`) and extracted on first use to
`%LOCALAPPDATA%\ViitorX\PointForge\worker\<content-hash>\` when no `pf360\` folder sits
beside the exe (a `pf360\` folder or the source tree still wins, for development). So the
exe alone is enough. The worker runs on the pack's Python first, then the user's.

Needs: Windows 10 1803+ (uses the system `curl.exe` and `tar.exe`), an NVIDIA GPU
with a current driver, ~17 GB free disk during install (~10.5 GB afterwards). No
Python, Visual Studio or CUDA Toolkit is needed on the user's PC: nothing is compiled.

**Trust chain** (`src/viewer/EnginePacks.{h,cpp}`):
1. `catalog.json` + `catalog.json.sig` are fetched over HTTPS. The signature
   (ECDSA P-256 / SHA-256, raw r‖s base64) must verify against the public key
   compiled into the build (`src/viewer/EnginePackKey.h`). An empty key = the build
   refuses every catalog.
2. The catalog's `sequence` must not be lower than the newest one this PC accepted
   (`engines\.engine-state.json`) — an old, validly signed catalog cannot be replayed.
3. Every file is pinned by size + SHA-256; pack URLs are pinned to a Hugging Face
   commit. Downloads resume (`curl --continue-at -`, up to 8 attempts) from
   `engines\.downloads\<pack>\`; a hash mismatch deletes the file.
4. Fields are validated even when signed: `installDir` must be a relative path of
   `[A-Za-z0-9._-]` parts (no `..`), URLs must be `https://`, `postInstall` is one of a
   fixed list (`python-relocate`) — the catalog cannot name a command to run.
5. Zips are unpacked with `tar.exe` (refuses absolute / `..` entries) into
   `engines\.staging\`, then swapped in. A folder the installer did not create (a
   developer setup) is kept as `<dir>.backup-<time>`; a previous managed install is
   restored if anything fails, else deleted. `<dir>\.pfpack.json` records id/version.
6. `pf360-runtime`'s post-install step runs `python.exe -I pf360_relocate.py`, which
   regenerates the `ns-*.exe` launchers for the folder it was unpacked into (pip bakes
   absolute interpreter paths into them, so they are not shipped).

**Live catalog** (published 2026-11-21): `https://huggingface.co/vinayaklakhani/pointforge-engines/resolve/main/catalog.json`,
packs pinned to commit `fdb6a84…`; public key in `EnginePackKey.h` (private key kept by the
publisher, outside the repo). Configure builds with
`-DPF_ENGINE_CATALOG_URL=https://huggingface.co/vinayaklakhani/pointforge-engines/resolve/main/catalog.json`.

The worker treats a pack runtime specially (`environment.pack_env`): child
processes get `PYTHONNOUSERSITE=1` and `TORCH_HOME=<pack>\torch-home`, so neither a
Python the user installed nor a missing download.pytorch.org can break training.
The pack FFmpeg is preferred over `PATH`.

Install runs as a job (`Kind::EnginePacks`) on the same queue as reconstructions,
so packs are never swapped while a job uses them. *Pause download* cancels; starting
again resumes. The catalog URL is the CMake cache variable `PF_ENGINE_CATALOG_URL`
(baked in), overridable in *Engines ▸ Download source*.

**Publishing (developers)** — `scripts/pf360/engine_packs.py`, run with the pf360 env's Python:
1. Once: `engine_packs.py keygen --private-key %USERPROFILE%\.pointforge\engine-signing.pem`
   (asks for a passphrase; writes the public key into `EnginePackKey.h`; keep the private
   key out of the repo and backed up — losing it means shipping a new build with a new key).
2. `engine_packs.py build --out <dir>` zips the working engines from this PC
   (base CPython + venv site-packages merged; ~5.2 GB, ~15 min).
3. `huggingface-cli login`, then `engine_packs.py publish --repo <org>/<name> --packs-dir <dir> --private-key <pem>`
   uploads the zips, pins their URLs to the upload commit, signs and uploads the catalog,
   and prints the catalog URL for `-DPF_ENGINE_CATALOG_URL=...`.
4. Licences: the FFmpeg pack is a GPL-3.0 build — PointForge only runs it as a
   separate program, but the GPL source-offer obligations apply to distributing those
   binaries (or switch to an LGPL build). The CUDA DLLs in the PyTorch and COLMAP packs
   are NVIDIA redistributables under the CUDA EULA. Review before public release.

### 8.2 Developers: manual setup

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
5. gsplat: the setup script installs the **prebuilt** `gsplat 1.4.0+pt21cu118` wheel, so
   no CUDA compile is needed. (Falling back to the JIT build fails on Windows + CUDA 11.8.)
6. Optional masking add-on: `powershell -ExecutionPolicy Bypass -File scripts\pf360\setup_masking.ps1`
   — asks first, ~0.5 GB: `transformers==4.46.3` (+ safetensors, huggingface_hub) into the
   pf360 env with torch pinned, and `openmmlab/upernet-convnext-base` (MIT, **safetensors
   only**, pinned revision) into `engines\models\upernet-convnext-base-ade20k`. Jobs never
   download models; the doctor reports `Masking (optional)` PASS/SKIP and the wizard
   disables the masking controls until it is installed.

> **Note when installing from inside a packaged app (e.g. the Claude desktop app):**
> Windows may redirect that app's writes to `%LOCALAPPDATA%` into
> `%LOCALAPPDATA%\Packages\<app>\LocalCache\Local\…`, where PointForge started normally
> cannot see them. Run the setup scripts from a normal PowerShell window.

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
| "Only N of M perspective images could be placed" on real walking footage | COLMAP 3.11 has no rig model, so it tends to register only some view directions. pf360 now (a) poses every view of a panorama from any registered one (`ns_helpers/rig_complete.py`, exact rotations of Nerfstudio's projection, verified to 0.06° against a full COLMAP model), (b) matches across 5 panoramas, (c) starts the mapper at 4° instead of 16°, (d) **joins COLMAP's separate models** through panoramas they share (`rig_complete.py --sparse`: rotation from the rig, scale + offset from the panorama centres; real walking footage splits into forward-facing / backward-facing models over the same stretch), then re-triangulates the joined model with `colmap point_triangulator` (poses fixed). A failed job resumes with its COLMAP features + matches kept. If it still fails: bottom crop (the person / stick moving with the camera creates bogus matches), mask People, 8 views, fewer frames |
| "registered cameras barely move" | the camera rotated in place; recapture walking through the scene |
| Low registration % | too little overlap/texture; try 14 views, more frames, or `exhaustive` matching |
| CUDA out of memory | raise downscale factor, fewer frames, standard Splatfacto, close GPU apps |
| Upside-down result | file not recognised as pf360 output (report moved?) — use Properties ▸ Up axis |
| Masking controls greyed out | add-on not installed: run `setup_masking.ps1`, then *Full diagnostic* |
| "Masking is enabled but the segmentation model is not installed" | same; or turn masking off |
| Warning "N view(s) are more than 60% masked" | usually Sky on indoor scenes (textured ceilings can be classed as sky) or a crowd filling the view; drop that category or crop |
| Engines section: "this build has no engine signing key" | developer build without `EnginePackKey.h` populated: run `engine_packs.py keygen`, rebuild |
| "catalog signature is INVALID" | catalog edited after signing, or signed with a different key than this build embeds; re-run `publish` |
| "OLDER engine catalog than this PC already accepted" | the server serves a stale catalog (cache/mirror) or an old one was re-uploaded; publish again (new sequence) |
| "failed its SHA-256 check" | interrupted/corrupted download or the pack changed without a new catalog; it is deleted — install again |
| "cannot replace … a program is using it" | a reconstruction or diagnostic still runs from that folder; wait or cancel it, then install again |
| Download stops at the same point | network drops large transfers; *Install* again resumes from the partial file |
| Ghosts / smeared people remain | add Vehicles, raise *Grow mask*; masks are per-view semantic segmentation, so tiny distant people can be missed |

## 10. Tests

* Python: `cd tools\pf360 && python -m unittest discover -s tests -t .` — 80 tests
  (incl. masking manifest/switch/cache keys/model discovery and masked worker runs)
  (manifest, presets, Nerfstudio view-count parity, ffprobe parsing + projection
  classification, quality metrics, protocol, PLY validation incl. NaN/truncation/
  types, process-tree cancellation, Unicode/space paths, CLI flag construction
  for old/new tyro, COLMAP registration checks, export resolution, full worker
  orchestration with a fake backend incl. cache reuse/failure/cancel). Uses real
  ffmpeg/ffprobe when present (skips otherwise); no GPU.
* C++: `pf360hosttest` (built with `PF_BUILD_TESTS`) — 57 checks: view port,
  presets, manifest UTF-8 round trip + masking fields, real worker launch + JSONL parsing, probe,
  frames-only job + cache reuse, failure propagation, cancel (worker exit 3,
  no surviving `ffmpeg.exe`), Z-up detection, mask preview through the real model
  (skipped when the add-on is absent) and rejection of invalid group strings.
* Integration media needed for the GPU chain (not in the repo): a 20–60 s,
  ≥ 4K stitched equirectangular MP4 walked slowly through a static, textured
  room, camera at head height. Run it through the wizard with the *Preview*
  preset first, then *High Quality*.

## 11. Verification status

| Item | Status |
|---|---|
| Worker CLI, manifest, protocol, cache, report, validation | tested (80 Python tests) |
| Host launch / JSONL / cancel / Job Object / mask preview | tested (pf360hosttest 57/57) |
| Masking end to end | **done 2026-11-21** on the synthetic clip (people + sky, dilate 8): 240 masks in ~30 s on GPU, COLMAP 240/240 with masks, training with per-frame masks, 190,969 Gaussians, validated PLY. Segmentation quality checked on one real street photo (pedestrians, auto-rickshaw, truck, distant traffic masked; one tiny distant figure missed) |
| Static release build (VS 18, MSVC 14.51) incl. `pftest` PASS | done |
| Viewer starts and exits cleanly with the new code | done |
| Wizard UI interaction | **not exercised** by an automated run (native SDL window) |
| Nerfstudio projection → COLMAP → Splatfacto → export → load | **done 2026-11-21** on a *synthetic* 30 s 2880×1440 clip (`scripts/pf360/make_synthetic_360.py`), Preview preset, RTX 3060: 240/240 views registered, 30 distinct positions, 7000 its in 3.2 min, 259,599 Gaussians, validated PLY, loaded in PointForge (`--export-glb` smoke). Versions: `compat.json` |
| Z-up orientation of a real export | **verified** on that run: camera centres vary only in X/Y, mean camera-up = +Z in export space |
| Engine packs | **done 2026-11-21**: packs built (ffmpeg 0.10 GB, colmap 1.88 GB, pf360-runtime 3.49 GB; 523 s); installed by the C++ installer into a separate folder (193 s from a loopback server); relocated Python had no path outside its folder, CUDA + gsplat `_C` OK, launchers rewritten; doctor READY with `LOCALAPPDATA` pointed there and FFmpeg removed from PATH; masked Preview run (30 frames, 2000 its): 240/240 registered, 203,477 Gaussians, validated PLY. Mask pack downloaded from Hugging Face for real (466 MB, ~10.7 MB/s, SHA OK). `pf360hosttest` 100/100, Python 86. **Live**: real key + Hugging Face repo published; the installer fetched the signed catalog and installed all 4 packs from Hugging Face (~5.9 GB) in 609 s into a separate folder; doctor READY from it. **Not done**: a clean second PC; clicking through the Engines section in the wizard |
| Pause / resume | **done 2026-11-21** on the GPU (synthetic clip, masking on, 3000 its): paused in projection (112 views kept), masking (72 masks kept), COLMAP matching (database kept, extraction skipped on resume), training twice (stopped 1–2 s after Pause with checkpoints at steps 1356 and 2115); resumed each time, training continued 1356 → 2115 → 3000 (no extra steps), final 360,952 Gaussians, validated PLY. Found + fixed on the way: Nerfstudio 1.1.5 Splatfacto resume crash (optimizers bound to pre-load tensors) — `train_resumable.py` re-binds them. `pf360hosttest` 126/126 (manifest → job round trip, unfinished list), Python 91 |
| Panorama rig completion | math verified against a full COLMAP model (0.03° spread); knock-out test: from 1 of 8 views per panorama all 240 views restored within 0.06° / 0.16 % of the scene extent. **Not yet run on the failing real clip** |
| Model joining | **done 2026-11-21**: a real COLMAP model of the synthetic clip cut into 5 pieces (different view directions, overlapping time ranges, random scale/rotation/offset) joined back: 240/240 views, max 0.04° / 0.11 % of the walk; `point_triangulator` kept every pose exactly (38,693 points, all views linked); full worker run passes; `tests/test_rig_merge.py`. **Not yet run on the real clips** |
| Real-world batch (5 clips, 14 views, High) | 3 succeeded; 2 failed at 26 % / 16 % placed — logs showed COLMAP split each walk into many direction-wise models (largest 604 / 289 views), only the largest was used → model joining above; re-run pending |
| Real-world 360 footage | first real run (5.7K, 192 s, 14 views, High) **failed at camera reconstruction: 347/5600 views (6 %)** — fixes above, re-run pending |
| Real-world 360 footage (older note) | **not run yet** (masking not yet exercised on real moving people/traffic) |
| Wizard masking controls + mask preview overlay | built; the preview call path is tested by pf360hosttest, the ImGui controls were **not clicked through** |

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

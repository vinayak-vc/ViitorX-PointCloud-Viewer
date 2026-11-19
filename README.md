# PointForge

An out-of-core point cloud **importer + viewer**, built to ingest very large point
clouds (LAS/LAZ, E57, PLY, PTS/XYZ) — including data sets far larger than RAM
(billions of points) — and to display them interactively. The viewer also
renders **3D Gaussian Splats** (3DGS `.ply`, `.splat`, `.vxpc`).

It deliberately mirrors the technology stack of *Axiom Present*:

| Concern            | Technology                                  |
|--------------------|---------------------------------------------|
| Language           | C++17                                        |
| Windowing / input  | SDL2                                         |
| Rendering          | OpenGL 3.3 core (GLSL `#version 330 core`)   |
| GUI overlay        | Dear ImGui (docking branch)                  |
| Math               | GLM                                          |
| LAS/LAZ I/O        | LASzip (`laszip_api`)                        |
| E57 I/O            | libE57Format                                 |
| PLY I/O            | own parser (no external dep)                 |
| PTS/XYZ I/O        | own text parser                              |
| Node compression   | zstd (optional, `--compress` / Convert dialog)|
| Web remote         | civetweb + libdatachannel (WebRTC) + libjpeg-turbo |
| Photogrammetry     | ODM (Docker) / COLMAP — external engines, orchestrated (not linked in) |

PointForge is split into two programs that share a common core (`pfcore`):

* **`pfconvert`** — the *importer*. Converts a source point cloud into a
  streamable, disk-based **octree** (LOD pyramid). Runs out-of-core: memory use
  is bounded regardless of input size.
* **`pfview`** (built as `ViitorXPCViewer.exe`) — the *viewer*. Streams octree
  nodes on demand using frustum culling and screen-space-error LOD selection,
  so only the visible/relevant detail is ever resident on the GPU. Also draws
  Gaussian splat clouds (see §3, *Gaussian splats*).

A third target, **`PointForgeNative.dll`**, exposes the octree reader and a
Gaussian-splat API as a flat C interface for engine integrations (the Unreal
plugin *PointForge Viewer* uses it). See §2, *Native DLL*.

The viewer can also start from **plain photos** (drone/camera): the Convert
wizard reconstructs a point cloud with an external photogrammetry engine
(ODM or COLMAP, auto-installed on first consent) and then converts it like
any other scan — see §3.

Builds and runs on MSVC 2022 + vcpkg. See `CLAUDE.md` and `docs/ARCHITECTURE.md`
for the deeper why/how; this file is the quick-start.

---

## 1. Prerequisites

* MSVC 2022 (Windows) — the project is developed and tested on this toolchain.
  Other C++17 compilers (Clang 12+, GCC 10+) may work but are untested.
* [CMake](https://cmake.org/) ≥ 3.20
* [vcpkg](https://github.com/microsoft/vcpkg) — supplies every third-party
  library through the manifest in `vcpkg.json` (Dear ImGui is the one
  exception: it's pulled via CMake `FetchContent`, pinned to the
  `v1.91.5-docking` tag, because the vcpkg `imgui` port dropped its SDL2
  binding in 1.92.x and the viewer's docked UI needs the docking branch).

## 2. Building

Two build scripts live at the project root; both always build into a
directory **at the repo root** (`build\` or `build-static\`), regardless of
where you invoke them from. Both pick up `VCPKG_ROOT` from your environment,
falling back to `C:\vcpkg` if unset.

```powershell
.\build.bat          # dynamic build, fast iteration -> build\Release\
.\build-static.bat   # single-file release exe       -> build-static\Release\
```

Or run the underlying CMake commands directly:

```powershell
cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=C:/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build build --config Release
```

The vcpkg manifest (`vcpkg.json`) pulls in `sdl2`, `glew`, `glm`, `laszip`,
`libe57format`, `zstd`, `civetweb`, `nlohmann-json`, `libjpeg-turbo`, and
`libdatachannel` (web remote / WebRTC). Binaries land in `build\Release\`
(`pfconvert.exe`, `ViitorXPCViewer.exe` — plus `shaders\`, copied next to the
exe for the dynamic build only).

Watch the configure output for `-- LASzip: ...` and `-- libE57Format: ...` to
confirm those readers were detected; if a library is missing, the project
still builds — that reader becomes a graceful stub instead of a hard failure.

### Native DLL (`PointForgeNative.dll`)

Built with the rest of the project (CMake option `PF_BUILD_NATIVE_DLL`, on by
default). To build just the DLL:

```powershell
cmake --build build --config Release --target PointForgeNative
```

Headers: `src/library/unity/PointForgeC.h` (octree API) and
`src/library/native/PointForgeSplatC.h` (splat API). The splat API opens a
cloud (`PF_Splat_Open`), culls/sorts/projects it per camera on the calling
thread (`PF_Splat_Update`, keeps the nearest splats when capped), and exports
view-independent records for GPU renderers (`PF_Splat_CopyGpuRecords`). Copy the
DLL and headers into `Plugins/PointForgeViewer/ThirdParty/PointForgeNative/` of
the Unreal project when they change.

LASzip is looked up in vcpkg's own install folder (`VCPKG_INSTALLED_DIR`), so a
build directory that shares another vcpkg tree still links the release
library.

### Single-file release build

`build-static.bat` links everything statically (static CRT, static
SDL2/GLEW/LASzip/E57Format/zstd) and embeds the shaders + icon, producing one
self-contained `ViitorXPCViewer.exe` with zero third-party DLL dependencies —
verified with `dumpbin /DEPENDENTS` (only OS DLLs: OPENGL32, KERNEL32, USER32,
GDI32, WINMM, IMM32, ole32, OLEAUT32, VERSION, ADVAPI32, SETUPAPI, SHELL32).

The static vcpkg triplet must be compiled with the same MSVC toolset version
CMake's default generator links against, or you'll hit `LNK2019` on `__std_*`
STL symbols. `triplets\x64-windows-static.cmake` (an overlay triplet, wired in
by `build-static.bat`) pins that toolset version — see `docs/decisions.md`
("Single-File Static Release") if it ever needs re-pinning on a new machine.

## 3. Usage

### Import (convert to octree)

```powershell
pfconvert input.laz --out scans\mysite --chunk-depth 4 --spacing 0.0
```

* `--out <dir>`        output octree directory (created if missing, **required**)
* `--chunk-depth <n>`  coarse grid depth L for the disk-chunking pass; grid is (2^L)³ (default 4, range 0..10)
* `--spacing <meters>` root-node sample spacing; `0` = auto (cubeSize / 128)
* `--leaf <n>`         target max points per leaf node (default 50000)
* `--max-depth <n>`    hard octree depth cap (default 24)
* `--flush <points>`   chunker memory budget in points (default 16M ≈ 320 MB)
* `--keep-chunks`      keep intermediate chunk files (debugging)
* `--verbose`          debug logging

> The subtree-build phase (Phase C) runs on a worker pool (one thread per core
> by default, `--threads`/wizard-configurable) with byte-identical output to
> the sequential build — see `docs/decisions.md`.

Produces:

```
scans/mysite/
  meta.bin          FileMetadata (magic "PFO1") - binary metadata
  metadata.json     human-readable copy: bounds, scale/offset, attributes, spacing, point count
  hierarchy.bin     packed octree node descriptors (see docs/architecture.md)
  octree.bin        concatenated per-node point payloads
```

Conversion can also be started from inside the viewer — see below. (From the
viewer the default output is a single-file **`.vxpc` package** instead of the
loose folder; both load identically.)

### From photos (photogrammetry)

The Convert wizard's Source step has **"Or start from photos"**: pick a folder
of overlapping drone/camera photos (JPG/PNG/TIFF, 60–80% overlap) — or just
drag the folder onto the window. The wizard scans the images and recommends an
engine:

* **GPS EXIF tags found → ODM** — georeferenced, metric-scale LAZ straight
  from the tags. Runs via Docker.
* **No GPS (or no NVIDIA GPU) → COLMAP** — densest per-pixel reconstruction
  for ground-level/object captures; output has arbitrary scale/origin.
  Dense stereo needs CUDA. (Pinned 4.1.0, native install.)

Both engines stay external — nothing is linked into the exe. On first use one
consent dialog installs everything automatically (COLMAP release zip into app
data; Docker Desktop via winget + the ODM image), no manual steps.
Reconstruction runs as a cancelable background job that chains straight into
the octree conversion. If CPU virtualization is disabled in the BIOS (Docker
can't run), the wizard detects it, falls back to COLMAP and shows
board-specific BIOS steps to unlock ODM.

### View

```powershell
ViitorXPCViewer.exe scans\mysite
```

The viewer opens as a **docked shell**: menu bar + toolbar across the top, the
3D viewport in the center (never covered by panels), a *Properties* panel on
the right, and a bottom dock holding *Jobs* / *Console* / *Performance*
(closed by default — click their names under the *Window* menu, or let a
running conversion job auto-open *Jobs*). A status bar across the bottom shows
FPS, points drawn/total, GPU memory, and the current tool/mode.

To convert a scan **without leaving the viewer**: `Ctrl+I`, or drag a
`.las`/`.laz`/`.e57`/`.ply`/`.pts`/`.xyz` file onto the window. Conversion runs
as a background job — the dialog closes immediately, and progress shows in the
status-bar pill and the *Jobs* panel. Dragging an already-converted octree
**folder** onto the window loads it directly.

### Gaussian splats

```powershell
ViitorXPCViewer.exe D:\scans\room.ply
```

Or drag a 3DGS `.ply` / `.splat` onto the window. Splats are loaded directly
(no conversion). In *Properties › Display*:

* **Up axis** — *Y down (3DGS / COLMAP)* (default), *Y up (glTF / OpenGL)*,
  *Z up (as stored)*. 3DGS training output is Y down; the viewer's world is
  Z up, so the default opens such scenes upright. Changing it re-orients the
  loaded clouds in place and reframes.
* **LOD** — skip splats smaller than *Min splat size* (px). Off draws every
  splat.
* **CPU pre-cull** (off by default) — also cull on the sort thread. Faster for
  very large clouds; very fast turns can briefly show gaps.

These settings are saved. **F** frames splat clouds like point clouds.

How it renders: splat records are uploaded once; a background thread keeps a
back-to-front draw order (re-sorted when the view direction changes); the
vertex shader culls against the current camera every frame, so a lagging sort
never leaves holes. Each splat is drawn as a quad sized to its ellipse (and to
its visible extent for faint splats). Splats larger than the screen (camera
inside them) are skipped.

Controls:

| Input                  | Action                                    |
|-------------------------|-------------------------------------------|
| LMB drag                | Orbit around pivot                        |
| Double-click            | Focus pivot on point under cursor         |
| RMB drag                | Free look                                 |
| Wheel / Ctrl+wheel       | Zoom to cursor / point size               |
| WASD, Q/E                | Fly (Shift = fast)                        |
| F                       | Frame all                                 |
| 1 / 3 / 7 / 5           | Front / Side / Top view / toggle ortho    |
| M / C                   | Measure / Clip tool                       |
| Ctrl+O / Ctrl+I         | Open octree folder / Convert a scan       |
| Ctrl+, / Ctrl+P / F1    | Preferences / Command palette / Shortcuts |
| F3 / F5 or Shift+Space  | Stats overlay / hide all UI (zen mode)    |
| **F9**                  | **Stereoscopic SBS** — hides *all* UI so each eye sees only the cloud; F9 or Esc exits |
| F11 / F12               | Fullscreen / screenshot                   |
| Esc Esc                 | Quit                                      |

Gamepad (Xbox auto-mapped, or a custom joystick/ESP32-Bluetooth controller) is
supported too — configure it under *Edit > Preferences > Input*.

## 4. Layout

```
PointForge/
  CMakeLists.txt
  vcpkg.json
  build.bat            dynamic build -> build\
  build-static.bat      single-file release build -> build-static\
  triplets/             vcpkg overlay triplet (static-build toolset pin)
  README.md
  CLAUDE.md
  docs/
    architecture.md      the why: out-of-core build, octree/LOD, on-disk format, UI shell
    project-overview.md, roadmap.md, tasks.md, decisions.md, ai_handoff.md
  src/
    common/      Vec3d, AABB, Point, PointFormat, OctreeFormat, Log (+ setLogSink)
    io/          PointReader interface + LAS/E57/PLY/text readers + factory;
                 SplatReader (3DGS .ply / .splat, up-axis conversion)
    library/
      unity/     PointForgeC.h — flat C octree API
      native/    PointForgeSplatC.h — flat C Gaussian-splat API (PointForgeNative.dll)
    indexer/     out-of-core octree builder (chunking + sampling + writer)
    viewer/      Camera, Shader, OctreeStore, PointRenderer, SplatRenderer, Controller,
                 SerialController, Jobs.h (background convert job queue),
                 UiLog.h (Console panel log buffer), main.cpp (docked UI shell)
    tools/
      pfconvert/ importer CLI entry point
  shaders/       point.vert / point.frag (GLSL 330 core)
```

## 5. Tests / lint

`pftest` (built with the project) generates a synthetic `.xyz` cloud and
round-trips it through the indexer + `OctreeStore`, checking sequential vs
parallel byte-identity and hierarchy invariants:

```powershell
ctest --test-dir build-static -C Release --output-on-failure
```

No linter/CI yet. MSVC builds with `/W3 /permissive- /MP`; GCC/Clang with
`-Wall -Wextra`.

## 6. License of dependencies

LASzip (LGPL/permissive), libE57Format (BSD), SDL2 (zlib), GLEW (modified
BSD), GLM (MIT), Dear ImGui (MIT), zstd (BSD/GPLv2 dual). PLY and PTS/XYZ
readers are original code with no external dependency. Review each before
shipping commercially.

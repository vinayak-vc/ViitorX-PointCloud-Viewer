# pf360 — 360 video → 3D Gaussian Splat worker

Standalone, standard-library-only Python (≥ 3.8) worker used by PointForge's
360 Reconstruction Studio. It drives FFmpeg, COLMAP and Nerfstudio Splatfacto as
child processes and reports progress as JSON Lines. Full documentation:
[`docs/360-reconstruction.md`](../../docs/360-reconstruction.md).

```powershell
cd tools\pf360
python -m pf360 doctor                    # environment diagnostic (--json, --quick)
python -m pf360 probe D:\cap\room.mp4 --quality
python -m pf360 estimate --width 5760 --height 2880 --frames 90 --views 14 --crop 0 0.2 0 0
python -m pf360 run --manifest job.json   # JSONL events on stdout; Ctrl+C cancels
python -m pf360 validate-ply out.ply
python -m unittest discover -s tests -t . # 62 tests, no GPU needed
```

Minimal manifest (everything else comes from the preset):

```json
{"schemaVersion": 1, "jobId": "room_001",
 "input": {"videoPath": "D:/cap/room.mp4", "projectionConfirmed": true},
 "training": {"qualityPreset": "balanced"},
 "output": {"directory": "D:/PointForgeProjects/room"}}
```

Tool discovery: manifest `environment` → `PF360_NS_PYTHON` / `PF360_COLMAP` /
`PF360_FFMPEG` / `PF360_FFPROBE` → `%LOCALAPPDATA%\ViitorX\PointForge\engines\pf360-env`
→ PATH. Set up Nerfstudio with `scripts\pf360\setup_environment.ps1`
(isolated venv, asks first). Tested versions: `compat.json`.

Layout: `worker.py` (orchestration), `backend.py` (extension point),
`nerfstudio_backend.py`, `video_probe.py`, `views.py` (Nerfstudio view-geometry
port), `ply_validate.py`, `procrun.py`, `protocol.py`, `manifest.py` +
`presets.json`, `workspace.py`, `environment.py`, `ns_helpers/planar.py`
(runs inside the Nerfstudio env).

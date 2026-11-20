"""pf360 - PointForge 360-degree video -> 3D Gaussian Splat reconstruction worker.

The worker is deliberately standard-library only so it runs on any CPython
>= 3.8. Heavy tools (FFmpeg, COLMAP, Nerfstudio + PyTorch/CUDA) are external
executables discovered by `pf360.environment` and driven as child processes;
nothing here imports torch or nerfstudio.

Entry points:
    python -m pf360 doctor            environment diagnostic
    python -m pf360 probe <video>     video metadata + projection classification
    python -m pf360 run --manifest j  run a reconstruction job (JSONL on stdout)
"""

__version__ = "0.1.0"

# Version of the JSON Lines event protocol spoken on stdout (see docs/360-reconstruction.md).
PROTOCOL_VERSION = 1
# Version of the job manifest schema accepted by `run`.
MANIFEST_SCHEMA_VERSION = 1

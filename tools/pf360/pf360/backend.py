"""Reconstruction backend extension point.

The worker owns the backend-independent stages (video validation, frame
extraction, final PLY validation, reporting, caching). A backend owns
everything between "a folder of equirectangular frames" and "a 3DGS PLY".

A future backend (e.g. an omnidirectional-native Gaussian method that needs no
COLMAP, or a 4D/dynamic method) implements this interface and declares its own
stage list; the host UI renders whatever stages the worker reports.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from .environment import Tools
from .procrun import CancelToken
from .protocol import Emitter


@dataclass
class StageContext:
    manifest: Dict[str, Any]
    tools: Tools
    workspace: "Any"            # workspace.Workspace
    emitter: Emitter
    cancel: CancelToken
    video_info: Dict[str, Any] = field(default_factory=dict)
    results: Dict[str, Dict[str, Any]] = field(default_factory=dict)   # stage -> data of completed stages


class StageFailed(Exception):
    """A stage ran but its result is unusable (message is user-facing)."""

    def __init__(self, message: str, data: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.data = data or {}


class ReconstructionBackend:
    name = "abstract"

    def stages(self) -> Sequence[str]:
        """Backend-owned stage ids, in execution order."""
        raise NotImplementedError

    def stage_enabled(self, stage: str, ctx: StageContext) -> bool:
        """False -> the worker reports stage_skipped (optional stages such as masking)."""
        return True

    def check_environment(self, tools: Tools, manifest: Dict[str, Any]) -> List[str]:
        """Problems that make this backend unable to run the manifest (empty = OK)."""
        raise NotImplementedError

    def stage_inputs(self, stage: str, ctx: StageContext) -> Dict[str, Any]:
        """Settings that, if changed, invalidate this stage's cached output."""
        raise NotImplementedError

    def stage_outputs_exist(self, stage: str, ctx: StageContext) -> bool:
        raise NotImplementedError

    def run_stage(self, stage: str, ctx: StageContext) -> Dict[str, Any]:
        """Runs one stage; returns structured result data. Raises StageFailed / procrun.Cancelled."""
        raise NotImplementedError

    def result_ply(self, ctx: StageContext) -> Optional[Path]:
        raise NotImplementedError

"""Child-process execution with streaming output and cancellation.

* Arguments are always passed as a list (no shell, no string concatenation).
* stdout+stderr are merged and streamed line by line to a callback and to a
  per-tool log file, so memory stays bounded regardless of tool verbosity.
* Cancellation kills the WHOLE process tree (Nerfstudio spawns COLMAP and
  ffmpeg as grandchildren): `taskkill /T /F` on Windows, process-group
  SIGKILL elsewhere. The PointForge host additionally wraps the worker in a
  Windows Job Object as a last line of defence.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
from typing import Callable, Dict, IO, List, Optional, Sequence


class CancelToken:
    def __init__(self) -> None:
        self._ev = threading.Event()
        self._lock = threading.Lock()
        self._procs: List[subprocess.Popen] = []

    def cancel(self) -> None:
        self._ev.set()
        with self._lock:
            procs = list(self._procs)
        for p in procs:
            kill_tree(p)

    @property
    def cancelled(self) -> bool:
        return self._ev.is_set()

    def wait(self, timeout: float) -> bool:
        return self._ev.wait(timeout)

    def _register(self, p: subprocess.Popen) -> None:
        with self._lock:
            self._procs.append(p)
        if self.cancelled:
            kill_tree(p)

    def _unregister(self, p: subprocess.Popen) -> None:
        with self._lock:
            if p in self._procs:
                self._procs.remove(p)


class Cancelled(Exception):
    pass


def kill_tree(p: subprocess.Popen) -> None:
    if p.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), check=False)
        else:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
    except (OSError, ProcessLookupError):
        pass
    try:
        p.kill()
    except OSError:
        pass


def child_env(path_prepend: Sequence[str] = (), extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    env = dict(os.environ)
    dirs = [d for d in path_prepend if d]
    if dirs:
        env["PATH"] = os.pathsep.join(dirs + [env.get("PATH", "")])
    # UTF-8 everywhere so non-ASCII paths survive child Python processes.
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    # rich (used by Nerfstudio) must not emit live cursor animations into a pipe.
    env.setdefault("TERM", "dumb")
    env["COLUMNS"] = env.get("COLUMNS", "200")
    if extra:
        env.update(extra)
    return env


def run(args: Sequence[str], *, cancel: Optional[CancelToken] = None,
        on_line: Optional[Callable[[str], None]] = None, log: Optional[IO[str]] = None,
        cwd: Optional[str] = None, env: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None) -> int:
    """Runs a command to completion. Raises Cancelled if the token fires."""
    if cancel is not None and cancel.cancelled:
        raise Cancelled()
    args = [str(a) for a in args]
    if log is not None:
        log.write("\n$ " + subprocess.list2cmdline(args) + "\n")
        log.flush()
    kwargs = {}
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0) | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    p = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         cwd=cwd, env=env, **kwargs)
    if cancel is not None:
        cancel._register(p)
    timer = None
    if timeout:
        timer = threading.Timer(timeout, kill_tree, args=(p,))
        timer.daemon = True
        timer.start()
    try:
        assert p.stdout is not None
        # Split on both \n and \r: tqdm/rich progress bars rewrite with \r.
        buf = b""
        while True:
            chunk = p.stdout.read1(65536) if hasattr(p.stdout, "read1") else p.stdout.read(4096)
            if not chunk:
                break
            buf += chunk
            while True:
                idx = min([i for i in (buf.find(b"\n"), buf.find(b"\r")) if i >= 0], default=-1)
                if idx < 0:
                    break
                raw, buf = buf[:idx], buf[idx + 1:]
                _deliver(raw, on_line, log)
            if len(buf) > 1 << 20:      # pathological no-newline output: flush in pieces
                _deliver(buf, on_line, log)
                buf = b""
        if buf:
            _deliver(buf, on_line, log)
        rc = p.wait()
    finally:
        if timer is not None:
            timer.cancel()
        if cancel is not None:
            cancel._unregister(p)
        if p.poll() is None:
            kill_tree(p)
        if p.stdout is not None:
            p.stdout.close()
    if cancel is not None and cancel.cancelled:
        raise Cancelled()
    return rc


def _deliver(raw: bytes, on_line: Optional[Callable[[str], None]], log: Optional[IO[str]]) -> None:
    line = raw.decode("utf-8", errors="replace").rstrip()
    line = _strip_ansi(line)
    if not line.strip():
        return
    if log is not None:
        try:
            log.write(line + "\n")
        except (OSError, ValueError):
            pass
    if on_line is not None:
        on_line(line)


def _strip_ansi(s: str) -> str:
    out = []
    i = 0
    n = len(s)
    while i < n:
        c = s[i]
        if c == "\x1b":
            i += 1
            if i < n and s[i] == "[":
                i += 1
                while i < n and not ("@" <= s[i] <= "~"):
                    i += 1
            i += 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


def capture(args: Sequence[str], timeout: float = 60.0, env: Optional[Dict[str, str]] = None) -> subprocess.CompletedProcess:
    """Short-lived probe command; returns CompletedProcess (never raises on non-zero)."""
    kwargs = {}
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return subprocess.run([str(a) for a in args], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, timeout=timeout, env=env, check=False, **kwargs)

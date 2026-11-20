"""3DGS PLY validation, including PointForge SplatReader compatibility rules.

PointForge's SplatReader (src/io/SplatReader.cpp) reads:
  * `format binary_little_endian 1.0`
  * the `vertex` element as the first element
  * properties of type float/float32 (4 bytes) or double/float64 (8 bytes) only
  * x,y,z, scale_0..2 (log scale), opacity (logit), rot_0..3 (w,x,y,z),
    f_dc_0..2 (SH degree-0), optional f_rest_* (higher-order SH)
Nerfstudio's `ns-export gaussian-splat` (default sh_coeffs colour mode) writes
exactly this layout.

Numeric checks use numpy when importable (full scan, chunked); otherwise an
evenly-spaced sample of records is checked and the report says so.
"""
from __future__ import annotations

import math
import os
import struct
from typing import Any, Dict, List, Optional, Tuple

REQUIRED = ("x", "y", "z", "scale_0", "scale_1", "scale_2", "opacity", "rot_0", "rot_1", "rot_2", "rot_3",
            "f_dc_0", "f_dc_1", "f_dc_2")
TYPE_SIZES = {"float": 4, "float32": 4, "double": 8, "float64": 8}
OTHER_TYPES = {"char": 1, "int8": 1, "uchar": 1, "uint8": 1, "short": 2, "int16": 2, "ushort": 2, "uint16": 2,
               "int": 4, "int32": 4, "uint": 4, "uint32": 4}
MAX_HEADER = 64 * 1024


class PlyError(ValueError):
    pass


def parse_header(f) -> Tuple[Dict[str, Any], int]:
    head = b""
    while b"end_header\n" not in head and b"end_header\r\n" not in head:
        chunk = f.read(4096)
        if not chunk:
            raise PlyError("no end_header found - not a PLY file or truncated header")
        head += chunk
        if len(head) > MAX_HEADER:
            raise PlyError("PLY header larger than 64 KB")
    marker = b"end_header\r\n" if b"end_header\r\n" in head else b"end_header\n"
    end = head.index(marker) + len(marker)
    text = head[:end].decode("ascii", errors="replace").replace("\r", "")
    lines = text.split("\n")
    if lines[0].strip() != "ply":
        raise PlyError("file does not start with 'ply'")
    hdr: Dict[str, Any] = {"format": None, "elements": []}
    for ln in lines[1:]:
        tok = ln.split()
        if not tok or tok[0] in ("comment", "obj_info"):
            continue
        if tok[0] == "format":
            hdr["format"] = tok[1] if len(tok) > 1 else None
        elif tok[0] == "element":
            if len(tok) < 3 or not tok[2].isdigit():
                raise PlyError("malformed element line: %r" % ln)
            hdr["elements"].append({"name": tok[1], "count": int(tok[2]), "props": []})
        elif tok[0] == "property":
            if not hdr["elements"]:
                raise PlyError("property before any element")
            if tok[1] == "list":
                hdr["elements"][-1]["props"].append({"name": tok[-1], "type": "list"})
            else:
                hdr["elements"][-1]["props"].append({"name": tok[2] if len(tok) > 2 else "", "type": tok[1]})
    return hdr, end


def validate(path: str, sample_records: int = 200000) -> Dict[str, Any]:
    """Returns a report dict; report['ok'] is False with report['errors'] on failure."""
    rep: Dict[str, Any] = {"path": path, "ok": False, "errors": [], "warnings": []}
    errs: List[str] = rep["errors"]
    if not os.path.isfile(path):
        errs.append("file does not exist")
        return rep
    size = os.path.getsize(path)
    rep["fileSizeBytes"] = size
    if size == 0:
        errs.append("file is empty")
        return rep
    with open(path, "rb") as f:
        try:
            hdr, data_off = parse_header(f)
        except PlyError as e:
            errs.append(str(e))
            return rep
    if hdr["format"] != "binary_little_endian":
        errs.append("format is %r; PointForge SplatReader needs binary_little_endian" % hdr["format"])
    if not hdr["elements"] or hdr["elements"][0]["name"] != "vertex":
        errs.append("first element must be 'vertex'")
        return rep
    vert = hdr["elements"][0]
    props = vert["props"]
    names = [p["name"] for p in props]
    missing = [r for r in REQUIRED if r not in names]
    if missing:
        errs.append("missing Gaussian attributes: " + ", ".join(missing))
    bad_types = [p["name"] + ":" + p["type"] for p in props if p["type"] not in TYPE_SIZES]
    if bad_types:
        errs.append("unsupported property types for SplatReader (float/double only): " + ", ".join(bad_types))
    count = vert["count"]
    rep["gaussianCount"] = count
    if count <= 0:
        errs.append("Gaussian count is zero")
    n_rest = sum(1 for n in names if n.startswith("f_rest_"))
    rep["shRestCoefficients"] = n_rest
    deg = None
    for d in range(0, 5):
        if 3 * ((d + 1) ** 2 - 1) == n_rest:
            deg = d
    rep["shDegree"] = deg
    if deg is None:
        rep["warnings"].append("f_rest_* count %d does not match a whole SH degree" % n_rest)
    if errs:
        return rep
    stride = sum(TYPE_SIZES[p["type"]] for p in props)
    expected = data_off + stride * count
    if size < expected:
        errs.append("file truncated: %d bytes, header implies at least %d" % (size, expected))
        return rep
    if len(hdr["elements"]) == 1 and size > expected:
        rep["warnings"].append("%d trailing bytes after vertex data" % (size - expected))
    rep["stride"] = stride
    _numeric_checks(path, data_off, props, count, stride, sample_records, rep)
    rep["ok"] = not errs
    return rep


def _numeric_checks(path: str, off: int, props: List[Dict[str, Any]], count: int, stride: int,
                    sample_records: int, rep: Dict[str, Any]) -> None:
    idx = {p["name"]: i for i, p in enumerate(props)}
    try:
        import numpy as np  # type: ignore
    except Exception:
        np = None
    if np is not None:
        dt = np.dtype([(p["name"], "<f4" if TYPE_SIZES[p["type"]] == 4 else "<f8") for p in props])
        stats = _NpStats(np)
        chunk = max(1, (64 << 20) // stride)
        with open(path, "rb") as f:
            f.seek(off)
            left = count
            while left > 0:
                n = min(chunk, left)
                arr = np.frombuffer(f.read(n * stride), dtype=dt, count=n)
                stats.update(arr)
                left -= n
        rep["numericCheck"] = "full"
        stats.finish(rep)
        return
    # Pure-python sampled check.
    fmt = "<" + "".join("f" if TYPE_SIZES[p["type"]] == 4 else "d" for p in props)
    st = struct.Struct(fmt)
    n = min(count, sample_records)
    step = count / float(n)
    nonfinite = zero_q = 0
    lo = [math.inf] * 3
    hi = [-math.inf] * 3
    with open(path, "rb") as f:
        for k in range(n):
            i = int(k * step)
            f.seek(off + i * stride)
            rec = st.unpack(f.read(stride))
            if any(not math.isfinite(v) for v in rec):
                nonfinite += 1
                continue
            q = [rec[idx["rot_%d" % j]] for j in range(4)]
            if sum(v * v for v in q) < 1e-12:
                zero_q += 1
            for a, nm in enumerate("xyz"):
                v = rec[idx[nm]]
                lo[a] = min(lo[a], v)
                hi[a] = max(hi[a], v)
    rep["numericCheck"] = "sampled %d of %d records (numpy unavailable)" % (n, count)
    _finish_common(rep, nonfinite, zero_q, lo, hi, n)


class _NpStats:
    def __init__(self, np) -> None:
        self.np = np
        self.nonfinite = 0
        self.zero_q = 0
        self.n = 0
        self.lo = [math.inf] * 3
        self.hi = [-math.inf] * 3
        self.max_log_scale = -math.inf

    def update(self, arr) -> None:
        np = self.np
        finite = np.ones(len(arr), dtype=bool)
        for nm in arr.dtype.names:
            finite &= np.isfinite(arr[nm])
        self.nonfinite += int((~finite).sum())
        a = arr[finite]
        self.n += len(arr)
        if len(a) == 0:
            return
        qn = a["rot_0"] ** 2 + a["rot_1"] ** 2 + a["rot_2"] ** 2 + a["rot_3"] ** 2
        self.zero_q += int((qn < 1e-12).sum())
        for i, nm in enumerate("xyz"):
            self.lo[i] = min(self.lo[i], float(a[nm].min()))
            self.hi[i] = max(self.hi[i], float(a[nm].max()))
        self.max_log_scale = max(self.max_log_scale, float(max(a["scale_0"].max(), a["scale_1"].max(), a["scale_2"].max())))

    def finish(self, rep: Dict[str, Any]) -> None:
        _finish_common(rep, self.nonfinite, self.zero_q, self.lo, self.hi, self.n)
        if math.isfinite(self.max_log_scale):
            rep["maxScale"] = math.exp(min(self.max_log_scale, 80.0))


def _finish_common(rep: Dict[str, Any], nonfinite: int, zero_q: int, lo: List[float], hi: List[float], n: int) -> None:
    if nonfinite:
        rep["errors"].append("%d records contain NaN/Inf values" % nonfinite)
    if zero_q:
        rep["warnings"].append("%d Gaussians have a zero-length rotation quaternion" % zero_q)
    if all(math.isfinite(v) for v in lo + hi):
        rep["boundsMin"] = lo
        rep["boundsMax"] = hi
        ext = max(h - l for l, h in zip(lo, hi))
        rep["extent"] = ext
        if ext <= 0:
            rep["errors"].append("all Gaussians share one position (degenerate)")
    rep["checkedRecords"] = n

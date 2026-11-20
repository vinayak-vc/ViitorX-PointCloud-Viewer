"""Shared test helpers (no GPU / Nerfstudio needed)."""
import os
import shutil
import struct
import subprocess
from typing import Dict, List, Optional, Sequence

GS_PROPS = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2"] + ["f_rest_%d" % i for i in range(45)] + \
           ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]


def write_ply(path: str, n: int = 100, props: Optional[Sequence[str]] = None, fmt: str = "binary_little_endian",
              ptype: str = "float", values=None, truncate: int = 0) -> None:
    props = list(props or GS_PROPS)
    hdr = "ply\nformat %s 1.0\nelement vertex %d\n" % (fmt, n)
    hdr += "".join("property %s %s\n" % (ptype, p) for p in props) + "end_header\n"
    code = {"float": "f", "double": "d", "uchar": "B"}[ptype]
    st = struct.Struct("<" + code * len(props))
    with open(path, "wb") as f:
        f.write(hdr.encode("ascii"))
        body = bytearray()
        for i in range(n):
            if values is not None:
                rec = values(i, props)
            else:
                rec = []
                for p in props:
                    if p in ("x", "y", "z"):
                        rec.append(float(i % 10) * (1 + "xyz".index(p)))
                    elif p == "rot_0":
                        rec.append(1.0)
                    elif p.startswith("scale"):
                        rec.append(-4.0)
                    else:
                        rec.append(0.1)
            body += st.pack(*rec)
        if truncate:
            body = body[:-truncate]
        f.write(bytes(body))


def ffmpeg() -> Optional[str]:
    return shutil.which("ffmpeg")


def make_video(path: str, w: int = 1024, h: int = 512, seconds: int = 4) -> bool:
    ff = ffmpeg()
    if not ff:
        return False
    cp = subprocess.run([ff, "-v", "error", "-y", "-f", "lavfi", "-i",
                         "testsrc2=size=%dx%d:rate=30:duration=%d" % (w, h, seconds), "-pix_fmt", "yuv420p", path],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return cp.returncode == 0 and os.path.isfile(path)

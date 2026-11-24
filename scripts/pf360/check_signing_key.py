"""Check that a passphrase unlocks the engine-pack signing key, and that the key matches the
public key built into PointForge (src/viewer/EnginePackKey.h). Uploads nothing.

usage: python check_signing_key.py [path-to-pem]
The passphrase comes from PF_ENGINE_KEY_PASSPHRASE (set it with Read-Host -AsSecureString),
so no terminal password prompt is involved."""
import os
import re
import sys
from pathlib import Path

from cryptography.hazmat.primitives import serialization

REPO = Path(__file__).resolve().parents[2]
pem = Path(sys.argv[1] if len(sys.argv) > 1 else Path.home() / ".pointforge" / "engine-signing.pem")
pw = os.environ.get("PF_ENGINE_KEY_PASSPHRASE", "")
print("key file      :", pem, "(%d bytes)" % pem.stat().st_size)
print("passphrase    : %d characters, %s" % (len(pw), "ASCII only" if pw.isascii() else "contains non-ASCII characters"))
if pw != pw.strip():
    print("note          : the passphrase has leading/trailing spaces")
try:
    key = serialization.load_pem_private_key(pem.read_bytes(), password=pw.encode() if pw else None)
except Exception as e:                                   # noqa: BLE001 - report, never print the passphrase
    print("RESULT        : does NOT unlock the key (%s)" % type(e).__name__)
    sys.exit(1)
nums = key.public_key().public_numbers()
hexkey = "%064x%064x" % (nums.x, nums.y)
built = re.search(r'"([0-9a-f]{128})"', (REPO / "src" / "viewer" / "EnginePackKey.h").read_text())
print("RESULT        : passphrase OK")
print("matches build : %s" % ("YES" if built and built.group(1) == hexkey else "NO - this key is not the one PointForge trusts"))

"""Post-install fix-up for the PointForge "pf360-runtime" engine pack.

Shipped inside the pack as pf360-env/pf360_relocate.py and run once by
PointForge's engine installer, right after unpacking, with the pack's own
interpreter:

    <engines>/pf360-env/python.exe -I <engines>/pf360-env/pf360_relocate.py

Why it exists: pip writes console-script launchers (ns-train.exe, ...) with the
ABSOLUTE path of the interpreter baked in, so the launchers from the packaging
machine cannot be shipped. The pack therefore contains no Scripts/ folder and
this script regenerates every console/gui script for the folder the pack was
unpacked into. Re-running it is harmless (launchers are overwritten).

Standard library + pip's vendored distlib only (pip is part of the pack).
Exit code 0 = success; prints one summary line.
"""
import os
import sys
from importlib import metadata


def main() -> int:
    root = os.path.dirname(os.path.abspath(sys.executable))
    if os.path.normcase(os.path.dirname(os.path.abspath(__file__))) != os.path.normcase(root):
        print("pf360_relocate: run me with the python.exe that sits next to me", file=sys.stderr)
        return 2
    scripts = os.path.join(root, "Scripts")
    os.makedirs(scripts, exist_ok=True)
    try:
        from pip._vendor.distlib.scripts import ScriptMaker
    except ImportError as e:     # pip is part of the pack; a missing pip means a broken pack
        print("pf360_relocate: pip/distlib not importable: %s" % e, file=sys.stderr)
        return 3

    console, gui = set(), set()
    for dist in metadata.distributions():
        for ep in dist.entry_points:
            if ep.group == "console_scripts":
                console.add("%s = %s" % (ep.name, ep.value))
            elif ep.group == "gui_scripts":
                gui.add("%s = %s" % (ep.name, ep.value))

    maker = ScriptMaker(None, scripts)
    maker.executable = sys.executable      # absolute path of THIS install (quoted by distlib if needed)
    maker.clobber = True
    maker.variants = {""}                  # ns-train.exe only, no ns-train-3.10.exe
    made = []
    made += maker.make_multiple(sorted(console), {"gui": False})
    made += maker.make_multiple(sorted(gui), {"gui": True})
    missing = [n for n in ("ns-train", "ns-process-data", "ns-export")
               if not os.path.isfile(os.path.join(scripts, n + ".exe"))]
    if missing:
        print("pf360_relocate: launchers not created: %s" % ", ".join(missing), file=sys.stderr)
        return 4
    print("pf360_relocate: wrote %d launchers into %s" % (len(made), scripts))
    return 0


if __name__ == "__main__":
    sys.exit(main())

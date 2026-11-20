import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest

from pf360 import procrun

# Child that spawns a long-lived grandchild, prints its pid, then sleeps.
CHILD = r"""
import subprocess, sys, time
g = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
print("GRANDCHILD", g.pid, flush=True)
time.sleep(120)
"""


def pid_alive(pid: int) -> bool:
    if os.name == "nt":
        out = subprocess.run(["tasklist", "/FI", "PID eq %d" % pid, "/NH"], capture_output=True, text=True).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


class CancellationTests(unittest.TestCase):
    def test_cancel_kills_process_tree(self):
        token = procrun.CancelToken()
        seen = {}

        def on_line(l):
            if l.startswith("GRANDCHILD"):
                seen["pid"] = int(l.split()[1])
                threading.Timer(0.3, token.cancel).start()

        t0 = time.time()
        with self.assertRaises(procrun.Cancelled):
            procrun.run([sys.executable, "-c", CHILD], cancel=token, on_line=on_line)
        self.assertLess(time.time() - t0, 30)
        self.assertIn("pid", seen)
        deadline = time.time() + 10
        while pid_alive(seen["pid"]) and time.time() < deadline:
            time.sleep(0.2)
        self.assertFalse(pid_alive(seen["pid"]), "grandchild survived cancellation")

    def test_cancel_before_start(self):
        token = procrun.CancelToken()
        token.cancel()
        with self.assertRaises(procrun.Cancelled):
            procrun.run([sys.executable, "-c", "print(1)"], cancel=token)

    def test_exit_code_and_lines(self):
        lines = []
        with tempfile.TemporaryFile("w+", encoding="utf-8") as log:
            rc = procrun.run([sys.executable, "-c", "import sys;print('a');print('b\\rc');sys.exit(3)"],
                             on_line=lines.append, log=log)
        self.assertEqual(rc, 3)
        self.assertEqual(lines, ["a", "b", "c"])

    def test_unicode_and_spaces_in_args(self):
        tmp = tempfile.mkdtemp(prefix="pf 360 ünï ")
        lines = []
        rc = procrun.run([sys.executable, "-c", "import sys;print(sys.argv[1])", tmp], on_line=lines.append,
                         env=procrun.child_env())
        self.assertEqual(rc, 0)
        self.assertEqual(lines, [tmp])

    def test_ansi_stripped(self):
        self.assertEqual(procrun._strip_ansi("\x1b[1;33mhi\x1b[0m"), "hi")


if __name__ == "__main__":
    unittest.main()

"""COLMAP mapper progress: monotonic percent over split models + heartbeat while COLMAP is quiet."""
import io
import os
import unittest

from pf360 import protocol
from pf360.nerfstudio_backend import _MapperProgress, _Ticker, _fmt_dur

P = "I20261123 12:55:19.437625 15788 incremental_pipeline.cc:"


def reg(n, iid=100):
    return P + "390] Registering image #%d (%d)" % (iid, n)


class MapperProgressTests(unittest.TestCase):
    def test_percent_never_drops_when_colmap_starts_a_new_model(self):
        mp = _MapperProgress(1000, 55.0, 90.0)
        seen = []
        logs = []
        lines = [P + "282] Finding good initial image pair", P + "306] Initializing with image pair #1 and #2"]
        lines += [reg(n) for n in range(3, 401)]
        lines += [P + "42] Retriangulation and Global bundle adjustment"]
        lines += [P + "282] Finding good initial image pair", P + "306] Initializing with image pair #5 and #9"]
        lines += [reg(n) for n in range(3, 201)]
        for l in lines:
            got = mp.feed(l)
            if got:
                seen.append(got[0])
                logs += got[2]
        self.assertEqual(seen, sorted(seen))                       # monotonic
        self.assertAlmostEqual(seen[-1], 55.0 + 35.0 * 600 / 1000, places=3)
        self.assertIn("COLMAP mapper started", logs[0])
        self.assertIn("COLMAP model 1 finished with 400 views (400 of 1000 views", logs[1])
        self.assertEqual(len(logs), 2)
        self.assertEqual(mp.models, 2)
        self.assertIn("2 model(s) from 2 attempt(s), 600 of 1000 views", mp.summary())

    def test_failed_initial_pairs_do_not_count_as_models(self):
        mp = _MapperProgress(100, 0.0, 100.0)
        for _ in range(3):
            mp.feed(P + "282] Finding good initial image pair")
            mp.feed(P + "306] Initializing with image pair #1 and #2")
        mp.feed(reg(3))                                            # a 3-view attempt COLMAP discards
        mp.feed(P + "306] Initializing with image pair #1 and #2")
        self.assertEqual((mp.models, mp.attempts, mp.done), (0, 4, 0))
        pct, msg, _ = mp.feed(P + "42] Retriangulation and Global bundle adjustment")
        self.assertIn("bundle adjustment - model 1", msg)

    def test_percent_is_capped_when_views_sit_in_several_models(self):
        mp = _MapperProgress(20, 55.0, 90.0)
        for _ in range(3):
            mp.feed(P + "306] Initializing with image pair #1 and #2")
            for n in range(3, 16):
                mp.feed(reg(n))
        self.assertEqual(mp.feed(reg(16))[0], 90.0)

    def test_unrelated_lines_are_ignored(self):
        self.assertIsNone(_MapperProgress(10, 0, 1).feed("W2026 levenberg_marquardt_strategy.cc:123] Linear solver"))

    @unittest.skipUnless(os.environ.get("PF360_COLMAP_LOG"), "set PF360_COLMAP_LOG to replay a real colmap.log")
    def test_replay_real_log(self):
        mp = _MapperProgress(int(os.environ.get("PF360_TOTAL", "28000")), 55.0, 90.0)
        last, n = 55.0, 0
        with open(os.environ["PF360_COLMAP_LOG"], encoding="utf-8", errors="replace") as f:
            for l in f:
                got = mp.feed(l)
                if got:
                    self.assertGreaterEqual(got[0], last)
                    last, n = got[0], n + 1
        print("\nreplay: %d updates, %s, final %.1f%%" % (n, mp.summary(), last))


class TickerTests(unittest.TestCase):
    def test_heartbeat_only_when_quiet_and_logs_every_period(self):
        now = [0.0]
        buf = io.StringIO()
        t = _Ticker(protocol.Emitter(buf, "j"), "camera_reconstruction", quiet_after=30.0, log_every=300.0,
                    clock=lambda: now[0])
        t.set(70.0, "bundle adjustment - model 3")
        now[0] = 10.0
        t.tick()                                                   # not quiet yet: nothing
        now[0] = 45.0
        t.tick()
        now[0] = 301.0
        t.tick()
        now[0] = 320.0
        t.tick()                                                   # same 5-min period: no second log
        t.seen()
        now[0] = 330.0
        t.tick()                                                   # output resumed: nothing
        evs = [protocol.parse_event(l) for l in buf.getvalue().splitlines()]
        kinds = [(e["type"], e.get("percent")) for e in evs]
        self.assertEqual(kinds, [("progress", 70.0), ("progress", 70.0), ("progress", 70.0), ("log", None),
                                 ("progress", 70.0)])
        self.assertIn("still working, no output for 45s", evs[1]["message"])
        self.assertIn("no output from the tool for 5m 01s", evs[3]["message"])

    def test_thread_starts_and_stops(self):
        buf = io.StringIO()
        with _Ticker(protocol.Emitter(buf, "j"), "s", interval=0.01, quiet_after=0.0) as t:
            t.set(1.0, "x")
            import time
            time.sleep(0.05)
        self.assertIn("still working", buf.getvalue())

    def test_duration_format(self):
        self.assertEqual(_fmt_dur(5), "5s")
        self.assertEqual(_fmt_dur(125), "2m 05s")
        self.assertEqual(_fmt_dur(3725), "1h 02m")


if __name__ == "__main__":
    unittest.main()

import io
import json
import unittest

from pf360 import protocol


class ProtocolTests(unittest.TestCase):
    def test_events_are_single_line_json_with_job_id(self):
        buf = io.StringIO()
        em = protocol.Emitter(buf, job_id="j1")
        em.emit("stage_started", stage="training")
        em.progress("training", 150.0, "step")
        em.progress("training", None)
        em.log("hello\nworld")
        lines = buf.getvalue().splitlines()
        self.assertEqual(len(lines), 4)
        evs = [protocol.parse_event(l) for l in lines]
        self.assertTrue(all(e["jobId"] == "j1" and e["v"] == protocol.PROTOCOL_VERSION for e in evs))
        self.assertEqual(evs[1]["percent"], 100.0)          # clamped
        self.assertIn("percent", evs[2])                    # unknown progress is explicit null
        self.assertIsNone(evs[2]["percent"])
        self.assertEqual(evs[3]["message"], "hello\nworld")

    def test_unknown_type_rejected(self):
        with self.assertRaises(ValueError):
            protocol.Emitter(io.StringIO()).emit("bogus")
        with self.assertRaises(ValueError):
            protocol.parse_event(json.dumps({"v": 1, "type": "bogus", "jobId": ""}))
        with self.assertRaises(ValueError):
            protocol.parse_event(json.dumps({"v": 99, "type": "log", "jobId": ""}))
        with self.assertRaises(ValueError):
            protocol.parse_event(json.dumps({"v": 1, "type": "progress", "jobId": "", "percent": 101}))

    def test_required_event_types_exist(self):
        for t in ("worker_ready", "job_started", "stage_started", "progress", "log", "warning", "error",
                  "stage_completed", "cancellation_requested", "job_completed", "job_failed", "job_cancelled"):
            self.assertIn(t, protocol.EVENT_TYPES)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Unit tests for analyze-cupti-gaps.py."""

import importlib.util
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
import zipfile

sys.dont_write_bytecode = True
SPEC = importlib.util.spec_from_file_location("analyze_cupti_gaps", Path(__file__).with_name("analyze-cupti-gaps.py"))
gaps = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gaps)


def make_cdw1_frame(kind, payload_dict, session=b"\x00" * 16, pid=1, gen=1, seq=1):
    payload = json.dumps(payload_dict).encode("utf-8")
    hdr = bytearray(80)
    hdr[:4] = b"CDW1"
    struct.pack_into(">HHI", hdr, 4, 1, kind, len(payload))
    hdr[12:28] = session
    struct.pack_into(">QQQQ", hdr, 28, pid, gen, seq, 0)
    # timestamp at 60:68
    struct.pack_into(">Q", hdr, 60, 1000)
    return bytes(hdr) + payload


class CuptiGapsTest(unittest.TestCase):
    def setUp(self):
        root = Path.home() / "tmp"
        root.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="cupti-gaps-test-", dir=root)
        self.addCleanup(self.temp.cleanup)
        self.dir = Path(self.temp.name)

    def test_reconstructs_gaps_and_classifies_host_concurrency(self):
        # Op 0: kernel 0 (GPU: 100 to 200), launched at host: 50 to 80
        # Op 1: kernel 1 (GPU: 220 to 300), launched at host: 150 to 180 (Host Ahead: submitted before Op 0 ended at 200)
        #   Gap between Op 0 and Op 1: 200 to 220 (20 ns). Since api1 ended at 180 <= 200, it is HW Queue delay.
        # Op 2: kernel 2 (GPU: 350 to 450), launched at host: 290 to 320 (Host During: started at 290 <= 300 < 320)
        #   Gap between Op 1 and Op 2: 300 to 350 (50 ns). Driver submission: 320 - 300 = 20 ns. HW Queue: 350 - 320 = 30 ns.
        # Op 3: kernel 3 (GPU: 550 to 650), launched at host: 480 to 510 (Host Behind: started at 480 > 450)
        #   Gap between Op 2 and Op 3: 450 to 550 (100 ns). Host delay: 480 - 450 = 30 ns. Driver: 510 - 480 = 30 ns. HW Queue: 550 - 510 = 40 ns.

        frames = []
        # APIs
        frames.append(make_cdw1_frame(gaps.KIND_API, {"kind": "api", "domain": "driver", "callback_id": 307, "start": 50, "end": 80, "correlation": 1}, seq=1))
        frames.append(make_cdw1_frame(gaps.KIND_API, {"kind": "api", "domain": "driver", "callback_id": 307, "start": 150, "end": 180, "correlation": 2}, seq=2))
        frames.append(make_cdw1_frame(gaps.KIND_API, {"kind": "api", "domain": "driver", "callback_id": 307, "start": 290, "end": 320, "correlation": 3}, seq=3))
        frames.append(make_cdw1_frame(gaps.KIND_API, {"kind": "api", "domain": "driver", "callback_id": 307, "start": 480, "end": 510, "correlation": 4}, seq=4))

        # Activities
        frames.append(make_cdw1_frame(gaps.KIND_ACTIVITY, {"kind": "activity", "subtype": "kernel", "name": "k0", "valid": True, "start": 100, "end": 200, "correlation": 1}, seq=5))
        frames.append(make_cdw1_frame(gaps.KIND_ACTIVITY, {"kind": "activity", "subtype": "kernel", "name": "k1", "valid": True, "start": 220, "end": 300, "correlation": 2}, seq=6))
        frames.append(make_cdw1_frame(gaps.KIND_ACTIVITY, {"kind": "activity", "subtype": "kernel", "name": "k2", "valid": True, "start": 350, "end": 450, "correlation": 3}, seq=7))
        frames.append(make_cdw1_frame(gaps.KIND_ACTIVITY, {"kind": "activity", "subtype": "kernel", "name": "k3", "valid": True, "start": 550, "end": 650, "correlation": 4}, seq=8))

        raw_bytes = b"".join(frames)
        res = gaps.analyze_journal(raw_bytes, warmup_ms=0.0)

        self.assertEqual(res["total_activities"], 4)
        self.assertEqual(res["total_apis"], 4)
        self.assertEqual(res["correlation_match_rate"], 1.0)

        ov = res["overall"]
        self.assertEqual(ov["gap_count"], 3)
        # Total active time: (200-100) + (300-220) + (450-350) + (650-550) = 100 + 80 + 100 + 100 = 380 ns
        self.assertAlmostEqual(ov["active_ms"], 380 / 1e6)
        # Total gap time: 20 + 50 + 100 = 170 ns
        self.assertAlmostEqual(ov["gap_ms"], 170 / 1e6)

        # Classification counts
        lcounts = ov["launch_state_counts"]
        self.assertEqual(lcounts["host_ahead"], 1)
        self.assertEqual(lcounts["host_during"], 1)
        self.assertEqual(lcounts["host_behind"], 1)

        # Attribution:
        # Host delay: 30 ns
        # Driver submission: 0 + 20 + 30 = 50 ns
        # HW Queue: 20 + 30 + 40 = 90 ns
        # Total = 30 + 50 + 90 = 170 ns
        attr = ov["attribution"]
        self.assertAlmostEqual(attr["host_delay_ms"], 30 / 1e6)
        self.assertAlmostEqual(attr["driver_submission_ms"], 50 / 1e6)
        self.assertAlmostEqual(attr["hardware_queue_ms"], 90 / 1e6)

    def test_detects_omitted_files_trap_in_bundle(self):
        # A bundle with omitted_files containing capture.journal must raise JournalError
        status = {
            "run_completeness": "complete",
            "dropped_records": 0,
            "omitted_files": [{"name": "worker-process-1-capture.journal", "bytes": 67383147, "sha256": "abc"}]
        }
        bundle_path = self.dir / "test_bundle.pprof"
        with zipfile.ZipFile(bundle_path, "w") as zf:
            zf.writestr("capture-status.json", json.dumps(status))
            zf.writestr("profile.pb", b"dummy")

        with self.assertRaises(gaps.JournalError) as ctx:
            gaps.extract_journal_bytes(bundle_path)
        self.assertIn("dropped from bundle", str(ctx.exception))

    def test_reads_embedded_journal_from_bundle(self):
        frame = make_cdw1_frame(gaps.KIND_ACTIVITY, {"kind": "activity", "subtype": "kernel", "name": "k0", "valid": True, "start": 100, "end": 200, "correlation": 1})
        bundle_path = self.dir / "valid_bundle.pprof"
        with zipfile.ZipFile(bundle_path, "w") as zf:
            zf.writestr("worker-process-1-capture.journal", frame)

        extracted = gaps.extract_journal_bytes(bundle_path)
        self.assertEqual(extracted, frame)

    def test_reads_from_directory(self):
        frame = make_cdw1_frame(gaps.KIND_ACTIVITY, {"kind": "activity", "subtype": "kernel", "name": "k0", "valid": True, "start": 100, "end": 200, "correlation": 1})
        sub = self.dir / "process-42"
        sub.mkdir()
        (sub / "capture.journal").write_bytes(frame)

        extracted = gaps.extract_journal_bytes(self.dir)
        self.assertEqual(extracted, frame)

    def test_reads_from_pprof_gzip_comment(self):
        import gzip, base64
        frame = make_cdw1_frame(gaps.KIND_ACTIVITY, {"kind": "activity", "subtype": "kernel", "name": "k0", "valid": True, "start": 100, "end": 200, "correlation": 1})
        payload = {
            "manifest": {"version": 3},
            "files": {
                "worker-process-1-capture.journal": base64.b64encode(frame).decode("ascii")
            }
        }
        comment = b"#cudaprof:capture:v3\n" + json.dumps(payload).encode("utf-8")
        # create gzip stream with this comment
        gz_data = gzip.compress(comment)
        pprof_file = self.dir / "capture.pprof"
        pprof_file.write_bytes(gz_data)

        extracted = gaps.extract_journal_bytes(pprof_file)
        self.assertEqual(extracted, frame)


if __name__ == "__main__":
    unittest.main()

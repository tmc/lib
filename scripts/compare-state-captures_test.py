#!/usr/bin/env python3
"""Model-free controls; temporary fixtures are kept under ~/tmp."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
SPEC = importlib.util.spec_from_file_location("compare_states", Path(__file__).with_name("compare-state-captures.py"))
compare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compare)


class StateComparisonTest(unittest.TestCase):
    def setUp(self):
        root = Path.home() / "tmp"
        root.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="compare-states-", dir=root)
        self.addCleanup(self.temp.cleanup)
        self.before = Path(self.temp.name) / "before"
        self.after = Path(self.temp.name) / "after"
        self.before.mkdir()
        self.after.mkdir()
        for i, name in enumerate(compare.NAMES):
            logits = [0] * compare.VOCAB
            logits[7] = 1
            state = {"accepted_decode_count": i, "session_position": 10 + i, "vocab": compare.VOCAB, "argmax": 7, "logits": logits}
            data = json.dumps(state).encode()
            for directory in (self.before, self.after):
                (directory / name).write_bytes(data)

    def change(self, name="state-00.json", **values):
        path = self.after / name
        state = json.loads(path.read_bytes())
        state.update(values)
        path.write_text(json.dumps(state))

    def test_positive_and_actual_hashes(self):
        report = compare.compare(self.before, self.after)
        self.assertEqual(report["result"], "EXACT")
        self.assertEqual(len(report["files"]), 10)
        for item in report["files"]:
            data = (self.before / item["name"]).read_bytes()
            self.assertGreater(item["before_bytes"], 0)
            self.assertEqual(item["before_sha256"], hashlib.sha256(data).hexdigest())
        joined = b"".join((self.before / name).read_bytes() for name in compare.NAMES)
        self.assertEqual(report["before_concatenation_sha256"], hashlib.sha256(joined).hexdigest())
        self.assertNotEqual(report["before_concatenation_sha256"], hashlib.sha256(b"").hexdigest())

    def test_missing_file(self):
        (self.after / "state-09.json").unlink()
        with self.assertRaises(compare.CaptureError):
            compare.compare(self.before, self.after)

    def test_empty_file(self):
        (self.after / "state-00.json").write_bytes(b"")
        with self.assertRaisesRegex(compare.CaptureError, "empty"):
            compare.compare(self.before, self.after)

    def test_wrong_names(self):
        (self.after / "state-09.json").rename(self.after / "state-9.json")
        with self.assertRaises(compare.CaptureError):
            compare.compare(self.before, self.after)

    def test_extra_file(self):
        (self.after / "state-10.json").write_bytes((self.after / "state-09.json").read_bytes())
        with self.assertRaises(compare.CaptureError):
            compare.compare(self.before, self.after)

    def test_known_sidecars(self):
        for directory in (self.before, self.after):
            (directory / "states.sha256").write_text("not trusted as hash evidence\n")
            (directory / "trace.log").write_text(str(directory))
        report = compare.compare(self.before, self.after)
        self.assertEqual(report["result"], "EXACT")
        self.assertEqual(report["excluded_sidecars"], ["states.sha256", "trace.log"])

    def test_unknown_sidecar(self):
        (self.after / "unexpected.log").write_text("extra")
        with self.assertRaises(compare.CaptureError):
            compare.compare(self.before, self.after)

    def test_byte_change_with_same_values(self):
        path = self.after / "state-00.json"
        path.write_bytes(path.read_bytes() + b"\n")
        report = compare.compare(self.before, self.after)
        self.assertEqual(report["result"], "DIFFERENT")
        self.assertFalse(report["files"][0]["equal"])

    def test_float_word_change(self):
        path = self.after / "state-00.json"
        state = json.loads(path.read_bytes())
        state["logits"][0] = -0.0
        path.write_text(json.dumps(state))
        self.assertEqual(compare.compare(self.before, self.after)["result"], "DIFFERENT")

    def test_invalid_json_and_duplicate_keys(self):
        path = self.after / "state-00.json"
        original = path.read_bytes()
        for data in (b"{", b"[]", original.replace(b'"vocab": 129280', b'"vocab": 129280, "vocab": 129280')):
            with self.subTest(data=data[:20]):
                path.write_bytes(data)
                with self.assertRaises(ValueError):
                    compare.compare(self.before, self.after)

    def test_nonfinite_or_overflow_logits(self):
        path = self.after / "state-00.json"
        original = json.loads(path.read_bytes())
        for value in (float("nan"), float("inf"), 1e100, True, "0"):
            with self.subTest(value=value):
                original["logits"][0] = value
                path.write_text(json.dumps(original))
                with self.assertRaises(compare.CaptureError):
                    compare.compare(self.before, self.after)

    def test_nonfinite_leader(self):
        for directory in (self.before, self.after):
            path = directory / "state-00.json"
            state = json.loads(path.read_bytes())
            state["logits"][7] = float("inf")
            path.write_text(json.dumps(state))
        with self.assertRaisesRegex(compare.CaptureError, "non-finite"):
            compare.compare(self.before, self.after)

    def test_metadata_and_position(self):
        path = self.after / "state-00.json"
        original = path.read_bytes()
        for values in ({"accepted_decode_count": 1}, {"session_position": 9}, {"session_position": 10.0}, {"vocab": 1}, {"argmax": 0}, {"logits": []}):
            with self.subTest(values=values):
                path.write_bytes(original)
                self.change(**values)
                with self.assertRaises(compare.CaptureError):
                    compare.compare(self.before, self.after)

    def test_matching_sequential_positions_required(self):
        for i, name in enumerate(compare.NAMES):
            self.change(name, session_position=20 + i)
        with self.assertRaisesRegex(compare.CaptureError, "positions differ"):
            compare.compare(self.before, self.after)

    def test_self_directory_and_alias(self):
        with self.assertRaisesRegex(compare.CaptureError, "self-comparison"):
            compare.compare(self.before, self.before)
        alias = Path(self.temp.name) / "alias"
        alias.symlink_to(self.before, target_is_directory=True)
        with self.assertRaisesRegex(compare.CaptureError, "self-comparison"):
            compare.compare(self.before, alias)


class RetainedFixtureTest(unittest.TestCase):
    def test_retained_captures(self):
        fixture = os.environ.get("DS4_RETAINED_ROUTER_FIXTURE")
        if not fixture:
            self.skipTest("DS4_RETAINED_ROUTER_FIXTURE is unset")
        fixture = Path(fixture)
        before, after = fixture / "capture-base", fixture / "capture-candidate"
        report = compare.compare(before, after)
        self.assertEqual(report["result"], "EXACT")
        self.assertEqual(len(report["files"]), 10)
        self.assertTrue(all(item["equal"] for item in report["files"]))
        self.assertEqual(report["before_concatenation_sha256"], report["after_concatenation_sha256"])


if __name__ == "__main__":
    unittest.main()

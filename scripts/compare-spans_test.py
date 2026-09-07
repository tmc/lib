#!/usr/bin/env python3
"""Model-free controls; temporary fixtures are kept under ~/tmp."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
SPEC = importlib.util.spec_from_file_location("compare_spans", Path(__file__).with_name("compare-spans.py"))
spans = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(spans)


def record(run_id, index, name, **overrides):
    item = {
        "schema": "gputrace.host-event/v1", "kind": "interval", "run_id": run_id,
        "id": str(index), "name": name, "clock_domain": "cpu_uptime_ns",
        "timestamp_ns": 1000 + index, "duration_ns": 1,
    }
    item.update(overrides)
    return item


class SpanCountTest(unittest.TestCase):
    def setUp(self):
        root = Path.home() / "tmp"
        root.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="compare-spans-", dir=root)
        self.addCleanup(self.temp.cleanup)
        self.dir = Path(self.temp.name)

    def write(self, name, records):
        path = self.dir / name
        path.write_text("".join(json.dumps(item) + "\n" for item in records))
        return path

    def arms(self, before_syncs, after_syncs):
        before = self.write("before.jsonl", [record("base", 0, "prefill")] + [record("base", i + 1, "stream.sync") for i in range(before_syncs)])
        after = self.write("after.jsonl", [record("cand", 0, "prefill")] + [record("cand", i + 1, "stream.sync") for i in range(after_syncs)])
        return before, after

    def compare(self, before, after, dropped_before=0, dropped_after=0, expectations=()):
        return spans.compare(before, after, dropped_before, dropped_after, list(expectations))

    def test_counts_by_name_and_reports_delta(self):
        report = self.compare(*self.arms(10, 3))
        self.assertEqual(report["result"], "CHANGED")
        self.assertEqual(report["names_with_nonzero_delta"], ["stream.sync"])
        entry = next(item for item in report["spans"] if item["name"] == "stream.sync")
        self.assertEqual((entry["before"], entry["after"], entry["delta"]), (10, 3, 7))

    def test_reports_unchanged_when_no_count_moves(self):
        self.assertEqual(self.compare(*self.arms(5, 5))["result"], "UNCHANGED")

    def test_met_expectation_passes_and_unmet_fails(self):
        before, after = self.arms(10, 3)
        self.assertEqual(self.compare(before, after, expectations=[("stream.sync", 7)])["result"], "CHANGED")
        report = self.compare(before, after, expectations=[("stream.sync", 6)])
        self.assertEqual(report["result"], "UNMET")
        self.assertEqual(report["unmet_expectations"], [{"name": "stream.sync", "want": 6, "got": 7}])

    def test_expectation_on_absent_name_is_refused(self):
        before, after = self.arms(4, 2)
        with self.assertRaises(spans.SpanError):
            self.compare(before, after, expectations=[("missing.op", 0)])

    def test_dropped_spans_invalidate_the_count(self):
        before, after = self.arms(10, 3)
        with self.assertRaises(spans.SpanError):
            self.compare(before, after, dropped_before=1)
        with self.assertRaises(spans.SpanError):
            self.compare(before, after, dropped_after=1)

    def test_mixed_run_ids_within_one_file_are_refused(self):
        before = self.write("before.jsonl", [record("base", 0, "prefill"), record("other", 1, "stream.sync")])
        after = self.write("after.jsonl", [record("cand", 0, "prefill")])
        with self.assertRaises(spans.SpanError):
            self.compare(before, after)

    def test_identical_run_ids_across_arms_are_refused(self):
        before = self.write("before.jsonl", [record("same", 0, "prefill")])
        after = self.write("after.jsonl", [record("same", 0, "prefill")])
        with self.assertRaises(spans.SpanError):
            self.compare(before, after)

    def test_self_comparison_is_refused(self):
        before, _ = self.arms(4, 2)
        with self.assertRaises(spans.SpanError):
            self.compare(before, before)

    def test_duplicate_span_id_is_refused(self):
        before = self.write("before.jsonl", [record("base", 0, "prefill"), record("base", 0, "stream.sync")])
        after = self.write("after.jsonl", [record("cand", 0, "prefill")])
        with self.assertRaises(spans.SpanError):
            self.compare(before, after)

    def test_unknown_or_missing_field_is_refused(self):
        extra = record("base", 0, "prefill")
        extra["surprise"] = 1
        before = self.write("before.jsonl", [extra])
        after = self.write("after.jsonl", [record("cand", 0, "prefill")])
        with self.assertRaises(spans.SpanError):
            self.compare(before, after)
        missing = record("base", 0, "prefill")
        del missing["clock_domain"]
        before = self.write("before.jsonl", [missing])
        with self.assertRaises(spans.SpanError):
            self.compare(before, after)

    def test_wrong_schema_or_kind_is_refused(self):
        after = self.write("after.jsonl", [record("cand", 0, "prefill")])
        for override in ({"schema": "other/v1"}, {"kind": "counter"}):
            before = self.write("before.jsonl", [record("base", 0, "prefill", **override)])
            with self.assertRaises(spans.SpanError):
                self.compare(before, after)

    def test_nonpositive_duration_is_refused(self):
        before = self.write("before.jsonl", [record("base", 0, "prefill", duration_ns=0)])
        after = self.write("after.jsonl", [record("cand", 0, "prefill")])
        with self.assertRaises(spans.SpanError):
            self.compare(before, after)

    def test_empty_file_is_refused(self):
        before = self.write("before.jsonl", [])
        after = self.write("after.jsonl", [record("cand", 0, "prefill")])
        with self.assertRaises(spans.SpanError):
            self.compare(before, after)

    def test_record_limit_is_refused(self):
        before = self.write("before.jsonl", [record("base", i, "stream.sync") for i in range(spans.RECORD_LIMIT)])
        after = self.write("after.jsonl", [record("cand", 0, "prefill")])
        with self.assertRaises(spans.SpanError):
            self.compare(before, after)

    def test_expectation_parsing(self):
        self.assertEqual(spans.parse_expectation("stream.sync=574"), ("stream.sync", 574))
        for text in ("stream.sync", "=5", "stream.sync=x"):
            with self.assertRaises(spans.SpanError):
                spans.parse_expectation(text)


if __name__ == "__main__":
    unittest.main()

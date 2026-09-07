#!/usr/bin/env python3
"""Count host-event spans by name and compare two arms of one experiment."""

import argparse
import json
from pathlib import Path
import sys

SCHEMA = "gputrace.host-event/v1"
KIND = "interval"
FIELDS = {"schema", "kind", "run_id", "id", "name", "clock_domain", "timestamp_ns", "duration_ns"}
# span.go caps the recorder at these bounds and only increments a counter when
# it drops. A truncated file is indistinguishable from a quiet run by content
# alone, so reaching either bound invalidates the count.
RECORD_LIMIT = 100_000
BYTE_LIMIT = 8 << 20


class SpanError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise SpanError(message)


def decode(line, label):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f"{label}: duplicate JSON key {key}")
            result[key] = value
        return result

    def constant(value):
        raise SpanError(f"{label}: non-finite JSON constant {value}")

    return json.loads(line, object_pairs_hook=pairs, parse_constant=constant)


def read_spans(path, dropped):
    require(path.is_file(), f"{path}: not a regular file")
    require(dropped == 0, f"{path}: recorder dropped {dropped} spans; the count is truncated and cannot be compared")
    data = path.read_bytes()
    require(bool(data), f"{path}: empty span file")
    require(len(data) <= BYTE_LIMIT, f"{path}: {len(data)} bytes at or above the {BYTE_LIMIT}-byte recorder bound")
    counts, ids, run_id = {}, set(), None
    for number, raw in enumerate(data.decode("utf-8").splitlines(), start=1):
        label = f"{path}:{number}"
        require(raw.strip(), f"{label}: blank line")
        record = decode(raw, label)
        require(isinstance(record, dict), f"{label}: record must be an object")
        require(set(record) == FIELDS, f"{label}: field set must be exactly {sorted(FIELDS)}")
        require(record["schema"] == SCHEMA, f"{label}: schema must be {SCHEMA}")
        require(record["kind"] == KIND, f"{label}: kind must be {KIND}")
        if run_id is None:
            run_id = record["run_id"]
        require(record["run_id"] == run_id, f"{label}: mixes run IDs {run_id!r} and {record['run_id']!r}")
        require(record["id"] not in ids, f"{label}: duplicate span id {record['id']}")
        ids.add(record["id"])
        for key in ("timestamp_ns", "duration_ns"):
            value = record[key]
            require(type(value) is int and value > 0, f"{label}: {key} must be a positive integer")
        name = record["name"]
        require(isinstance(name, str) and name, f"{label}: span name must be a non-empty string")
        counts[name] = counts.get(name, 0) + 1
    total = sum(counts.values())
    require(total > 0, f"{path}: no spans recorded")
    require(total < RECORD_LIMIT, f"{path}: {total} spans at or above the {RECORD_LIMIT}-record recorder bound")
    return {"run_id": run_id, "span_records": total, "dropped": dropped, "counts": counts}


def parse_expectation(text):
    name, _, value = text.partition("=")
    require(name and value, f"{text}: expectation must be NAME=DELTA")
    try:
        return name, int(value)
    except ValueError:
        raise SpanError(f"{text}: delta must be an integer") from None


def compare(before, after, dropped_before, dropped_after, expectations):
    require(not before.samefile(after), "self-comparison is not evidence: paths are identical")
    old = read_spans(before, dropped_before)
    new = read_spans(after, dropped_after)
    require(old["run_id"] != new["run_id"], f"both arms carry run ID {old['run_id']!r}; arms must be distinguishable")
    names = sorted(set(old["counts"]) | set(new["counts"]))
    spans = [{
        "name": name,
        "before": old["counts"].get(name, 0),
        "after": new["counts"].get(name, 0),
        "delta": old["counts"].get(name, 0) - new["counts"].get(name, 0),
    } for name in names]
    unmet = []
    for name, want in expectations:
        got = next((item["delta"] for item in spans if item["name"] == name), None)
        require(got is not None, f"{name}: expected a delta but the name appears in neither arm")
        if got != want:
            unmet.append({"name": name, "want": want, "got": got})
    moved = [item["name"] for item in spans if item["delta"] != 0]
    return {
        "result": "UNMET" if unmet else ("CHANGED" if moved else "UNCHANGED"),
        "before_run_id": old["run_id"], "after_run_id": new["run_id"],
        "before_span_records": old["span_records"], "after_span_records": new["span_records"],
        "before_dropped": dropped_before, "after_dropped": dropped_after,
        "spans": spans, "names_with_nonzero_delta": moved,
        "unmet_expectations": unmet,
        "scope": "host-event span record counts for two arms of one experiment",
        "limitation": "counts host waits and encoded operations only; establishes neither numerical equivalence nor elapsed-time attribution",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--dropped-before", type=int, required=True, help="spans dropped by the before arm; the run log reports a nonzero count")
    parser.add_argument("--dropped-after", type=int, required=True, help="spans dropped by the after arm")
    parser.add_argument("--expect", action="append", default=[], metavar="NAME=DELTA", help="pre-registered before-minus-after delta; repeatable")
    args = parser.parse_args()
    try:
        expectations = [parse_expectation(text) for text in args.expect]
        report = compare(args.before, args.after, args.dropped_before, args.dropped_after, expectations)
        print(json.dumps(report, indent=2))
        return 1 if report["result"] == "UNMET" else 0
    except (OSError, ValueError, TypeError) as err:
        print(f"compare-spans: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Validate observed prefill progress; never infer throughput or a stall."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

SCHEMA = "ds4-prefill-progress/v1"
FIELDS = {"schema", "completed_tokens", "elapsed_ns", "total_tokens"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def check(data):
    require(bool(data), "empty progress input")
    require(data.endswith(b"\n"), "incomplete final record")
    lines = data.splitlines()
    require(1 <= len(lines) <= 17, "invalid record count")
    records = []
    for index, line in enumerate(lines):
        record = json.loads(line, object_pairs_hook=pairs)
        require(type(record) is dict and set(record) == FIELDS, "invalid fields")
        require(record["schema"] == SCHEMA, "unregistered schema")
        for key in ("completed_tokens", "elapsed_ns", "total_tokens"):
            require(type(record[key]) is int, "non-integer " + key)
        require(record["total_tokens"] == 2048, "unregistered total")
        require(record["completed_tokens"] == index * 128, "missing or reordered token step")
        elapsed = record["elapsed_ns"]
        require(0 <= elapsed <= 2**63 - 1, "elapsed outside int64 range")
        if index == 0:
            require(elapsed == 0, "initial elapsed is not zero")
        else:
            require(elapsed > records[-1]["elapsed_ns"], "elapsed did not increase")
        records.append(record)
    last = records[-1]
    return {
        "result": "PREFILL_REACHED" if last["completed_tokens"] == 2048 else "OBSERVED_PREFIX",
        "records": len(records),
        "completed_tokens": last["completed_tokens"],
        "elapsed_ns": last["elapsed_ns"],
        "input_sha256": hashlib.sha256(data).hexdigest(),
        "intervals": [
            {"from_completed_tokens": a["completed_tokens"],
             "to_completed_tokens": b["completed_tokens"],
             "elapsed_ns": b["elapsed_ns"] - a["elapsed_ns"]}
            for a, b in zip(records, records[1:])
        ],
        "scope": "validated observations only; no workload completion, execution provenance, throughput, stall, or extrapolation claim",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("progress", type=Path)
    args = parser.parse_args()
    try:
        result = check(args.progress.read_bytes())
    except (ValueError, OSError, TypeError, OverflowError) as error:
        print("invalid prefill progress: " + str(error), file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

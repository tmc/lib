#!/usr/bin/env python3
"""Validate and compare the ten retained state-00..09.json capture files."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

VOCAB = 129280
NAMES = tuple(f"state-{i:02d}.json" for i in range(10))
SIDECARS = {"states.sha256", "trace.log"}


class CaptureError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise CaptureError(message)


def integer(value):
    return type(value) is int


def digest(data):
    require(bool(data), "refusing to hash an empty capture")
    return hashlib.sha256(data).hexdigest()


def decode(data, label):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f"{label}: duplicate JSON key {key}")
            result[key] = value
        return result

    def constant(value):
        raise CaptureError(f"{label}: non-finite JSON constant {value}")

    return json.loads(data, object_pairs_hook=pairs, parse_constant=constant)


def read_capture(directory):
    require(directory.is_dir(), f"{directory}: not a directory")
    names = {path.name for path in directory.iterdir()}
    require(set(NAMES) <= names and names <= set(NAMES) | SIDECARS, f"{directory}: expected state-00..09.json and optional states.sha256/trace.log; missing={sorted(set(NAMES) - names)}, extra={sorted(names - set(NAMES) - SIDECARS)}")
    payloads, positions, hashes = {}, [], {}
    for accepted, name in enumerate(NAMES):
        path = directory / name
        require(path.is_file(), f"{path}: not a regular file")
        data = path.read_bytes()
        require(bool(data), f"{path}: empty capture")
        state = decode(data, path)
        require(isinstance(state, dict), f"{path}: state must be an object")
        count = state.get("accepted_decode_count")
        require(integer(count) and count == accepted, f"{path}: accepted index must be {accepted}")
        position = state.get("session_position")
        require(integer(position) and position >= 0, f"{path}: invalid session position")
        if positions:
            require(position == positions[-1] + 1, f"{path}: session positions are not sequential")
        positions.append(position)
        require(integer(state.get("vocab")) and state["vocab"] == VOCAB, f"{path}: vocabulary must be {VOCAB}")
        logits = state.get("logits")
        require(isinstance(logits, list) and len(logits) == VOCAB, f"{path}: expected {VOCAB} logits")
        rounded = []
        for i, value in enumerate(logits):
            require(type(value) in (int, float), f"{path}: logit {i} is not numeric")
            try:
                value = float(value)
                require(math.isfinite(value), f"{path}: non-finite logit {i}")
                value = struct.unpack("<f", struct.pack("<f", value))[0]
            except (OverflowError, struct.error):
                raise CaptureError(f"{path}: logit {i} overflows float32") from None
            require(math.isfinite(value), f"{path}: non-finite float32 logit {i}")
            rounded.append(value)
        argmax = state.get("argmax")
        require(integer(argmax) and 0 <= argmax < VOCAB, f"{path}: invalid argmax index")
        expected = max(range(VOCAB), key=rounded.__getitem__)
        require(argmax == expected, f"{path}: argmax {argmax}, expected first maximum {expected}")
        payloads[name] = data
        hashes[name] = digest(data)
    joined = b"".join(payloads[name] for name in NAMES)
    return payloads, positions, hashes, digest(joined)


def compare(before, after):
    before, after = Path(before), Path(after)
    require(not before.samefile(after), "self-comparison is not evidence: directories are identical")
    old, old_positions, old_hashes, old_digest = read_capture(before)
    new, new_positions, new_hashes, new_digest = read_capture(after)
    require(old_positions == new_positions, "session positions differ between captures")
    files = []
    for name in NAMES:
        files.append({
            "name": name, "equal": old[name] == new[name],
            "before_bytes": len(old[name]), "after_bytes": len(new[name]),
            "before_sha256": old_hashes[name], "after_sha256": new_hashes[name],
        })
    return {
        "result": "EXACT" if all(item["equal"] for item in files) else "DIFFERENT",
        "files": files, "session_positions": old_positions,
        "concatenation_order": list(NAMES),
        "before_concatenation_sha256": old_digest,
        "after_concatenation_sha256": new_digest,
        "scope": "validated whole-file artifact equality for ten retained states",
        "excluded_sidecars": sorted(SIDECARS),
        "limitation": "does not establish source or executable provenance, model identity, throughput, or oracle parity",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    args = parser.parse_args()
    try:
        report = compare(args.before, args.after)
        print(json.dumps(report, indent=2))
        return 0 if report["result"] == "EXACT" else 1
    except (OSError, ValueError, TypeError) as err:
        print(f"compare-state-captures: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

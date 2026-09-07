#!/usr/bin/env python3
"""Compare retained prefill captures without loading a model or GPU runtime."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys

VOCAB = 129280
CASES = (2, 64, 2048)
ALLOWED_CASES = (2, 64, 65, 66, 2048)
LIFECYCLE = "one runtime; each case: reset, warm batch and fixed continuation, reset, captured batch and same continuation; close before COMPLETE"
COMPLETE = b"runtime closed; all selected captures written\n"
IDENTITY_FIELDS = (
    "schema", "backend", "goos", "goarch", "context", "cases",
    "continuation_token", "tokens_sha256", "lifecycle", "model_sha256", "model_bytes",
)


class CaptureError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise CaptureError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def integer(value):
    return type(value) is int


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f"{path}: duplicate JSON key {key}")
            result[key] = value
        return result

    return json.loads(path.read_bytes(), object_pairs_hook=pairs)


def read_capture(path, cases):
    require((path / "COMPLETE").read_bytes() == COMPLETE, f"{path}: invalid COMPLETE marker")
    identity = read_json(path / "identity.json")
    require(isinstance(identity, dict), f"{path}: identity must be an object")
    for key in IDENTITY_FIELDS + ("executable_sha256",):
        require(key in identity, f"{path}: missing {key}")
    require(integer(identity["schema"]) and identity["schema"] == 1, f"{path}: unsupported schema")
    require(identity["backend"] in ("metal", "cuda"), f"{path}: invalid backend")
    for key in ("goos", "goarch"):
        require(isinstance(identity[key], str) and identity[key], f"{path}: invalid {key}")
    require(integer(identity["context"]) and identity["context"] == 2049, f"{path}: invalid context")
    require(identity["cases"] == list(cases) and all(integer(n) for n in identity["cases"]), f"{path}: cases do not match requested order {list(cases)}")
    require(integer(identity["continuation_token"]) and 0 <= identity["continuation_token"] < VOCAB, f"{path}: invalid continuation token")
    require(identity["lifecycle"] == LIFECYCLE, f"{path}: unsupported lifecycle")
    require(integer(identity["model_bytes"]) and identity["model_bytes"] > 0, f"{path}: invalid model size")
    for key in ("tokens_sha256", "model_sha256", "executable_sha256"):
        require(sha(identity[key]), f"{path}: invalid {key}")
    tokens_data = (path / "tokens.json").read_bytes()
    require(digest(tokens_data) == identity["tokens_sha256"], f"{path}: tokens digest mismatch")
    tokens = json.loads(tokens_data)
    require(isinstance(tokens, list) and len(tokens) >= 2048 and all(integer(n) and 0 <= n < VOCAB for n in tokens), f"{path}: invalid token corpus")
    captures = {}
    for n in cases:
        for stage in ("prefill", "continuation"):
            name = f"n{n}-{stage}"
            metadata = read_json(path / (name + ".json"))
            require(isinstance(metadata, dict), f"{path}/{name}: invalid metadata")
            require(integer(metadata.get("vocab")) and metadata["vocab"] == VOCAB, f"{path}/{name}: invalid vocabulary")
            require(integer(metadata.get("bytes")) and metadata["bytes"] == 4 * VOCAB, f"{path}/{name}: invalid byte count")
            data = (path / (name + ".f32le")).read_bytes()
            require(len(data) == 4 * VOCAB, f"{path}/{name}: invalid payload size")
            require(sha(metadata.get("sha256")) and digest(data) == metadata["sha256"], f"{path}/{name}: payload digest mismatch")
            require(all(math.isfinite(value) for (value,) in struct.iter_unpack("<f", data)), f"{path}/{name}: non-finite logit")
            captures[name] = data
    return identity, captures


def compare(before, after, cases=CASES):
    before, after = Path(before), Path(after)
    require(not before.samefile(after), "self-comparison is not evidence: directories are identical")
    require(cases and len(set(cases)) == len(cases) and all(integer(n) and n in ALLOWED_CASES for n in cases), "cases must be distinct values from 2,64,65,66,2048")
    old, old_data = read_capture(before, cases)
    new, new_data = read_capture(after, cases)
    for key in IDENTITY_FIELDS:
        require(old[key] == new[key], f"identity mismatch: {key}")
    for name, data in old_data.items():
        require(data == new_data[name], f"bit mismatch: {name}")
    full = set(cases) == set(CASES)
    return {
        "result": "EXACT", "scope": "full requested capture suite" if full else "SCOPED capture comparison",
        "cases": list(cases), "backend": old["backend"], "vectors": len(old_data),
        "before_executable_sha256": old["executable_sha256"],
        "after_executable_sha256": new["executable_sha256"],
        "model_sha256": old["model_sha256"],
        "limitation": "capture bit-identity only; no throughput, cross-backend, or full parity claim",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--cases", default="2,64,2048", help="required capture cases in order (default: 2,64,2048)")
    args = parser.parse_args()
    try:
        cases = tuple(int(n) for n in args.cases.split(","))
        print(json.dumps(compare(args.before, args.after, cases), indent=2))
    except (OSError, ValueError, TypeError) as err:
        print(f"compare-prefill: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

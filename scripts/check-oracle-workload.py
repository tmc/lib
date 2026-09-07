#!/usr/bin/env python3
"""Check retained workload artifacts; does not establish execution or qualification."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys

PREFIX_SHA256 = "5fdf3b77e31c5cd253345ba1f8c4100e9f0b55eff30a1a5ed3029fb524c80380"
MODEL_SHA256 = "ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0"
COMPLETE = b"runtime closed; 2048 prefill and 128 non-EOS decodes completed\n"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError("non-finite JSON constant: " + value)


def read_json(path):
    return json.loads(path.read_bytes(), object_pairs_hook=pairs,
                      parse_constant=reject_constant)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def token(value):
    return type(value) is int and 0 <= value < 129280


def sha256(value):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


def check(directory, backend, host, executable_sha256, prompt):
    require(backend in ("cuda", "metal"), "invalid backend")
    require(bool(host), "empty expected host")
    require(sha256(executable_sha256), "invalid expected executable digest")
    require((directory / "COMPLETE").read_bytes() == COMPLETE, "invalid COMPLETE marker")
    identity = read_json(directory / "identity.json")
    result = read_json(directory / "result.json")
    require(isinstance(identity, dict) and isinstance(result, dict), "identity/result must be objects")
    expected = {
        "backend": backend, "host": host, "model_sha256": MODEL_SHA256,
        "executable_sha256": executable_sha256, "context": 65665,
        "token_ids_encoding": "little-endian uint32", "token_ids_count": 2048,
        "token_ids_sha256": PREFIX_SHA256, "eos": 1,
        "prompt_sha256": digest(prompt.read_bytes()),
    }
    for key, value in expected.items():
        require(type(identity.get(key)) is type(value) and identity[key] == value,
                "identity mismatch: " + key)
    require(identity.get("goos") == {"cuda": "linux", "metal": "darwin"}[backend], "backend/GOOS mismatch")
    require(identity.get("goarch") == "arm64", "expected arm64 host")
    for key in ("input_contract", "eos_provenance", "tokenizer_discrepancy_receipt", "lifecycle", "scope"):
        require(isinstance(identity.get(key), str) and bool(identity[key].strip()), "missing identity: " + key)
    corpus = read_json(directory / "tokens.json")
    require(isinstance(corpus, list) and len(corpus) >= 2048 and all(map(token, corpus)), "invalid token corpus")
    require(digest((directory / "tokens.json").read_bytes()) == identity.get("tokens_sha256"), "token file digest mismatch")
    prefix = struct.pack("<2048I", *corpus[:2048])
    require(prefix == (directory / "token-prefix.u32le").read_bytes(), "binary token prefix mismatch")
    require(digest(prefix) == PREFIX_SHA256, "unregistered token prefix")
    discrepancy = (directory / "tokenizer-discrepancy.json").read_bytes()
    require(bool(discrepancy) and digest(discrepancy) == identity.get("tokenizer_discrepancy_sha256"), "discrepancy digest mismatch")
    require(type(result.get("prefill_tokens")) is int and result["prefill_tokens"] == 2048, "wrong prefill count")
    generated = result.get("generated_tokens")
    require(isinstance(generated, list) and len(generated) == 128 and all(token(t) and t != 1 for t in generated), "invalid generated tokens")
    for key in ("prefill_seconds", "generation_seconds", "first_seconds", "steady_seconds"):
        value = result.get(key)
        require(type(value) in (int, float) and math.isfinite(value) and value > 0, "invalid duration: " + key)
    require(result["generation_seconds"] >= result["first_seconds"] + result["steady_seconds"], "generation window excludes evaluation time")
    for key in ("prefill_logits_sha256", "final_logits_sha256"):
        require(sha256(result.get(key)), "invalid logit digest: " + key)
    rates = (2048 / result["prefill_seconds"], 128 / result["generation_seconds"], 127 / result["steady_seconds"])
    require(all(math.isfinite(rate) and rate > 0 for rate in rates), "invalid derived rate")
    return {
        "result": "VALID_ARTIFACTS", "backend": backend, "host": host,
        "token_ids_sha256": PREFIX_SHA256,
        "prefill_tokens_per_second": 2048 / result["prefill_seconds"],
        "generation_tokens_per_second_including_selection": 128 / result["generation_seconds"],
        "steady_tokens_per_second_excluding_selection": 127 / result["steady_seconds"],
        "scope": "artifact consistency only; require separate actual-exit/test-reach, source binding, registered correctness and host qualification evidence",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--backend", required=True, choices=("cuda", "metal"))
    parser.add_argument("--host", required=True)
    parser.add_argument("--executable-sha256", required=True)
    parser.add_argument("--prompt", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = check(args.directory, args.backend, args.host, args.executable_sha256, args.prompt)
    except (ValueError, OSError, TypeError, OverflowError) as error:
        print("invalid workload capture: " + str(error), file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

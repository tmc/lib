#!/usr/bin/env python3
"""No-GPU mutation tests for output_agreement.py."""

import struct
import json
import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path


SCRIPT = Path(__file__).with_name("output_agreement.py")


def write(path, values, token_hash="history"):
    raw = struct.pack("<%df" % len(values), *values)
    path.write_bytes(raw)
    path.with_suffix(".json").write_text(json.dumps({
        "schema": 1, "phase": "prefill", "position": 2,
        "vocab": len(values), "bytes": len(values) * 4,
        "logits_sha256": hashlib.sha256(raw).hexdigest(), "token_ids_sha256": token_hash,
    }) + "\n")


def run(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def main():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        c = root / "c.f32le"
        same = root / "same.f32le"
        swapped = root / "swapped.f32le"
        scaled = root / "scaled.f32le"
        nan = root / "nan.f32le"
        write(c, [1, 5, 2, 4])
        write(same, [1, 5, 2, 4])
        write(swapped, [1, 5, 4, 2])
        write(scaled, [1, 5, 2, 40])
        write(nan, [1, 5, float("nan"), 4])
        common = ["--vocab", "4", "--gate", "fixed", "--min-top1", "1", "--max-kl", "0"]
        cases = [(same, 0), (swapped, 2), (scaled, 2), (nan, 1)]
        for target, want in cases:
            got = run(c, target, *common).returncode
            if got != want:
                raise SystemExit(f"{target.name}: got rc {got}, want {want}")
        no_gate = run(c, same, "--vocab", "4")
        if no_gate.returncode == 0:
            raise SystemExit("missing limits returned success")
        bad_sidecar = root / "bad.f32le"
        write(bad_sidecar, [1, 5, 2, 4], token_hash="different-history")
        if run(c, bad_sidecar, *common).returncode == 0:
            raise SystemExit("sidecar history mutation returned success")
        control = root / "control.f32le"
        write(control, [1, 5, 2, 3], token_hash="history")
        relative = run(c, same, "--vocab", "4", "--gate", "relative",
                       "--control-c", c, control, "--relative-kl-factor", "2")
        if relative.returncode != 0:
            raise SystemExit(f"relative gate rejected its control: {relative.stderr.decode()}")
        zero = run(c, same, "--vocab", "4", "--gate", "relative",
                   "--control-c", c, c, "--relative-kl-factor", "2")
        if zero.returncode == 0:
            raise SystemExit("relative gate accepted a zero C-vs-C control")
        missing_gate = run(c, same, "--vocab", "4", "--min-top1", "1", "--max-kl", "0")
        if missing_gate.returncode == 0:
            raise SystemExit("fixed limits without explicit gate returned success")
    print("output agreement mutation tests passed")


if __name__ == "__main__":
    main()

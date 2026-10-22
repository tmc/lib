#!/usr/bin/env python3
"""Compare retained C and Go logits without hiding an oracle mismatch.

The tool deliberately does not choose an acceptance bound from the C-vs-Go
result.  First run it with --control-c and preregister the resulting bound;
then run it with --max-kl and --min-top1 for the C-vs-Go gate.
"""

import argparse
import hashlib
import json
import math
import struct
import sys
from pathlib import Path

# accepted_decode_count is the number of generated tokens already consumed.
# The seventh generated token is therefore state 6 (the first token is state 0).
TOKEN7_STATE = 6


def read_f32(path):
    data = Path(path).read_bytes()
    if len(data) % 4:
        raise ValueError(f"{path}: {len(data)} bytes is not float32-aligned")
    return list(struct.unpack("<%df" % (len(data) // 4), data))


def read_dump(path, phase):
    raw_bytes = Path(path).read_bytes()
    raw = read_f32(path)
    meta_path = Path(path).with_suffix(".json")
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{meta_path}: invalid dump sidecar: {exc}")
    if meta.get("phase") != phase:
        raise ValueError(f"{path}: sidecar phase {meta.get('phase')!r}, want {phase!r}")
    if meta.get("vocab") <= 0 or meta.get("bytes") != len(raw) * 4:
        raise ValueError(f"{path}: sidecar size/vocab does not match raw dump")
    digest = meta.get("logits_sha256")
    if not digest:
        raise ValueError(f"{meta_path}: missing logits_sha256")
    if digest != hashlib.sha256(raw_bytes).hexdigest():
        raise ValueError(f"{path}: logits_sha256 does not match raw dump")
    return raw, meta


def validate_pair(left_path, right_path, left, right, left_meta, right_meta):
    for key in ("token_ids_sha256", "position", "vocab", "phase"):
        if left_meta.get(key) != right_meta.get(key):
            raise ValueError(f"dump sidecars differ in {key}: {left_meta.get(key)!r} vs {right_meta.get(key)!r}")
    if left_meta["vocab"] != len(left) or right_meta["vocab"] != len(right):
        raise ValueError("dump sidecar vocab does not match raw vector")


def finite(values, name):
    if not all(math.isfinite(v) for v in values):
        raise ValueError(f"{name}: non-finite logit")


def top1(values):
    return max(range(len(values)), key=values.__getitem__)


def softmax(values):
    peak = max(values)
    weights = [math.exp(v - peak) for v in values]
    total = math.fsum(weights)
    return [w / total for w in weights]


def kl_divergence(p_values, q_values):
    p, q = softmax(p_values), softmax(q_values)
    return math.fsum(x * math.log(x / y) for x, y in zip(p, q) if x > 0)


def rows(values, width):
    if len(values) % width:
        raise ValueError(f"logit count {len(values)} is not divisible by width {width}")
    return [values[i:i + width] for i in range(0, len(values), width)]


def report(left, right, width):
    if len(left) != len(right):
        raise ValueError(f"logit counts differ: {len(left)} vs {len(right)}")
    finite(left, "left")
    finite(right, "right")
    lrows, rrows = rows(left, width), rows(right, width)
    ltop = [top1(row) for row in lrows]
    rtop = [top1(row) for row in rrows]
    same = sum(a == b for a, b in zip(ltop, rtop))
    return {
        "positions": len(lrows),
        "vocab": width,
        "top1_agreement": same / len(lrows),
        "top1_matches": same,
        "max_abs_logit": max(abs(a - b) for a, b in zip(left, right)),
        "bit_identical": left == right,
        "final_kl_c_to_go": kl_divergence(lrows[-1], rrows[-1]),
        "left_top1": ltop,
        "right_top1": rtop,
    }


def load_state_dir(path):
    states = {}
    token_hashes = set()
    for item in sorted(Path(path).glob("state-*.json")):
        value = json.loads(item.read_text())
        accepted = int(value["accepted_decode_count"])
        logits = [float(v) for v in value["logits"]]
        finite(logits, str(item))
        states[accepted] = logits
        token_hash = value.get("forced_tokens_sha256")
        if not token_hash:
            raise ValueError(f"{item}: missing forced_tokens_sha256")
        token_hashes.add(token_hash)
    if not states:
        raise ValueError(f"{path}: no state-*.json files")
    if len(token_hashes) != 1:
        raise ValueError(f"{path}: state files disagree on forced token hash")
    expected_steps = set(range(len(states)))
    if set(states) != expected_steps:
        raise ValueError(f"{path}: teacher steps are not contiguous from zero")
    return states, token_hashes.pop()


def top2_margin(values):
    first, second = sorted(values, reverse=True)[:2]
    return first - second


def teacher_forced(left_dir, right_dir):
    left, left_hash = load_state_dir(left_dir)
    right, right_hash = load_state_dir(right_dir)
    if left_hash != right_hash:
        raise ValueError("teacher-forced token hashes differ")
    steps = sorted(set(left) | set(right))
    if set(left) != set(right):
        raise ValueError("teacher-forced state steps differ")
    out = []
    for step in steps:
        if len(left[step]) != len(right[step]):
            raise ValueError(f"step {step}: vocabulary differs")
        l, r = left[step], right[step]
        distance = max(abs(a - b) for a, b in zip(l, r))
        margin = top2_margin(l)
        out.append({
            "step": step,
            "top1_left": top1(l),
            "top1_right": top1(r),
            "top1_agree": top1(l) == top1(r),
            "left_top2_margin": top2_margin(l),
            "right_top2_margin": top2_margin(r),
            "max_abs_logit": distance,
            "kl_left_to_right": kl_divergence(l, r),
            "token7_interpretation": ("near_tie" if margin < distance else "mismatch") if step == TOKEN7_STATE else None,
        })
    return {"forced_tokens_sha256": left_hash, "steps": out}


def main(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("left", help="left/C little-endian float32 logits")
    p.add_argument("right", help="right/Go little-endian float32 logits")
    p.add_argument("--vocab", type=int, required=True)
    p.add_argument("--control-c", nargs=2, metavar=("C1", "C2"),
                   help="C-vs-C control dumps; reports its observed bound")
    p.add_argument("--gate", choices=("fixed", "relative"),
                   help="acceptance rule: preregistered fixed limits or limits relative to --control-c")
    p.add_argument("--min-top1", type=float,
                   help="required top-1 agreement fraction")
    p.add_argument("--max-kl", type=float,
                   help="required final-position KL(C||Go)")
    p.add_argument("--relative-kl-factor", type=float,
                   help="relative gate: candidate KL may be at most this multiple of C-vs-C KL")
    p.add_argument("--relative-top1-slack", type=float, default=0.0,
                   help="relative gate: allowed top-1 fraction below C-vs-C (default: 0)")
    p.add_argument("--teacher-left", help="teacher-forced state directory")
    p.add_argument("--teacher-right", help="teacher-forced state directory")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)
    if (args.min_top1 is None) != (args.max_kl is None):
        p.error("--min-top1 and --max-kl must be supplied together")
    if args.gate == "fixed" and args.min_top1 is None:
        p.error("fixed gate requires --min-top1 and --max-kl")
    if args.gate == "relative":
        if not args.control_c or args.relative_kl_factor is None:
            p.error("relative gate requires --control-c and --relative-kl-factor")
        if args.min_top1 is not None:
            p.error("relative gate does not use --min-top1 or --max-kl")
        if args.relative_kl_factor < 0 or args.relative_top1_slack < 0:
            p.error("relative gate factors must be non-negative")
    elif args.gate is None and (args.min_top1 is not None or args.max_kl is not None):
        p.error("--gate fixed is required when fixed limits are supplied")
    phase = "final" if ".final." in args.left or ".final." in args.right else "prefill"
    left, left_meta = read_dump(args.left, phase)
    right, right_meta = read_dump(args.right, phase)
    validate_pair(args.left, args.right, left, right, left_meta, right_meta)
    if args.vocab != left_meta["vocab"]:
        p.error(f"--vocab {args.vocab} disagrees with dump sidecar {left_meta['vocab']}")
    result = {"comparison": report(left, right, args.vocab), "provenance": {
        "phase": phase, "position": left_meta["position"],
        "token_ids_sha256": left_meta["token_ids_sha256"],
    }}
    if args.control_c:
        c1, c1_meta = read_dump(args.control_c[0], phase)
        c2, c2_meta = read_dump(args.control_c[1], phase)
        validate_pair(args.control_c[0], args.control_c[1], c1, c2, c1_meta, c2_meta)
        control = report(c1, c2, args.vocab)
        control["control_zero"] = control["bit_identical"]
        result["c_vs_c_control"] = control
    if args.teacher_left or args.teacher_right:
        if not args.teacher_left or not args.teacher_right:
            p.error("--teacher-left and --teacher-right must be supplied together")
        result["teacher_forced"] = teacher_forced(args.teacher_left, args.teacher_right)
    comparison = result["comparison"]
    if args.gate == "fixed":
        result["verdict"] = "PASS" if (comparison["top1_agreement"] >= args.min_top1 and
                                         comparison["final_kl_c_to_go"] <= args.max_kl) else "FAIL"
        result["gate"] = {"mode": "fixed", "min_top1": args.min_top1, "max_kl": args.max_kl}
    elif args.gate == "relative":
        control = result.get("c_vs_c_control")
        if control is None or control["control_zero"]:
            raise ValueError("relative gate requires a nonzero C-vs-C control")
        kl_limit = control["final_kl_c_to_go"] * args.relative_kl_factor
        top1_limit = control["top1_agreement"] - args.relative_top1_slack
        result["gate"] = {"mode": "relative", "control_top1": control["top1_agreement"],
                          "control_kl": control["final_kl_c_to_go"],
                          "top1_floor": top1_limit, "max_kl": kl_limit,
                          "kl_factor": args.relative_kl_factor,
                          "top1_slack": args.relative_top1_slack}
        result["verdict"] = "PASS" if (comparison["top1_agreement"] >= top1_limit and
                                         comparison["final_kl_c_to_go"] <= kl_limit) else "FAIL"
    else:
        result["verdict"] = "NO_VERDICT"
    print(json.dumps(result, indent=2, sort_keys=True) if args.json else json.dumps(result, indent=2))
    if args.gate is None:
        # A control-only report is evidence for preregistration, not a gate.
        return 3
    return 2 if result.get("verdict") == "FAIL" else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except (OSError, ValueError, struct.error) as exc:
        print(f"output-agreement: {exc}", file=sys.stderr)
        raise SystemExit(1)

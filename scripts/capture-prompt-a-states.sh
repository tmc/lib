#!/bin/sh
# capture-prompt-a-states.sh runs the prompt-A teacher-forced trace and
# succeeds only when it wrote a complete, well-formed set of ten logit states.
#
# TestMetalOracleTrace is expected-red: this port disagrees with the oracle at
# step 5, a pre-existing divergence. Its exit code therefore cannot gate a
# capture. But the capture is what a P1-vs-revert identity comparison needs,
# and a defective capture silently compares something other than what it
# claims to. So the gate here is the artifact, not the verdict.
#
# Counting files is not enough: ten unparseable files, ten one-logit vectors,
# NaN logits, a duplicated accepted_decode_count, or states numbered 10..19
# all satisfy a count of ten while making the comparison meaningless. Each
# check below exists because that shape passed an earlier version.
set -eu
[ $# -ge 3 ] || { echo "usage: capture-prompt-a-states.sh binary outdir oracledir" >&2; exit 2; }
bin=$1 out=$2 oracle=$3
[ -x "$bin" ] || { echo "capture: not executable: $bin" >&2; exit 1; }
[ -d "$oracle" ] || { echo "capture: no oracle dir: $oracle" >&2; exit 1; }
if [ -e "$out" ] && [ -n "$(ls -A "$out" 2>/dev/null)" ]; then
	echo "capture: output directory is not empty: $out" >&2; exit 1
fi
mkdir -p "$out"
DS4_ORACLE_STATE_OUT=$out DS4_ORACLE_STATE_DIR=$oracle \
	"$bin" -test.run TestMetalOracleTrace -test.v -test.timeout 40m \
	> "$out/trace.log" 2>&1 ||
	echo "capture: trace exited nonzero (expected: step-5 divergence)" >&2

CAPTURE_DIR=$out python3 - <<'PY' || exit 1
import json, math, os, sys

out = os.environ["CAPTURE_DIR"]
want = [f"state-{i:02d}.json" for i in range(10)]

def die(msg):
    print(f"capture: {msg}", file=sys.stderr)
    raise SystemExit(1)

got = sorted(f for f in os.listdir(out) if f.startswith("state-") and f.endswith(".json"))
if got != want:
    die(f"state files are {got or 'none'}, want state-00.json..state-09.json")

prev = None
for i, name in enumerate(want):
    try:
        with open(os.path.join(out, name)) as fh:
            d = json.load(fh)
    except (ValueError, OSError) as e:
        die(f"{name}: not readable JSON: {e}")
    for key in ("accepted_decode_count", "session_position", "vocab", "logits"):
        if key not in d:
            die(f"{name}: missing {key}")
    # The filename is what the comparison joins on, so a state whose recorded
    # count disagrees with its name would silently pair up mismatched steps.
    if d["accepted_decode_count"] != i:
        die(f"{name}: accepted_decode_count {d['accepted_decode_count']}, want {i}")
    if d["vocab"] != 129280:
        die(f"{name}: vocab {d['vocab']}, want 129280")
    logits = d["logits"]
    if not isinstance(logits, list) or len(logits) != 129280:
        n = len(logits) if isinstance(logits, list) else type(logits).__name__
        die(f"{name}: {n} logits, want 129280")
    for j, v in enumerate(logits):
        if not isinstance(v, (int, float)) or not math.isfinite(v):
            die(f"{name}: logit {j} is {v!r}, want finite")
    # Teacher forcing accepts exactly one token per step, so positions must
    # advance by one; a repeat or a gap means the trace did not run straight.
    pos = d["session_position"]
    if prev is not None and pos != prev + 1:
        die(f"{name}: session_position {pos} follows {prev}, want {prev + 1}")
    prev = pos

print("capture: 10 states validated")
PY

shasum -a 256 "$out"/state-*.json > "$out/states.sha256"

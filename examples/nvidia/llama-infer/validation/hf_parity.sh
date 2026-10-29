#!/bin/bash
# Run one bounded reference or engine phase under an existing GPU lock.
# Usage: hf_parity.sh CHECKPOINT OUT reference
#        hf_parity.sh CHECKPOINT OUT {egraph|bf16|f32} [CHUNK]
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
if [ $# -lt 3 ] || [ $# -gt 4 ]; then
	echo "usage: hf_parity.sh CHECKPOINT OUT {reference|egraph|bf16|f32} [CHUNK]" >&2
	exit 2
fi
ckpt=$(cd "$1" && pwd)
mkdir -p "$2"
out=$(cd "$2" && pwd)
phase=$3
case "$phase" in reference|egraph|bf16|f32) ;; *) echo "unknown phase: $phase" >&2; exit 2 ;; esac
here=$(cd "$(dirname "$0")" && pwd)
pkg=$(dirname "$here")
command -v timeout >/dev/null || { echo "GNU timeout is required" >&2; exit 2; }
# Require a real, bounded ancestor lock, not just an inherited label.
"${PYTHON:-python3}" - <<'PY'
import os, pathlib, re
owner = pathlib.Path.home() / 'ml-explore/.gpu-lock/lock/owner'
text = owner.read_text()
pid = int(re.search(r'(?:^|\s)pid=(\d+)', text).group(1))
limit = int(re.search(r'(?:^|\s)limit=(\d+)', text).group(1))
if limit != 1800:
    raise SystemExit('this phase requires a fresh GPU lock with limit 1800 seconds')
current = os.getpid()
while current > 1 and current != pid:
    stat = (pathlib.Path('/proc') / str(current) / 'stat').read_text()
    current = int(stat.rsplit(')', 1)[1].split()[1])
if current != pid:
    raise SystemExit('run this phase under gpu-lock; another process owns the lock')
PY
if [ "$phase" = reference ]; then
	[ $# -eq 3 ] || { echo "reference takes no CHUNK argument" >&2; exit 2; }
	stage="$out/reference"
else
	graph=${LLAMA_HF_PARITY_GRAPH:-1}
	case "$graph" in 0|1) ;; *) echo "LLAMA_HF_PARITY_GRAPH must be 0 or 1" >&2; exit 2 ;; esac
	if [ "$phase" = egraph ]; then
		chunk=${4:-all}
		case "$chunk" in all|0|1|4|8|40) ;; *) echo "CHUNK must be 0, 1, 4, 8, 40 or all" >&2; exit 2 ;; esac
	else
		[ $# -eq 3 ] || [ "$4" = 0 ] || { echo "$phase only supports CHUNK 0" >&2; exit 2; }
		chunk=0 graph=0
	fi
	"${PYTHON:-python3}" - "$here" "$out" "$ckpt" <<'PYREF'
import pathlib, sys
sys.path.insert(0, sys.argv[1])
from hf_checkpoint import completed_reference
completed_reference(pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3]))
PYREF
	stage="$out/$phase-graph$graph-chunk$chunk"
fi
# A failed phase is immutable; use a new OUT for another attempt.
mkdir "$stage"
cat "$HOME/ml-explore/.gpu-lock/lock/owner" > "$stage/lock-owner.txt"
started=$SECONDS
{
	date -u +%Y-%m-%dT%H:%M:%SZ
	if git -C "$pkg" rev-parse HEAD 2>/dev/null; then
		git -C "$pkg" status --porcelain -- .
	else
		: "${HF_PARITY_SOURCE:?exported trees require HF_PARITY_SOURCE receipt}"
		sha256sum "$HF_PARITY_SOURCE"
	fi
	go version
	nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
	printf 'checkpoint: %s\nphase: %s\n' "$ckpt" "$phase"
	printf 'graph: %s\nchunk: %s\natol_override: %s\nrtol_override: %s\ntile_override: %s\nref_args: %s\n' \
		"${graph:-n/a}" "${chunk:-n/a}" "${LLAMA_HF_PARITY_ATOL:-default}" "${LLAMA_HF_PARITY_RTOL:-default}" \
		"${LLAMA_EGRAPH_PF_TILE:-default}" "${REF_ARGS:-default}"
} > "$stage/env.txt"
if [ "$phase" = reference ]; then
	[ ! -e "$out/ref" ] || { echo "reference directory already exists" >&2; exit 2; }
	# Split option words without evaluating shell code or expanding globs.
	ref_args=()
	if [ -n "${REF_ARGS:-}" ]; then
		read -r -a ref_args <<< "$REF_ARGS"
	fi
	for arg in "${ref_args[@]}"; do
		case "$arg" in --checkpoint|--checkpoint=*|--output|--output=*) echo "REF_ARGS cannot override checkpoint or output" >&2; exit 2 ;; esac
	done
	if timeout --signal=TERM --kill-after=15s "$((1500 - (SECONDS - started)))s" "${PYTHON:-python3}" "$here/hf_reference.py" \
		--checkpoint "$ckpt" --output "$out/ref" "${ref_args[@]}" > "$stage/run.log" 2>&1; then
		rc=0
	else
		rc=$?
	fi
else
	binary="$stage/llama-infer.test"
	if (cd "$pkg" && timeout --signal=TERM --kill-after=15s "$((1500 - (SECONDS - started)))s" \
		env GOWORK=off go test -mod=readonly -c -o "$binary" .) > "$stage/build.log" 2>&1; then
		:
	else
		rc=$?
		printf '%s\n' "$rc" > "$stage/exit-code.txt"
		exit "$rc"
	fi
	remaining=$((1500 - (SECONDS - started)))
	[ "$remaining" -gt 0 ] || { echo "phase deadline reached during build" >&2; exit 124; }
	sha256sum "$binary" > "$stage/binary.sha256"
	if LLAMA_HF_PARITY="$out/ref" LLAMA_HF_PARITY_MODEL="$ckpt" LLAMA_HF_PARITY_ENGINE="$phase" \
		LLAMA_HF_PARITY_CHUNKS="$chunk" LLAMA_HF_PARITY_GRAPH="$graph" LLAMA_HF_PARITY_REPORT="$stage/report.json" \
		timeout --signal=TERM --kill-after=15s "${remaining}s" "$binary" -test.run '^TestHFParity$' -test.v -test.timeout 24m \
		> "$stage/run.log" 2>&1; then
		rc=0
	else
		rc=$?
	fi
	# Keep the hash/build receipt, not the throwaway test executable.
	"${PYTHON:-python3}" - "$binary" <<'PY'
import pathlib, sys
pathlib.Path(sys.argv[1]).unlink()
PY
fi
printf '%s\n' "$rc" > "$stage/exit-code.txt"
[ "$rc" -eq 0 ] || exit "$rc"
"${PYTHON:-python3}" - "$out" "$stage" "$phase" "$here" "$ckpt" <<'PY'
import json, pathlib, sys
out, stage, phase = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3]
sys.path.insert(0, sys.argv[4])
from hf_checkpoint import completed_reference
if phase == 'reference':
    suite = json.loads((out/'ref/suite.json').read_text())
    if not isinstance(suite.get('checkpoint'), str) or pathlib.Path(suite['checkpoint']).resolve() != pathlib.Path(sys.argv[5]).resolve():
        raise SystemExit('reference checkpoint differs from requested checkpoint')
    if not isinstance(suite.get('cases'), list) or not suite['cases']:
        raise SystemExit('reference has no cases')
else:
    suite = completed_reference(out, pathlib.Path(sys.argv[5]))
    report = json.loads((stage/'report.json').read_text())
    log = (stage/'run.log').read_text()
    expected = {c['name'] for c in suite['cases']}
    got = report.get('results', [])
    if '--- SKIP: TestHFParity' in log or report.get('engine') != phase:
        raise SystemExit('native parity test was skipped or used another engine')
    if {r['case'] for r in got} != expected or len(got) != len(expected):
        raise SystemExit('parity report does not cover every reference case')
    if report.get('rows', 0) <= 0 or any(r.get('rows', 0) <= 0 or not r.get('passed') for r in got):
        raise SystemExit('parity report contains missing or failing rows')
(stage/'PASS').write_text('phase completed; review tolerance and coverage labels separately\n')
PY

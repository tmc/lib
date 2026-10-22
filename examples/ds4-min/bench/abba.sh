#!/bin/bash
# Capture exactly five warm C, Go, Go, C prefill blocks under one GPU lease.
#
# Required environment:
#   DS4_MODEL, DS4_TOKENS, DS4_C_RUNNER, DS4_GO_RUNNER, CUDAPROF, CUDAPROF_AGENT
# Optional: DS4_ABBA_OUT (default: $HOME/tmp/ds4-abba-YYYYmmdd-HHMMSS)

set -eu

if [[ ${DS4_ABBA_LOCKED:-} != 1 ]]; then
  : "${GPU_LOCK:=/home/tmc/ml-explore/bin/gpu-lock}"
  exec "$GPU_LOCK" env DS4_ABBA_LOCKED=1 "$0" "$@"
fi

: "${DS4_MODEL:?set DS4_MODEL to the GGUF model}"
: "${DS4_TOKENS:?set DS4_TOKENS to the 2048-token JSON fixture}"
: "${DS4_C_RUNNER:?set DS4_C_RUNNER to the C prefill runner}"
: "${DS4_GO_PREFILL_DIGEST:?set DS4_GO_PREFILL_DIGEST}"
: "${DS4_GO_FINAL_DIGEST:?set DS4_GO_FINAL_DIGEST}"
if [[ ${DS4_ABBA_BUILD:-0} != 1 ]]; then
  : "${DS4_GO_RUNNER:?set DS4_GO_RUNNER to the Go ds4-min binary}"
  : "${CUDAPROF:?set CUDAPROF to cudaprof built from the campaign source}"
  : "${CUDAPROF_AGENT:?set CUDAPROF_AGENT to the explicitly built CUPTI agent}"
fi

benchroot=${DS4_BENCHROOT:-$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)}
root=${DS4_REPO_ROOT:-$(CDPATH= cd -- "$benchroot/.." && pwd)}
repo=${DS4_GIT_ROOT:-${DS4_REPO_ROOT:-$(CDPATH= cd -- "$benchroot/../../.." && pwd)}}
: "${DS4_ABBA_BUILD_COMMIT:=$(git -C "$repo" rev-parse HEAD)}"
stamp=$(date +%Y%m%d-%H%M%S)
out=${DS4_ABBA_OUT:-$HOME/tmp/ds4-abba-$stamp}
if [[ -e $out ]]; then
  echo "output directory already exists: $out" >&2
  exit 1
fi
mkdir -p "$out"

if [[ ${DS4_ABBA_BUILD:-0} == 1 ]]; then
  mkdir -p "$out/bin"
  (cd "$repo/examples" && GOWORK=off go build -o "$out/bin/ds4-min" ./ds4-min)
  (cd "$repo/nvidia" && GOWORK=off go build -o "$out/bin/cudaprof" ./x/cmd/cudaprof)
  "$repo/nvidia/internal/cudainject/native/build_agent.sh" "$out/bin"
  DS4_GO_RUNNER=$out/bin/ds4-min
  CUDAPROF=$out/bin/cudaprof
  CUDAPROF_AGENT=$out/bin/cupti_agent.so
fi

{
  printf 'source_commit='
  git -C "$repo" rev-parse HEAD
  printf 'build_commit=%s\n' "$DS4_ABBA_BUILD_COMMIT"
  printf 'c_warm_contract=DS4_PROFILE_PHASE=prefill (runner performs warm sync, resets session, then opens profiler range)\n'
  printf 'source_status_begin\n'
  git -C "$repo" status --porcelain
  printf 'source_status_end\n'
  for f in "$DS4_C_RUNNER" "$DS4_GO_RUNNER" "$CUDAPROF" "$CUDAPROF_AGENT" "$DS4_TOKENS"; do
    sha256sum "$f"
  done
  stat -c 'model_size=%s model_mtime=%y model=%n' "$DS4_MODEL"
  printf 'go_prefill_digest=%s\ngo_final_digest=%s\n' "$DS4_GO_PREFILL_DIGEST" "$DS4_GO_FINAL_DIGEST"
} >"$out/manifest"

go version -m "$DS4_GO_RUNNER" >"$out/go-version.txt" 2>&1 || {
  echo "cannot read Go build metadata from $DS4_GO_RUNNER" >&2
  exit 1
}
grep -Eq "vcs.revision(=|[[:space:]])+$DS4_ABBA_BUILD_COMMIT$" "$out/go-version.txt" || {
  echo "Go runner revision does not match DS4_ABBA_BUILD_COMMIT" >&2
  cat "$out/go-version.txt" >&2
  exit 1
}
grep -Eq "vcs.modified(=|[[:space:]])+true" "$out/go-version.txt" && {
  echo "Go runner was built from a modified tree" >&2
  exit 1
}

capture_c() {
  name=$1
  {
    printf 'C %s: ' "$name"
    printf '%q ' env CUDA_VISIBLE_DEVICES=0 CUDAPROF_CAPTURE_RANGE=cudaProfilerApi \
      "$CUDAPROF" capture --inject "$CUDAPROF_AGENT" --launch-manifest \
      --finalize profiler-stop --graph-trace=node --require-complete \
      -o "$out/$name.pprof" -- env DS4_PROFILE_PHASE=prefill \
      "$DS4_C_RUNNER" "$DS4_TOKENS" "$DS4_MODEL" 4096 2
    printf '\n'
  } >>"$out/manifest"
  env CUDA_VISIBLE_DEVICES=0 CUDAPROF_CAPTURE_RANGE=cudaProfilerApi \
    "$CUDAPROF" capture --inject "$CUDAPROF_AGENT" --launch-manifest \
    --finalize profiler-stop --graph-trace=node --require-complete \
    -o "$out/$name.pprof" -- env DS4_PROFILE_PHASE=prefill \
    "$DS4_C_RUNNER" "$DS4_TOKENS" "$DS4_MODEL" 4096 2 \
    >"$out/$name.stdout" 2>"$out/$name.stderr"
}

capture_go() {
  name=$1
  {
    printf 'Go %s: ' "$name"
    printf '%q ' env CUDA_VISIBLE_DEVICES=0 CUDAPROF_CAPTURE_RANGE=cudaProfilerApi \
      "$CUDAPROF" capture --inject "$CUDAPROF_AGENT" --launch-manifest \
      --finalize profiler-stop --graph-trace=node --require-complete \
      -o "$out/$name.pprof" -- env DS4_DISABLE_CUDA_GRAPH=1 \
      "$DS4_GO_RUNNER" -backend cuda -model "$DS4_MODEL" \
      -prompt-tokens "$DS4_TOKENS" -ctx 4096 -tokens 2 -warm-prefill \
      -cuda-profiler-phase prefill -cuda-profiler-stop \
      -expect-prefill-logits "$DS4_GO_PREFILL_DIGEST" \
      -expect-final-logits "$DS4_GO_FINAL_DIGEST"
    printf '\n'
  } >>"$out/manifest"
  env CUDA_VISIBLE_DEVICES=0 CUDAPROF_CAPTURE_RANGE=cudaProfilerApi \
    "$CUDAPROF" capture --inject "$CUDAPROF_AGENT" --launch-manifest \
    --finalize profiler-stop --graph-trace=node --require-complete \
    -o "$out/$name.pprof" -- env DS4_DISABLE_CUDA_GRAPH=1 \
    "$DS4_GO_RUNNER" -backend cuda -model "$DS4_MODEL" \
    -prompt-tokens "$DS4_TOKENS" -ctx 4096 -tokens 2 -warm-prefill \
    -cuda-profiler-phase prefill -cuda-profiler-stop \
    -expect-prefill-logits "${DS4_GO_PREFILL_DIGEST:?set DS4_GO_PREFILL_DIGEST}" \
    -expect-final-logits "${DS4_GO_FINAL_DIGEST:?set DS4_GO_FINAL_DIGEST}" \
    >"$out/$name.stdout" 2>"$out/$name.stderr"
}

gpu_stamp() {
  nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu,memory.used \
    --format=csv >"$out/$1"
}

capture_arm() {
  name=$1
  gpu_stamp "$name.gpu.before"
  case ${name#B*-} in
    C1|C2) capture_c "$name" ;;
    G1|G2) capture_go "$name" ;;
    *) echo "invalid arm name: $name" >&2; exit 1 ;;
  esac
  gpu_stamp "$name.gpu.after"
}

for block in 1 2 3 4 5; do
  capture_arm "B${block}-C1"
  capture_arm "B${block}-G1"
  capture_arm "B${block}-G2"
  capture_arm "B${block}-C2"
done

# Keep the capture list in the format consumed by cudaprof qualify.  The
# human-readable manifest above deliberately contains commands and hashes and
# is not a machine input to the qualification tool.
qualify_manifest="$out/qualify.manifest"
: >"$qualify_manifest"
for block in 1 2 3 4 5; do
  for suffix in C1 G1 G2 C2; do
    printf 'B%d %s %s\n' "$block" "$suffix" "$out/B${block}-${suffix}.pprof" >>"$qualify_manifest"
  done
done
set +e
"$CUDAPROF" qualify --warmup-until='hc_expand|hc_cached_expand' --json \
  "$qualify_manifest" >"$out/qualification.json"
qualify_rc=$?
set +e

for block in 1 2 3 4 5; do
  for suffix in C1 G1 G2 C2; do
    name="B${block}-${suffix}"
    "$CUDAPROF" gaps --warmup-until='hc_expand|hc_cached_expand' \
      "$out/$name.pprof" >"$out/gaps-boundary-$name.txt"
    if ! grep -q 'steady-state (activity before the first kernel matching' "$out/gaps-boundary-$name.txt"; then
      echo "$name: registered boundary kernel was not found" >&2
    fi
  done
done

for block in 1 2 3 4 5; do
  "$CUDAPROF" compare -phase 0 -ops "$root/bench/ds4-c-go.ops" \
    -stages "$root/bench/ds4.stages" \
    "$out/B${block}-C1.pprof" "$out/B${block}-G1.pprof" >"$out/compare-B${block}-C1-G1.txt"
  "$CUDAPROF" compare -phase 0 -ops "$root/bench/ds4-c-go.ops" \
    -stages "$root/bench/ds4.stages" \
    "$out/B${block}-C2.pprof" "$out/B${block}-G2.pprof" >"$out/compare-B${block}-C2-G2.txt"
  "$CUDAPROF" gaps --diff_base="$out/B${block}-C1.pprof" "$out/B${block}-G1.pprof" >"$out/gaps-B${block}-C1-G1.txt"
  "$CUDAPROF" gaps --diff_base="$out/B${block}-C2.pprof" "$out/B${block}-G2.pprof" >"$out/gaps-B${block}-C2-G2.txt"
  "$CUDAPROF" gaps --warmup-until='hc_expand|hc_cached_expand' \
    --diff_base="$out/B${block}-C1.pprof" "$out/B${block}-G1.pprof" >"$out/gaps-layers-B${block}-C1-G1.txt"
  "$CUDAPROF" gaps --warmup-until='hc_expand|hc_cached_expand' \
    --diff_base="$out/B${block}-C2.pprof" "$out/B${block}-G2.pprof" >"$out/gaps-layers-B${block}-C2-G2.txt"
  "$CUDAPROF" gaps --json --warmup-until='hc_expand|hc_cached_expand' \
    --diff_base="$out/B${block}-C1.pprof" "$out/B${block}-G1.pprof" >"$out/gaps-steady-B${block}-1.json"
  "$CUDAPROF" gaps --json --warmup-until='hc_expand|hc_cached_expand' \
    --diff_base="$out/B${block}-C2.pprof" "$out/B${block}-G2.pprof" >"$out/gaps-steady-B${block}-2.json"
done

set -e
set +e
python3 - "$out" <<'PY'
import json
import pathlib
import sys

out = pathlib.Path(sys.argv[1])
arms = {}
for block in range(1, 6):
  for suffix in ("C1", "G1", "G2", "C2"):
    name = f"B{block}-{suffix}"
    status = None
    for line in (out / f"{name}.stdout").read_text().splitlines():
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "phase_receipt" in obj:
            status = obj
            break
    if status is None:
        raise SystemExit(f"missing capture status for {name}")
    receipt = status["phase_receipt"]
    stop = next((e for e in receipt.get("edges", []) if e.get("edge") == "stop"), {})
    arms[name] = {"status": receipt.get("status", "UNREAD"),
                  "seconds": receipt.get("duration_nanos", 0) / 1e9,
                  "flush_nanos": stop.get("flush_nanos", 0),
                  "agent": status.get("capability", {}).get("agent_revision", "UNREAD")}
if any(a["status"] != "MEASURED" for a in arms.values()):
    (out / "summary.txt").write_text("gate_receipts=FAIL\n")
    raise SystemExit(4)
blocks = [(arms[f"B{b}-G1"]["seconds"] - arms[f"B{b}-C1"]["seconds"],
           arms[f"B{b}-G2"]["seconds"] - arms[f"B{b}-C2"]["seconds"])
          for b in range(1, 6)]
block_means = [sum(x) / 2 for x in blocks]
steady = []
for b in range(1, 6):
    for pair in (1, 2):
        x = json.loads((out / f"gaps-steady-B{b}-{pair}.json").read_text())
        c = x["base"]["devices"][0]["steady_state"]["span_ns"]
        g = x["target"]["devices"][0]["steady_state"]["span_ns"]
        steady.append((g - c) / 1e9)
def median(xs):
    xs = sorted(xs); n = len(xs)
    return xs[n//2] if n % 2 else (xs[n//2-1] + xs[n//2]) / 2
def mad(xs):
    m = median(xs)
    return median([abs(x - m) for x in xs])
c_noise = [arms[f"B{b}-C2"]["seconds"] - arms[f"B{b}-C1"]["seconds"] for b in range(1, 6)]
g_noise = [arms[f"B{b}-G2"]["seconds"] - arms[f"B{b}-G1"]["seconds"] for b in range(1, 6)]
c_mad, g_mad = mad(c_noise), mad(g_noise)
noise = max(c_mad, g_mad, 0.001)
gap = median(block_means)
steady_gap = median(steady)
qualification = json.loads((out / "qualification.json").read_text())
qualified_gap = qualification["steady"]["median_block_mean_ms"] / 1000
if abs(qualified_gap - steady_gap) > 0.000001:
    print(f"steady gap disagrees with cudaprof qualify: {steady_gap:.9f} vs {qualified_gap:.9f}", file=sys.stderr)
    raise SystemExit(5)
with (out / "summary.txt").open("w") as f:
    for b in range(1, 6):
        for suffix in ("C1", "G1", "G2", "C2"):
            name = f"B{b}-{suffix}"; a = arms[name]
            f.write(f"{name} status={a['status']} seconds={a['seconds']:.9f} flush_nanos={a['flush_nanos']} agent={a['agent']}\n")
    f.write("block_means_seconds=" + ",".join(f"{x:.9f}" for x in block_means) + "\n")
    f.write(f"median_gap_seconds={gap:.9f}\nc_mad_seconds={c_mad:.9f}\ng_mad_seconds={g_mad:.9f}\nnoise_floor_seconds={noise:.9f}\n")
    f.write("steady_gaps_seconds=" + ",".join(f"{x:.9f}" for x in steady) + f"\nsteady_median_gap_seconds={steady_gap:.9f}\n")
    f.write(f"qualify_verdict={qualification['verdict']}\nqualify_gap_seconds={qualified_gap:.9f}\n")
    f.write("gate_receipts=PASS\n")
PY
python_rc=$?
set -e

echo "ABBA artifacts: $out"
if [[ $python_rc -ne 0 ]]; then
  exit "$python_rc"
fi
if [[ $qualify_rc -ne 0 ]]; then
  exit "$qualify_rc"
fi

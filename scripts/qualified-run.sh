#!/bin/bash
# qualified-run.sh runs a command only while the host is quiet enough to be
# measured, and records the evidence that it was.
#
# The failure this exists to prevent: a throughput number taken on a contended
# host is indistinguishable from a real regression. Every published rate in this
# project that had to be withdrawn was withdrawn for that reason.
#
# It waits for the qualifier to pass, re-checks after the run, and marks the
# repetition DISQUALIFIED if the host became busy while the work was running --
# it does not discard it silently, because a disqualified rep is evidence about
# the host.
#
# Identity files given with -i are hashed before the whole run and again after
# every repetition. Paths given with -i are resolved to absolute paths first, so
# the comparison is over content and location, never over how the path happened
# to be spelled. A repetition whose identity moved is UNATTRIBUTABLE: it may
# have measured the code named in the receipt or something else, and there is no
# way to tell afterwards. This is not hypothetical in a shared worktree, where
# another agent's edit lands mid-run.
#
# Two limits on what that establishes. It compares endpoints, so an edit made
# and reverted inside one repetition is invisible: this detects a persistent
# change, not that the files held still throughout. And it hashes what it is
# given, so -i must name the actual sources, binary and inputs -- passing a
# manifest instead hashes the manifest twice and says nothing about whether its
# referents moved.
#
# The wait budget is per repetition: each rep waits up to -w for a quiet host.
#
# -q may be repeated. Every qualifier must pass for the host to be qualified,
# and they are tried in the order given so the cheap one can be listed first.
# They refuse for different reasons -- a load average says nothing about memory
# pressure -- and a run needs all of them to be satisfied, not any one.
#
# usage: qualified-run.sh -q qualifier... -o outdir [-n reps] [-w max-wait-sec]
#                         [-i identity-file]... -- command [args...]
set -u

qualifiers=(); outdir=; reps=3; maxwait=7200; identity=()
while [ $# -gt 0 ]; do
  case "$1" in
    -q) qualifiers+=("$2"); shift 2;;
    -o) outdir=$2; shift 2;;
    -n) reps=$2; shift 2;;
    -w) maxwait=$2; shift 2;;
    -i) identity+=("$2"); shift 2;;
    --) shift; break;;
    *) echo "qualified-run.sh: unknown flag $1" >&2; exit 2;;
  esac
done
[ ${#qualifiers[@]} -gt 0 ] && [ -n "$outdir" ] && [ $# -gt 0 ] || {
  echo "usage: qualified-run.sh -q qualifier... -o outdir [-n reps] [-w sec] [-i file]... -- cmd" >&2
  exit 2
}

case "$reps" in ''|*[!0-9]*|0) echo "qualified-run.sh: -n must be a positive integer" >&2; exit 2;; esac
case "$maxwait" in ''|*[!0-9]*) echo "qualified-run.sh: -w must be a non-negative integer" >&2; exit 2;; esac

# An identity file that cannot be hashed means the run is unattributable. Fail
# before spending the host, not after.
#
# Each path is resolved to an absolute one first, because shasum writes the
# path it was given beside each hash and the comparison is over that whole
# line. A relative path spelled one way before the run and another way after
# it compares unequal on identical bytes, which reports the strongest verdict
# this script has for a run that was in fact clean. Resolving once removes the
# spelling from the comparison without removing the path: two files that swap
# contents still differ.
resolved=()
for f in ${identity[@]+"${identity[@]}"}; do
  [ -f "$f" ] && [ -r "$f" ] || { echo "qualified-run.sh: identity must be a readable regular file: $f" >&2; exit 1; }
  d=$(cd "$(dirname "$f")" && pwd) || { echo "qualified-run.sh: cannot resolve identity path: $f" >&2; exit 1; }
  resolved+=("$d/$(basename "$f")")
done
identity=(${resolved[@]+"${resolved[@]}"})

# Refuse to write into a directory that already holds a run. Overwriting one
# silently destroys the evidence it exists to preserve.
if [ -e "$outdir" ] && [ -n "$(ls -A "$outdir" 2>/dev/null)" ]; then
  echo "qualified-run.sh: output directory is not empty: $outdir" >&2
  exit 1
fi
mkdir -p "$outdir"; chmod 700 "$outdir"
date -u +%Y-%m-%dT%H:%M:%SZ > "$outdir/start.txt"
printf '%s\0' "$@" > "$outdir/argv.nul"
uname -a > "$outdir/host.txt"
# A precheck cannot establish that identity capture succeeded; only the hash can.
if [ ${#identity[@]} -gt 0 ]; then
  shasum -a 256 "${identity[@]}" > "$outdir/identity.sha256" || {
    echo "qualified-run.sh: identity capture failed" >&2; exit 1; }
fi

# Run every qualifier, appending each one's output to the destination so a
# refusal names which gate refused. Returns the first non-zero exit.
qualify() {
  local dest=$1 q rc=0
  : > "$dest.stdout"; : > "$dest.stderr"
  for q in "${qualifiers[@]}"; do
    printf '== %s\n' "$q" >> "$dest.stdout"
    "$q" >> "$dest.stdout" 2>> "$dest.stderr" || { rc=$?; break; }
  done
  return $rc
}

# Wait for the qualifiers to pass, bounded. Returns 0 if they passed, 1 on timeout.
wait_qualified() {
  local dest=$1 budget=$2 spent=0
  while ! qualify "$dest"; do
    [ "$spent" -ge "$budget" ] && return 1
    sleep 60; spent=$((spent + 60))
  done
  printf '%s\n' "$spent" > "$dest.waited"
  return 0
}

qualified=0; failed=0; skipped=0; disqualified=0; unattributable=0
for rep in $(seq 1 "$reps"); do
  d=$outdir/rep$rep; mkdir -p "$d"
  # Gate every repetition, not just the first. A host that was quiet when rep 1
  # started is not evidence that it is quiet now.
  if ! wait_qualified "$d/qualify-before" "$maxwait"; then
    printf '%s\n' 1 > "$d/qualify-before.exit"
    echo SKIPPED > "$d/verdict.txt"; skipped=$((skipped + 1))
    continue
  fi
  printf '%s\n' 0 > "$d/qualify-before.exit"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$d/start.txt"
  "$@" > "$d/stdout.log" 2> "$d/stderr.log"
  rc=$?
  printf '%s\n' "$rc" > "$d/exit.txt"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$d/end.txt"
  qualify "$d/qualify-after"
  printf '%s\n' "$?" > "$d/qualify-after.exit"
  # Re-hash the identity files. A repetition whose inputs moved is judged
  # before its exit status, because a changed identity makes both a success
  # and a failure unattributable rather than making one of them interesting.
  moved=0
  if [ ${#identity[@]} -gt 0 ]; then
    if ! shasum -a 256 "${identity[@]}" > "$d/identity-after.sha256" 2>"$d/identity-after.stderr"; then
      moved=1
    elif ! cmp -s "$outdir/identity.sha256" "$d/identity-after.sha256"; then
      moved=1
    fi
  fi
  # A repetition counts only if its identity held, the host stayed quiet AND
  # the command succeeded. These are separate failures, reported separately.
  if [ "$moved" -ne 0 ]; then
    echo UNATTRIBUTABLE > "$d/verdict.txt"; unattributable=$((unattributable + 1))
  elif [ "$rc" -ne 0 ]; then
    echo FAILED > "$d/verdict.txt"; failed=$((failed + 1))
  elif [ "$(cat "$d/qualify-after.exit")" != 0 ]; then
    echo DISQUALIFIED > "$d/verdict.txt"; disqualified=$((disqualified + 1))
  else
    echo QUALIFIED > "$d/verdict.txt"; qualified=$((qualified + 1))
  fi
done

printf 'reps=%s qualified=%s disqualified=%s unattributable=%s failed=%s skipped=%s\n' \
  "$reps" "$qualified" "$disqualified" "$unattributable" "$failed" "$skipped" > "$outdir/summary.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$outdir/end.txt"
find "$outdir" -type f ! -name MANIFEST.sha256 -print0 | sort -z |
  xargs -0 shasum -a 256 > "$outdir/MANIFEST.sha256"
cat "$outdir/summary.txt"
# Report success only if something was actually measured on a quiet host and
# nothing else happened. Any kind of fail-open would misreport a bad run, and
# a disqualified repetition is one: with several reps, one that qualified used
# to be enough to exit 0 while another ran on a contended host.
[ "$qualified" -gt 0 ] && [ "$disqualified" -eq 0 ] && [ "$unattributable" -eq 0 ] &&
  [ "$failed" -eq 0 ] && [ "$skipped" -eq 0 ]

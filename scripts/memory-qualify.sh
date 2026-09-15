#!/bin/bash
# memory-qualify.sh passes only while the host has enough memory for a model
# run to finish. It is a qualifier in the sense qualified-run.sh means: it
# prints its reasoning and exits 0 when the host is fit, non-zero when it is not.
#
# The failure this exists to prevent: the load-average qualifier answers "may I
# start", and a host can be genuinely quiet by that measure and still be unable
# to finish. TestMetalDeepRegimeAnchor qualified at load 5.41 and was killed by
# memguard ten minutes later at compressor 0.473 against its 0.350 limit. Load
# average carries no information about memory pressure, and a launch sample
# carries none about minute nine.
#
# Compose it with the load qualifier rather than replacing one with the other;
# they refuse for different reasons and a run needs both to be false.
#
# -min-available is not a measured threshold. It is the wired peak a run of this
# size was observed to reach plus margin, and the caller supplies it, because the
# right value is a property of the workload and not of this script.
#
# usage: memory-qualify.sh [-min-available GiB] [-max-compressor FRACTION]
#
# VM_STAT and MEMSIZE_BYTES override the two host probes so the behaviour can be
# tested without arranging real host memory.
set -u

# Defaults come from the environment so the script is usable as a bare
# qualifier: qualified-run.sh invokes "$qualifier" unsplit, with no arguments,
# so a caller composing this into a run sets the thresholds in the environment.
# The flags are for running it by hand.
minavail=${DS4_MIN_AVAILABLE_GIB:-95}
maxcomp=${DS4_MAX_COMPRESSOR:-0.10}
while [ $# -gt 0 ]; do
  case "$1" in
    -min-available) minavail=$2; shift 2;;
    -max-compressor) maxcomp=$2; shift 2;;
    *) echo "memory-qualify.sh: unknown flag $1" >&2; exit 2;;
  esac
done

case "$minavail" in ''|*[!0-9.]*) echo "memory-qualify.sh: -min-available must be a number" >&2; exit 2;; esac
case "$maxcomp" in ''|*[!0-9.]*) echo "memory-qualify.sh: -max-compressor must be a number" >&2; exit 2;; esac

total=${MEMSIZE_BYTES:-$(sysctl -n hw.memsize)}
stat=$(${VM_STAT:-vm_stat}) || { echo "memory-qualify.sh: vm_stat failed" >&2; exit 2; }

# Parse once, in awk, so a missing field is an error rather than an empty string
# that silently reads as zero and qualifies a host with no memory at all.
parsed=$(printf '%s\n' "$stat" | awk -v total="$total" '
  /page size of/                 { for (i=1;i<=NF;i++) if ($i=="of") pg=$(i+1) }
  /Pages free/                   { free=$3+0; f=1 }
  /Pages inactive/               { inact=$3+0; i=1 }
  /Pages purgeable/              { purge=$3+0; p=1 }
  /Pages occupied by compressor/ { comp=$5+0; c=1 }
  END {
    if (!pg || !f || !i || !p || !c) exit 1
    printf "%.4f %.2f\n", comp*pg/total, (free+inact+purge)*pg/1073741824
  }') || {
  echo "memory-qualify.sh: could not read page counts from vm_stat" >&2
  exit 2
}
comp=${parsed%% *}
avail=${parsed##* }

printf 'compressor %s of host, available %s GiB; limits %s and %s GiB\n' \
  "$comp" "$avail" "$maxcomp" "$minavail"

if awk "BEGIN{exit !($comp <= $maxcomp && $avail >= $minavail)}"; then
  echo "host has memory to finish"
  exit 0
fi
echo "host does not have memory to finish" >&2
exit 1

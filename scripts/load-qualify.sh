#!/bin/bash
# load-qualify.sh runs the registered load gate as a qualifier for
# qualified-run.sh, so the gate's invocation lives in the tracked tree instead
# of being hand-written into each run's evidence directory. Every prior
# campaign wrote its own copy of this wrapper; they disagree about the limit.
#
# The limit is REQUIRED and has no default here on purpose. hostqualify's own
# -max-load defaults to NumCPU/4 -- 4.00 on a 16-CPU host -- so a wrapper that
# simply omits the flag quietly enforces a limit nobody registered, and the
# receipt records a pass against it. Every campaign registers its own limit
# (4.00 and 8.00 both appear in this repo's evidence) and the caller states it.
#
# usage: DS4_MAX_LOAD=8.00 load-qualify.sh
#        load-qualify.sh -max-load 8.00
#
# HOSTQUALIFY names the binary; the default is the one built beside its source.
set -u

maxload=${DS4_MAX_LOAD:-}
while [ $# -gt 0 ]; do
  case "$1" in
    -max-load) maxload=$2; shift 2;;
    *) echo "load-qualify.sh: unknown flag $1" >&2; exit 2;;
  esac
done

if [ -z "$maxload" ]; then
  echo "load-qualify.sh: set DS4_MAX_LOAD or pass -max-load; there is no default, because hostqualify's own default is NumCPU/4 and enforcing that silently is how a run passes a limit nobody registered" >&2
  exit 2
fi
case "$maxload" in ''|*[!0-9.]*) echo "load-qualify.sh: max-load must be a number" >&2; exit 2;; esac
awk "BEGIN{exit !($maxload > 0)}" || { echo "load-qualify.sh: max-load must be positive" >&2; exit 2; }

bin=${HOSTQUALIFY:-$(dirname "$0")/../examples/hostqualify/hostqualify}
[ -x "$bin" ] || {
  echo "load-qualify.sh: $bin is not executable; build it with 'go build -o examples/hostqualify/hostqualify ./examples/hostqualify'" >&2
  exit 2
}

# The limit is echoed, not just passed, so the receipt records which limit was
# enforced rather than only whether something passed.
echo "load gate: $bin -max-load $maxload"
exec "$bin" -max-load "$maxload"

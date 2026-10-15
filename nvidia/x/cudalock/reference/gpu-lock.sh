#!/usr/bin/env bash
# gpu-lock — serialize GPU-heavy work across agents on the single GPU.
#
# Usage:  gpu-lock [--] <command> [args...]
#
# The lock is a directory containing an owner file. It is created under a
# temporary name with the owner file already inside and renamed into place, so
# a permanent sibling .guard inode serializes inspection, publication and
# retirement with flock(2), and the lock never exists without naming
# its holder. Blocks until the lock is free, runs the command, releases on
# exit. A lock whose holder process is gone is reclaimed so a crashed holder
# can't wedge the queue; a lock whose holder is alive is never taken away from
# it, however long it has been held. A lock with no owner file cannot be
# probed, so it is removed once it has sat empty for 15s.
#
# Waiters are served in queue order. Each takes a ticket in a queue directory
# on arrival, and only the waiter holding the head ticket may try the mutex;
# the rest wait their turn. The head is chosen by the lane rules below; with
# one lane it is the oldest live ticket. Without the queue the lock went to
# whoever polled first after a release, and a driver that runs gpu-lock back
# to back tries the instant it starts, so a sweep starved every other waiter
# for hours. The queue decides only who may try; the permanent advisory guard
# serializes namespace transitions. Every participant must use this protocol.
# Older uncoordinated Go/script clients are unsafe and are not compatible.
# A ticket whose waiter is gone is reaped by the same rule as
# a lock: same host, pid dead. Never by age.
#
# A waiter may declare how long it will hold the lock. GPU_LOCK_EXPECT is an
# estimate, used only to predict start times (see gpu-lock-ls); it is never
# enforced. GPU_LOCK_LIMIT is a hard limit: once the lock has been held that
# long, the command's process group is stopped as on any other release and
# gpu-lock exits 124, as timeout(1) does. An EXPECT with no LIMIT implies a
# limit of 1.5 times the estimate, rounded up; an explicit LIMIT replaces it
# and must not be below the EXPECT. With neither the job is unbounded, as
# before. Both are recorded in the ticket and the lock's owner file, the limit
# as its effective value with limit_src=explicit|default.
#
# Tickets are in one of two lanes. A ticket whose effective limit is at most
# GPU_LOCK_SHORT_MAX is short; any other, including one with no limit, is
# batch. The estimate never decides the lane, so it cannot buy a place in the
# short lane: only a limit that will be enforced can. Within a lane tickets
# are served in arrival order. The head of the queue is the oldest short
# ticket, unless the oldest batch ticket arrived before it and has waited
# longer than its aging time; then that batch ticket is the head, so a stream
# of short jobs delays the oldest batch ticket by at most its aging time plus
# one short hold.
# The lane and the aging time are written into the ticket by its own waiter,
# so every waiter derives the same head from the same queue. They can still
# disagree for a poll at the instant a batch ticket becomes overdue, since
# each reads its own clock; that can make two waiters both try the mutex, or
# neither, but never grants it twice, because the queue only decides who may
# try while holding the same permanent guard. A ticket with no lane is batch.
#
# The command runs in its own process group, and releasing the lock means
# signalling that whole group and confirming it is gone. "released" is a
# statement about the lock, not about the GPU, but it is at least a statement
# that nothing gpu-lock started is still running: if anything survives TERM and
# then KILL, gpu-lock says so with pids, hands the lock to a survivor rather
# than freeing it, and exits 71. It never reports a release it cannot verify.
# A command that reads directly from a terminal should have its stdin
# redirected: a background process group reading the tty is stopped by SIGTTIN.
#
# Env:
#   GPU_LOCKDIR        lock directory (default $HOME/ml-explore/.gpu-lock/lock)
#   GPU_LOCK_MAXHOLD   warn about a lock held longer than this many seconds
#                      (3600). Warning only: it never authorizes a reclaim.
#   GPU_LOCK_POLL      whole seconds between acquire attempts (3)
#   GPU_LOCK_MAXWAIT   max seconds to wait, in the queue and for the mutex
#                      together; 0 = forever (0)
#   GPU_LOCK_QUEUE     ticket directory (default: tickets/ beside the lock)
#   GPU_LOCK_LABEL     human label recorded in the lock (default: pid)
#   GPU_LOCK_KILLWAIT  seconds to wait after TERM, and again after KILL, for
#                      the command's process group to exit (5)
#   GPU_LOCK_EXPECT    expected hold time, an estimate that is never enforced:
#                      whole seconds, or with an s, m or h suffix (30, 90s,
#                      5m, 2h); unset = unknown
#   GPU_LOCK_LIMIT     hard limit on the hold time, in the same form; the
#                      command is stopped once it is reached and gpu-lock exits
#                      124. Default: 1.5 x GPU_LOCK_EXPECT rounded up, or none
#                      when that is unset too.
#   GPU_LOCK_SHORT_MAX largest effective limit, in seconds, that puts a ticket
#                      in the short lane (300)
#   GPU_LOCK_BATCH_AGING
#                      seconds a batch ticket waits before it goes ahead of
#                      the short lane (1800). Recorded in the ticket, so it is
#                      the waiter's own setting that counts.
#   A nested gpu-lock (see GPU_LOCK_HELD) ignores EXPECT and LIMIT: it runs
#   inside the holder's process group, and the holder's limit covers it.
#
# Exit status: the command's own, or
#   2    bad usage or settings
#   70   guard/protocol failure, or a displaced live lock could not be restored
#   71   something the command started survived TERM and KILL; the lock was
#        handed to it rather than released
#   75   GPU_LOCK_MAXWAIT expired before the lock was acquired
#   124  the command was stopped at GPU_LOCK_LIMIT
#
# Exports to the command:
#   GPU_LOCK_HELD      pid of the holding gpu-lock. Set only while the lock is
#                      held, so a descendant can tell it is already covered and
#                      must not try to acquire again (it would self-deadlock:
#                      this lock has no owner recursion). nvidia/internal/gputest
#                      reads this, and so does gpu-lock: a nested gpu-lock whose
#                      GPU_LOCK_HELD names the live pid in the lock's owner file
#                      runs its command at once, with no ticket and no acquire.
#                      Cooperative among our own tooling, not a security
#                      boundary. It carries holder identity, not mere presence:
#                      do not invoke gpu-lock as `env GPU_LOCK_HELD=1 gpu-lock ...`;
#                      a value the lock does not corroborate is ignored.
set -euo pipefail

LOCKDIR="${GPU_LOCKDIR:-$HOME/ml-explore/.gpu-lock/lock}"
MAXHOLD="${GPU_LOCK_MAXHOLD:-3600}"
POLL="${GPU_LOCK_POLL:-3}"
MAXWAIT="${GPU_LOCK_MAXWAIT:-0}"
KILLWAIT="${GPU_LOCK_KILLWAIT:-5}"
LABEL="${GPU_LOCK_LABEL:-$$}"
SHORT_MAX="${GPU_LOCK_SHORT_MAX:-300}"
AGING="${GPU_LOCK_BATCH_AGING:-1800}"
WAIT_STARTED=$SECONDS

# The wait loop does integer arithmetic on these. Under set -e, a value like
# POLL=0.5 made that arithmetic fail, which ended the loop early: gpu-lock then
# printed "acquired" and ran the command without the lock.
for v in POLL:"$POLL" MAXWAIT:"$MAXWAIT" MAXHOLD:"$MAXHOLD" KILLWAIT:"$KILLWAIT" \
  SHORT_MAX:"$SHORT_MAX" BATCH_AGING:"$AGING"; do
  case "${v#*:}" in
  ""|*[!0-9]*|0[0-9]*)
    echo "gpu-lock: GPU_LOCK_${v%%:*}=${v#*:} must be a whole number of seconds, no leading zero" >&2
    exit 2
    ;;
  esac
done
if [ "$POLL" -eq 0 ]; then echo "gpu-lock: GPU_LOCK_POLL must be at least 1" >&2; exit 2; fi

duration() { # duration NAME VALUE — print VALUE in seconds, or complain and fail
  local n=$2 mult=1
  case $n in
  *s) n=${n%s} ;;
  *m) n=${n%m} mult=60 ;;
  *h) n=${n%h} mult=3600 ;;
  esac
  case $n in
  ""|*[!0-9]*|0*|??????????*)
    echo "gpu-lock: GPU_LOCK_$1=$2 must be a whole number of seconds from 1, optionally with an s, m or h suffix, no leading zero" >&2
    return 1
    ;;
  esac
  echo $((n * mult))
}
EXPECT="" LIMIT="" LIMIT_SRC=""
if [ -n "${GPU_LOCK_EXPECT:-}" ]; then EXPECT=$(duration EXPECT "$GPU_LOCK_EXPECT") || exit 2; fi
if [ -n "${GPU_LOCK_LIMIT:-}" ]; then
  LIMIT=$(duration LIMIT "$GPU_LOCK_LIMIT") || exit 2
  LIMIT_SRC=explicit
  if [ -n "$EXPECT" ] && [ "$EXPECT" -gt "$LIMIT" ]; then
    echo "gpu-lock: GPU_LOCK_EXPECT=$GPU_LOCK_EXPECT is longer than GPU_LOCK_LIMIT=$GPU_LOCK_LIMIT" >&2
    exit 2
  fi
elif [ -n "$EXPECT" ]; then
  LIMIT=$(((EXPECT * 3 + 1) / 2))
  LIMIT_SRC=default
fi
LANE="batch"
if [ -n "$LIMIT" ] && [ "$LIMIT" -le "$SHORT_MAX" ]; then LANE=short; fi
# The limit is enforced with wait -n -p, which is bash 5.1.
if [ -n "$LIMIT" ] && { [ "${BASH_VERSINFO[0]}" -lt 5 ] ||
   { [ "${BASH_VERSINFO[0]}" -eq 5 ] && [ "${BASH_VERSINFO[1]}" -lt 1 ]; }; }; then
  echo "gpu-lock: GPU_LOCK_LIMIT and GPU_LOCK_EXPECT need bash 5.1 or later" >&2
  exit 2
fi
# declared is the owner-file suffix naming the declared durations.
declared=""
if [ -n "$EXPECT" ]; then declared="$declared expect=$EXPECT"; fi
if [ -n "$LIMIT" ]; then declared="$declared limit=$LIMIT limit_src=$LIMIT_SRC"; fi
HOST=$(hostname -s)

[ "${1:-}" = "--" ] && shift
if [ "$#" -eq 0 ]; then echo "gpu-lock: no command given" >&2; exit 2; fi

# Protocol v2 is Linux-only: util-linux flock uses the same flock(2) locks as
# cudalock. There is no fallback to fcntl record locks or an unguarded path.
if [ "$(uname -s)" != Linux ] || ! command -v flock >/dev/null; then
  echo "gpu-lock: protocol v2 requires Linux and util-linux-compatible flock" >&2
  exit 2
fi
case $LOCKDIR in
/*) ;;
*) echo "gpu-lock: protocol v2 requires an absolute GPU_LOCKDIR" >&2; exit 2 ;;
esac
# Match Go filepath.Abs/Clean before resolving the physical parent.
# Resolving symlinks before lexical .. would select a different guard.
LOCKDIR=$(realpath -m -s -- "$LOCKDIR")
if [ "$LOCKDIR" = / ]; then echo "gpu-lock: lock cannot be a filesystem root" >&2; exit 2; fi

# Preflight: an unwritable or missing parent makes the acquire loop
# unsatisfiable, and because the acquire's stderr is discarded the loop would
# spin silently forever, printing nothing and running nothing. Fail loudly
# instead. (This is exactly how a stale macOS default wedged four lanes on
# 2026-08-28: /Users does not exist on Linux.)
PARENT=$(dirname "$LOCKDIR")
if ! mkdir -p "$PARENT" 2>/dev/null || [ ! -w "$PARENT" ]; then
  echo "gpu-lock: cannot use lock parent '$PARENT' (missing or unwritable)." >&2
  echo "gpu-lock: set GPU_LOCKDIR to a writable path. Refusing to spin." >&2
  exit 2
fi
QUEUE="${GPU_LOCK_QUEUE:-$PARENT/tickets}"
if ! mkdir -p "$QUEUE" 2>/dev/null || [ ! -w "$QUEUE" ]; then
  echo "gpu-lock: cannot use queue '$QUEUE' (missing or unwritable)." >&2
  echo "gpu-lock: set GPU_LOCK_QUEUE to a writable path. Refusing to spin." >&2
  exit 2
fi

PARENT=$(cd -P "$PARENT" && pwd)
LOCKDIR="$PARENT/$(basename "$LOCKDIR")"
export GPU_LOCKDIR="$LOCKDIR"
GUARD="$LOCKDIR.guard"
PARENT_ID=$(stat -c '%d:%i' "$PARENT")
if [ -L "$GUARD" ] || { [ -e "$GUARD" ] && [ ! -f "$GUARD" ]; }; then
  echo "gpu-lock: guard must be a regular file, not a symlink" >&2; exit 70
fi
# Create once, without truncation. Neither release nor reclaim ever unlinks it.
(umask 077; : >>"$GUARD")
exec 9<>"$GUARD"
GUARD_ID=$(stat -Lc '%d:%i' "/proc/$$/fd/9")
guard_check() {
  if [ -L "$GUARD" ] || [ ! -f "$GUARD" ] ||
     [ "$(stat -c '%h' "$GUARD")" != 1 ] ||
     [ "$(stat -c '%d:%i' "$GUARD")" != "$GUARD_ID" ] ||
     [ "$(stat -c '%d:%i' "$PARENT")" != "$PARENT_ID" ] ||
     [ -L "$LOCKDIR" ] || { [ -e "$LOCKDIR" ] && [ ! -d "$LOCKDIR" ]; }; then
    echo "gpu-lock: permanent guard or namespace identity changed; refusing" >&2
    return 70
  fi
}
check_wait() { # check_wait LIMIT; LIMIT=0 does not bound retirement
  if [ "$1" -gt 0 ] && [ $((SECONDS - WAIT_STARTED)) -ge "$1" ]; then
    echo "gpu-lock: acquisition deadline expired" >&2
    exit 75
  fi
}
guard_lock() { # guard_lock [WAIT_LIMIT]; 0 means wait indefinitely for retirement
  local limit=${1:-0} status
  while true; do
    check_wait "$limit"
    if flock -x -n -E 75 9; then
      if ! guard_check; then flock -u 9 || true; exit 70; fi
      check_wait "$limit"
      return
    else
      status=$?
      if [ "$status" -ne 75 ]; then
        echo "gpu-lock: cannot lock permanent guard" >&2
        exit 70
      fi
    fi
    if [ "$limit" -gt 0 ] && [ $((SECONDS - WAIT_STARTED)) -ge "$limit" ]; then
      echo "gpu-lock: timed out waiting for permanent guard" >&2
      exit 75
    fi
    sleep 0.05
  done
}
guard_unlock() {
  if ! flock -u 9; then echo "gpu-lock: cannot unlock permanent guard" >&2; exit 70; fi
}
# Only ESRCH is evidence of death. EPERM and unrecognized probe errors keep
# the owner alive; an arbitrary failed kill -0 is not permission to reclaim.
dead_pid() {
  local failure
  case $1 in ''|0|*[!0-9]*) return 1 ;; esac
  if failure=$(LC_ALL=C kill -0 "$1" 2>&1); then return 1; fi
  case $failure in *'No such process'*) return 0 ;; *) return 1 ;; esac
}

# TOKEN identifies this holder for the life of the process. Ownership is what
# authorizes a delete, so it must survive pid reuse: a pid alone does not.
TOKEN="$HOST-$$-$(date +%s)-${RANDOM}${RANDOM}"

# field must succeed when the file is gone. A lock or ticket can vanish between
# finding it and reading it, and under set -e a failed read in an assignment
# ended gpu-lock silently, rc 1, without ever running its command.
field() { # field NAME DIR — print NAME= from DIR/owner, empty if unreadable
  sed -n "s/.*[ 	]*$1=\([^ 	]*\).*/\1/p" "$2/owner" 2>/dev/null | head -1 || true
}

# A gpu-lock run by the holder's own command is already covered, and waiting
# would deadlock it behind itself. The environment names the holder and the
# lock must agree, as in nvidia/x/cudalock; anything less falls through to a
# normal wait, which blocks on a real holder rather than running unlocked.
if [ -n "${GPU_LOCK_HELD:-}" ]; then
  guard_lock "$MAXWAIT"
  if [ "$(field pid "$LOCKDIR")" = "$GPU_LOCK_HELD" ] &&
     [ "$(field host "$LOCKDIR")" = "$HOST" ] && kill -0 "$GPU_LOCK_HELD" 2>/dev/null; then
    check_wait "$MAXWAIT"
    echo "gpu-lock: already held by gpu-lock pid $GPU_LOCK_HELD; running nested: $*" >&2
    guard_unlock
    exec "$@" 9>&-
  fi
  guard_unlock
fi

# release deletes the lock only if it is still ours. A lock that is missing or
# names somebody else is not ours to delete: an earlier gpu-lock that lost its
# lock must not take out the live lock that replaced it. Nothing but a
# reclaim takes a lock from a live holder, and reclaim requires the holder to
# be dead, so the check cannot go stale. The lock is renamed aside before it
# is deleted: rm -rf empties a directory before removing it, and a waiter's
# rename landing in that window nests inside it, so the rm failed and left an
# empty lock with no owner that stopped every waiter.
release() {
  local owner aside
  guard_lock
  if [ "$(field token "$LOCKDIR")" = "$TOKEN" ]; then
    aside=$(mktemp -d "$PARENT/.gpu-lock-released.XXXXXX")
    rmdir "$aside"
    mv "$LOCKDIR" "$aside"
    rm -rf "$aside"
    echo "gpu-lock: released (label=$LABEL, rc=${rc:-interrupted})" >&2
    guard_unlock
    return
  fi
  owner=$(cat "$LOCKDIR/owner" 2>/dev/null || echo "no owner file")
  echo "gpu-lock: not releasing '$LOCKDIR': no longer ours ($owner)" >&2
  guard_unlock
}

# reclaim takes the lock of a holder already found dead. It renames the lock
# aside first and only then re-reads the owner, so the thing it deletes is
# provably the thing it inspected; if the rename caught a different lock (a
# second reclaimer beat us and a new holder acquired) it is put back untouched.
# It compares the whole owner line rather than the token so that it can also
# clear a lock left by a gpu-lock predating the token field.
# The caller holds the permanent guard from the original owner read through
# this retirement and the next root publication. Restore is defense against
# uncoordinated clients, not the mutual-exclusion mechanism.
reclaim() { # reclaim DEAD_OWNER
  local dead=$1 aside
  aside=$(mktemp -d "$PARENT/.gpu-lock-stale.XXXXXX")
  rmdir "$aside"
  mv "$LOCKDIR" "$aside" 2>/dev/null || return 0
  if [ "$(cat "$aside/owner" 2>/dev/null)" = "$dead" ]; then
    echo "gpu-lock: reclaimed lock of dead holder: $(cat "$aside/owner" 2>/dev/null)" >&2
    rm -rf "$aside"
    return 0
  fi
  # Not the lock we diagnosed: a second reclaimer beat us and a new holder
  # acquired. Put it back exactly as we found it.
  mv "$aside" "$LOCKDIR" 2>/dev/null || true
  if [ -d "$aside" ] || [ -d "$LOCKDIR/${aside##*/}" ]; then
    echo "gpu-lock: could not restore a live lock displaced during reclaim;" >&2
    echo "gpu-lock: it is at '$aside'. Refusing to run." >&2
    exit 70
  fi
  return 0
}

# A ticket is a directory with an owner file, named by arrival time so that
# the queue's sorted order is arrival order. Like the lock it is built under a
# dot name, which the queue scan skips, and renamed into place already naming
# its waiter.
ns=$(date +%s%N)
case $ns in *[!0-9]*) ns="$(date +%s)000000000" ;; esac
TICKET="$QUEUE/$ns.$$"
take_ticket() {
  local staging
  staging=$(mktemp -d "$QUEUE/.ticket.XXXXXX")
  printf 'pid=%s host=%s label=%s token=%s epoch=%s at=%s lane=%s aging=%s%s\n' \
    "$$" "$HOST" "$LABEL" "$TOKEN" "$(date +%s)" "$(date -Iseconds)" "$LANE" "$AGING" \
    "$declared" >"$staging/owner"
  mv "$staging" "$TICKET"
}

# ofield sets VAR to NAME= from an owner line, or to empty. The label may
# contain spaces, so pid and host are read from before " label=" and every
# other field from " token=" on, where a label cannot reach.
ofield() { # ofield VAR NAME LINE
  local rest=$3 re=" $2=([^ ]*) "
  case $2 in
  pid|host) rest=${rest%%" label="*} ;;
  *) case $rest in *" token="*) rest=" token=${rest#*" token="}" ;; esac ;;
  esac
  if [[ " $rest " =~ $re ]]; then printf -v "$1" %s "${BASH_REMATCH[1]}"; else printf -v "$1" %s ""; fi
}

# scan_queue reaps the tickets of dead waiters and sets head (the ticket that
# may try the mutex now), pos (ours, in grant order counting from 1; 0 if it
# is missing) and n. Reaping follows the reclaim rule: a waiter on another
# host cannot be probed, so its ticket is left alone, and age alone never
# condemns a ticket.
#
# Grant order merges the two lanes, each in arrival order (the tickets' sorted
# names). At each step the oldest batch ticket goes first if it is overdue and
# arrived before the oldest short one, otherwise the oldest short ticket does.
# It depends only on the tickets' names and contents and on now.
scan_queue() {
  local t o p h l a name arr now i b s k
  local -a live=() short=() batch=() order=()
  head="" pos=0 n=0
  now=$(date +%s)
  for t in "$QUEUE"/*; do
    [ -d "$t" ] || continue
    o=$(cat "$t/owner" 2>/dev/null || true)
    if [ "$t" != "$TICKET" ]; then
      ofield p pid "$o"; ofield h host "$o"
      if [ "$h" = "$HOST" ] && [ -n "$p" ] && dead_pid "$p"; then
        echo "gpu-lock: reaped ticket of dead waiter: $o" >&2
        rm -rf "$t"
        continue
      fi
    fi
    ofield l lane "$o"
    if [ "$l" = short ]; then
      short+=("${#live[@]}")
    else
      ofield a aging "$o"
      case $a in ""|*[!0-9]*) a=1800 ;; esac
      name=${t##*/} arr=${name%%.*}
      case $arr in ??????????*[0-9]) arr=${arr%?????????} ;; *) arr=$now ;; esac
      case $arr in *[!0-9]*) arr=$now ;; esac
      # A batch entry is "index:overdue".
      batch+=("${#live[@]}:$((now - arr > a ? 1 : 0))")
    fi
    live+=("$t")
  done
  b=0 s=0
  while [ $((b + s)) -lt "${#live[@]}" ]; do
    if [ "$b" -lt "${#batch[@]}" ] && {
         [ "$s" -ge "${#short[@]}" ] ||
         { [ "${batch[b]#*:}" -eq 1 ] && [ "${batch[b]%:*}" -lt "${short[s]}" ]; }; }; then
      order+=("${batch[b]%:*}"); b=$((b + 1))
    else
      order+=("${short[s]}"); s=$((s + 1))
    fi
  done
  n=${#order[@]}
  for k in "${!order[@]}"; do
    i=${order[k]}
    if [ "$k" -eq 0 ]; then head=${live[i]}; fi
    if [ "${live[i]}" = "$TICKET" ]; then pos=$((k + 1)); fi
  done
}

# Until the lock is ours the only thing to undo is the ticket. The signal
# handlers must exit: a bare handler runs and then resumes the wait, leaving a
# waiter that is invisible to the queue. The acquire replaces all of these.
trap 'rm -rf "$TICKET"' EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM
take_ticket

warned_maxhold=0
waited=0
announced=
while true; do
  check_wait "$MAXWAIT"
  waited=$((SECONDS - WAIT_STARTED))
  scan_queue
  if [ "$pos" -eq 0 ]; then
    # Nothing removes a live waiter's ticket but a person. Put it back under
    # the same name, which keeps our place in line.
    echo "gpu-lock: ticket '$TICKET' vanished from the queue; retaking it" >&2
    take_ticket
    continue
  fi
  if [ "$head" != "$TICKET" ]; then
    if [ "$announced" != "$pos" ]; then
      echo "gpu-lock: queued at position $pos of $n (label=$LABEL lane=$LANE); head: $(cat "$head/owner" 2>/dev/null)" >&2
      announced=$pos
    fi
    if [ "$MAXWAIT" -gt 0 ] && [ $((SECONDS - WAIT_STARTED)) -ge "$MAXWAIT" ]; then
      echo "gpu-lock: timed out after ${waited}s at queue position $pos of $n" >&2
      exit 75
    fi
    sleep "$POLL"; waited=$((waited + POLL))
    continue
  fi

  guard_lock "$MAXWAIT"
  while true; do
    guard_check || exit 70
    staging=$(mktemp -d "$PARENT/.gpu-lock-acquire.XXXXXX")
    printf 'pid=%s host=%s label=%s token=%s epoch=%s at=%s lane=%s%s\n' \
      "$$" "$HOST" "$LABEL" "$TOKEN" "$(date +%s)" "$(date -Iseconds)" "$LANE" "$declared" >"$staging/owner"
    # Inspection and publication share the permanent guard. Do not rename over
    # even an empty directory: ownerless locks still have their grace period.
    # Verify the token as defense against unsupported uncoordinated clients.
    check_wait "$MAXWAIT"
    if [ ! -e "$LOCKDIR" ] && mv "$staging" "$LOCKDIR" 2>/dev/null && [ "$(field token "$LOCKDIR")" = "$TOKEN" ]; then
      trap cleanup EXIT INT TERM
      trap - HUP
      rm -rf "$TICKET"
      guard_unlock
      break 2
    fi
    rm -rf "$staging" "$LOCKDIR/${staging##*/}"
  
    if [ -f "$LOCKDIR/owner" ]; then
      held_owner=$(cat "$LOCKDIR/owner" 2>/dev/null || true)
      held_pid=$(field pid "$LOCKDIR")
      held_host=$(field host "$LOCKDIR")
      held_epoch=$(field epoch "$LOCKDIR")
      # Liveness, not elapsed time, is what says a lock is abandoned. A slow but
      # healthy holder keeps its lock. A holder on another host cannot be probed
      # from here, so it is always left alone.
      case $held_pid in ''|0|*[!0-9]*) malformed=1 ;; *) malformed=0 ;; esac
      if [ "$malformed" -eq 1 ] && { [ -z "$held_host" ] || [ "$held_host" = "$HOST" ]; }; then
        now=$(date +%s)
        mtime=$(stat -c %Y "$LOCKDIR/owner")
        if [ $((now - mtime)) -gt 15 ]; then reclaim "$held_owner"; continue; fi
      fi
      if { [ -z "$held_host" ] || [ "$held_host" = "$HOST" ]; } &&
         dead_pid "$held_pid"; then
        reclaim "$held_owner"
        continue
      fi
      if [ "$announced" != head ]; then
        echo "gpu-lock: first in queue (label=$LABEL); waiting for holder: $held_owner" >&2
        announced="head"
      fi
      if [ "$warned_maxhold" -eq 0 ] && [ -n "$held_epoch" ] && [ "$held_epoch" -gt 0 ]; then
        now=$(date +%s)
        if [ $((now - held_epoch)) -gt "$MAXHOLD" ]; then
          echo "gpu-lock: holder alive but has held the lock $((now - held_epoch))s: $(cat "$LOCKDIR/owner")" >&2
          warned_maxhold=1
        fi
      fi
    else
      # The lockdir exists but has no owner file. A holder of this version never
      # leaves one that way, but the release of an earlier one could (see
      # release), and nvidia/x/cudalock holds one for the instant between its
      # mkdir and its owner write. Nobody can be probed for liveness, so follow
      # cudalock: an empty lock unchanged for 15s is abandoned. rmdir refuses a
      # lock that has gained an owner file.
      now=$(date +%s)
      mtime=$(stat -c %Y "$LOCKDIR" 2>/dev/null || stat -f %m "$LOCKDIR" 2>/dev/null || echo "$now")
      if [ $((now - mtime)) -gt 15 ] && rmdir "$LOCKDIR" 2>/dev/null; then
        echo "gpu-lock: removed empty lock with no owner file, unchanged for $((now - mtime))s" >&2
        continue
      fi
      echo "gpu-lock: '$LOCKDIR' exists with no owner file; waiting (${waited}s)" >&2
    fi
    break
  done
  guard_unlock
  if [ "$MAXWAIT" -gt 0 ] && [ $((SECONDS - WAIT_STARTED)) -ge "$MAXWAIT" ]; then
    echo "gpu-lock: timed out after ${waited}s waiting for: $(cat "$LOCKDIR/owner" 2>/dev/null)" >&2
    exit 75
  fi
  sleep "$POLL"; waited=$((waited + POLL))
done

group_members() { # group_members PGID — live pids in that process group
  ps -eo pid=,pgid=,stat= 2>/dev/null |
    awk -v g="$1" '$2 == g && $3 !~ /^Z/ { print $1 }'
}

# stop_group signals the command's process group and prints the pids that
# outlive it. Empty output means the group is verifiably gone; anything else is
# a survivor that the lock was guarding and that the next holder would collide
# with. A process that ignores TERM passes that disposition to everything it
# forks, so TERM alone proves nothing and KILL has to follow.
stop_group() { # stop_group PGID
  local pgid=$1 sig waited_ticks ticks
  # A pgid of 0 means "my own process group" to kill(2), and on Linux kernel
  # threads report pgid 0 to ps, so an unset child would look like a group with
  # members and take out the caller. Refuse it.
  case $pgid in ''|0|*[!0-9]*) return 0 ;; esac
  ticks=$((KILLWAIT * 5))
  for sig in TERM KILL; do
    [ -n "$(group_members "$pgid")" ] || return 0
    kill -"$sig" -"$pgid" 2>/dev/null || true
    waited_ticks=0
    while [ "$waited_ticks" -lt "$ticks" ]; do
      [ -n "$(group_members "$pgid")" ] || return 0
      sleep 0.2
      waited_ticks=$((waited_ticks + 1))
    done
  done
  group_members "$pgid"
}

# cleanup stops the work before it touches the lock. The order is the point: a
# lock released while the work it guarded is still running hands the next
# holder a machine that is not free.
cleaned=0
cleanup() {
  local survivors keep
  if [ "$cleaned" -eq 1 ]; then return; fi
  cleaned=1
  if [ -n "${watchdog:-}" ]; then kill "$watchdog" 2>/dev/null || true; fi
  survivors=$(stop_group "${child:-0}")
  if [ -z "$survivors" ]; then
    release
    return
  fi
  # The pids go in the message itself. A detail line from ps is a nicety that
  # may come back empty; what must never be missing is which pids are still up.
  echo "gpu-lock: NOT RELEASED (label=$LABEL): survived TERM and KILL:$(echo "$survivors" | tr '\n' ' ' | sed 's/ *$//;s/^/ /')" >&2
  ps -o pid=,pgid=,stat=,etime=,args= -p "$(echo "$survivors" | tr '\n' ',' | sed 's/,$//')" 2>/dev/null |
    sed 's/^/  /' >&2
  # Hand the lock to a survivor instead of freeing it. Reclaim is gated on
  # liveness, so the lock now stays held exactly as long as the orphan runs and
  # frees itself when the last one is gone.
  guard_lock
  if [ "$(field token "$LOCKDIR")" = "$TOKEN" ]; then
    keep=$(echo "$survivors" | head -1)
    printf 'pid=%s host=%s label=%s token=%s epoch=%s at=%s orphan-of=%s\n' \
      "$keep" "$HOST" "$LABEL" "$TOKEN" "$(date +%s)" "$(date -Iseconds)" "$$" >"$LOCKDIR/owner"
    echo "gpu-lock: lock left held by surviving pid $keep; kill it to free the GPU" >&2
  fi
  guard_unlock
  exit 71
}

echo "gpu-lock: acquired (label=$LABEL lane=$LANE${LIMIT:+ limit=${LIMIT}s}) — running: $*" >&2
# Descendants inherit this and skip acquiring; see the header note.
export GPU_LOCK_HELD=$$
# Job control puts the command in a process group of its own, so a signal can
# reach descendants that a signal to this script would never touch.
set +e
rc=
set -m
"$@" 9>&- &
child=$!
set +m
trap cleanup EXIT INT TERM
if [ -z "$LIMIT" ]; then
  wait "$child"
  rc=$?
  set -e
  exit $rc
fi
# The limit is a sleep beside the command, outside its process group, and
# whichever finishes first decides. A signal interrupts the wait with neither
# (finished stays empty); the trap has already stopped the command by then.
sleep "$LIMIT" 9>&- &
watchdog=$!
finished=
wait -n -p finished "$child" "$watchdog"
rc=$?
if [ "$finished" = "$watchdog" ]; then
  watchdog=
  rc=124
  echo "gpu-lock: killed at its declared limit: held ${LIMIT}s (limit_src=$LIMIT_SRC, label=$LABEL) — $*" >&2
fi
set -e
exit $rc

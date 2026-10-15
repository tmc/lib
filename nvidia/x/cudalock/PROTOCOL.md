# Local lock protocol v2

## Participants and namespace

Go clients support Linux and Darwin using OS flock. The repository reference
script supports Linux, GNU coreutils, Bash (5.1 when declared limits are used),
and util-linux-compatible flock. The script requires an absolute lock path
and uses GNU realpath -m -s before resolving its physical parent, matching
Go lexical cleaning without resolving symlinks before `..`. It refuses other
platforms before starting
work. There is no fcntl-record-lock or unguarded fallback. The script is a
repository successor, not an installed tool update.

Each lock has one permanent sibling `<lock-directory>.guard`. Resolve the
parent to its physical path and keep the leaf name; parent symlink aliases
therefore select the same guard. The guard must be a regular file with one
link. Creation does not truncate it. Clients never unlink, rotate or replace
the guard, even after the last holder releases. Owner lines retain their
existing pid/host/label/epoch/at representation; script tokens and queue fields
are unchanged. The guard path is independent of owner-directory generations.

Keep the physical parent, its ancestors and guard namespace stable while any
client runs. An administrator must stop all participants before moving or
removing it. Path replacement by a nonparticipant is outside this cooperative
protocol: an identity check cannot make arbitrary concurrent unlink/replacement
atomic with the next operation. Detected old descriptors or changed parents
are refused rather than reopened onto a second lock domain.

## Namespace transaction

1. Open the guard without following a final symlink (Go), recording descriptor
   identity and physical parent identity. The Linux script records the open
   descriptor through `/proc/$$/fd/9` and compares the regular, single-link path.
2. Take exclusive flock on that inode. Go uses nonblocking attempts so context
   cancellation bounds waiting; guard errors refuse acquisition. The script's
   acquisition wait observes MAXWAIT and refuses lock errors. OS process death
   releases flock, so a crashed guard holder cannot strand a permanent lock.
3. Under the guard, recheck descriptor/path/parent identity. Inspect the current
   owner. A live process (including permission-denied liveness probes) or a
   foreign host is never a stale candidate. Unknown script liveness errors are
   retained, not classified as death. Missing/malformed owners retain the
   existing grace policy. WithNoReclaim never retires contention.
4. Retire an eligible stale directory and publish a complete replacement owner
   while retaining the same guard. A second acquirer cannot enter the temporary
   vacancy. Owner-byte revalidation and restoration remain defensive checks for
   uncoordinated modifications; they are not the exclusion mechanism.
5. Register the Go process reference before unlocking. Unlock and close the Go
   descriptor before returning the public handle. Script descriptors remain
   open but unlocked, and are closed in launched work and watchdogs.

Every admission, including local reentrancy, inherited Acquire and shell
nested exec, verifies the current owner under the same guard. Go process
references retain exact owner bytes and namespace identity. Public Inherited
keeps its API and performs a nonblocking guard/process-state verification;
ErrNotInherited on contention means unverified ownership, not an absent owner.

Go root retirement takes the same guard before owner verification, rename and
removal. It preserves the owner file until all other contents are removed.
On failure, restoration verifies the original aside inode and exact owner bytes
and requires an absent canonical path. It never rewrites a missing owner or
overwrites/adopts a newer one. If the snapshot cannot be restored, the retained
handle has explicit pending-retirement cleanup custody and cannot authorize a
new local Acquire/nesting/Inherited verification. Retrying cleanup verifies only
its held aside inode and leaves any newer canonical owner intact. Handle
references bind to the exact process-hold generation.

Acquire rechecks the real context after the process-state mutex wait and before
admitting local or inherited references or publishing a root. A final context
check runs under the namespace guard after root-reference registration. A
canceled publication is retired with that exact generation through the same
root-retirement implementation. If automatic cleanup fails, the returned error
wraps both cancellation and cleanup causes and implements io.Closer so callers
can retry exact-generation cleanup with Close (obtain it using errors.As). This
reference is cleanup only: a restored canonical owner cannot admit further work.
Successful automatic cleanup returns cancellation without retaining an owner.
Root work authority (GPU_LOCK_HELD) is exported only after the final successful
admission check and guard unlock. Canceled cleanup never exported that authority
and does not clear an unrelated or subsequently changed environment value.

The Go lock order is handle mutex, then guard, then the brief process
map mutex; no code waits for the guard while holding the process map mutex.
Reentrant/inherited count-only releases do not take the guard or remove an
ancestor's directory. A failed Go root identity check leaves the handle and
reference retriable. Retirement cleanup errors retain either restored canonical
ownership or explicit refusal state, never a successful ghost reference.
Guard/unlock/close errors are returned, not ignored as successful acquisition. A result refused after publication leaves its live
owner directory in place and drops its process reference, failing toward stuck.
The script also guards survivor ownership handoff after stopping its own work.
Script retirement occurs after stopping its work; filesystem errors can leave
retired debris and a failed script exit. The Go rollback/retry state described
above does not claim a shell cleanup handle or shell rollback implementation.
Its acquisition deadline is checked before an immediate successful guard, after
guard admission, at queue-head attempts and immediately before publication.
Retirement uses no acquisition deadline after work has been stopped.

Do not hold the advisory guard during GPU work. Existing owner liveness and
inherited references cover that interval. No changes to owner schema, public
API, process-group shutdown policy or device admission are supplied here.

## Qualification and mixed versions

Owner-line compatibility does not establish protocol compatibility. A legacy
client can rename a live owner's directory without taking the guard. New
clients cannot prevent that with an advisory lock. Mixing old Go, old installed
scripts, gpulease, or any other uncoordinated client remains unsafe.

Before native admission, identify every actual participant, coordinate the
same protocol and guard inode, qualify the actual installed reference script,
and independently verify local exclusion/recovery/cleanup on the target host.
This source/host repair does not authorize installed-tool changes or live lock
experiments and does not establish device, network-filesystem, cross-host,
crash-durability or arbitrary namespace-replacement guarantees.

The host controls use private directories, synthetic checked-absent stale PIDs,
and explicitly owned child processes. A held guard blocks real public Acquire;
an owned child's crash releases its guard; simultaneous public roots never
enter the active-work marker together; old inode/parent, symlink and hard-link
cases refuse; root retirement verifies the original namespace identity. The
Darwin gate checks script syntax and unsupported-platform refusal, not Linux
script interoperability. Linux script/Go interop remains a distinct required
gate, and installed tools remain untouched.

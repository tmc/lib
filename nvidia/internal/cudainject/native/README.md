# Native CUDA injection bootstrap and CUPTI producer

This directory contains two related native components. `agent.c` and
`buffer.c` provide the header-only C11 bootstrap and bounded callback queue
used by native unit tests. `cupti_agent.c` is the loadable Linux CUPTI
activity producer used by `cudaprof capture`; it is compiled separately
against a qualified CUDA toolkit and records versioned bootstrap, capability,
kernel, memcpy, API, failure, and finalized records. Each exec-created
process owns `CUDAPROF_TRANSPORT/process-PID-generation/` with its handshake
and journal; the controller reports these streams separately and never merges
them by PID alone.

Neither component is a Go shared library or embeds a Go runtime. The CUPTI
producer uses its constructor only to open the controller-provided transport,
register CUPTI callbacks, and start a bounded native drain worker. Callback
paths do not perform blocking transport I/O or unbounded allocation. The Go
controller validates session/process identity, record ordering, checksums and
terminal state before declaring a capture collected.

## Drain and admission

CUPTI hands the buffer-completed callback a whole 8 MiB activity buffer in one
burst. The drain answers a burst with one gathered `writev` per `DRAIN_BATCH`
records rather than one `write` per record, and it serializes those records
outside the admission mutex: the single consumer owns the slots between tail
and head until it publishes a new tail, so producers keep admitting while a
batch is in flight. When the ring is nevertheless full, admission waits with an
absolute deadline instead of refusing, which spends callback latency rather
than losing a record; the wait is abandoned once callbacks are closed, so it
cannot delay finalization. The finalized record reports offered, retained and
dropped-by-category counts that reconcile exactly, alongside drain batch and
callback latency totals.

The ring holds `PENDING_RECORDS` entries and that bound is deliberate: the
throughput fix is the batched drain, not a larger queue.

## Termination

Finalization runs from the ELF destructor, from `atexit`, from a terminator
thread woken by `SIGTERM`, from a `cudaProfilerStop` or `cuProfilerStop`
callback under `CUDAPROF_FINALIZE=profiler-stop`, or from a host program
calling the exported `cudaprof_finalize`. The `SIGTERM` handler does nothing
but an atomic store and a one-byte self-pipe write; every CUPTI call,
allocation, lock and journal write happens in the terminator's normal
execution context. Exactly one context runs the ordered sequence and the
others wait for it, so the process cannot exit over a half-written journal.
The producer preserves the target's termination semantics: it chains to a
handler the target installed before load, and otherwise restores the default
disposition and re-raises once the journal is closed.

The first three triggers are libc's, and a Go target reaches none of them: the
Go runtime leaves through `exit_group`, which runs no ELF destructor and no
`atexit` handler, and nothing sends the process a `SIGTERM`. CUPTI hands a
buffer to `buffer_completed` only when that buffer fills or when something
flushes it, so such a run ends with its records still inside CUPTI -- measured
on one workload, 0 activity records where the same work built as a C program
collected 51. The other two triggers exist for that target, and both are
points the target itself reaches.

`CUDAPROF_FINALIZE=profiler-stop` finalizes the capture when the target calls
`cudaProfilerStop` or `cuProfilerStop`, from inside that callback, before the
runtime can leave. It is off by default because a stop is not by itself the
end of a capture: a target may stop collecting and start again, and finalizing
on every stop would end the journal at the first one. The mode asks less of
the profiler-API subscription than gating does -- it opens no range, so a
surviving stop trigger is its whole requirement -- and it is a separate
request: a target that calls only `cudaProfilerStop`, to hand the agent the
flush its exit path will never run, has no range to gate.

`cudaprof_finalize` is exported for a target that reaches neither the profiler
API nor a libc exit path. It flushes CUPTI and closes the journal, returns
only once the finalized record is on disk -- whether this call ran the
sequence or found another trigger already running it -- and reports 0 when the
capture is finalized and -1 when this process has no initialized agent to
finalize, which is an unconfigured run or a forked child. A caller that
returns from it and then calls `exit_group` loses nothing.

Two cases remain unfinalized by construction. `SIGKILL` cannot be handled at
all. A target that installs its own `SIGTERM` handler after load -- CPython
does -- replaces the producer's handler; such a target still finalizes through
`atexit` when it winds down normally, but not if it is then force-killed. A
target that runs no libc exit path and asks for neither of the other two
triggers is a third: its journal holds whatever was published before it left,
with no finalized record after it.

## Fork

`fork(2)` copies only the calling thread, so a child inherits the producer's
initialized state, its descriptors and its signal disposition, but none of its
threads. Every remaining entry point is harmful there: finalizing joins a
`pthread_t` that names no thread in the child and hangs; a record it wrote
would interleave with the parent's into one journal and break the decoder's
per-stream sequence; and the inherited `SIGTERM` handler writes the *parent's*
self-pipe, so the parent's terminator wakes with its own flag clear, returns,
and the parent loses its termination service for the rest of its life. A
`pthread_atfork` child handler therefore disowns the agent outright: it shuts
every entry point, restores the disposition the target had before load, and
closes the inherited journal descriptor and both ends of the termination pipe
rather than merely forgetting their numbers. The disposition is restored only
while the producer's own handler is still the installed one: a target that
replaced it after load owns that handler, and overwriting it in the child would
change how the child terminates. If that handler cannot be
registered, the producer says so -- a failure record and `atfork_guard` in the
capability record -- rather than leave a reader assuming a guard that is not
there.

This is inertness, not capture. A fork-without-exec child is not profiled. A
child that execs runs the constructor again and gets a real agent of its own,
which is why a forked-and-exec'd worker is captured and a forked-only one is
not; the journal descriptor is `O_CLOEXEC`, so nothing leaks across that exec.

## Per-kind accounting

`queue_pending` is the single admission path for every record kind, and the
aggregate `dropped`, `queue_dropped` and `queue_capacity_dropped` counters say
nothing about which kind was lost. Dividing a retained activity count by an
all-kind drop count compares two different populations; a receipt that did so
overstated activity loss by an order of magnitude. The finalized record
therefore carries `by_kind`, an offered/retained/dropped triple per kind,
counted with the record's own kind at admission and at drain:

    "by_kind":{"activity":{"offered":N,"retained":N,"dropped":N},"api":{...},
               "external":{...},"failure":{...}}

The identity that holds for each kind is `retained + dropped == offered`, and
summing the kinds returns the aggregate `offered`, `retained` and the sum of
the queue-drop categories. `native_dropped` is CUPTI's own loss and never
reaches admission, so it is outside this identity. `activity.retained` is also
counted independently on the drain side and must equal `activity`. Records the
producer writes outside the ring -- bootstrap, capability, finalized, a
capture-range edge, and a constructor-time failure -- are in no kind's
counters.

## Configuration

`CUDAPROF_TRANSPORT` and `CUDAPROF_SESSION` are required; without them the
producer does not initialize. `CUDAPROF_API_ACTIVITY=0` disables runtime and
driver API attribution, which dominates journal volume on serving workloads.
The capability record states which attribution was enabled, so an activity-only
run is a recorded configuration rather than a silent degradation.

`CUDAPROF_MEMSETS=0` leaves memset activity off. With it on, CUDA 13
`cuGraphLaunch` has faulted inside libcuda, called from CUPTI, for an MLX
workload that relaunches cached graph execs updated with captured memset
nodes. The capability record's `memsets_requested` says which was asked for.

`CUDAPROF_CAPTURE_RANGE=cudaProfilerApi` collects only between the target's
profiler-API start and stop, and `CUDAPROF_FINALIZE=profiler-stop` ends the
capture at a stop. Each is answered on its own: one subscription serves both,
because CUPTI allows one subscriber per process, and either request can be met
while the other is not.

`CUDAPROF_FLUSH_INTERVAL_MS` asks for a `cuptiActivityFlushAll` every so many
milliseconds. The default, 0, is off, because a flush a run does not need
spends CUPTI callback latency on the target's own threads for nothing. It
bounds how much of a run is held inside CUPTI for a target that neither fills
a buffer nor reaches any trigger. The flush runs on its own thread rather than
in the drain: `cuptiActivityFlushAll` delivers the buffer on the thread that
calls it, so a drain thread that flushed would be inside `buffer_completed`
offering a whole buffer to a ring only it can empty, and every record past
`PENDING_RECORDS` would wait out the admission deadline and then be dropped
for capacity. Records published this way reach the journal with no finalized
record after them, which is what they are -- a partial capture -- and the
consumer reads them as one.

The capability record states both policies the way it states
`sigterm_finalize` and `atfork_guard`: `"finalize"` is `exit`,
`profiler-stop`, or `requested_unavailable` where the mode was asked for and
no stop trigger could be armed, and `"flush_interval_ms"` is the interval this
process is actually running, which is 0 if the flush thread could not be
started. Both say what this process will do rather than what was requested.

The identity in every event includes protocol and ABI versions, compiled
architecture, process generation, PID, nonzero session nonce, and a CUPTI
version field (zero until a real CUPTI loader records one). The queue
uses caller-owned fixed storage. `cudaprof_native_agent_emit` copies a bounded
payload and never waits; queue exhaustion changes the state to
`BUFFER_EXHAUSTED` and increments a loss counter. Callback admission is
rejected after shutdown begins. Finalization is allowed only after callbacks
have left and the controller has drained the queue; otherwise the state is
`SHUTDOWN_UNRESOLVED`.

Build and run the native-only tests without writing a repository binary:

```sh
mkdir -p "$HOME/tmp/cudaprof-native"
cc -std=c11 -Wall -Wextra -Werror -pedantic -I. \
  agent.c buffer.c agent_test.c \
  -o "$HOME/tmp/cudaprof-native/agent-test"
"$HOME/tmp/cudaprof-native/agent-test"
```

This test proves protocol validation, bootstrap/ready transitions, callback
shutdown ownership, bounded payload copying, queue overflow accounting, and
architecture rejection. It is not a CUDA hardware witness.

The CUPTI producer has its own controls in `../nativecontrol`. They compile
`cupti_agent.c` against the stub CUPTI in `testcupti/` and drive it through
`cupti_agent_control.c`, so the sustained, burst, termination-race,
forced-kill, fork, capture-range, finalization and configuration scenarios run
on any host with a C toolchain:

```sh
go test ./internal/cudainject/nativecontrol/
```

Those controls decode the journal the producer actually wrote and reconcile
offered against retained and dropped. They are not a CUDA hardware witness
either: the stub delivers synthetic buffers through the producer's registered
callbacks, and a real CUPTI run remains the only evidence about CUPTI. A CUPTI build
must record the exact CUDA toolkit, CUPTI version, compiler, target
architecture, loader path, and collector capability result before claiming
activity capture. The producer currently qualifies kernel, memcpy, and API
activity only; deeper counters, code images, PC sampling and child-process
aggregation remain separate dispositions. Queue loss is reported by origin
(oversize, contention, or capacity) in the finalized record.

## Building the CUPTI Producer Shared Library (.so)

`build_agent.sh` is this recipe as a script, and `go generate
./x/cmd/cudaprof` runs it. With no argument it installs `cupti_agent.so`
where `go install` puts binaries, which is one of the places `cudaprof
capture` looks, so a capture then needs no `--inject`:

```sh
go generate ./x/cmd/cudaprof
cudaprof capture -o cap.pprof -- ./target
```

The recipe itself, to build elsewhere or by hand:

```sh
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
CUDA_TARGET="$(uname -m)-linux"
[ "$CUDA_TARGET" = "aarch64-linux" ] && CUDA_TARGET="sbsa-linux"

# The revision this agent was built from, stamped into its capability record.
# Omit it and the record simply carries no agent_revision, which is honest but
# means a reader cannot tell this agent from one built before the capability it
# reports on existed.
REV="$(git rev-parse --short HEAD)$(git diff --quiet || echo -dirty)"

cc -std=c11 -O2 -fPIC -shared \
  -DCUDAPROF_AGENT_REVISION="\"${REV}\"" \
  -I"${CUDA_HOME}/targets/${CUDA_TARGET}/include" -I"${CUDA_HOME}/include" -I. \
  cupti_agent.c -o "$HOME/tmp/libcupti_agent.so" \
  -L"${CUDA_HOME}/targets/${CUDA_TARGET}/lib" -L"${CUDA_HOME}/lib64" -lcupti -lpthread
```

The capability record carries two identities and needs both: `agent_revision`
names the source this `.so` was built from, and `producer` names the `cudaprof`
that read it. They are separate artifacts resolved separately -- the `.so` is
found next to the executable -- so either can be older than the other, and a
record naming only one still cannot say what it describes. An identity the
build did not supply is written as an absent field rather than as `"unknown"`,
because a reader will treat any present value as an answer.

A capture that spans processes loads one agent per process and merges their
activity records into one set, but publishes one capability record, taken from
the target's journal. When the processes ran different agents that record
describes part of the capture in the same words it would use for all of it, so
`cudaprof` adds a `producer_scope` block naming the agent the record describes,
the processes it does not, and how many records they contributed. The block is
absent when every contributing process ran the same agent: a caveat present on
every capture is one a reader learns to skip.

The library exports `cudaprof_push_external`, `cudaprof_pop_external` and
`cudaprof_finalize` for a target that drives them itself. A target that does
not link against the library resolves them with `dlsym(RTLD_DEFAULT, ...)`
once it is loaded; a resolution that fails means no agent is present, which is
how an uninjected run of the same binary behaves.

**Version compatibility and layout gating**:
The agent enforces major version compatibility between compile-time headers and the target's active runtime (e.g. CUDA 12 `Kernel9` vs CUDA 13 `Kernel10`). Within CUDA 13 (13.0 through 13.4+), activity record structures grow append-only; the agent dynamically verifies via `cuptiActivityGetStructSize` (on runtimes exporting it) that the active runtime's record sizes satisfy the compiled layout requirements (`sizeof(Kernel10) == 216` and `sizeof(Memcpy6) == 96`), allowing minor- and update-version skews to collect cleanly while safely rejecting genuine layout incompatibilities. Both `compile_cupti` and `runtime_cupti` versions are recorded in the capability record.


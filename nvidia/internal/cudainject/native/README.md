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

Finalization runs from the ELF destructor, from `atexit`, or from a terminator
thread woken by `SIGTERM`. The `SIGTERM` handler does nothing but an atomic
store and a one-byte self-pipe write; every CUPTI call, allocation, lock and
journal write happens in the terminator's normal execution context. Exactly one
context runs the ordered sequence and the others wait for it, so the process
cannot exit over a half-written journal. The producer preserves the target's
termination semantics: it chains to a handler the target installed before load,
and otherwise restores the default disposition and re-raises once the journal
is closed.

Two cases remain unfinalized by construction. `SIGKILL` cannot be handled at
all. A target that installs its own `SIGTERM` handler after load -- CPython
does -- replaces the producer's handler; such a target still finalizes through
`atexit` when it winds down normally, but not if it is then force-killed.

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
producer writes outside the ring -- bootstrap, capability, finalized, and a
constructor-time failure -- are in no kind's counters.

## Configuration

`CUDAPROF_TRANSPORT` and `CUDAPROF_SESSION` are required; without them the
producer does not initialize. `CUDAPROF_API_ACTIVITY=0` disables runtime and
driver API attribution, which dominates journal volume on serving workloads.
The capability record states which attribution was enabled, so an activity-only
run is a recorded configuration rather than a silent degradation.

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
forced-kill, fork and configuration scenarios run on any host with a C
toolchain:

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

To compile `cupti_agent.c` into the loadable CUPTI producer library (`libcupti_agent.so`) for use with `cudaprof capture --inject`:

```sh
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
CUDA_TARGET="$(uname -m)-linux"
[ "$CUDA_TARGET" = "aarch64-linux" ] && CUDA_TARGET="sbsa-linux"

cc -std=c11 -O2 -fPIC -shared \
  -I"${CUDA_HOME}/targets/${CUDA_TARGET}/include" -I"${CUDA_HOME}/include" -I. \
  cupti_agent.c -o "$HOME/tmp/libcupti_agent.so" \
  -L"${CUDA_HOME}/targets/${CUDA_TARGET}/lib" -L"${CUDA_HOME}/lib64" -lcupti -lpthread
```

**Version compatibility and layout gating**:
The agent enforces major version compatibility between compile-time headers and the target's active runtime (e.g. CUDA 12 `Kernel9` vs CUDA 13 `Kernel10`). Within CUDA 13 (13.0 through 13.4+), activity record structures grow append-only; the agent dynamically verifies via `cuptiActivityGetStructSize` (on runtimes exporting it) that the active runtime's record sizes satisfy the compiled layout requirements (`sizeof(Kernel10) == 216` and `sizeof(Memcpy6) == 96`), allowing minor- and update-version skews to collect cleanly while safely rejecting genuine layout incompatibilities. Both `compile_cupti` and `runtime_cupti` versions are recorded in the capability record.


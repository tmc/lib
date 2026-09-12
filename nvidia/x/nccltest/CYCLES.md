# Cycles and shared-resource sweeps

Pinned nccl-tests `common.cu:953–962` repeats the size sweep inside `TimeTest`.
It retains communicators and buffers, reruns initialization/warmup/checking via
`BenchTime`, and prints a row for each size and cycle. `-N 0` repeats forever.

## Implementation

`RunCycles` repeats one case. `RunSweep` visits every supplied size in order
within each cycle. Both retain communicators, contexts, streams, events,
allocations, and registrations until completion. The largest case determines
allocation capacity. Each sample updates its layout and argument offsets,
resets inputs, poison and comparison counts, then checks, warms up and times
independently. Graphs are destroyed before yielding the sample result.

Every size is validated before CUDA initialization; the size slice is copied.
RunCycles and RunSweep require positive finite cycle counts. A sample or callback error stops
execution. Cleanup occurs once after the sequence, and its failures remain
return errors even if every sample was yielded successfully.

`RunSweepContext` additionally accepts zero cycles for continuous operation.
It checks cancellation before CUDA initialization and between samples. A sample
already in progress completes and is yielded; native calls and callbacks cannot
be interrupted. Cancellation returns the context error after resource cleanup,
joined with cleanup failures. A subprocess remains necessary for a hard timeout.

## Command receipts

Without `-sizes`, the default single-cycle command emits its ordinary result.
`-cycles` above one emits `Kind: cycle` records with one-based `Cycle` and
`Result`, followed by a `Kind: summary` record with `Requested`, `Completed`,
`Complete`, and `Error`.

`-sizes` accepts comma-separated byte sizes and cannot be combined with
`-bytes`. It emits `Kind: sample` records with one-based `Cycle` and
`SizeIndex`, plus `Result`. Its summary contains `RequestedCycles`,
`RequestedSamples`, `CompletedSamples`, `Complete`, and `Error`.
`-cycles 0` emits sweep samples (also for a single `-bytes` size) until an
interrupt or SIGTERM requests cancellation. It then emits a terminal summary
with zero requested cycles/samples, the actual completed count, `Complete: false`,
and the cancellation/cleanup error, and exits nonzero. A continuous run has no
finite completion target; stopping it is not reported as a complete comparison.

Consumers must require a successful terminal summary and process exit zero;
earlier sample records do not establish cleanup success.

## Upstream comparison

The comparator's `-cycles` passes upstream `-N` and checks each ordered cycle
row. `-shared-sweep` additionally runs the complete size sequence in one
process on each side. The reference can represent increasing linear or
geometric sweeps; incompatible lists are rejected before launch. Each sample
must match its effective layout, controls, cycle, and correctness counts.
Missing summaries, malformed ordering, trailing records, or sample failures
prevent a complete receipt. The comparator records `SharedSweep` and `Cycles`.

## Evidence

Host tests cover sequencing, callback failures, allocation/count overflow,
pre-CUDA validation of later sizes, and malformed or incomplete JSON streams.
A local stub integration in `sweep-cli-20260909` completed eight samples and
rejected a missing terminal summary. Its audit script initially mishandled
JSON null comparison lists; the retained receipt audit records that separately.
Stub results establish command routing only, not CUDA behavior.

`cycles-20260909-01` passed physical ordinary, graph, and registered-offset
cycle tests plus callback cancellation. Its upstream matrix passed eleven
operations with three cycles per mode: 57 cases and 29,184 comparisons with
zero mismatches. Scope: one physical rank, float/sum, 1024 bytes. Cleanup and
fresh locked inventories were empty before release at 2026-09-09T21:11:02Z.
Later overflow, cleanup-drain, and shared-sweep changes are outside that binary.

`sweep-20260909-01` subsequently passed physical qualification as recorded below.
It gates a 114-sample shared-sweep comparison on `TestSweep`, which exercises
all eleven operations with non-monotonic sizes and supported in-place modes.
This qualifies only the measured single-rank configuration.

Artifacts are under `/Users/tmc/tmp/nccl-go-testing/` in the named directories.
See [DGX.md](DGX.md) for historical campaign and cleanup receipts.

## Shared-sweep qualification, 2026-09-09

`sweep-20260909-01` passed non-monotonic physical sweeps for all eleven
operations and supported in-place modes. The upstream comparison passed
114 samples across sizes 16/128/1024 and two cycles, retaining communicators
and allocations across sizes. It recorded 22,192 comparisons and zero
mismatches. Scope: one physical rank, float/sum.

The campaign exited zero in 48.6 seconds. Both owned groups and a fresh
locked compute/scoped inventory at 2026-09-09T21:26:16Z were empty. The lane
was explicitly released; the Mac quiet window was preserved. Frozen sources,
binary hashes, logs, and receipts: `/Users/tmc/tmp/nccl-go-testing/sweep-20260909-01/`.
The later parser section-order changes are outside this frozen build and
still await host testing. No multi-rank or performance-equivalence claim.


## Cancellation implementation

Host tests pass for cancellation before CUDA, between sizes, after the final
finite sample, and during a sample. Continuous receipt tests reject a missing
stop reason and preserve cancellation joined with cleanup failure. The physical
`TestSweepContextCancellationCUDA` covers ordinary and graph sweeps stopped
after three samples, followed by a fresh run; it has not yet executed on DGX.
The comparator remains finite and does not treat interrupted runs as equivalence.


Cancellation qualification is prepared in `cancel-20260909-01`: frozen ARM64
runner/test binaries, a 45-second native test gate, and a 45-second CLI signal
gate. The CLI verifier requires at least three ordered samples, SIGINT, exit
one, and an exact cancellation-only summary; a joined cleanup error fails it.
Local stub controls accepted clean cancellation and rejected cleanup failure
(`cancel-cli-stub-20260909-01/audit.json`). No physical result is claimed yet.


The initial cancellation freeze was withdrawn before transfer: its physical
assertion incorrectly required error identity despite successful cleanup wrapping
cancellation with `errors.Join`. Replacement `cancel-20260909-02` requires
`errors.Is(err, context.Canceled)` and cancellation-only error text, so added
cleanup failures still reject the run. The CLI binary is unchanged; the test
binary was rebuilt and all host tests passed again. Physical execution awaits
a new handoff.


The current source also retains a rank's buffers, events, stream, and context
if buffer deregistration or communicator destruction fails. Successful
registration releases clear their handles; failures prevent communicator
destruction, while other registrations still receive a cleanup attempt.
Host failure-injection tests cover send/receive deregistration and destruction
failures. This change is newer than the pending `cancel-20260909-02` binaries
and is not covered by that campaign's physical evidence.


## Cancellation qualification, 2026-09-09

`cancel-20260909-02` passed ordinary and graph continuous-sweep cancellation
at three callback samples, followed by successful fresh runs. Both CLI SIGINT
checks emitted six valid samples, then cancellation-only terminal summaries
with `Complete: false` and expected exit one. The verifier and campaign exited
zero; elapsed campaign time was 5.4 seconds, not a performance measurement.
The signal was sent after receiving the third sample; additional completed
samples reflect signal-delivery scheduling and are permitted by the contract.

Owned groups 215984 and 216010 were empty; cleanup errors were empty. Fresh
HOST-shared/UUID-exclusive compute and scoped inventory was empty at
2026-09-09T21:53:19Z. Evidence: `/Users/tmc/tmp/nccl-go-testing/cancel-20260909-02/`
(`run.json`, `cancel-tests.log`, `cli-signal.log`, `cli-0.json`, `cli-2.json`,
`release.json`). Scope is one physical rank, ordinary and graph cancellation.
The later deregistration/destruction failure-retention change is outside this
frozen build and has host failure-injection evidence only.


A subsequent source review found that failure to abort a partially initialized
communicator clique did not prevent releasing rank resources. The current
source marks all ranks retained if any abort fails, while still attempting
other non-null communicators. Tests now check retained ranks on failure and
normal release eligibility on success. These changes are untested while the
Mac Metal quiet hold is active; no new binary or physical result includes them.


The repaired host batch passed all three packages at 2026-09-09T22:28:19Z,
including spread counts/routing/snapshot tests, comparator spread matching,
and cleanup failure-injection tests. Source hashes were unchanged during the
run; process group 91403 was empty afterward. Evidence:
`/Users/tmc/tmp/nccl-go-testing/host-repaired-20260909/result.json` and
`test.log`. No GPU tests ran in this CPU-only batch. The spread changes and
new cleanup failure paths still lack physical qualification.


## Spread and cleanup qualification, 2026-09-11 PDT

Frozen campaign `spread-20260911-01`, runner `c0e9fda2`, passed native registered
buffers, ordinary/graph cancellation, generated-spread sweeps, mutation
isolation, and omitted-transfer rejection. Upstream comparisons for spreads
0, 0.5, and 1 each passed six ordered samples at 16/128/1024 bytes over two
cycles: 18 samples and 3,504 checked values total, zero wrong values. Requested
spread, reference child environment, Go flags, and effective cases were checked.

The campaign exited zero in 14.8 seconds. All four owned process groups were
empty, with no cleanup errors. Fresh HOST-shared/UUID-exclusive compute and
scoped inventory at 2026-09-12T00:40:54Z was empty; locks were released and the
coordinator received RELEASE. Evidence is under
`/Users/tmc/tmp/nccl-go-testing/spread-20260911-01/`: `build.json`, `manifest.json`,
`run.json`, `native-tests.log`, the three `spread-*/comparison.json` receipts,
and `release.json`.

This includes the latest cleanup implementation's successful native paths;
forced deregistration/destruction/abort failures have host injection evidence.
One physical rank cannot distinguish distributed spread routing. The four-rank
host fixtures distinguish those counts and segments, but do not prove transport
or multi-GPU equivalence. No performance-equivalence claim.

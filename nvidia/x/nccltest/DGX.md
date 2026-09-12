# DGX NCCL comparisons

Status: baseline validated; full goal remains active.

Host: `spark-31db`, Linux arm64, one NVIDIA GB10. NCCL header/runtime 2.31.2,
CUDA 13.0.88, Go 1.26.7. The native library and executable SHA-256 hashes are
in `environment.txt`; each process attempt also records its executable hash.

Reference: NVIDIA/nccl-tests `b4d5beebca8a76cf01335f724d154b9b9d394d96`.
Every transferred reference source file was checked against `reference.sha256`.
The original baseline's Go driver/NCCL/harness snapshot is recorded by
`candidate.sha256`; later milestones have their own source and binary manifests.

Canonical evidence: matrix-03, with its twelve gather/scatter rows superseded
by matrix-04 after selecting the native NCCL entry points instead of the older
send/receive fallback. Earlier matrices remain retained, including matrix-01's
empty-reduction-column parser failure. Do not count superseded rows twice.

- **129 checked cases; 1,431,384 value comparisons; zero errors.**
- Eleven host operation patterns, 16/1024/65536-byte requests, sum/prod/min/max/avg
  where applicable, and every buffer mode upstream validates.
- All canonical cases used one actual rank and NCCL runtime version 23102.
- Physical Go testing: all collective subtests passed. The omitted-collective
  negative control rejected 16,384/16,384 values.
- All eleven Go benchmarks passed with fixed 20-iteration CUDA-event metrics.
- Local race tests and vet passed for the harness, commands, and NCCL package.

## Observed 64 KiB float32 sum timings

One matched attempt per row, out of place. These are observations, not confidence
intervals or a claim of performance equality. Short cases show scheduling noise.

| Operation | Native µs | Go µs | Go/native |
|---|---:|---:|---:|
| all_reduce | 2.660 | 2.905 | 1.092 |
| all_gather | 2.640 | 13.143 | 4.978 |
| broadcast | 2.640 | 2.900 | 1.098 |
| reduce_scatter | 2.660 | 3.465 | 1.303 |
| reduce | 2.720 | 2.896 | 1.065 |
| alltoall | 2.630 | 3.155 | 1.200 |
| alltoallv | 6.320 | 6.159 | 0.975 |
| scatter | 2.660 | 2.897 | 1.089 |
| gather | 2.680 | 2.897 | 1.081 |
| sendrecv | 6.380 | 6.177 | 0.968 |
| hypercube | 2.760 | 2.903 | 1.052 |

## Remaining proof and implementation

This DGX has one GPU. The tests establish local buffer/ABI behavior and reporting
for the listed cases. Single-rank reductions are identity operations; these runs
do not prove multi-rank reduction arithmetic, inter-GPU routing, or transport.
CPU multi-rank oracle/negative-control tests are separate evidence.

The follow-ups below extend this baseline with pre-multiplied sum, aggregation,
and datatype work. [COVERAGE.md](COVERAGE.md) tracks the remaining requirements.
Keep the goal active until the full scope and hardware boundary are resolved.

Raw receipts are retained locally at `~/tmp/nccl-go-testing/receipts` and on the
DGX at `/home/tmc/tmp/nccl-go-testing/receipts`. The source is in the isolated
worktree `~/tmp/nccl-go-testing/worktree`, branch `codex/nccl-go-testing`; it has
not been copied over unrelated changes in the shared main tree.

## Aggregation follow-up

`Options.Aggregation` and both commands' `-aggregation` flag now implement the
reference `-m` behavior. Timed iterations open one outer NCCL group when the
aggregation is greater than one, submit that many collectives, and close the
group. Warmup and correctness passes remain separate. Event time is divided by
`Iterations * Aggregation`. Zero library-option aggregation means one; invalid
or overflowing counts are rejected before native execution.

DGX `aggregation-02` verified all eleven host operation patterns and their
nineteen checked buffer modes at 64 KiB float32, sum where applicable,
20 iterations, aggregation 4, five warmups, and two correctness passes. Every
case reports 80 timed operations. **622,592 value comparisons passed.**
`aggregation-default-01` separately passed the two all-reduce buffer modes at
aggregation 1. `go-aggregation-test.log` records the physical Go test.

The comparison tool now parses and checks upstream's printed thread, GPU,
warmup, iteration, aggregation, and validation controls. It also requires the
Go result to report the expected iteration, aggregation, warmup, check, and
operation counts. This prevents unlike benchmark controls from passing solely
because both runs report zero wrong values.

`aggregation-source.sha256` and `aggregation-binaries.sha256` retain the DGX
source and executable provenance. These receipts extend the earlier baseline;
they do not prove additional datatypes, physical multi-rank communication,
advanced graph/RMA/device modes, or equal performance.

## Pre-multiplied sum follow-up

Float32 `mulsum` now follows `verifiable/verifiable.h`: even communicator ranks
use scalar 1, odd ranks use scalar 2. Each collective creates a host-immediate
custom operator and destroys it after the collective API returns, before any
outer `GroupEnd`, as upstream does. The Go scalar is pinned through the FFI call;
NCCL copies it before returning.

Retained checks:

- `mulsum-01`: all-reduce, reduce, and reduce-scatter; three message sizes;
  both buffer modes. Eighteen cases, 199,728 value comparisons, zero errors.
- `mulsum-pinned-01`: the six 64 KiB aggregated cases, four collectives per
  iteration and 80 timed operations per case. 196,608 comparisons, zero errors.
  This supersedes `mulsum-aggregation-01` for the scalar-pinning path.
- `go-mulsum-final.log`: physical Go tests for all three pre-multiplied-sum
  collectives, plus aggregation and the omitted-collective negative control.
- Local race/vet checks, a runnable example, and an oracle test that rejects
  ordinary sum 32 when weighted sum 44 is required for three ranks.

`mulsum-source.sha256` and `mulsum-binaries.sha256` retain DGX provenance.
The DGX still has one GPU: its rank-zero scalar is 1. The native runs establish
custom-operator creation, use, destruction, and grouped lifetime behavior, but
not physical odd-rank multiplication or multi-rank weighted reduction. Those
remain separate from the CPU oracle evidence.
## Comparator environment regression

`environment-01/comparison.json` retains three 65,536-byte sum cases with the
stricter comparator: all-reduce out/in-place and all-to-all-v out-of-place.
All passed with 98,304 Go value comparisons and zero wrong values. These are
regression cases, not additional unique operation coverage.

`environment-rejection.log` records rejection of
`NCCL_TESTS_ALLTOALLV_SPREAD=0` with exit status 2; the requested output directory
was absent afterward. The comparator rejects reference-only environment
overrides before launching either child. It also checks the Go result's Case,
including operation, reduction, requested bytes, root, and buffer mode.
`environment-source.sha256` and `environment-binaries.sha256` identify the
tested comparator source and both executables. Local race tests and vet pass.

## Float64 coverage

`double-01/comparison.json` contains 147 checked cases across all eleven host
operations, all six reductions where applicable, 16/1024/65536-byte requests,
and every reference-checked buffer mode. Its 225 child processes succeeded;
815,556 Go value comparisons passed with zero wrong values. Every case used
one rank and NCCL runtime 23102. Commands include `-datatype double` for Go
and `-d double` for upstream. This extends datatype coverage; it does not
establish physical multi-rank arithmetic or transport.

Float64 inputs contain fractions below float32 precision. Transfers, native
collective arguments, in-place and peer offsets, layout alignment, bandwidth,
and pre-multiplied-sum scalar storage use the selected element width. The host
oracle computes in float64, and unit controls reject float32-rounded results.
The checker now rejects nonfinite expected values instead of allowing an
infinite tolerance to accept a finite wrong result. Large-rank products that
overflow remain unsupported pending bounded input generation.

`double-01-source.sha256` and `double-01-binaries.sha256` identify this first
matrix's snapshot. Subsequent source changes remove internal float32 wrappers,
reject nonfinite oracle values, and improve invalid-datatype diagnostics.
`double-final-source.sha256` and `double-final-binaries.sha256` identify the
follow-up snapshot. `double-environment.txt` records the host, GPU UUID, tools,
and NCCL library hash.

On the follow-up snapshot, `go-double-tests.log` records passing native tests
for both datatypes and all eleven host paths. The omitted-operation controls
reject 16,384/16,384 float32 values and 8,192/8,192 float64 values.
`go-double-benchmarks.log` records 22 passing fixed-20-iteration benchmarks
(eleven per datatype). These CUDA-event observations do not establish equal
performance; the paired matrix retains the actual timing ratios.

The follow-up snapshot also passed `double-aggregation-01`: 49 checked cases,
all eleven operations and six applicable reductions at 65,536 bytes with
aggregation four, 80 timed operations per case, and 802,816 value comparisons.
`float-regression-01` passed the corresponding 49 float32 cases with aggregation
one and 1,605,632 value comparisons. Neither run reported wrong values. Report
these controls separately; repeated operation/size rows are not new unique
operation coverage. The final harness source manifest matches the local worktree.
No test process or GPU compute process remained after this sequence completed.

## Integer coverage

The implementation now supports signed and unsigned 8-, 32-, and 64-bit
integers. `CheckBytes` compares their bit patterns without floating conversion.
The arithmetic follows `verifiable/verifiable.cu` in the pinned reference:
width-wrapped sums/products, signed min/max ordering, and division of the
wrapped sum for average. Complemented expected outputs replace floating NaN
poisoning for integer buffers.

The six `integer-<datatype>-01/comparison.json` receipts passed all 882 checked
cases and 17,942,232 Go value comparisons. Each datatype covered eleven host
operations, six applicable reductions, three sizes, and supported buffer modes.
All 1,350 child processes succeeded; every case used one rank and NCCL 23102.

| Datatype | Checked cases | Value comparisons | Wrong |
| --- | ---: | ---: | ---: |
| int8 | 147 | 6,524,448 | 0 |
| uint8 | 147 | 6,524,448 | 0 |
| int32 | 147 | 1,631,112 | 0 |
| uint32 | 147 | 1,631,112 | 0 |
| int64 | 147 | 815,556 | 0 |
| uint64 | 147 | 815,556 | 0 |

`integer-source.sha256`, `integer-binaries.sha256`, and
`integer-harness.tar.gz` preserve that snapshot. All 18 archived Go-file hashes
match the source manifest. `integer-verification.json` records the receipt audit.

The latest-source follow-up passed 36 integer pre-multiplied-sum cases with
aggregation four: all six types, all three reduction operations, and both
buffer modes at 65,536 bytes. It checked 2,162,688 values with zero errors and
80 timed operations per case. Twelve float32/float64 product regression cases
also passed. These are separate controls, not additional unique operations.

`go-integer-tests.log` records 152 passing physical collective modes across all
eight supported datatypes and eight passing omitted-operation controls.
`go-integer-benchmarks.log` records 88 passing fixed-20-iteration benchmarks.
The optional CUDA-header codec test skipped in this invocation; its separate
explicit validation is described below. The follow-up script verified exit
status 2 for root one in a one-rank communicator before an output directory was
created; `integer-invalid-root.log` retains the rejection message. Physical
nonzero-root execution remains unmeasured.

`integer-final-source.sha256`, `integer-final-binaries.sha256`, and
`integer-final-harness.tar.gz` preserve the follow-up snapshot, including bounded
floating-product inputs and comparator root selection. The Go source hashes
match the local worktree. `integer-final-verification.json` records the audit;
`integer-environment.txt` records host/device/tool and library information.
Local race tests and vet pass. No test or GPU compute process remained after
this sequence completed. Physical multi-GPU arithmetic and transport remain open.

## Remaining floating-format codecs

The private half, bfloat16, E4M3, and E5M2 codecs passed 1,992,584 encoding and
1,992,584 decoding checks against CUDA's host routines, covering 498,146 distinct
double input patterns. These include all representable finite values, rounding
boundaries and their neighbors, both signs, overflow, infinities, and NaNs.
Non-NaN results agree bit for bit; NaNs agree by classification.

The first fixture used local CUDA 13.3 headers and Clang. The same source and
input corpus then ran on the DGX with its CUDA 13.0 headers and GCC; the entire
output file matched byte for byte. `cuda-float-codec-final.log`,
`cuda-float-codec-provenance.json`, `cuda-float-codec-dgx.txt`, and
`cuda-float-codec-dgx-headers.json` retain results, commands, and hashes, including
20 CUDA header dependencies on each host. Raw fixtures remain under
`~/tmp/nccltest-codec/cuda-1275101908` locally and
`~/tmp/nccl-go-testing/float-codec-dgx` on the DGX.

This was CPU-only conversion validation. These four datatypes are not yet
accepted by Case or the commands. [SMALLFLOAT.md](SMALLFLOAT.md) records the
remaining runner/oracle integration and physical-validation requirements.

## Reduced-precision native tests, 2026-09-09

The frozen `native-20260909` binary passed 76 collective modes across half,
bfloat16, E4M3, and E5M2, with 7,471,104 comparisons and zero wrong values.
All four omitted-operation controls rejected every poisoned receive element.
These runs used one physical GB10 rank. They do not prove inter-GPU routing
or multi-rank reduction arithmetic.

The subsequent matched half matrix exceeded its 100-second process limit.
The campaign exited 1; remaining datatype matrices and stream/registration
tests were not executed. This incomplete matrix receives no equivalence credit.
The immediate post-kill inventory still listed a terminating GPU process;
a fresh probe under both locks verified empty compute inventory and no owned
process-group members before the cleanup release.

Raw logs, binary/source hashes, `run.json`, and `release.json` are retained in
`/Users/tmc/tmp/nccl-go-testing/native-20260909/`. The next matrix uses a new
receipt directory and a larger bounded timeout; previous results are preserved.

## Half comparison and stream/registration follow-up

`half-20260909-02` completed all 147 half cases with 3,262,224 comparisons,
225 successful child processes, and zero reported mismatches. The default-stream
physical test passed. The following registration test failed because it expected
an omitted one-rank in-place sum to be detectable, although its initial input
already equals the expected output. Both positive registration runs had returned
success before that assertion. The campaign still exited 1 and remains retained
as failed; this diagnosis does not rewrite its outcome.

The corrected test retains positive in-place/out-of-place coverage and requires
all 32,768 comparisons in the two-pass disjoint-buffer omission control to fail.
A host test verifies the identity case and poison distinction. The corrected GPU
test has not yet run. Both cleanup groups and fresh locked compute inventory
were empty before release. Evidence is retained in
`/Users/tmc/tmp/nccl-go-testing/half-20260909-02/`.

## BF16 comparison and corrected registration control

`bfloat16-20260909-01` passed 78 requested workloads, 147 matched cases,
3,262,224 comparisons, and 225 child processes. The structured receipt has
`Complete: true`, zero comparison errors, and no failed attempts. Both the
default-stream test and corrected registered-buffer test passed. The runner
exited zero, with empty owned process groups and no cleanup errors; a fresh
probe under both locks confirmed empty compute inventory before release.

The corrected registration omission control covers disjoint receive buffers;
one-rank in-place identity is not credited as omission detection. These tests
still exercise only one physical rank. Receipts, frozen source/binary hashes,
and logs are in `/Users/tmc/tmp/nccl-go-testing/bfloat16-20260909-01/`.
Complete matrices for E4M3 and E5M2 remain pending.

## E4M3 matrix stopped at upstream average

`f8e4m3-20260909-01` completed 12 of 78 requested workloads before
upstream `all_reduce_perf` exited 3 on the 16-byte average case. Its log
reports NCCL `invalid argument` at `all_reduce.cu:507`. The comparison
receipt remains `Complete: false`; this run does not establish full E4M3
matrix equivalence. No retry or remaining matrix was launched.

The runner exited 1, with its owned process group empty and no cleanup
errors. A fresh HOST-shared then UUID-exclusive probe at
2026-09-09T20:39:54Z confirmed empty compute inventory and owned process
group output. Locks were released and the DS4 owner received the explicit
cleanup callback. Raw logs and receipts are retained in
`/Users/tmc/tmp/nccl-go-testing/f8e4m3-20260909-01/`.

The pinned NCCL source translates FP8 average to `ncclDevPreMulSum`
(`src/enqueue/enqueue.cc`, `hostToDevRedOp`). Its single-rank path calls
`ncclLaunchOneRank`, whose FP8 premultiplication cases are conditional on
`__CUDA_ARCH__ >= 900` in `src/device/onerank.cu`. This is a candidate
explanation for the observed invalid argument, not proof of the installed
library's build configuration. Ordinary FP8 collective tests used the default
sum reduction, so their success does not contradict this failure.

A separate, unlaunched diagnostic is frozen in
`/Users/tmc/tmp/nccl-go-testing/fp8-dispatch-20260909-01/`. It compares Go
and reference sum, average, and premultiply calls for both FP8 formats with
`NCCL_DEBUG=WARN`. Sum must succeed; average/premultiply exits and logs are
retained diagnostically, never counted as numerical passes. The comparator
continues to require every requested workload to pass without exclusions.

## Paired FP8 dispatch diagnostic

`fp8-dispatch-20260909-01` completed all twelve children in 13.5 seconds.
For both E4M3 and E5M2, the 16-byte one-rank all-reduce sum succeeded in Go
and upstream. Average and premultiply failed in both implementations with
NCCL invalid argument; Go retained error code 4 and zero comparisons.
These are matching error observations, not successful numerical comparisons.
`NCCL_DEBUG=WARN` added no more specific diagnostic. The source-level dispatch
explanation remains a hypothesis about the installed build.

The capture runner exited zero because all planned diagnostic calls completed;
the eight failing children remain explicitly recorded. All twelve owned groups
were empty, cleanup errors were empty, and a fresh probe under both locks at
2026-09-09T20:48:23Z confirmed empty compute and owned-group inventories.
The lane was released with no further launch queued. Raw logs and `run.json`:
`/Users/tmc/tmp/nccl-go-testing/fp8-dispatch-receipts-20260909-01/`.
The frozen runner, source snapshot, manifest, and fresh `release.json` are in
`/Users/tmc/tmp/nccl-go-testing/fp8-dispatch-20260909-01/`.

## Graph qualification, 2026-09-09

`graph-20260909-01` passed graph replay tests for all eleven operations and
the omitted-collective control. The subsequent upstream comparison completed
11 workloads, 19 cases, and 30 successful child processes: 19,456 checked
elements, including 9,728 replay comparisons, with zero mismatches. This
qualifies the measured one-rank float/sum/1024-byte configuration with three
replays. It does not qualify other datatypes/reductions or multi-device capture.

The campaign exited zero in 44.3 seconds. Both owned groups were empty,
cleanup errors were empty, and a fresh locked compute/scoped-group inventory
at 2026-09-09T20:52:22Z was empty. The lane was explicitly released with no
additional work queued. Frozen sources/binary hashes, raw logs, comparison
receipt, and release receipt: `/Users/tmc/tmp/nccl-go-testing/graph-20260909-01/`.
No performance-equivalence claim is made.

## Offset qualification, 2026-09-09

`offset-20260909-01` passed physical positive and disjoint-buffer omission
tests for one-, two-, four-, and eight-byte types at offset three. The matched
float/sum/1024-byte matrix passed all eleven operations and nineteen cases
with 9,728 comparisons and zero mismatches. This is single-rank evidence.
Both process groups were empty, cleanup errors were empty, and a fresh
locked compute/scoped-group inventory at 2026-09-09T20:59:20Z was empty.
The lane was explicitly released. Raw logs, frozen hashes, comparison and
release receipts: `/Users/tmc/tmp/nccl-go-testing/offset-20260909-01/`.
The later host-only allocation preflight and duplicate-banner checks are
not in this frozen binary; they passed separate local tests.

## Cycle qualification, 2026-09-09

`cycles-20260909-01` passed physical ordinary, graph, and registered-offset
cycle tests and callback cancellation after the second cycle. The upstream
comparison passed all eleven operations with three cycles per mode: 57 cases,
29,184 comparisons, zero mismatches, and complete terminal summaries.
This covers a single physical rank, float/sum, and a 1024-byte case.

The campaign exited zero in 40.1 seconds. Both owned groups and a fresh
locked compute/scoped inventory at 2026-09-09T21:11:02Z were empty. The lane
was explicitly released. Frozen sources/binary hashes, logs, and receipts:
`/Users/tmc/tmp/nccl-go-testing/cycles-20260909-01/`. Later local comparison
overflow and clique-drain changes are not in this binary; those have only
host test evidence. Shared-communicator size sweeps remain unqualified.

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


## Synchronized iterations, 2026-09-09

Frozen runner `c6a5e919fd00c26d2f0e47b5ce2ae506a8fa7e37c47f0998da596a49d0d72931`
and binaries in `sync-20260909-01/manifest.json` passed native synchronized
aggregation-one/four tests, followed by all 11 operations at float/sum,
1024 bytes, aggregation four, against upstream `-z 3`. All 19 cases passed:
9,728 checked values, zero wrong values, 30 successful children. Runner
exit zero; elapsed 39.1 seconds is campaign duration, not a performance claim.
Owned groups 212569 and 212587 were empty; cleanup errors were empty.
Fresh HOST-shared/UUID-exclusive inventory at 2026-09-09T21:43:22Z was empty.

Evidence: `/Users/tmc/tmp/nccl-go-testing/sync-20260909-01/run.json`,
`compare-sync/comparison.json`, and `release-repaired.json`. The earlier
pre-transfer shell-quoting failure remains in `ssh.log` and `release.json`.
The repaired wrapper uses shell argument quoting and passed local syntax,
argument-round-trip, and frozen-hash checks before the fresh handoff.
Single physical rank only; CUDA-event and reference host timing definitions
differ, so these results do not establish timing or transport equivalence.


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

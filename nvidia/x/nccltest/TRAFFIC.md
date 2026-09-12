# All-to-all-v traffic controls

Source audit: pinned nccl-tests `b4d5beebca8a76cf01335f724d154b9b9d394d96`,
`src/alltoallv.cu`. This describes missing implementation requirements;
it is not evidence that custom traffic is supported.

## Generated traffic

The current Go path implements the default spread of one. Upstream lines
361–370 blend uniform and distance-weighted byte counts using a spread in
[0, 1], then truncate each peer's bytes to a multiple of 16. The requested
size is first divided across ranks and aligned; peer counts use that effective
total, not the original requested byte count. With one rank every spread has
the same count, so one-rank correctness cannot distinguish spread controls.

Adding spread must update layout totals, source-element lookup, NCCL send/recv
counts, bandwidth, CLI controls, and comparator provenance together. The
comparator currently rejects reference-only environment overrides; a supported
spread flag must set the matching upstream environment explicitly and record
its value. Do not simply relax the environment rejection.

## Matrix traffic

Upstream lines 397–403 set ParamCount to zero in matrix mode. Lines 381–393
compute independent maximum send and receive counts across all ranks. This
is incompatible with the Go assumption that a positive ParamCount establishes
nonzero work. The matrix's actual transferred elements must establish work.

Rank send/receive totals can differ. Matrix support therefore needs per-rank
counts and prefix offsets, while allocation uses the maxima. The current
source-element fallback returns rank zero/element zero after the received
segments; it must not be used for an inactive tail. Upstream lines 420–423
zero the full receive and expected buffers, including tails. The Go checker
must distinguish transferred elements from untouched tails, and omission
controls must poison real receive segments without crediting inactive tails
as successful communication.

Matrix input is square, can be larger than the communicator, and uses its
leading rank-sized submatrix. Upstream parses nonnegative byte values and a
scale, checks overflow, and aligns each cell to 16 bytes. Matrix mode forbids
size sweeps and requires allocation capacity at least as large as every row
and column total. Parsing, scale arithmetic, alignment, capacity checks, and
matrix provenance must be tested independently before native execution.

## Required evidence

Use asymmetric host fixtures with different row/column sums, an inactive rank,
a self edge, and nonzero off-diagonal edges. Check every source segment and
tail; include corruption and omitted-transfer controls. Generated fixtures
must distinguish spread zero, one-half, and one at multiple rank counts and
native element widths. Reject invalid shape, nonfinite/negative values,
overflow, undersized capacity, and unsupported matrix sweeps before CUDA.

The available single-GPU DGX can exercise matrix parsing, self traffic,
allocation, CLI routing, and cleanup. It cannot prove off-diagonal routing,
imbalance across devices, or custom-spread equivalence. These require distinct
physical ranks and remain part of the open qualification scope.


## Implementation in progress

`Case.Spread` now optionally selects generated traffic; nil preserves spread
one. The layout, source lookup, native send/receive counts, and bandwidth use
the selected value. `nccltest -spread` sets it explicitly. Invalid/nonfinite
values and use with other operations are rejected. The runner snapshots the
optional value and copies it into each result to isolate callback mutations.
Host fixtures distinguish spread zero, one-half, and one across four ranks;
they are added but unexecuted during the active Metal quiet hold.

The comparator now accepts `-spread` when `-operations alltoallv` is selected.
It sets `NCCL_TESTS_ALLTOALLV_SPREAD` only on the reference child, passes the Go
flag, and records the requested spread and explicit child environment in the
receipt. Ambient NCCL_TESTS overrides remain rejected. Decoded Case values
compare the optional spread by value, rejecting an omitted or changed value.
These edits and their tests remain unexecuted during the quiet hold. Matrix
support is unchanged and remains absent. No custom-spread physical result is
claimed.


Fixed byte-count fixtures now cover every receive segment at four ranks for
1-, 2-, 4-, and 8-byte elements, independently of the generated-count helper.
A physical test is added for all three spread settings, reused sweeps, caller
and result mutation isolation, and omitted-transfer rejection. These tests
remain unexecuted under the Mac quiet hold. Single-rank execution can validate
those runtime contracts but cannot distinguish the distributed patterns.


Generated layouts now reject negative or over-capacity peer counts before
accumulating them. A boundary test covers a 64-bit requested size whose weighted
double count rounds past the signed integer limit. This protects allocation
arithmetic; the test is added but has not run during the quiet hold.


The authorized host batch in `host-pending-20260909` failed only
`TestSpreadCountOverflow`; comparator and CLI package tests passed. On this
platform the out-of-range conversion did not produce a negative count, so the
post-conversion check was insufficient. Peer counts now reject the signed
integer boundary before conversion. The failed receipt is retained; the repair
has not been retested and requires a fresh CPU handoff.


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

# Buffer offsets

The `-offset` option matches upstream `-u`: it counts elements, not bytes.
Pinned nccl-tests common.cu shifts both ordinary send/receive buffer pointers
by `unalign * wordSize(type)` before the run (lines 948–949), then restores
them afterward. Host RMA and device paths add the equivalent byte offset to
registered-window offsets. Those paths are not yet implemented here.

The Go runner allocates enough space for the leading elements and payload,
retains allocation bases for freeing and registration, and shifts submission,
input, poison, and readback addresses. In-place rank offsets are added after
the leading offset. It rejects negative or overflowing allocation sizes before
CUDA initialization. The comparator also bounds offsets to avoid overflow in
upstream's C-int `8*unalign` padding expression, passes `-u`, and requires the
printed `unalign` control and Go result offset to match the request.

Host allocation and parser tests pass. `TestBufferOffsets` is prepared for
one-, two-, four-, and eight-byte types, in-place/out-of-place positives, and
shifted disjoint-buffer omission rejection. It has not yet run on CUDA.
Physical comparison against upstream remains pending. No misalignment
performance or multi-rank claim is supported.

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

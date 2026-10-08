# Producer facts and limits

The source inspected is NVIDIA/nccl commit
12df1a11afad322be5a204a2db890161cbf8131d (2.32.3):
plugins/profiler/inspector/{inspector.cc,inspector_json.cc,inspector_ring.cc,inspector_ring.h}.
Official immutable source: https://github.com/NVIDIA/nccl/tree/12df1a11afad322be5a204a2db890161cbf8131d/plugins/profiler/inspector

1. inspector.cc creates communicator records from supplied commHash/rank, while
   inspector_json.cc writes op->sn as coll_sn. These selected files do not prove
   same-ID recreation, split/shrink sequence-reset semantics or global identity
   uniqueness. Join requires the caller's explicit job/capture/epoch selection.
2. inspector_ring.cc overwrites oldest entries and increments dropped;
   inspector_json.cc snapshots total and since-last-dump counts. Counts alone
   cannot identify a specific missing sequence. Join retains every statistics
   record, never sums repeated cumulative counts or assigns them to an operation.
3. The ring drains completed entries, not a demonstrated cross-rank collective
   sequence order. Join accepts arbitrary input order and sorts facts itself.
4. Draining consumes ring entries; repeated dumping is not evidence that every
   record repeats. Re-reading or overlapping caller captures can repeat facts.
   Join retains each observation; differing same-rank operation/process facts
   are conflicts, not silently selected winners. Dump timestamps can differ.

Tests and examples are caller-constructed facts, not measured multi-GPU captures.
Required lane72 G22 same-run Inspector/RAS captures remain physical qualification.
No start/end times, synchronized clocks, rank-specific dropped operation, health
classification or transport identity is inferred.

Within the caller-selected epoch, a communicator/rank must retain its observed
host/PID across operations and statistics. A conflict refuses the input rather
than joining observations from different process identities. Membership storage
is bounded both per communicator and across all collective groups; a large rank
count cannot multiply into unbounded unobserved-rank slices over many sequences.

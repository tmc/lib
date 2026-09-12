# Synchronized timed iterations

`Options.SyncIterations` and command `-sync-iterations` synchronize every
rank's stream after each aggregated timed iteration. The comparator passes
upstream `-z 3` and requires the Go result to record the requested mode.

In pinned nccl-tests `common.cu:765–770`, mode three closes the aggregation
group, synchronizes streams, then performs a host barrier. The Go harness
currently uses one host thread, so it has no cross-thread barrier. Warmup
and correctness retain their separate existing synchronization boundaries.
Graph replay is rejected for synchronized iterations; upstream also disables
this combination. Modes one and two, which synchronize within `startColl`,
are not implemented.

Upstream mode three accumulates host elapsed time around each iteration.
The Go runner retains CUDA-event timing across the synchronized sequence.
These are different timing definitions: matching correctness under this
control does not establish performance equivalence.

Host tests are added for group/end/synchronization ordering, synchronization
failure stopping later submissions, and graph incompatibility before CUDA.
`TestSynchronizedIterations` is added for aggregation one and four. The host tests passed in the bounded CPU-only run recorded at
`/Users/tmc/tmp/nccl-go-testing/sync-host-20260909.log`. Physical synchronized
iteration tests remain unexecuted. The first transfer wrapper failed shell
parsing before transfer or launch; its failure and empty locked inventory
are retained in `sync-20260909-01/ssh.log` and `release.json` under the same
scratch root. The repaired wrapper passes shell syntax, argument round-trip,
and frozen-file hash checks locally; it requires a fresh remote handoff.


The repaired launch passed on DGX: native aggregation-one/four tests and all
11 operations at float/sum, 1024 bytes, aggregation four. The comparator
retained 19 matched cases, 9,728 checked values, zero wrong values, and 30
successful child processes. Both owned groups were empty after cleanup;
a fresh locked inventory was empty at 2026-09-09T21:43:22Z. Receipts are in
`/Users/tmc/tmp/nccl-go-testing/sync-20260909-01/` (`run.json`,
`compare-sync/comparison.json`, and `release-repaired.json`). This proves
only the stated single-rank correctness configuration.

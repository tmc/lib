# CUDA graph execution contract

Source audit: nccl-tests b4d5beebca8a76cf01335f724d154b9b9d394d96,
`src/common.cu`, `BenchTime` at lines 722–824. Graph capture/replay is implemented locally but has not yet passed
physical qualification. Both commands accept `-graph-launches`; the comparator requires matching
reference `-G` controls and explicit replay correctness counts.

The reference warms up and synchronizes before capture. It starts thread-local
capture on every device stream, captures all `iterations * aggregation`
collectives, ends every capture, and instantiates one executable per device.
It then launches each executable `-G` times. Reported per-operation duration
divides by iterations, aggregation, and graph launches. Capture and
instantiation are outside the reported replay duration. The existing Go
`TimedOperations` calculation must therefore include the replay multiplier;
merely replaying one collective per graph would not match the reference.

Available driver bindings are `Stream.BeginCapture`, `Stream.EndCapture`,
`Graph.Instantiate`, `GraphExec.GraphLaunch`, and both graph destruction
methods. `StreamCaptureModeThreadLocal` is available. The reference uses this
mode because P2P connection setup can allocate during capture without warmup.

Implementation must retain two distinct correctness checks: ordinary submission
and replayed submission, resetting poison before each. A successful ordinary
collective does not establish that a captured graph contains the operation.
The omission control must exercise the replay path as well. Check arithmetic
from a single replay independently of repeated in-place timing, whose inputs
are mutated by repeated reductions.

For each rank, retain capture state, graph, and executable separately. If setup
fails after some streams begin capture, end all begun captures in their owning
contexts before stream cleanup. Synchronize replay before destroying executable
graphs, then destroy source graphs before registrations, communicators, buffers,
and streams. Reject default-stream capture before initializing CUDA.

Required validation includes positive replay operation counts, overflow checks
for the additional multiplier, cleanup failures, omitted graph work, and matched
physical cases. Single-rank replay tests do not prove multi-device capture or
transport. No graph equivalence or performance claim is currently supported.

Replay correctness counts are exposed as `GraphComparisons` and `GraphWrong`
and are also included in total `Comparisons` and `Wrong`. This distinguishes
ordinary submission evidence from replay evidence. Invalid graph counts and
default-stream capture are rejected before CUDA initialization. If cleanup
leaves any graph live, all ranks retain their resources because a captured
collective may reference peer buffers.

`TestGraphCollectives` covers all eleven operation paths and verifies the
iteration/aggregation/replay multiplier. `TestGraphMissingCollective` requires
both ordinary and replay checkers to inspect and reject every poisoned receive
element; a capture or launch error alone cannot satisfy it. These physical tests
are added but remain unexecuted.

Host validation on 2026-09-09: uncached `GOMAXPROCS=2 go test -p=1
-count=1 ./x/nccltest/...` passed with native CUDA and codec gates disabled.
This covers graph option validation, retained-resource cleanup checks, reference
control parsing, and missing/mutated replay receipt rejection. Log:
`/Users/tmc/tmp/nccl-go-testing/graph-host-20260909.log`. It does not exercise
CUDA capture, replay, or the physical omission control.

Additional host checks: `go vet -p=1 ./x/nccltest/...` passed. A five-second
single-worker `FuzzParseReference` run completed 282,163 executions without
failure after parsing became strict about the reference graph-launch field.
This fuzz target checks parser robustness and basic accepted-row invariants;
the table tests separately verify malformed graph controls and replay-count
receipt mutations. Neither check establishes native graph behavior.

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

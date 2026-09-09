# Optional NCCL operations in the portable CUDA capture

Status: completed (software, CLI, oracle, error handling, metadata, and pprof reporting); multi-GPU physical execution deferred (host `dgx` possesses exactly 1 physical GPU, NVIDIA GB10). Priority: P2. Dependency: plan 001's public capture API. Planned on 2026-09-09.

## Outcome and limits

Extend `examples/ptx-sass-trace` with optional, single-process, single-host
NCCL AllReduce and AllGather operations. Keep the existing CUDA pipeline and
single-file `capture.pprof`. The default invocation remains the existing
single-GPU workload and must not load NCCL or initialize communicators.

Build this after [plan 001](001-ptx-sass-trace.md), as a separate increment.
It must not delay or weaken plan 001's default-workload acceptance gates.
Do not add NCCL types or workload fields to `x/cudacapture`. A future NCCL-only
example can use the same public API, but is outside this increment.

Multi-node launch, MPI, network bootstrap, CUDA graph capture, overlapping
compute/communication, communicator recovery, and MIG qualification are
deferred. CUPTI PC sampling and hardware-counter collection belong to
[plan 003](003-cupti-feature-coverage.md), with integration rules below. The first hardware
qualification uses distinct physical GPUs. A hostname such as `dgx` is not
evidence that two usable GPUs exist; inspect visible device identities first.

## Source baseline and references

Paths are relative to `nvidia/` unless noted otherwise:

- `nccl/funcs.gen.go`: `CommInitAll`, `GroupStart`, and `GroupEnd` bindings.
- `nccl/comm.gen.go`: `AllReduce`, `AllGather`, rank/device/count queries,
  `GetAsyncError`, `Abort`, and `Destroy`.
- `nccl/handles.gen.go`: buffer and stream handles do not own allocations.
- `nccl/comm_array_test.go`: caller-owned communicator array ABI regression.
- `nccl/initall_physical_test.go`: opt-in two-GPU AllReduce, distinct UUIDs,
  pinned arrays, context switching, full result checking, and bad-oracle control.
  This is useful source evidence, not a claim that its physical test passed here.
- `x/cuda/context.go`: `Context.Run` preserves the current context on a locked
  OS thread. `x/cuda/stream.go` and `event.go` provide completion queries.
- `examples/ptx-sass-trace/`: current capture, validation, and timing behavior.

Before implementation, check git state and both source baselines from the
repository root. Plan 001 will intentionally change some recorded files;
review those differences and record a new implementation baseline without
replacing the historical hash files.

```sh
git status --short
shasum -a 256 -c plans/001-ptx-sass-trace.baseline.sha256
shasum -a 256 -c plans/002-optional-nccl-capture.baseline.sha256
```

NVIDIA documents that each communicator rank maps to a distinct CUDA device;
`CommInitAll` initializes local communicators in one process. See
[Creating a communicator](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html).
Single-thread multi-device collectives require grouping, and grouped enqueue
must finish before stream synchronization. See
[Group calls](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/groups.html).
AllReduce returns the reduction to every rank; AllGather returns rank-ordered
concatenation. See
[Collective operations](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html).
Recheck these rules against the installed NCCL version during implementation.

## CLI and availability contract

Add these flags to the existing example:

| Flag | Proposed behavior |
| --- | --- |
| `-nccl=off` | Default. Allowed values: `off`, `allreduce`, `allgather`, `both`. |
| `-devices=0,1` | Required when NCCL is enabled; ordered visible CUDA ordinals, with rank equal to list position. Reject when NCCL is off to avoid an ignored flag. |

Reuse `-elements` as the number of input elements per rank and `-passes` as
measured repetitions per selected collective, with one unmeasured warmup per
collective. `-rounds` continues to control only the CUDA compute stage.
When enabled, run the original CUDA pipeline on the first selected device,
then run the NCCL operations; do not implicitly multiply the CUDA workload
across every GPU. With `-nccl=both`, use AllReduce then AllGather consistently
on all ranks. Reset inputs and receive sentinels before each operation.

Reject malformed lists, duplicate/out-of-range ordinals, fewer than two
selected devices, unknown modes, overflowed sizes, and unexpected positional
arguments. Check resolved device UUIDs and reject duplicate identities even
if the ordinal strings differ. Record visible ordinal-to-UUID/PCI mapping;
`CUDA_VISIBLE_DEVICES` may remap ordinals. Exclude MIG for this first gate.

An explicit NCCL request on a host without sufficient devices, the NCCL
library, or required symbols must fail clearly with nonzero status. Do not
silently downgrade it to the single-GPU capture or manufacture samples.
Perform preflight before creating output files where possible. Default mode
and `-h` must work without NCCL installed; retained recordings must remain
viewable without NCCL or CUDA. Runtime loading must remain lazy.

## Ownership and enqueue sequence

Use the existing public NCCL bindings, with a small example-local per-rank
owner containing device/context, communicator, stream, buffers, and events.
Keep calls explicit; no general collective runner interface or new wrapper
package is needed. Reuse the public CUDA owners where possible and make
raw-handle conversions explicit at the NCCL call boundary.

1. Validate all selected devices and buffer sizes. Retain each primary context,
   then create per-device streams, timing events, and out-of-place buffers.
2. Allocate communicator output storage for all ranks before `CommInitAll`.
   Follow the existing ABI test's stable/pinned array pattern for Go memory
   passed to native code. Verify non-null handles and communicator rank,
   device, and count against the requested mapping. Track partial ownership.
   Preserve the caller's thread-local CUDA context around communicator setup
   as well as the grouped calls; do not assume initialization leaves it intact.
3. Prepare rank-specific input and receive sentinels. Finish uploads before
   measured work. Do not include allocation, initialization, or correctness
   checking in the event interval.
4. Lock one OS thread across the complete `GroupStart`/`GroupEnd` section.
   Record a start event on each rank's own stream before opening the group.
   Enqueue the same collective once per rank, using matching count/type/op
   and that rank's buffers and stream. Keep context-sensitive calls in the
   correct primary context. Pair each successful group start with one group
   end, retaining both enqueue and group-end errors.
5. Only after successful group completion, record each stop event on its
   matching stream. Observe completion for all ranks, checking communicator
   async errors while waiting. Do not wait for rank 0 before enqueueing rank 1.
6. Download and validate every result on every rank, including warmup. Publish
   measured samples/results only after the corresponding operation validates.
   Publish the final capture only when the whole run and cleanup succeed.

The first implementation uses blocking communicator initialization/enqueue
semantics. Completion queries should be bounded, but an in-process timeout
cannot interrupt a blocked native initialization or `GroupEnd`. Run hardware
checks in a child process under a fixed external deadline (proposed: 120 s
per small/default invocation, declared before running). Retain timeout
failures; do not enlarge the deadline to turn a failed gate green.

On success, complete all queued work before destroying communicators and
freeing buffers, events, streams, and context references. On enqueue/async
failure or timeout, coordinate abort across all owned communicators before
releasing storage that NCCL may still use. Audit abort semantics for the
installed version. If completion/abort cannot be established, terminate the
isolated child rather than continue freeing live device storage. Do not
blindly synchronize a failed collective during deferred cleanup. Never
claim fault recovery from killing the child. No retries or recovery protocol
are part of this example.

## Workload and independent result checks

Use `int32` payloads for exact validation, independent of the floating-point
CUDA compute output. For zero-based rank `r` and element `i`, initialize:

```text
input[r][i] = 1024*(r+1) + (i mod 1024)
```

Compute expected values on the CPU using checked `int64` arithmetic, then
check representability in `int32` before launch. Validate all count/byte
multiplications before allocation, including `ranks * elements * 4` for
AllGather receive buffers. Report planned allocation sizes per device;
allocation failures must be ordinary errors with no final profile published.

| Operation | Receive elements per rank | Expected value |
| --- | --- | --- |
| AllReduce, `Int32`/`Sum` | `N` | `1024*R*(R+1)/2 + R*(i mod 1024)` |
| AllGather, `Int32` | `R*N` | At offset `r*N+i`, the original input of rank `r` |

Check every output after every warmup/measured pass, on every rank. A sentinel
and rank-dependent input must catch an untouched buffer, a rank permutation,
and a local-copy implementation. Add CPU-only tests showing that the oracle
rejects each of those mutations. Keep CUDA `verified_results` unchanged from
plan 001; record NCCL verified-result counts separately by operation/rank.

For `R` ranks, `N` elements, and `P` measured passes:

- Each selected collective contributes `R*P` profile samples and event intervals.
- AllReduce checks `R*N*P` measured outputs.
- AllGather checks `R*R*N*P` measured outputs.
- Warmup checks the same layouts but contributes no measured samples/counts.

For two ranks, `N=257`, `P=2`, `both` contributes eight NCCL samples,
1,028 AllReduce outputs, and 2,056 AllGather outputs. The original CUDA stage
still contributes four samples and 1,028 checked outputs, recorded separately.
For default `N=1048576`, `P=12`, `both` contributes 48 NCCL samples,
25,165,824 AllReduce outputs, and 50,331,648 AllGather outputs.

## Capture, timing, and source semantics

Use the public `x/cudacapture` from plan 001. Put an optional NCCL object in
example-owned opaque metadata: requested mode, selected devices and stable
identities, library version/path/build identity where obtainable, communicator
rank mapping, operation/type/reduction/count, warmup/pass counts, per-rank
CUDA event intervals, verified counts, and host completion intervals.
Record the relevant NCCL/CUDA environment and topology information with
provenance. Do not infer which transport or algorithm actually ran from the
physical topology; absent observed transport data is `not_recorded`.

NCCL samples use existing `gpu_time` nanoseconds and labels such as `library`,
`collective`, `rank`, `device_uuid`, `device_ordinal`, `phase`, and `pass`.
Retain comparable device identity labels on the original CUDA samples. One
sample is one completed per-rank collective interval; it is not a sampled
internal NCCL kernel. Rank filtering must work in pprof.

The event pair must belong to the same device and stream. Durations include
stream waits and possible host enqueue skew around the grouped operation;
label them as per-rank collective event intervals, not pure NCCL kernel time.
The sum across GPUs is aggregate device time, not end-to-end latency. Never
subtract timestamps from different GPUs. A separately recorded host interval
can cover grouped enqueue through completion of every rank, before validation;
keep it distinct from the sum or maximum of GPU intervals.

Add Go trace tasks/regions for operation, pass, and rank with event timing and
validation logs. Go timeline spans describe host activity. For pprof, record
source-only frames for the Go enqueue site and `nccl.allreduce` or
`nccl.allgather`; do not attach the application CUBIN address or fabricated
`.PTX`/`.SASS` frames to NCCL calls. The existing NVRTC kernels retain their
real embedded CUDA/PTX/SASS listings. Clearly report that internal NCCL
PTX/SASS is not recorded in this increment. Acquiring and correlating code
from the loaded NCCL library is a separate future feature, not a reason to
mislabel the example's assembly.

The reusable profile package retains all sample types. This mode does not
populate DRAM, tensor, or stall counters, or claim PC sampling. Payload sizes
are operation metadata, not measured DRAM or interconnect traffic. Do not
present a calculated payload/time ratio as measured bus bandwidth.

## Implementation slices and allowed changes

### Integration with CUPTI coverage (plan 003)

The initial NCCL mode uses CUDA events. It must remain independently usable
while plan 003 develops collectors. Its public capture schema must follow
plan 003's descriptor/availability design rather than freeze the nine current
convenience fields as a coverage limit. Do not add a second NCCL metric registry.

When plan 003's collection selectors are available, inventory support per
participating GPU/context and requested collector. Preserve device/rank,
communicator, operation/pass, correlation IDs, module identity, and replay
identity through native records and exports. A per-rank event interval, a
CUPTI internal-kernel interval, and a process-wide collective interval are
distinct observations; keep their profiles/scopes separate to prevent double
counting. Measured zero stays visible; missing or unsupported data keeps its
explicit disposition.

Do not apply single-kernel replay independently to one rank of a collective.
Before enabling a replaying collector for NCCL, prove a supported coordinated
schedule that restores all ranks' inputs and matching collective sequences
for every pass under a bounded process deadline. Otherwise reject that
collector/workload combination with an explicit reason. Full collection must
retain the gap and report partial/nonzero status according to plan 003; it
must not silently fall back to event-only measurement or redefine `all`.

Only verified native module/PC/source correlation may add NCCL-internal SASS
or PTX attribution. Until then, NCCL rows remain Go/operation attribution.
Dedicated integration tests must cover rank-correlation collisions, replay
pass identity, missing collector data, and unchanged baseline collective
results. This is a follow-on gate for each claimed collector combination,
not a prerequisite to shipping the correctly labeled event-only NCCL mode.

### Implementation sequence

1. Add pure configuration, checked-size, rank-layout, oracle, sample-label,
   and metadata tests. Extend the existing CLI txtar harness from plan 001.
2. Implement AllReduce ownership and timing in a focused example file; pass
   the two-device hardware and failure gates before adding AllGather.
3. Add AllGather using the same concrete owners and enqueue ordering. Avoid
   an abstraction solely to combine two short operation loops.
4. Extend the viewer's example-metadata summary, documentation, capture tests,
   and optional packager for mixed CUDA/NCCL recordings. Verify that application
   assembly views still work and NCCL source-only rows display honestly.

Allowed changes: the trace example, its public-client check and packager;
`cmd/cudaprof` metadata presentation/tests; focused compatibility tests in
`x/cudacapture` and `x/cudaprof`; NVIDIA README/DESIGN documentation; plan
status rows. Use `nccl/` and CUDA owner source as references. If a binding or
owner defect blocks the work, record a separate, scoped fix with its owning
source generator where applicable; do not hand-edit generated bindings or
fold unrelated cleanup into this change. Do not add workload fields to the
public capture codec. Preserve v1 reads and v2 single-file round trips.

## Acceptance gates

Software-only checks, run with `GOWORK=off` and temporary output under `~/tmp`:

- Existing plan 001 tests/default-mode behavior remain passing. `-h` and mode
  `off` perform no NCCL initialization. Use a host with NCCL unavailable to
  verify the default capture path on one CUDA GPU when available.
- CLI tests reject invalid mode/device lists/counts before submission; explicit
  NCCL unavailability produces an error and no final capture.
- CPU tests check both oracles, rank order, overflow, unchanged sample values,
  per-rank labels, and warmup exclusion. Do not substitute a mock collective
  result for real multi-GPU validation.
- The unrelated-module copy/build check from plan 001 still builds all example
  source and runs help without CUDA/NCCL. Dynamic loading adds no new C link
  requirement to the copied example or offline viewer.
- A synthetic mixed recording round-trips directory/embedded forms and pprof
  parse/write without losing rank labels, optional metadata, trace, or source.

Hardware gates require an explicitly selected host with at least two distinct
physical GPUs, NCCL, CUDA/NVRTC, and nvdisasm. Record source hashes, toolchain,
loaded-library identity, device UUID/PCI mapping, and topology before running.
Do not provision paid hardware as part of this planning change. Example
commands after implementation, from the NVIDIA module on that host:

```sh
GOWORK=off go build -o "$HOME/tmp/ptx-sass-trace-nccl" ./examples/ptx-sass-trace
timeout 120s "$HOME/tmp/ptx-sass-trace-nccl" \
  -nccl=both -devices=0,1 -elements=257 -rounds=8 -passes=2 \
  -out "$HOME/tmp/nccl-small-NEW" -nvdisasm /verified/path/to/nvdisasm
timeout 120s "$HOME/tmp/ptx-sass-trace-nccl" \
  -nccl=both -devices=0,1 -out "$HOME/tmp/nccl-default-NEW" \
  -nvdisasm /verified/path/to/nvdisasm
```

Choose fresh directories and verify the timeout utility/path before execution.
Run `allreduce` and `allgather` individually too, plus a small reversed-device
list (`1,0`) to exercise rank/device separation. Four-or-more-rank runs are
additional qualification only when such hardware is available. Check exact
counts above and compare pprof sums to saved per-rank intervals, never prior
performance magnitudes.

Required negative controls: duplicate devices rejected before native work;
wrong CPU oracle rejected after a completed collective; a test-owned early
failure after partial rank setup exercises cleanup; a bounded child-process
hang/abort control verifies failure reporting and absence of a final profile.
Do not deliberately issue mismatched counts or omit a collective participant
in the production path. Keep a cleanup failure or timeout as a failed gate.

Finally copy only `capture.pprof` to a host without CUDA/NCCL. Check stock pprof
sample totals, rank filtering, both application PTX/SASS views, NCCL Go source
attribution, trace extraction, the matching Go trace tool, and optional archive
viewers/offline rebuilds. Report cross-builds separately from executed viewers.
Retained single-GPU v1/v2 recordings must still open when NCCL is absent.

Done means all required software and two-GPU gates pass, all outputs are
checked, metadata states measurement limits, and plan status links retained
receipts. If hardware is unavailable, leave multi-GPU acceptance unverified
and implementation status partial/TODO as appropriate; default-mode success
or a skipped physical test does not qualify NCCL.

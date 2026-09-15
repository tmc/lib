# Qualification status

Local development snapshot: 2026-09-14. This is an implementation audit, not a
hardware-support or performance claim.

## Source layout

The nvidia changes are isolated in `~/tmp/cuptibench-implementation`, based on
`b2764a4643d30ad7eb4584df9ca2dd4cee9f42a0`. The real DS4 adapter is isolated in
`~/tmp/cuptibench-ds4-integration`, based on
`cbc78f8cfec4c205aee11e0ee8c1026e13cbae5d`. Its temporary module file selects the
new nvidia module. Neither shared source checkout is modified by this work.
The core is integrated into main by the cuptibench landing series. The DS4
adapter remains a dependent change until its runtime is available on main.
The revisions above identify the hardware qualification source bases; no GPU
rerun of the main integration is claimed. Local integration checks are separate.

## Requirement audit

| Requirement | Evidence | Remaining work |
|---|---|---|
| Bounded prepared replay and nested user ranges | cuptirange capture/replay lifecycle tests; native PushRange/PopRange wrapper | Synthetic controls and DS4 Regions state checks pass; kernel capture fails |
| Logical-operation normalization independent of replay passes | Collect accounting tests at 1, 2, and 4 passes | Actual native host 1/2/4 nesting controls pass |
| Whole, attributed, and independently selected isolated benchmarks | Actual Go test-binary selection and repeat tests; separate Whole aggregate for Kernels | Synthetic identity/views and real DS4 Regions benchmark/profile pass; Kernels fails |
| Multi-stream completion timing | Event dependency graph and failure-path tests; dependent/overlap physical fixture | Native overlap/dependency controls pass |
| Profile numeric/schema integrity | Integer boundaries, schema partition, missing values, profile round trips | Native round trip passes |
| Cleanup invalidates failed measurements | Capture, timer, override retry tests and actual caller-owned Close failure/retry/open-plan subprocess tests | Native failure qualification |
| DS4 uses actual forward work | Production stage callbacks preserve graph boundaries; independently prepared prefixes | All 260 token-0 Regions state/logit receipts pass; original failure unexplained |
| Cache semantics are explicit | CachePolicy admission, documented no-reset metadata, cold/warm raw-sector fixture compiled for Linux AMD64/ARM64 | Native cold/warm and two-pass controls pass |
| Faithful C benchmark formatting | Versioned JSON decoder, 64 MiB input bound, normalization and injection tests, CLI tests | Source-specific reader and checked comparison are separate proposed work |
| C family mapping is qualified independently of formatting | Adapter alternatives spec includes coverage and mapping gates | Retained Nsight mapping is known wrong and remains format-only |
| Cross-collector exclusion | Device lease integration for Range, PM, SASS and PC; conflict and failed teardown/retry race tests pass | PC exclusion and benchmark timing/export pass on GB20B; physical failure injection remains open; external collection remains caller-controlled |

## Local checks

From the nvidia module, with `GOWORK=off GOFLAGS=-mod=readonly`:

```sh
go test -race ./x/cuda ./x/cuptibench/... ./x/cuptirange ./x/cudaprof ./internal/cuptiowner ./x/cuptipmsampling ./x/cuptisass ./cmd/cuptibench-import
GOOS=linux GOARCH=arm64 go test -exec=true ./x/cuptibench ./x/cuptirange ./x/cuda
```

Both commands passed. The second command compiles Linux tests but substitutes a
no-op executor: it does not execute GPU code. DS4's selector race tests and Linux
ARM64 test-binary compilation also passed; exact build commands are in the DS4
adapter's `CUPTIBENCH.md`. The caller-owned export lifecycle fixes additionally passed the targeted race
gate covering TestExportRequiresSuccessfulCallerCleanup, TestExportTestMain,
TestOverrideCleanupRetainsChildUntilRetry, TestProcessExportSnapshotAndFinish,
and TestFinishNoResultsAndNoOutput. The new traffic fixture compiled for both
Linux AMD64 and ARM64. The combined race gate above passed after the ownership and export lifecycle
fixes. A retained Linux ARM64 cuptibench test binary was also compiled under
~/tmp/cuptibench-qualification; the initial and v4 revisions have been executed on hardware.

## Hardware results and remaining gates

DGX use was authorized and every run acquired the existing directory lease.
Product NVIDIA GB10; measured CUPTI chip GB20B, driver API 13000, CUPTI 130202.
Clock snapshots, lease records, binaries and source hashes are retained under
`~/tmp/cuptibench-qualification`. No clock settings were changed.

The synthetic v4 suite passed: native actual host 1/2/4-pass normalization via
Regions nesting, independent range-count loss detection, graph/direct kernel
identity forward/reverse controls, full/isolated views, multi-stream timing,
filtered L2 cold/warm controls, and native profile round trip. The optional
metric-schedule-specific 1/2/4 test was skipped. Kernel replay internal pass
counts remain unobserved. Receipt: `FC0AB720-20260915T033656Z-v4`.

The original failures remain in `FC0AB720-20260915T030500Z-full`: numeric native
range names and silent omission with zero reported drops. Numeric names are
preserved as ordinals; current association uses independent launch evidence.
Zero drops alone never establishes completeness.

The first DS4 run failed final full capture with non-finite logit 5480 after
all 129 isolated stages passed twice. That failure remains unexplained. The
current all-stage rerun passes: 258 isolated state receipts and two full-logit
receipts match their baselines, using three actual host replay passes for each
full capture. Receipt: `FC0AB720-20260915T035700Z-ds4-full-v5`. No model math or
Reset fix was applied. These tests cover token 0 at position 0; kernel benchmark
profiles and other token positions remain separate gates.

The real DS4 per-kernel path fails at native RangeProfilerStop with
CUPTI_ERROR_INVALID_PARAMETER, followed by a Disable failure. Both full and
isolated benchmarks fail; a direct isolated Collect reproduces the Stop failure
without benchmark timing. Receipts: FC0AB720-20260915T042242Z-ds4-benchmark-v8
and FC0AB720-20260915T042800Z-ds4-stage-v9. No kernel profile was published.
Tiny controls pass with DS4 collection bounds, read-only shared HMM, captured
cuBLAS HC projection, and both blocking/nonblocking streams and graph warmup
orders. The cause remains open; these are not equivalent workload qualification.

The real DS4 Regions benchmark now passes both full and independently selected
isolated/layer00/qkv invocation, with readable pprof and JSON exports. Receipt:
FC0AB720-20260915-ds4-regions-v10. Each is a single benchtime=1x execution, not
a performance estimate. Both captures have three host passes. Full root CTA is
1,261,287; the terminal-region pprof projection totals 1,244,592 and omits work
outside leaves. Root counters remain in JSON. DS4 defaults explicitly to Regions;
Kernels remains selectable, fails on this workload, and never falls back.

Native cleanup-failure and cross-collector conflict injection remain locally
tested contracts, without a separate physical failure-injection qualification.
The C adapter remains a format adapter; retained Nsight family mapping is known
wrong and supplies no performance comparison. See [hardware status](design/hardware-status.md) and the
receipt root for exact run paths and bounded conclusions.

## Portability: attributed L4 / CUDA 12.9.41 report

D304CF8D's 2026-09-15 report, retained at
`~/tmp/agent-collab/lib/20260915-D304CF8D-cuptibench-on-l4-ad104.md`, describes an
independent NVIDIA L4 (AD104) run with driver 580.178.04, toolkit 12.9.41 and
CUPTI runtime/API version 27. This report was read locally; these measurements
were not rerun here. Its profiler-host header MD5 is
`6d969e3b23411a1c39c9804f4182b6e0`, matching the admitted 24/26/28 header.
Its measured CallbackData, ResourceData, ModuleResourceData, GraphData,
SynchronizeData, NvtxData and StreamAttrData sizes are respectively
72, 24, 24, 56, 16, 24 and 24 bytes. Both layout tables now admit 27 on that
evidence. Unknown versions still fail closed. KernelReplay admission remains
130200/130202; header compatibility does not establish replay support.

The report records passing counter, two-stream timing, filtered L2 cold/warm,
Regions nesting and graph Whole/Regions controls. AutoRange Kernels returned
zero ranges, so kernel attribution and insufficient-capacity behavior remain
unqualified on that configuration. The report's summary test totals and detailed
list differ; no aggregate pass count is inferred here. Driver 580.178.04 differs
from the GB10 host's driver, so this is not a driver-controlled comparison.

The initial `libnvrtc.so.13` failure comes from physical fixtures calling
`nvrtcx.Compile`, not from cuptibench runtime collection. The generated NVRTC
Linux loader selects that soname. The package's production dependency graph
contains neither NVRTC nor nvrtcx. D304CF8D reports using NVRTC 13.0.88 to run
these fixtures and rejecting newer emitted PTX on that driver. This establishes
a fixture/toolchain prerequisite, not a universal CUDA 13 requirement for
cuptibench workloads using precompiled kernels.

## Completion follow-up: collector ownership (2026-09-15)

PC sampling now shares device exclusion with Range, PM and SASS collection,
including the standalone reason-inventory enable path. Failed native disable
retains ownership for retry. If startup fails before a full session exists and
cleanup also fails, Start returns a session whose Close retries that cleanup.
The standalone Enable helper likewise returns a cleanup function alongside an
error when a successful native enable returns malformed parameter storage.

Benchmark event timing also holds device exclusion until timer cleanup succeeds.
This prevents cooperating collectors from instrumenting the supposedly
uninstrumented timing interval. It does not exclude external profilers.

Focused race tests cover competing ownership, constructor failures, partial
construction, cleanup retry and concurrent standalone disable. An early-release
mutation fails both startup-cleanup cases with "failed cleanup released device".
The native PC exclusion fixture enables sampling once in a fresh process and
queries a nonempty reason inventory before releasing ownership.

The initial hardware attempt waited 45 seconds for the directory lease and
exited 75 while DF6906F6 held it for the T2 census; no GPU test ran in that
attempt. After the lease became available, revision
`375de4ea98b76ddb130e43254935a2e942b60375` passed all six selected top-level tests
on NVIDIA GB10 / CUPTI chip GB20B, CUPTI 130202, driver 580.173.02:

- Native PC exclusion, nonempty reason inventory (36 reasons), and release.
- Counter-volume control (12 versus 24 CTAs).
- Multistream event timing: overlapping 20.06 ms, dependent 40.02 ms.
- Insufficient range capacity rejected by independent launch evidence.
- Actual benchmark full/isolated profile export and pprof validation.
- Actual host 1/2/4-pass replay through nesting, with invariant root CTA count 24.

Both test processes exited zero. The PC test enabled sampling once in a fresh
process; it did not sample a model workload or inject a native Disable failure.
The timing/profile check executes the new owned timer through Benchmark.
Lease pid 1461350, label `FC0AB720-cuptibench-375de4ea`, was acquired at
2026-09-15T15:51:59Z and released after the checks. Toolchain pins were NVRTC
13.0 and the CUDA 13.2 CUPTI library directory. No clocks were changed.

Receipt: `~/tmp/cuptibench-completion-receipts/375de4ea/`, including verified
binary/source hashes, source manifest, before/after device snapshots, lease,
exit statuses, test logs and the exported profile. These are synthetic physical
controls, not DS4 or performance-parity qualification. The earlier DS4 native
Stop failure, later-position correctness, native cleanup-failure injection and
L4 AutoRange gates remain open.

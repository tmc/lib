# cuptibench hardware qualification: first receipts

2026-09-15 UTC; authorized by Travis's “Using the dgx is fine” relayed in-thread.
Primary runner FC0AB720. GPU leases were acquired with the existing directory
protocol and released after each run. No clock settings were changed.

## Identity

Host spark-31db, nvidia-smi product NVIDIA GB10, CUPTI chip name GB20B.
Driver 580.173.02 / driver API 13000, CUPTI 130202. Compiler CUDA13.0.88,
library path /usr/local/cuda-13.2/targets/sbsa-linux/lib. Do not silently equate
the native chip spelling with the peer report's gb10b spelling.

Source bases: core b2764a4643d30ad7eb4584df9ca2dd4cee9f42a0; DS4 hooks
cbc78f8cfec4c205aee11e0ee8c1026e13cbae5d. The per-run local-checks.json retains
all changed source hashes and binary hash. Later runs also retain a changed-source
archive against these bases.

## Priority run: PASS, narrower than all requested gates

Receipt directory: FC0AB720-20260915T030500Z (local beside this file and remote
/home/tmc/tmp/cuptibench-qualification/FC0AB720-20260915T030500Z).
Lease pid1265148, epoch1789441501, label identical to directory.
priority.exit=0. Counter grid change yielded CTA12 then24 for3 logical operations.
These captures required ONE actual native pass; this does not qualify2/4passes.

Multi-stream event/host times: overlapping20.054527/20.073141ms;
dependent40.017246/40.030169ms. Per-stream event controls are in priority.log.
Clock snapshots and compute process preflight are retained in nvidia-before.txt,
nvidia-after.txt and processes-before.txt. No compute applications were present
before the run; graphical desktop processes are visible in the full inventory.

## Full run: FAIL; retain before fixing

Receipt directory: FC0AB720-20260915T030500Z-full. full.exit=1.

- Whole and nested user ranges pass. Default region MaxDepth16 required16 actual
  passes despite the CTA metric configuration itself requiring1.
- Automatic ranges expose names "0", "1", etc. Kernel identity/isolated/profile
  checks fail; numeric range names are not proven kernel function names.
- MaxRanges1 silently admits a two-kernel workload with one decoded range and
  DroppedRanges0. The capacity negative control correctly fails. This is a
  release-blocking loss-accounting gap, not evidence of complete collection.
- Cold/warm filtered traffic passes with TWO actual passes: driver L2 attribute
  25,165,824bytes; target1,572,864bytes; eviction100,663,296bytes. Exact read sectors
  49,152; evicted misses49,152; warm misses0. Three operations read147,456sectors,
  with only the first cold read missing. Half-sized target reads24,576sectors.
  Native dimension is l2_sector. This is not a DRAM-byte conversion or bandwidth.

Implementation fixes and explicit1/2/4pass qualification are in progress.
The first-run binary remains retained; later receipts must use distinct binary
and source hashes. Do not treat this file or the passing priority command as
qualification of the failed full suite.


## Synthetic v4: PASS

Receipt: FC0AB720-20260915T033656Z-v4, full.exit=0. Sources and binary are
identified by the retained sources.json and changed-sources.tar.gz. This supersedes
the first suite failures for the tested synthetic workloads; the old failures
remain retained.

Native callback/graph-order association passes unequal-CTA forward and reverse
controls. Exact-capacity collection passes with an independent count witness;
the two-launch/one-range negative now fails collection as required even when
CUPTI reports zero drops. Whole/Regions graph execution, isolated preparation,
profile publication/round trip, stream timing and filtered L2 controls pass.

Regions with nesting limits 1, 2 and 4 required actual host native pass counts
1, 2 and 4. Raw CTA24 / three logical operations = CTA8 in each case. The optional
metric-schedule-specific pass test was skipped because no explicit schedules
were supplied. Kernel replay's internal passes are not observed by this test.

## DS4: full sequence FAIL; parent-only control PASS

FC0AB720-20260915T030500Z-ds4 retains the failed original run. All 129 isolated
stage subtests passed, each repeated twice. Final full capture then failed with
non-finite logit 5480. Passing stages do not qualify the complete adapter.

FC0AB720-20260915T034426Z-ds4-parent uses the same immutable binary and skips
all child subtests with -test.run '^TestCUDAProfileReplay$/^$'. The parent executes
baseline and two full captures and passes in 21.18 seconds. The runner's
'no tests to run' warning refers to the child selection. This control changes
isolation history and narrows the investigation; it does not establish a cause.

A bounded before/after isolated diagnostic is in progress. Full DS4 qualification
remains open. No performance comparison is claimed.


## Synthetic v6: PASS; graph allocations covered

Receipt FC0AB720-20260915T035500Z-v6, full.exit=0. All required synthetic controls
pass, including captured alloc/project5/decode2/free graph: four native nodes,
two kernel ranges with exact CTA5/2 and output equality over two repetitions.
Source graph destruction does not invalidate the retained executable snapshot.
This also exercises exact capacity with an independent kernel count. Optional
metric-schedule discovery remains skipped.

The bounded DS4 diagnostic passes full-before and full-after two isolated final
FFN captures, with identical logits hashes. Another run passes layer00 QKV,
attention and FFN plus layer03 QKV and attention, checking full execution after
each isolated pair. These intervening full runs differ from the original long
sequence; they do not discharge its failure. Current full129-stage replay is
running with retained per-stage counter and state-hash receipts.

A setup-only failure is retained at FC0AB720-20260915T034533Z-ds4-resume
(binary copied without executable mode); retry used a new directory. The next
setup failure, -ds4-resume-exec, rejected MemAlloc graph nodes before execution.
Regions fixtures now avoid optional graph identity retention; allocation support
was separately tested by v6. No Reset or model arithmetic changes have been made.


## Local verification after pointer and portability fixes

Combined race tests pass for driver, cuda, cudagraph, cuptibench and benchfmt,
cuptirange, cupticallback, cudaprof, cuptiowner, PM, SASS and the importer CLI.
The same package set passes go vet. Native range-name scanning now reads the
native pointer field with its pointer type and bounds scanning with unsafe.Add;
strict checkptr tests pass, including the final allowed byte and unterminated
buffer controls. The v7 binaries await their next native run.

A source audit found no established cuBLAS workspace-lifetime fault in DS4:
one handle uses one stream, graph execution completes before isolation stops,
and graphs are destroyed before the handle/buffers. This is source evidence,
not a refutation of a native driver/library defect. No speculative workspace or
Reset patch was applied.


## DS4 all-stage rerun v5: PASS

Receipt FC0AB720-20260915T035700Z-ds4-full-v5, ds4.exit=0, 1592.36 seconds.
All 129 stages produced two matching state receipts; two final full captures
also matched baseline logits. There are 260 receipts and zero hash mismatches.
Both full captures used three actual host replay passes. This covers token0,
position0, the pinned model and retained toolchain/source snapshot only.

The original non-finite-logit failure remains unexplained and retained; this
passing rerun is not a causal fix claim. No model math or Reset change was used.
The kernel benchmark/profile path is being qualified separately.


## Real DS4 per-kernel capture: FAIL at native Stop

Receipts FC0AB720-20260915T042242Z-ds4-benchmark-v8 (full and isolated,
fresh processes) and FC0AB720-20260915T042800Z-ds4-stage-v9 (direct Collect,
layer00/qkv) fail with CUPTI_ERROR_INVALID_PARAMETER. The direct-stage receipt
identifies RangeProfilerStop, after Run completes; native Disable then also
fails. This is not a qualified per-kernel DS4 measurement, and no profile was
published. It does not invalidate the separately retained Regions results.

Controls narrowing this failure all pass:

- FC0AB720-20260915T042600Z-native-v9: tiny direct and captured graphs with
  DS4's exact collection bounds; current native pointer scanner also exercised.
- FC0AB720-20260915T043100Z-hmm-v10: device and read-only private HMM input,
  each with direct UserReplay, direct KernelReplay, graph Whole, graph Kernels.
- FC0AB720-20260915T043700Z-controls-v11: read-only MAP_SHARED HMM with those
  same four modes, and four standalone cuBLAS HC-projection modes, including
  a graph warmed before creating the CUPTI plan. All eight fresh processes pass.
- FC0AB720-20260915-warm-v13: blocking/nonblocking streams crossed with plan
  creation before/after graph warmup. All four fresh processes pass with exact
  kernel count, CTA5/2, identities and output checks.

These controls rule out the corresponding categorical explanations; they do
not establish a cause or a fix for the real mixed-kernel DS4 graph. No stream,
model arithmetic, Reset, workspace or production replay fallback was changed.
AutoRange UserReplay graphs remain rejected under the documented CUPTI gate.


## Real DS4 Regions benchmark and profile: PASS

Receipt FC0AB720-20260915-ds4-regions-v10. Both full and independently selected
isolated/layer00/qkv benchmark processes exit zero with explicit Regions mode,
benchtime=1x, count=1. Both publish pprof and JSON manifests, and go tool pprof
parses both. This is an execution/export qualification sample, not a performance
estimate or C/Go comparison. Whole and Kernels are explicit alternative modes;
there is no fallback on a failed capture.

Full root: 1,261,287 CTA; isolated QKV: 4,635 CTA. Both captures have three actual
host passes. Full terminal-region pprof sum: 1,244,592 CTA; the 16,695 difference
is work outside exported terminal regions. The JSON retains the full root.
The pprof projection's total is not the whole-pass total; no residual was
invented or silently attributed to a kernel. Timing is retained in benchmark
logs and JSON, with preparation/counter collection outside the timed loop.

The source archive, binary hashes, host/toolchain/model identity, lease and clock
snapshots accompany the receipt. Lease was released. No setting was tuned.

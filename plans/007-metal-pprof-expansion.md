# Metal pprof expansion after the CUDA/CUPTI review

Date: 2026-09-09. Status: proposed implementation sequence.

User direction: stay close to the core Go toolchain. Use upstream pprof and go tool trace for browser views, and JSON for non-additive observations. The custom web inspector proposed below is superseded; no custom HTTP server or dashboard is planned.

## Recommendation

Make Metal's portable viewer and attribution as useful as cudaprof, then expand collectors according to actual device capabilities. The highest-value work is standard pprof source navigation, explicit host-submission stacks, and complete JSON export of available replay statistics. Do not copy CUDA's metric names into empty Metal sample columns.

This review inspected current source in lib/nvidia and lib/apple, plus the local gputrace decoder and a retained Apple M4 Max capture. It did not execute CUDA collectors on NVIDIA hardware. Untracked CUDA launcher/injection work was changing during review; it is not a stable capability baseline. No CUDA files were changed.

## What CUDA actually provides

| Area | Current source evidence | Interpretation for Metal |
| --- | --- | --- |
| Upstream pprof integration | `nvidia/cmd/cudaprof/main.go` passes both `Obj` and `Fetch` to `driver.PProf`; `object.go` implements Open, Disasm, SourceLine, Symbols and ObjAddr | Metal only supplies Fetch. Add an object adapter to improve offline source/code navigation and eliminate the unresolved MetalPipeline lookup |
| Code identity and coordinates | `nvidia/cmd/cudaprof/object.go:38` hashes CUBIN/PTX; SASS uses kernel and instruction offsets; PTX uses explicitly logical text coordinates | Preserve MSL, AIR and exact pipeline ISA as different representations. Navigation coordinates must never become sampled hardware PCs |
| Rich sample schema | `nvidia/x/cudaprof/proto.go:67` defines 23 additive columns, including GPU time, memory traffic, warp counts, stalls and delays; SetSampleTypes selects measured columns | These fields are caller-supplied capacity, not proof of 23 active collectors. Metal should support selected additive metrics with units and evidence, not blindly adopt all 23 |
| Stack and labels | `nvidia/x/cudaprof/profile.go:230` records explicit frames and context labels; RecordTraceback builds CUDA/PTX/SASS frames; `do.go:17` brackets labels with NVTX | Add explicit submission stacks and label propagation. Neither NVTX nor a debug group by itself proves CPU-to-GPU correlation |
| Activity collector | `nvidia/x/cuptiactivity/record.go:16` decodes API/kernel/copy/memset records with timestamps and correlation IDs; `session.go:14` exposes losses, incomplete records and pending buffers | Build an owned-workload Metal submission/command-buffer/encoder ledger with explicit clock domains and loss reporting |
| Range counters | `nvidia/x/cuptirange/doc.go` and `evaluate.go:18` describe bounded user replay and named counter evaluation | Keep replay separate from live timing. Discover native metrics, retain configuration and raw values, and project only metrics with known units and aggregation |
| Flight recording | `nvidia/x/cudaprof/flightrecorder.go` retains caller-supplied records and exports pprof/Perfetto | Metal already has a ring; add coverage/loss and real event timestamps before claiming a device timeline |
| PC/SASS sampling | `nvidia/x/cudaprof/profile.go:41` explicitly requires callers to record samples; Start does not initialize a native PC sampler | Do not treat WithPCSampling or RecordSASSMetric as an end-to-end CUPTI collection witness |
| Public example | `nvidia/examples/ptx-sass-trace/capture.go:183` selects only gpu_time and labels attribution as whole-launch entry with no PC sampling | CUDA's polished code browser is compatible with honest whole-launch timing. Metal can reach that level without inventing instruction hotness |

The CUDA range collector exports named metric values to Perfetto separately; its existence does not establish automatic conversion of every counter into cudaprof sample columns. The third-party CUDA launcher currently rejects replay and cannot require complete data before a collector is configured. Plans 005/006 describe broader product work; they are not delivered functionality.

## Current Metal evidence

`apple/x/metal/metalprof/proto.go:58` has samples and gpu_time only. AddSample already accepts a stack of location IDs, but the example supplies one kernel source anchor (`examples/metal-trace/workload.go:320`). `labels.go` supports pprof labels and `labels_darwin.go` exposes debug groups/signposts. The portable capture carries source, AIR, Go trace, metadata, observations and a separate replay profile. The CLI supports live/replay selection and text observations; its upstream pprof view does not expose the complete observation dataset.

Fresh retained recording: `/Users/tmc/tmp/metal-metrics-current/capture.pprof`.

| Evidence | Value | Boundary |
| --- | ---: | --- |
| Elements / FMA rounds / measured passes | 1,048,576 / 512 / 12 | Owned example workload |
| Validated output values | 25,165,824 | Warmup excluded |
| Maximum absolute error | 0 | Compared with CPU reference |
| Live compute stage time | 20,367,458 ns | Sum of 12 hardware stage-counter intervals |
| Live vector-add stage time | 1,318,833 ns | Sum of 12 hardware stage-counter intervals |
| Replay command-buffer active time | 7,493,743 ns | Separate replay, includes captured warmup |
| Replay command-buffer wall span | 28,314,833 ns | Not active time |
| Replay dispatch offset deltas | 397 us | Inferred cumulative-offset timing; not measured per-dispatch duration |
| Pipelines / replay commands | 2 / 26 | Replay includes warmup |
| Compute temporary / uniform registers | 3 / 12 | Compiler statistics, not dynamic register occupancy |
| Compute compiled instructions | 20 | Static instruction count, not executed instruction count |

Raw source: `extracted/manifest.json` and `extracted/replay_export.json` beside the capture. The latter contains pipeline instruction mixes, register/spill statistics, compiler remarks, counter archives and timeline data beyond today's typed exporter. Presence must be checked against `recorded_statistics`: gputrace's numeric struct fields can otherwise serialize missing values as zero. Its source of truth is `gputrace/internal/counter/streamdata.go`, particularly assignPipelineStatFields.

The CLI reports that profile-replay does not produce derived utilization/limiter/occupancy counters for this GPU generation. The existing capability ledger records an ISA helper target mismatch. Those are explicit gaps; no real numbers may be fabricated to fill them. Revalidate the exact mode/toolchain before claiming a changed capability.

## Implementation sequence

### 1. Complete the measurement contract and available statistics

Extend observations with source field/path, capture/run identity, clock domain, owner identity, availability reason, and derivation. Separate measured, compiler-reported and inferred values. Represent unavailable values without a numeric zero. Add capture/run scope rather than assigning whole-run totals to one dispatch.

Export all recorded pipeline statistics, compiler diagnostics, command-buffer active/wall timing, encoder timings, dispatch geometry and available raw counter records. Preserve native IDs and units. Retain counter records at their actual owner; do not attach them to a pipeline by array position. Expose unmatched owners and unknown timebases. Keep instruction mix categories separate: overlapping classes need not sum to total instructions.

A local draft already adds 21 compiler-metric descriptors with recorded-statistic presence checks. Finish consistent presence handling for the existing instruction/spill exports before accepting this slice. Compiler statistics stay in observations, never additive pprof costs.

Acceptance: every exported numeric raw field has a documented disposition (typed observation, identity, raw-only, unsupported or unresolved). Positive recorded-zero and negative absent-field controls; exact large integers; invalid/fractional/negative counts; duplicate owners; same-name specializations; raw-to-observation reconciliation. Keep the original raw replay payload as well as JSON so decoding can be independently revisited.

### 2. Use upstream pprof and Go trace directly

Keep the standard profile.proto output and existing capture extraction. Open gpu.pprof with `go tool pprof -http=127.0.0.1:8097 -symbolize=none gpu.pprof` from the extracted directory so embedded MSL source is available. Use `go tool trace go.trace` for the Go execution trace. Non-additive observations stay in manifest.json and the retained replay_export.json; no custom metric dashboard or HTTP server.

Improve the existing pprof object adapter only if required for accurate source/code navigation. MSL navigation works with extracted files. AIR remains an offline compilation artifact unless exact pipeline identity is proven; native ISA remains unavailable until qualified. Never fabricate an executable or sampled PC to suppress a symbolization warning.

Acceptance: extract only capture.pprof in a fresh directory; use stock pprof to inspect totals, graph, flame graph, labels and MSL source without the original source tree. Verify zero and missing metrics in JSON. Ensure whole-launch source anchors cannot be mistaken for measured instruction-level hotness. No native GPU tools are required for viewing.

### 3. Add explicit submission stacks and workload correlation

Provide a small recording API that captures Go frames at submission and binds them to run/device/queue/command-buffer/encoder IDs. Preserve request/model/layer/pass labels through completion. Build pprof stacks ending at the measured kernel, with Go frames clearly described as submission attribution rather than a simultaneously executing GPU stack.

Record CPU encoding, submit, scheduled/completed events and device intervals in their own domains. Only derive latency across clocks after a recorded calibration establishes the mapping and uncertainty. Do not use callback-arrival time as GPU completion time. Never copy the current flight recorder's wall-time Record timestamp onto device work as though it were a hardware timestamp.

Acceptance: concurrent queues and repeated same-name kernels keep correct labels; asynchronous completion preserves caller attribution; deliberately swapped IDs fail reconciliation; calibrated intervals and host callback delays remain separate. Measure collection overhead with collection enabled and disabled.

### 4. Expand additive pprof profiles with real collectors

Add sample-type selection or validated descriptors for measured quantities. Candidate separate profiles: live GPU stage duration, command-buffer active intervals, host encoding duration, observed copy bytes, allocation bytes/events, and exact sampled counts if a collector supplies them. Keep differing observation populations and overlapping scopes separate; never sum stage timing with its enclosing command-buffer timing.

Record buffer allocation/release and explicit blit sizes in the owned workload. Requested bytes are API/workload quantities, not DRAM traffic. Report unified-memory resource storage mode and ownership; do not label it VRAM or infer PCIe transfers. Derive bandwidth only from matched byte/time windows. Ratios, peaks, occupancy, compiler registers and compiled instructions remain non-additive observations.

Acceptance: stock pprof -sample_index totals match retained records; omitted collectors create no false zero columns; allocation lifetime and copy-direction controls; overlapping intervals tested without double-counting; integer unit conversion and overflow checks.

### 5. Qualify deeper replay and hardware counters

Inventory each capture/replay mode independently, retain raw shards and exact tool/device identity, and bound replay before execution. Validate resource/workload fidelity where possible and label any unverified replay-output equivalence. Investigate native ISA/PC attribution and derived counters as separate capability experiments, not prerequisites for useful timing/source browsing.

Acceptance: positive hardware witness and deliberately missing/stale/truncated payload controls for each collector; source/pipeline hash mismatch must block attribution; no inferred offset delta becomes measured GPU duration; unresolved native helper target remains visible. Do not widen a timeout after failure.

### 6. Harden flight recording and external capture later

Add bounded retention with dropped/overwritten counts, snapshot consistency, explicit start/stop/drain lifecycle, and collector ownership. Start with instrumented Go Metal workloads. Capturing unchanged third-party programs is a separate investigation: a Metal debug group wrapper is not a CUPTI-style process injection/callback subsystem.

Acceptance: loss accounting under pressure, concurrent snapshot/stop, failed drain retention, and portable offline export. Arbitrary process capture and multi-device support need separate authorization/scope and physical evidence.

## Order and completion

Deliver 1 and the upstream object/source integration from 2 first: they expose the substantial real dataset already collected and directly improve the requested browser output. Then 3 and 4 make profiles useful for real Go applications. Run 5 as bounded capability work; 6 follows the owned-workload path.

For each slice retain source revision, test output, hardware receipt and browser checks. Use rsc.io/script for new CLI contract tests, targeted Go unit tests for schema/aggregation, and real hardware only for collection claims. No CUDA hardware qualification is implied by this review. Do not mark the full Metal goal complete until the metric-family disposition and browser coverage have been audited; platform gaps must remain explicit.

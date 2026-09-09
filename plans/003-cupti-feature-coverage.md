# Cover CUPTI features and all representable measurements

Status: completed (bindings/feature census, survey tooling, extensible dynamic sample schema, logical image base export, and `RecordSASSMetric` tensor/warp attribution fixes verified). Priority: P1.

## Required outcome

Cover the complete CUPTI feature surface for each declared supported toolkit/runtime/OS/GPU tuple. Discover the metrics available on the actual device, collect every reachable requested metric through its correct API, and export every measurement that the destination can represent faithfully. Preserve the rest as typed records or metrics in the same capture, with an explicit reason when a pprof projection is unsuitable.

Keep the simple two-kernel example as a readable public-API client. Add a full-collection mode and small public collectors; do not put all of CUPTI inside `main.go`. The baseline CUDA-event run remains available for comparison. Full collection can require separate runs or replay passes; it does not mean enabling every collector simultaneously.

The nine fields in `cudaprof.SampleValues` are a convenience schema, not the definition of CUPTI coverage. A feature is not complete merely because a generated symbol exists, a flag is accepted, or its value can be placed in a pprof field.

## Current evidence

Paths are relative to `$HOME/go/src/github.com/tmc/lib` unless an absolute path is given.

| Evidence | What it establishes |
| --- | --- |
| `nvidia/cupti/init.gen.go:409–603` | 193 distinct registered native symbol names in the inspected source. This is a binding declaration inventory, not 193 verified runtime capabilities. See `003-cupti-bindings.tsv`. |
| `nvidia/cupti/enums.gen.go` | 62 `ActivityKind` constants, including invalid/sentinel values. A count is not an activity coverage claim. |
| `nvidia/x/cuptiactivity/layout.go:14–34`, `record.go:72–107` | Only five kinds currently decode: driver API, runtime API, concurrent kernel, memcpy, and memset. Layout admission is restricted to version 130200. Only a subset of their fields is copied into the public record. |
| `nvidia/x/cuptirange/plan.go:40`, `collect.go:29–33`, `evaluate.go:18–54` | A reusable range collector already exists: named input metrics, automatic kernel ranges, user replay, finite float64 evaluation. Extend it rather than replacing it. |
| `nvidia/x/cuptirange/trace.go:12–25` | Named float64 metrics already export to Perfetto, but unit and aggregation metadata are absent. |
| `nvidia/x/cudaprof/proto.go:65–108` | A fixed nine-metric schema, with `samples` ambiguously described as cycles or sample hits. |
| `nvidia/x/cudaprof/profile.go:37–80` | `WithPCSampling` and `WithSourceAttribution` store values that `Start` does not use to collect or resolve anything. |
| `nvidia/x/cudaprof/profile.go:167–192` | `RecordSASSMetric` copies `ExecutedWarps` to both `Samples` and `TensorWarps`, regardless of opcode, and ignores `IssuedWarps`. This must be corrected before claiming richer metrics. |
| `nvidia/x/cudaprof/flightrecorder.go:57–90` | The sampling-period option is stored but does not start a native collector; `Start` enables an in-memory recorder. |
| `nvidia/cupti/provenance.json` | Explicit header hashes exist, but upstream version is `unrecorded` and the hash explicitly excludes the transitive header closure. |
| `$HOME/go/src/github.com/tmc/codegen/contrib/cuda/cupti_config.go:8–37` | The generator deliberately excludes `cupti_pcsampling_util.h`, whose `CuptiUtil*` entry points belong to a separate static library. Its header list also lacks the Clock Control header listed in current NVIDIA documentation. |

Local baseline: `GOWORK=off TMPDIR="$HOME/tmp" go test -mod=readonly -count=1 ./cupti ./x/cuptiactivity ./x/cuptirange ./x/cudaprof` passed on darwin/arm64. Linux collector tests and the `cupti_physical` build-tag tests were not run. No GPU feature was qualified during this planning turn.

Source drift check: compare the files and SHA-256 values in `003-cupti-source-baseline.json` before implementation. The companion TSV is the inspected binding denominator only; the SDK inventory below must find missing declarations too.

## Full feature ledger

Build the denominator from the pinned SDK headers, their complete include closure, versioned documentation, and native exports. Include everything below and every additional declared family discovered by that inventory. NVIDIA's [CUPTI documentation](https://docs.nvidia.com/cupti/) describes distinct tracing, profiling, and sampling APIs; its [module index](https://docs.nvidia.com/cupti/main/main.html) also includes utility and clock-control APIs. Runtime reachability is version/device specific.

| Family | Required coverage | Starting point / gap |
| --- | --- | --- |
| Activity | Every supported record kind/version and meaningful field: kernels, APIs, copies, memset, memory/pools/UM, synchronization/events, devices/contexts/streams, graph/JIT/module/source/function metadata, markers, overhead, topology/interconnect/environment, and supported language-runtime records | Extend `x/cuptiactivity`; five decoded kinds today. Discover the exhaustive set, including deprecated kinds that cannot be enabled on newer runtimes. |
| Callback | Runtime/driver entry and exit, resource, synchronization, state, NVTX and other supported domains/IDs; subscription lifecycle, concurrency, reentrancy, and payload ownership | Raw subscriber/callback bindings exist; add a public lifecycle owner where no suitable package exists. Do not infer callback support from activity API records. |
| Correlation and clocks | Scoped external IDs, process/context/stream/graph identities, CUPTI timestamps, explicit host-clock calibration with uncertainty, module/CUBIN/source identity | Raw functions and some correlation tests exist; no global identity or clock alignment may be guessed. |
| Profiler Host | Enumerate chips, base metrics, valid submetrics, properties, units, rollups, counter availability, config images, required passes, and single-pass sets | `cuptirange.NewPlan` accepts names; add discovery and descriptors using existing host bindings. |
| Range Profiling | Every supported range/replay mode, nesting, counter images, completed passes, dropped ranges, configuration/evaluation | Existing collector covers automatic kernel ranges with user replay only. |
| PC Sampling | Capability/configuration, all enumerated stall reasons, per-PC counts, collection period, range/context identity, dropped/non-user samples, start/stop/drain, and source correlation | Raw `PCSampling*` bindings exist; no collector use found in `nvidia/x`. Implement a real collector. [PC Sampling API](https://docs.nvidia.com/cupti/api/group__CUPTI__PCSAMPLING__API.html). |
| SASS Metrics | Enumerate every supported metric and granularity; config/enable/flush/disable; preserve native metric ID/name, instance, value and instruction identity | Raw `SassMetrics*` bindings exist; current `SASSMetric` is a caller-supplied record, not this collector. [SASS Metrics API](https://docs.nvidia.com/cupti/api/group__CUPTI__SASS__METRICS__API.html). |
| PM Sampling | Discover schedulable metric sets, time/cycle triggers, device samples and intervals, buffer/decode/drain/loss, metric evaluation | Raw `PmSampling*` bindings exist; implement public collection. Do not attribute a device-wide sample to a kernel without sufficient correlation. |
| Checkpoint | Save/restore/free, supported state scope, limits, cleanup and replay equivalence | Raw bindings exist; exercise controlled replay through a checked workload. |
| Legacy Profiling/Event/Metric | Inventory the older target profiler and event-group/metric APIs; exercise supported legacy configurations or record a documented runtime refusal and modern replacement | Do not erase old APIs from the ledger or enable a deprecated API blindly on a newer GPU/runtime. |
| PC Sampling utilities | Utility file operations and source/SASS correlation, including the separate library and result type | Deliberately outside the current `libcupti` binding target. Design a separate adapter/build path for the utility library; do not add nonexistent `libcupti` dynamic symbols. |
| Clock Control | Query/set/reset operations and supported modes; observe and restore original state in dedicated tests | No Clock symbol is registered in the current binding source; reconcile against the supported SDK version. These are control operations, not pprof samples. [Clock Control API](https://docs.nvidia.com/cupti/api/group__CUPTI__CLOCK__CONTROL__API.html). |
| Support and lifecycle | Version/device/permission queries, native errors, attributes, callback ABI/layout, subscriber ownership, finalize/unload, diagnostics and process isolation | Existing loader, layout and physical tests are starting points. Keep support checks separate from successful collection receipts. |

Each row must link API symbols, structs/fields, collector entry points, tests, output fields and receipts. Version-inapplicable, hardware-unsupported, permission-denied, not implemented, not exercised, failed, and measured are distinct states. A missing implementation never becomes “unsupported.” A claimed supported feature needs a positive run and a meaningful failure/control case.

## Every existing sample type needs an honest producer

| Existing field | Required producer and meaning |
| --- | --- |
| `samples` | Split distinguishable native measures into names such as `pc_samples`, `executed_warps`, or `sm_cycles`; sample hits, cycles and warps cannot share one meaning. Preserve the old convenience API only where its meaning is explicit. |
| `vram_alloc` | Allocation observations/records or explicitly scoped allocator accounting. Define allocated bytes versus live bytes versus reserved pool bytes; none is DRAM traffic. |
| `dram_read`, `dram_write` | Device-enumerated range/PM metrics with documented byte conversion and scope. Do not substitute memcpy payload bytes or `elements * sizeof(T)` estimates. |
| `shared_mem` | Split static/dynamic shared-memory allocation from measured shared-memory traffic; preserve separate names/units and the actual counter conversion. |
| `tensor_warps` | A tensor-specific native metric with a verified definition. General executed warps must never populate it. |
| `stall_scoreboard`, `stall_wait` | Explicit native PC stall-reason counts or well-defined range metrics; preserve all reasons and their units, not only these two buckets. Do not interpret sample counts as nanoseconds. |
| `gpu_time` | CUDA events or complete CUPTI kernel intervals, identified by source and scope. Retain both as distinct series if both are collected; do not double-count the same launch. |

Measured zero is a valid value and should stay visible. Missing, unsupported, dropped, not scheduled, or inapplicable is not zero. A scalar FMA kernel may correctly have zero tensor instructions; proving tensor measurement needs a separate checked tensor workload. A workload's expected memory accesses are an oracle for a controlled experiment, not replacement counter data.

## Data model and export

Replace the fixed schema as the underlying representation with runtime descriptors and ordered values. Keep only small typed convenience methods for established metrics. Start with a constructor/configuration taking a slice of `{Name, Unit}` and an error-returning sample method taking aligned `[]int64` values plus existing frames/labels; test the caller shape before fixing names. Carry richer native descriptors in capture metadata: collector, exact metric name/ID and suffix, numeric kind, native/export unit, scale, aggregation rule, scope, clock, configuration and replay identity, availability and loss.

The [pprof schema](https://github.com/google/pprof/blob/main/proto/profile.proto) defines arbitrary type/unit names, signed 64-bit sample values, equal-width sample vectors, and summation when samples aggregate. Therefore:

- Export all valid additive integral counts, bytes, cycles and durations with their true names/units. Check signed overflow, vector lengths and descriptor identity. Different schemas cannot share a sample vector accidentally.
- Preserve uint64 and floating-point native values exactly in the typed capture format. Use scaled integer pprof projections only with an explicit scale, overflow/precision checks and valid aggregation semantics. Never cast a fraction to int64 silently.
- Keep ratios, percentages, averages, rates and gauges in a metric table or appropriate trace counter view when ordinary pprof summation would misrepresent them. Preserve numerator/denominator and weighting when available; they may support an additional correct aggregate projection. “Not suitable for pprof” must not drop the measurement from the capture.
- Use profiles grouped by collector/schema/scope when values have different availability. Zero-fill only a known measured zero, never an absent measurement. Keep correlations between profiles explicit.
- Retain raw typed activity/callback records, range/PM images and configuration identity, PC/SASS records, and diagnostics so exports can be audited. Decode native images using the matching collector/toolchain; do not feed untrusted native images into evaluators implicitly.
- A single shareable recording may contain a baseline profile plus `range.pprof`, `pc.pprof`, `sass.pprof`, other faithful projections, a metric table and Perfetto tracks. The viewer chooses a profile and metric dynamically. Update plan 001's capture metadata to inventory those members and schemas; a GPU/driver installation must not be required for the evaluated offline views.

The current embedded payload in pprof comments is a project convention. The upstream schema reserves comments for human-friendly text and says a profile must remain functional if they are cleared. Document that transport limitation and test the pinned pprof round trip; do not claim arbitrary tools guarantee preservation of the full recording. If comments are stripped, baseline samples must still work and the viewer must report missing attachments clearly.

## Ordered implementation slices

### 1. Establish an exhaustive versioned inventory

Extend the existing `$HOME/go/src/github.com/tmc/codegen/cmd/cupti-survey` and `contrib/cuda/cupti_config.go`. The survey's include scanning/parser fallback must report incomplete closure or fallback evidence; do not treat a partial parse as a complete denominator. Pin toolkit/header release and hashes, generator/toolchain revision, native library identity, OS/architecture and GPU capabilities. Reconcile the 193-symbol snapshot with every declaration, record kind/field, callback ID, metric family and separate utility library.

Add a public discovery path for all reachable metrics and their properties. Preserve unavailable candidates with reasons. Proposed CLI: `-list-features` and `-list-metrics` (read-only queries); export the inventory as JSON. Capability discovery must not change device clocks or run profiling as a side effect.

**Verify:** generated inventory has no unclassified SDK declarations or omitted enum members; unknown symbols and an intentionally omitted declaration make the inventory checker fail. Baseline source-only command above still passes. Regenerate bindings in an owned generator checkout when gaps are found; never patch `.gen.go` directly. Retain full reproduction arguments with outputs under `~/tmp`.

### 2. Correct existing semantics and make sample schemas extensible

Fix `RecordSASSMetric`'s unconditional tensor attribution and ignored issued-warp field. Remove or explicitly reject no-op hardware options until connected to an actual collector; document the flight recorder as storage. Add dynamic descriptors, valid-zero handling, typed native values and faithful projections. No-op options must never silently report collection success.

**Verify:** table tests cover each of the nine existing fields plus a previously unknown metric, mixed units, wrong vector length, duplicate descriptors, values above int64, fractional values, incompatible aggregation, measured zero and absence. A non-tensor SASS observation produces no tensor claim. Preserve real source coordinates and labels through parse/write/viewer round trips. `go test -count=1 ./x/cudaprof ./x/cuptirange ./cmd/cudaprof` passes under `GOWORK=off`.

### 3. Complete activity/callback and range collection

Extend `x/cuptiactivity` and `x/cuptirange` in place. Add a small callback-owner package only if needed. Decode all admitted record variants with generated field-offset/size evidence and copied native payload lifetimes. Preserve complete fields rather than squeezing all kinds into the current small `Record` struct. Add metric discovery, units and every supported range/replay mode. Keep shared ownership and loss accounting explicit.

**Verify:** per-kind decoding tests, unknown-layout refusal, dropped/incomplete records, callback-after-stop, cleanup retry, correlation collisions across contexts, replay state restoration, missing passes and range overflow. Existing `CUPTI_ACTIVITY_TEST=1` and `CUPTI_RANGE_TEST=1 CUPTI_CHIP=<queried-chip>` Linux tests are minimum starting gates, not full-family proof. Extend them with positive/control receipts for every ledger row.

### 4. Add PC, SASS and PM collectors, then remaining control/utility coverage

Use separate focused public packages where ownership differs, for example `x/cuptipcsampling`, `x/cuptisass`, and `x/cuptipmsampling`; keep shared metric discovery with its natural host API owner. Implement the complete bounded native lifecycle, copied/pinned callback data, context ownership, source identity, loss/drain and cleanup retry. Enumerate PC reasons and SASS metric IDs instead of hard-coding names. Complete checkpoint, utility and clock-control adapters and supported legacy routes from the feature ledger.

**Verify:** compare native outputs to NVIDIA's matching-version samples on the same target/configuration, with per-feature controls. Include nonzero PC samples, multiple stall reasons where produced, SASS integer/FP versus tensor controls, PM buffer overflow/insufficient-data cases, wrong-context/module mappings, checkpoint restore equivalence, and clock restoration. An unavailable target leaves that feature unqualified; it does not allow a blanket completeness claim.

### 5. Collect all reachable requested data without conflating runs

Add a full mode such as `-collect=all`; retain explicit selectors and the small baseline mode. `all` means the full discovered inventory for the declared support tuple, with a retained disposition for each item. Required-but-unimplemented or failed items make the result partial/nonzero. Do not silently reduce `all` to the existing nine sample fields.

Plan compatible collector groups and replay passes before running. CUPTI PM sampling requires a single-pass configuration, and PC sampling has context restrictions; consult the matching-version [CUPTI usage guide](https://docs.nvidia.com/cupti/main/main.html) and runtime support queries. Collect incompatible sets in separate identified executions. A budget limit leaves uncollected entries marked partial; it must not redefine the denominator. Restore input state for every replay and validate outputs after it. Keep uninstrumented event timing separate from profiled/replayed timing.

Expand the capture/viewer with collector and metric selection, metadata, loss, unsupported reasons and typed tables. Keep example orchestration thin. Use separate test workloads for allocation/free, managed memory, graphs, synchronization, memory traffic, shared memory and tensor instructions when the original two kernels cannot exercise a feature. All workloads need numerical or structural correctness checks; avoid making every workload part of the example's default path.

**Verify:** the full mode attempts every planned group and produces a coverage report whose measured plus explicitly dispositioned entries exactly equal the discovered inventory. Every representable measurement appears in its exported schema, including measured zeros; non-projectable values remain in the typed recording. Removing a collector/export adapter must fail the coverage test rather than disappear from the list.

### 6. Qualify end to end

Retain the unrelated-module build gate from plan 001 for the example and every new public collector. Run unit, race and native lifecycle tests on the platforms each feature supports. Preserve binary/source/module identities, raw collection receipts, evaluated native values, schema descriptors, target capabilities, loss and every replay pass. Verify copied recordings offline through CLI and browser views, extraction, and the supported pinned pprof round trip.

Use fresh small/default runs for the baseline and dedicated feature workloads for the full mode. `-collect=all` must include every sample type discovered as reachable and faithfully exportable; unsupported or inapplicable capabilities remain visible with evidence. A format that can express a value but has no working collector is still a missing feature. A working collector whose values cannot be surfaced is still an export gap.

**Verify:** machine-readable ledger joins SDK declaration → generated API → runtime support → collector receipt → raw value → exported value/view. All claimed-supported cells need positive and negative controls. No unclassified or silently omitted items. Report coverage separately for bindings, collection, representation, example reachability and hardware qualification; never collapse them to one percentage.

## Scope, ownership and completion

This plan explicitly extends plan 001's scope to `nvidia/cupti` (generator-produced changes), `x/cuptiactivity`, `x/cuptirange`, `x/cudaprof`, the named new collector/utility packages, the example, capture format and viewer, plus the CUPTI-specific generator survey/configuration/tests in the sibling codegen repository. Establish dirty-state ownership in both repositories first. No unrelated binding or generator cleanup. Preserve the user's commit procedure, do not stage binaries or write sums by hand, and keep scratch/captures under `~/tmp`.

Inventory and schema design should happen before finalizing plan 001's public capture format. Collector implementations can follow its public-package cleanup. The original plan's one-new-package bound applies to capture refactoring, not to the distinct native collector lifecycles authorized here. Its deferral of new profiling backends is superseded by this plan.

Stop and reconcile when a header/library tuple cannot be identified, layouts disagree, the utility-library boundary requires an unplanned linking strategy, or a collector combination changes correctness/ownership. Record the specific gap and continue independent ledger work. Do not widen a runtime version gate from size equality alone or substitute generated symbol presence for ABI evidence.

The writeup is complete as a plan; implementation is TODO. Completion of the implementation requires every feature in the declared support matrix to be accounted for, all reachable requested measurements to be collected, all faithfully expressible sample types to be exported, and the remainder retained with explicit reasons. Hardware-unsupported cases may be documented as such; not-implemented and not-tested cases remain work to do.

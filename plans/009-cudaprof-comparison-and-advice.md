# Cudaprof comparison and advice semantics

Status: proposed. Date: 2026-09-16. Priority: P1. Scope: output and rendering; this plan does not record an implementation or GPU qualification.

## Decision

Use the [CUDA C++ Best Practices Guide 13.4](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) to order investigations and explain possible optimizations. Do not turn its general recommendations into automatic diagnoses without the measurements that establish the cause. Its APOD cycle calls for representative measurements, a change, correctness verification, and a measured comparison. Cudaprof should make that sequence easy to follow.

The first repair is semantic: a PC sample is a statistical observation, not elapsed GPU time, a memory transaction, or a completed instruction. Keep `cudaprof -diff_base` (upstream pprof), `cudaprof diff --web` (SASS comparison), and `cudaprof gaps --diff_base` (whole-run journal comparison) distinct in the UI and help. Give them one shared capture identity and quality summary, but retain their different metric contracts.

## Reproduction and current code

These are local fixtures from a peer's DGX GB10 2048-token prefill captures. The peer reports matching PC sampling settings and no-loss accounting; the implementation must verify that provenance from the artifacts. Reproduce with the captured files, not by claiming a new measurement. The live web servers at ports 8090 and 8091 used `$HOME/go/bin/cudaprof` built from `6004b336`; recheck against current HEAD before fixing or accepting output.

| Comparison | Base | Target | PC samples, base → target | Example raw sample shift |
| --- | --- | --- | --- | --- |
| Gather vectorization | `$TMPDIR/ds4-prefill-before-gather-20260916.pprof` (`12c146df`) | `$TMPDIR/ds4-prefill-pcsampled-20260916.pprof` (`f6607361`) | 4,078,809 → 3,911,984 | Q2 gather 1,338,579 → 1,083,682 (−254,897); IQ2 gather 444,134 → 528,815 (+84,681) |
| IQ2 control | full-vectorization file above | `$TMPDIR/ds4-prefill-q2-only-20260916.pprof` (`14d3da35`) | 3,911,984 → 3,830,823 | Q2 gather 1,083,682 → 1,084,161 (+479); IQ2 gather 528,815 → 446,586 (−82,229) |

The corresponding profile `DurationNanos` values are 50.70s, 49.36s, and 48.69s. They describe capture windows, not a measured prefill latency. In the existing upstream pprof web diff, the displayed duration is the sum of the two profile windows, while “Total samples” uses the base count. The web top header shows a filtered-node subtotal; the full delta is different. None of these figures measures elapsed-time improvement. Upstream pprof percentage conventions and filtering should be documented at the entry point instead of relabeled as cudaprof performance results.

Current paths to inspect and change: `cmd/cudaprof/main.go` delegates pprof output to `driver.PProf`; `cmd/cudaprof/diff_view.go` reads captures with ignored errors and renders SASS summaries; `cmd/cudaprof/diff.go:findTimeIndex` accepts any sample type containing `time` or `ns`, then falls back to sample type zero; `internal/cudaview/diff.html` prints `Runtime` and `× speedup`; `cmd/cudaprof/gaps.go` already refuses incomplete journals for whole-run claims. `cmd/cudaprof/advise.go` contains useful occupancy caveats, but also scalar-load suggestions to use 128-bit accesses/TMA and coalescing findings whose evidence needs reclassification. Preserve concurrent edits to those files.

## Measurement contract

Build one typed comparison model used by text, JSON, and HTML. Every value carries name, unit, producer, scope, collection window, denominator, and disposition (`measured`, `measured_zero`, `unavailable`, `unsupported`, `dropped`, `partial`). Never infer a unit from a loose substring or a default sample index. Require the same sample type and unit on both sides, then compare compatible collector configuration, workload identity, GPU/device scope, code image, and capture completeness. If a field differs, show both inputs and a reason the comparison is limited. A descriptive PC-sample shift can remain visible when a performance verdict is disallowed.

For PC sampling, show base count, target count, signed raw delta, base share, target share, and percentage-point shift. Raw counts are sensitive to sampling exposure; shares answer a different question and can change when other kernels change. If offering normalized counts, state the exact scale factor and baseline. Show capture duration and total samples separately for each side; never sum durations in a comparison heading. Display filtered-node subtotals separately from full totals and expose the filter threshold. Negative samples in a pprof diff mean a decrease relative to the base; red/green legends must say which direction each means without equating either with better performance.

Show `gpu_time`/runtime and speedup only for a documented native duration producer with compatible scope and complete required observations. Distinguish summed kernel execution time from wall span: overlapping kernels make the sum larger than elapsed wall time. Whole-run wall span, idle gaps, duty cycle, and throughput require the journal quality gates already used by `gaps`; absence of a journal is `unavailable`, not zero. Give zero denominators and unobserved kernels explicit states. Pair kernels by stable code/module identity and launch scope where available; same display name alone does not prove equivalence. Static PTX/SASS instructions and source lines are navigation anchors, not extra sampled work.

## Views and output

1. Add an explicit **Comparison: base → target** header with capture paths, revisions if retained, device, workload/window, collector and settings, sample totals, and quality badges. Where metadata is missing, show `unknown` and the blocked claim. Preserve links to raw pprof exploration.
2. Make a dedicated PC-sample comparison table and flamegraph explanation that agree with CLI and JSON. Default sort by absolute shift while retaining the sign. Put raw counts and shares beside each other; use “more/fewer PC samples” instead of “faster/slower.” Keep upstream pprof routes labeled as upstream pprof and explain their merged duration, base denominator, and node filtering.
3. In SASS diff, label static code changes separately from measured execution changes. Replace permissive `findTimeIndex` with exact metric recognition and provenance. Hide runtime/speedup when the native metric or scope is absent; surface capture-read errors instead of silently falling back to disassembly. A source-only or disassembly-only comparison remains useful with an explicit static label.
4. Make advice items cite their measured evidence and required follow-up in both Markdown and JSON. Add a concise “what to measure next” link/command when a hypothesis lacks its deciding counter. Use the same disposition words in advice and comparison views. Avoid a “no optimization issues” conclusion when the necessary collectors were unavailable.

## How far to lean on NVIDIA guidance

| Guide topic | Cudaprof behavior | Evidence required for a diagnosis |
| --- | --- | --- |
| [Assess and verify](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#assess-parallelize-optimize-deploy) | Rank measured cost first and present a repeatable A/B and correctness check. | Comparable representative workload, time/throughput producer, reference-output test. PC counts alone rank sample attribution only. |
| [Timing and bandwidth](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#performance-metrics) | Keep CPU, GPU-event/activity, capture-window, and PC-sample measures separate. Compute effective bandwidth only when bytes and elapsed time refer to the same work. | Native bytes/requests plus matched timing and scope; denominator and overlap disclosed. |
| [Coalesced access](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#coalesced-access-to-global-memory) | Prioritize investigation of costly memory access sites. A scalar load, `LDG`, or memory stall is a hypothesis, not proof of uncoalesced transactions. Vector/TMA suggestions must be conditional on access pattern, architecture, and measured benefit. | Requested versus actual global traffic, sectors/transactions per request, or an equivalent supported counter with a qualified window. |
| [Occupancy and block size](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#occupancy) | Show theoretical occupancy as a resource ceiling and measured occupancy separately. Suggest a block-size/register experiment, not a promised speedup. | Device limits, launch geometry, register/shared-memory use; active-warp and stall/latency evidence for a causal claim, plus A/B timing. |
| [Transfers and overlap](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#data-transfer-between-host-and-device) | Surface transfer bytes/counts and timeline gaps where recorded; suggest batching or overlap conditionally. | Direction, bytes, transfer and kernel intervals, streams, engine capability, and complete journal. |
| [Correctness](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#verification) | Put numerical/reference validation beside performance comparisons. | Named positive and negative controls; tolerance appropriate to the algorithm. |

Advice should use three levels: **observed** (direct native measurement and scope), **suspected** (static code or correlated samples, with deciding measurement named), and **unmeasured** (no diagnosis). A guide rule supplies a candidate experiment, never a substitute for the observation. Remove fixed “critical” severity for generic scalar loads and architecture-only occupancy cliffs; severity requires measured impact or a demonstrated correctness/launch failure. Check architecture-specific mechanisms (for example TMA) from the actual device/code path, not an opcode heuristic.

## Delivery slices and acceptance

1. **Contract and audit:** inventory every current use of `findTimeIndex`, `DeviceTimeNanos`, `ShareOfProfile`, `Runtime`, `speedup`, sample percentages, and advice severity. Define exact recognized metrics and JSON schema. Acceptance: a table of producers, units, scopes, and blocked inferences; no fallback from arbitrary sample type zero to nanoseconds.
2. **Comparison core:** build pure offline base/target calculation and capture-quality admission. Acceptance: table tests for swapped sides, unequal sample totals/windows, zero and absent values, incompatible collector settings, lost/partial data, same-name code collisions, and negative deltas. Keep per-side totals unchanged.
3. **Renderers:** wire CLI/JSON/HTML to that model; label upstream pprof UI and add a comparison landing/legend without forking pprof unless its extension points cannot support the contract. Acceptance: text, JSON, and HTML render the same numbers and disposition; full and filtered totals cannot be confused; zero-duration or missing-time inputs never show a speedup.
4. **Advice evidence audit:** reclassify memory, occupancy, spill, transfer, and architecture rules; attach source links and named next measurements. Acceptance: synthetic negative controls prove scalar load and PC memory stalls cannot assert coalescing, static occupancy cannot assert throughput, and missing counters cannot become measured zero. Preserve real legality and proven spill findings.
5. **Fixture and live qualification:** run CPU-only regression checks on the two retained pprof pairs and snapshot the Q2/IQ2 sample shifts above. Compare the checked-in binary/HEAD behavior with the older live binary. Separately, if hardware execution is authorized, run controlled repeated A/B prefill timing and output-exactness checks; only those results may support speedup claims. Stop after the named gates pass.

No new collector is required to correct the labels and prevent false claims. Strong coalescing, bandwidth, occupancy, or overlap diagnoses are deferred until the corresponding native counters and their collection quality are present.

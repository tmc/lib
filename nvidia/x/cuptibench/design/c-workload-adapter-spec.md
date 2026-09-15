# C workload adapters for cuptibench and benchstat

Status: design alternatives under consideration, 2026-09-14.
Companion: `cuptibench-spec-final.md`.

## 1. Objective

Compare a C implementation and a Go implementation of the same workload using
benchstat, including a complete DS4 decode pass and selected operations within
it. Support retained measurements as well as future instrumented C runs.

A retained Nsight Systems CSV fixture is now available (section 13). It is a
candidate for the first source reader; its family mapping is unqualified. The
canonical JSON importer provides a reusable format-only boundary. Acquisition
and checked comparison remain separate choices under consideration.

A successful adapter preserves what a source measured. It does not make two
measurements equivalent merely by assigning the same benchmark name.

## 2. Separate acquisition, normalization, and presentation

    C executable or retained artifact
        -> source-specific reader / measurement harness
        -> normalized samples plus provenance
        -> validated comparison selection
        -> Go benchmark text
        -> benchstat

Acquisition may execute code and require a GPU. An offline importer only reads
artifacts and writes normalized data or benchmark text. Keep those actions
separate so importing an old capture never launches a workload implicitly.

The normalized sample is the common boundary. Direct C instrumentation, CUPTI
injection, and Nsight readers can all feed it. pprof export is a separate view
for drill-down, not the interchange format required for benchmark statistics.

## 3. Acquisition options

| Option | C source changes | Best use | Main limitation | Current position |
|---|---|---|---|---|
| A. C emits normalized JSON | Small harness changes | Repeatable full-pass and isolated-operation measurements | Requires access to build/source and explicit timing boundaries | Preferred new measurement path |
| B. Parse existing C output | None | Reuse retained oracle timings or structured logs | Missing fields may prevent comparison; human logs are fragile | First concrete reader once an artifact is chosen |
| C. Import Nsight Systems exports | None, optional NVTX ranges | In-context kernel attribution and stream timeline | Traced timing differs from uninstrumented timing; summaries lose boundaries | Useful retained-trace path |
| D. CUPTI injection into C process | Usually no source change; optional region markers | Native counters and kernel attribution | Replay/state restoration and logical-operation boundaries remain hard | Investigate after timing import |
| E. Call a C workload through a shared-library ABI | A small callable wrapper | Shared preparation/iteration harness around comparable C and Go operations | ABI, thread/context ownership, and wrapper overhead affect scope | Optional controlled microbenchmark path |

These options are composable. For example, A may emit uninstrumented timing,
while D collects counters in a separately identified run. The resulting bandwidth
estimate must retain that cross-run provenance.

### A. Direct C JSON emitter

Add a small C harness that owns setup, preparation, timed execution, completion,
and correctness checks. It emits one record for each independently acquired
sample. The C implementation remains responsible for its streams and mutable
state.

For CUDA event timing, enqueue events around the actual operation and join all
participating streams before the end event. For end-to-end latency, use a host
monotonic clock and explicitly include or exclude submission, completion waits,
logit copying, token selection, and other host work.

The emitter need not link to Go or reproduce testing.B. It supplies operation
count and measured values; normalization produces the benchmark row.

Consider two output forms: one JSON document containing samples, or one JSON
record per line for long runs. A document is simpler for all-or-nothing validation;
JSON Lines supports incremental checkpoints but requires explicit completed-run
markers. Start with the document format and version it.

### B. Existing C output reader

Implement a reader against an actual retained artifact and its producer code.
Accept known schema/version variants explicitly. Use caller-supplied mappings
only for identities or documented missing context; retain those fields as
caller-stated rather than measured.

A run-level generation duration and generated-token count can yield one average
per-token timing sample if the timing boundary is known. It cannot yield one
independent sample per generated token without retained individual durations.
A report containing only mean, minimum, and maximum cannot reconstruct raw runs.

For DS4, inspect the exact oracle's prefill/first-token/steady-decode boundaries.
Do not infer them from field names. Retain token positions, actual operation
count, model identity, input token hash, and observed correctness results.
An expected hash in source is not proof that a run reproduced it.

### C. Nsight Systems import

Prefer a versioned SQLite export when event-level relationships are required.
A documented CSV statistics export is sufficient only for measurements it
actually contains. Retain the original report and export-tool version.

Possible measurements include:

- Sum of kernel durations in a selected range.
- First-start to last-end device span, including intervening gaps.
- Union of active kernel intervals.
- Individual kernel invocation durations and launch counts.
- Host-side NVTX range duration when recorded.

These are distinct quantities. Preserve their names; none automatically becomes
uninstrumented full-pass CUDA-event time. GPU-idle gaps and overlapping streams
make the differences significant.

If only a kernel summary table exists, operation boundaries and ordering may be
unrecoverable. Export a labeled kernel aggregate or decline the requested
full-pass conversion. Do not infer token counts from a recurring kernel name.

### D. CUPTI injection

Explore the existing repository injection machinery and NVIDIA's injection
examples before adding another collector. Injection may observe launches without
recompiling C, but it cannot invent a reset operation or identify a logical token
boundary that the process does not expose.

For single-pass counter sets, a marked range may be collected without application
replay, subject to the selected backend. Multi-pass metrics require an explicit
supported replay strategy: application-controlled restoration, qualified kernel
replay, or rerunning an equivalent application workload. Record which was used.

Whole-pass and per-kernel collection may need different native range modes.
Preserve graphs and stream concurrency where supported; report any instrumentation
that changes execution topology. Native collection success is not evidence that
all replayed application state was equivalent.

### E. Shared-library workload ABI

A possible C ABI exposes opaque workload ownership and explicit lifecycle:

    create(config, out_handle)
    prepare(handle, operation_selector)
    run(handle, operation_selector)
    synchronize(handle)
    inspect_result(handle, out_result)
    destroy(handle)

The exact ABI is not selected. Return error codes and retained error text; do not
let C++ exceptions cross it. Define the owning thread, CUDA context, streams,
allocation ownership, and whether run submits asynchronously.

The Go adapter can drive C callbacks through the same cuptibench workload model.
That measures the C operation under a Go-hosted harness. Label it accordingly;
it is not automatically equivalent to running the native C executable.

Keep process startup and model loading outside per-operation measurements unless
the benchmark explicitly measures those actions. Report FFI overhead separately
when it is material, using both direct-C and wrapped controls.

## 4. Normalized sample contract

The initial JSON envelope has a schema version and a list of sample records.
Final field spelling should follow the implemented importer; this section defines
the required semantics, not a second incompatible wire schema.

Each sample needs:

- Stable benchmark path and independent sample identity.
- Source artifact reference and preferably content hash.
- Implementation identity: C executable/library or Go revision and build options.
- Logical operation definition and positive operation count.
- View: full pass, attributed in-context operation, or isolated operation.
- Workload identity: shape, model, input sequence, token position/batch regime.
- Measurements with values, units, scope, acquisition method, and normalization.
- Hardware/software identity and preparation policy where known.
- Completion, loss, and correctness status, with unknown values preserved.

Represent omitted values explicitly; zero is a measured value. Reject NaN,
infinities, nonpositive denominators, duplicate metric units in one row, and
malformed identifiers. No text field may inject another benchmark row or file
configuration line.

For each measurement, distinguish:

- **Total:** divide an additive quantity by the stated operation count.
- **Per operation:** retain the value; do not divide again.
- **Raw ratio/rate:** retain its native meaning and denominator description.

Keep operation count separate from independent sample count, replay-pass count,
kernel launch count, and profiler sample count. They are not interchangeable.

## 5. Comparison identity and units

C and Go rows use the same semantic benchmark path. Put implementation identity
in separate files and provenance, not a gratuitous `/C` versus `/Go` suffix that
prevents rows from matching.

For example, two illustrative rows in separate files might be:

    BenchmarkDecode/full 128 250000 gpu-ns/op
    BenchmarkDecode/full 128 240000 gpu-ns/op

Each line represents ONE independent run averaged over 128 logical operations.
The numbers are synthetic examples, not retained DS4 measurements.

Align Go's processor suffix deliberately when matching raw go test output.
Do not fabricate a C GOMAXPROCS setting. Prefer a documented name projection or
normalization applied consistently to both inputs, retaining original names in
provenance. Benchmark path aliases must be supplied explicitly and checked for
collisions.

Suggested timing quantities are:

| Quantity | Suggested unit token | May compare with |
|---|---|---|
| Uninstrumented completed device event interval | gpu-ns/op | Same event/completion scope in Go |
| Host end-to-end operation interval | wall-ns/op | Same host boundary in Go |
| Sum of traced kernel durations | kernel-sum-ns/op | Same traced kernel-sum definition |
| Span of traced kernels | kernel-span-ns/op | Same span and range selection |
| Union of traced kernel activity | kernel-active-ns/op | Same union and device selection |
| Exact hardware counter per operation | Metric-specific counter unit | Same counter definition, filters, and operation scope |

Do not relabel a C host duration as Go's gpu-ns/op. Go's ordinary ns/op may
include wrapper/event bookkeeping and may describe a sequence of operations;
normalize and align its scope before comparison. In the current cuptibench
adapter, Config.Operations greater than one makes Go ns/op a sequence duration,
while gpu-ns/op remains per logical operation. A checked comparison must either
use the same sequence as its host-time denominator or explicitly normalize both
arms to logical operations. It must reject an undeclared denominator mismatch.

A filtered L2 read-miss byte estimate remains filtered L2 traffic. It does not
become DRAM bytes because the comparison tool prefers a familiar column heading.

## 6. Aggregate and individual-operation comparison

Use the same three-view distinction as cuptibench. A full-pass result executes the
actual dependency chain. An isolated result restores appropriate intermediate
state. In-context attribution retains the full run's schedule and preparation.

Map implementations by semantic work and shape, not just mangled kernel name.
C may fuse operations that Go launches separately. Compare a common enclosing
region when possible. A one-to-many kernel mapping is an explicit aggregation
with stated coverage, not an automatic name match.

Only sum compatible additive values over disjoint scopes. Do not sum inclusive
parent and child ranges, independent isolated timings into full-pass latency,
or percentages into an aggregate efficiency value.

Dynamic expert routing and sequence growth require matching actual executed work.
A benchmark averaging positions 2048 through 2175 is different from repeatedly
restoring position 2048, even if each operation emits one token.

## 7. Sampling and statistical use

One source sample produces one output row. Never repeat an average to manufacture
sample count. Preserve failed/incomplete runs as diagnostics and exclude them
through an explicit admission rule; do not silently cherry-pick successes.

A single retained C sample can be displayed alongside Go samples. The adapter
must not imply that formatting it creates enough independent observations for
confidence intervals or significance claims. benchstat owns statistical analysis;
the importer owns sample identity and faithful formatting.

For new runs, use a predeclared repetition and ordering policy. Interleave C and
Go runs when practical, with equivalent preparation and hardware conditions.
Retain all acquired samples and stop conditions. Repeated tokens inside one run
may be correlated and are not automatically independent experimental runs.

## 8. Compatibility checking options

Three policies are under consideration:

1. **Format only:** validate sample structure and emit rows; users judge semantic
   comparability from provenance. Simple, but easy to misuse.
2. **Checked comparison:** require a matching comparison contract for operation,
   timing/counter semantics, inputs, shape, and preparation. Missing required
   evidence prevents a comparable row while leaving the imported sample usable.
3. **Exploratory comparison:** emit differences with explicit unmatched/unknown
   conditions in a companion report; make no qualification claim.

Recommended: retain format-only import as a mechanical operation and make checked
comparison a separate command. A formatting library should not invent an
application-specific DS4 correctness policy. An exploratory report must not
silently downgrade a failed checked comparison.

Host identities need not always match: cross-device comparison can be intentional.
The comparison contract states which dimensions must match and which are the
experimental variable. Never enforce universal same-host rules by accident.

## 9. CLI and package choices

The current implementation direction is a small Go formatting/import package and
`cuptibench-import` CLI accepting normalized JSON from a file or stdin and writing
Go benchmark text. It has no CUDA execution requirement and does not run commands
embedded in the input.

Proposed later commands, names not committed:

    cuptibench-import -format oracle-json < c-run.json > c.txt
    cuptibench-import -format nsys-sqlite -mapping ranges.json capture.sqlite > c.txt
    cuptibench-compare -contract comparison.json c-samples.json go-samples.json
    benchstat c.txt go.txt

The first two require selected source formats and verified readers; these command
lines describe options, not presently implemented capabilities.

Consider whether the generic formatter should ultimately live outside the
NVIDIA module. Its data model does not inherently require CUDA. Initially keep
it near cuptibench and avoid introducing a broad benchmark framework prematurely.

Retain raw inputs and normalized provenance alongside benchmark text. Comments
in benchmark text help inspection but are not the sole machine-readable evidence
store. Import whole documents with bounded input sizes before emitting output,
so invalid trailing records do not leave a plausible partial comparison file.

## 10. Recommended sequence

1. Finish the format-only normalized JSON reader/writer and CLI. Verify real
   benchstat parsing, stable names, unit handling, and sample preservation.
2. Select one actual retained C artifact. Inspect its producer and schema, then
   implement the smallest faithful reader with retained fixtures.
3. Define the DS4 full-decode comparison contract: exact timing scope, token
   sequence/positions, graph mode, state preparation, and observed correctness.
4. If C source is available, add a direct emitter for repeated measurements.
5. Add in-context attribution through retained Nsight data or CUPTI injection,
   depending on whether timing or counters are the next unanswered question.
6. Add a shared-library bridge only if a common callback harness provides a
   concrete benefit over direct C runs and offline conversion.

Do not implement all acquisition options before choosing a real use case.
The normalized boundary allows the later choice without changing benchstat output.

## 11. Acceptance tests

- One source sample yields exactly one benchmark row; Operations is not replication.
- Total and already-normalized values produce the expected per-operation values.
- Duplicate identities, missing values, unsupported units, overflow, and injected
  newlines fail before output.
- Full timing, traced kernel sums, spans, and isolated timings remain distinct.
- Filtering a single kernel does not invent its input state or parent operation count.
- Fused/unfused mappings reject ambiguous or overlapping coverage.
- C and Go fixtures with the same contract match in actual benchstat output.
- Active-cycle and elapsed-cycle throughput bases cannot be treated as equivalent
  merely because both values are percentages. Unknown bases fail checked comparison.
- Deliberately mismatched operation counts, token positions, or timer methods fail
  checked comparison instead of producing an apparently qualified result.
- A retained real artifact round-trips through its reader with independently
  checked totals and provenance once that format is selected.
- No source-only or synthetic fixture test is reported as a measured C/Go speedup.

## 12. Open decisions

- Should the retained four-capture Nsight CSV fixture be the first source reader?
- Is the primary target end-to-end decode latency, device span, or kernel counters?
- Can the C implementation be rebuilt with explicit ranges and JSON output?
- Are full-pass and isolated-operation state fixtures already available?
- Should the next phase be checked comparison or richer acquisition?
- Which kernel/region mapping is stable across the C and Go implementations?

## 13. Retained Nsight candidate and mapping requirements

Evidence received 2026-09-14: [peer fixture report](agent-collab/lib/20260914-DF6906F6-benchstat-fixture.md).
The four retained files are `c08_cuda_gpu_kern_sum.csv`,
`c40_cuda_gpu_kern_sum.csv`, `g08_cuda_gpu_kern_sum.csv`, and
`g40_cuda_gpu_kern_sum.csv`. The accompanying `c.txt` and `go.txt` demonstrate
formatting only. Their family mapping does not reproduce the cited census.
These are attributed measurements, not qualification of this adapter.

### Differential estimates

The proposed estimate subtracts the 8-token capture from the 40-token capture
and divides by 32. This produces one differential estimate per side, not 32
independent observations. Retain both source hashes, operation counts, subtraction
order, and the derivation. Reject a nonpositive operation-count difference.

A fixed prefill length does not prove its measured duration cancels. Separate
captures may differ in setup, scheduling, cache state, routing, and growing KV
state. Label the result a differential kernel-duration estimate. Confirm the
producer's token-count semantics and position interval before admitting a
per-decode-token interpretation. Negative differences remain diagnostics; never
clamp them to zero or silently drop them.

### Explicit mapping

A source reader preserves exact kernel signatures first. Family assignment is a
separate, caller-supplied, versioned mapping with a content hash. It is a semantic
assertion, not a measured field. Each rule states its predicate and destination.
Default to rejecting multiple matches. If ordered precedence is explicitly
selected, retain the order and report shadowed matches; a broad q8 substring
must not silently consume an expert kernel.

Produce a coverage report for each original capture and each side, including:

- Total, mapped, and unmapped durations and invocation counts.
- Every unmatched signature and its contribution, ordered by duration.
- Multiple matches, precedence resolutions, and destination membership.
- Reconciliation of disjoint mapped plus unmapped totals with the source.
- Mapping version/hash and comparison-contract acceptance threshold.

Do not compute coverage only on the difference of two captures: cancellation and
negative differences can conceal omitted work. A high coverage percentage proves
accounting coverage, not semantic correctness. Validate named families against
an independently established census or reviewed operation inventory as well.
A threshold is contract-specific; no universal percentage makes a comparison
qualified. A failed gate leaves diagnostic artifacts but emits no checked result.

The peer's C experts mapping yielded 22.39 ms against a cited census 15.309 ms;
residual yielded 0.90 ms against 4.634 ms. These discrepancies are a negative
fixture for the mapping gate, not a performance comparison to publish.

### Statistical presentation

Record the independent sample count alongside operation count in the companion
report and an informational benchmark-text comment. Preserve one row per
independent estimate. benchstat may display an aggregate change even when there
are too few observations for useful statistical inference; successful parsing
and a printed percentage do not qualify the comparison.

### Cache metadata

Retain requested and effective cache control, replay mode, preparation, and
whether behavior was documented or measured. Flushed and unflushed measurements
are a sensitivity comparison, not universal upper and lower bounds. Do not
merge those conditions into repeated samples of one benchmark configuration.

### Additional acceptance cases

- The four-capture fixture produces one estimate per side, with both inputs linked.
- A fixed prefill count alone cannot pass a cancellation or timing-equivalence gate.
- Wrong family membership fails despite valid benchmark text and high coverage.
- Overlapping rules, order changes, and unmatched material work are visible.
- Fused C kernels and split Go kernels require a reviewed common semantic scope.
- The known wrong mapping stays labeled format-only through export and reports.

The normalized formatter is implemented; this source reader, mapping validator,
and checked-comparison command are proposed work. No option is selected merely
because this fixture is available.

## References

- [benchstat documentation](https://pkg.go.dev/golang.org/x/perf/cmd/benchstat):
  Go benchmark input format and statistical comparison.
- [Go benchmark format](https://go.dev/design/14313-benchmark-format):
  benchmark records, measurements, and configuration.
- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/):
  retained reports, SQLite export, and statistics processing.
- [CUPTI usage](https://docs.nvidia.com/cupti/main/main.html):
  native range profiling and injection mechanisms.

Local precedents inspected: examples/llama-infer/gpuprof/metrics.go contains an
existing benchmark-line formatter; the read-only DS4 sibling worktree contains
oracle_workload_bench_test.go with explicit timing and token-count boundaries.
Neither is evidence that a specific retained C artifact has been selected or
qualified for conversion.

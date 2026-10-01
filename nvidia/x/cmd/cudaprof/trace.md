# Trace commands

`cudaprof trace export` and `trace compare` invoke the existing export and
export-compare implementations. Legacy commands remain available. Top-level
`compare` still performs timing decomposition. Use `cudaprof help trace check`
for grouped command help.

```
cudaprof trace check -json run.pprof
cudaprof trace check -iteration-marker decode -expect-markers 128 -require coverage run.pprof
cudaprof trace check -perfetto /path/to/trace_processor_shell run.pftrace
```

Input detection uses content rather than the filename. Native capture input is
hashed and parsed from one bounded buffer. The collector gate checks target
failure, retained records, generation completeness, and journal finalization.
Incomplete native inputs keep their failure reason; they supply no qualified
records to downstream analysis.

Checks report independent integrity, collector, import, coverage, stationarity,
shape, and correctness dispositions. A check exits 2 if requested evidence is
failed or unavailable, 1 for malformed input or usage/I/O failure, and 0 when
applicable requested checks pass. Success does not establish every dimension of
performance qualification. Shape and correctness remain unavailable without a
supported independently bound receipt.

Standalone Perfetto packet framing establishes input structure only. Import
health requires a local Trace Processor. Explicit `-perfetto` takes precedence
over `CUDAPROF_TRACE_PROCESSOR` and PATH. The supported adapter is v58.2; it
records executable hash/version and runs the documented query CSV interface.
Its input is a private snapshot of the same hashed bytes. No network download,
browser, or installation occurs. Positive error/data-loss statistics fail the
import check; empty results do not prove collector completeness. Import warnings
remain in the JSON receipt.

Default bounds are 512 MiB input, 256 MiB journals, one million records per
journal, 30 seconds per external invocation, 4 MiB stdout and stderr each, and
10,000 importer rows. External snapshots use `~/tmp/cudaprof-trace` and are
removed after querying. Native summaries need no external executable.

Comparison archives are inspected without extraction. Supported alignment
receipts require two framed trace members, BOOTTIME clocks, matching fixed
offsets, and valid capture identities. Historical archives lack trace-member
hashes and source journals; bound collector qualification remains unavailable.
Standalone traces have no bound native collector, workload, or correctness
receipts. External executable compatibility is deliberately limited to the
validated v58.2 interface; other versions report unavailable evidence.

Summary and named queries are diagnostics: generating an unavailable or partial
view exits 0 and retains the reason prominently in the result. They do not
reconstruct a qualified numerical timeline from an incomplete capture. Invalid
input, usage, I/O, and external execution/CSV/resource failures exit 1. For
`check`, a missing or unsupported importer is unavailable required evidence and
exits 2; an importer that fails to execute or exceeds its bounds exits 1.

Summary and query accept the same input/backend resource override flags as check,
plus `-output-bytes` (default 1 GiB) for diagnostics and HTML. Native
analysis honors input/journal/archive bounds; external analysis also honors
`-backend-timeout`, `-backend-output-bytes`, and `-backend-rows`. Receipts retain
the normalized bounds. HTML writes are atomic and reject input/output aliases,
including hard links and symlinks.

Versioned comparison member bindings verify every retained artifact hash and
source association. Standalone exports with versioned receipts additionally
reconcile imported raw kernel/copy/memset counts. A count mismatch fails the
import qualification; a missing export receipt leaves reconciliation
unavailable. Reconciliation never promotes supplied metadata to native collector
completeness. Shape receipts are recomputed from retained hash-matched original
captures; correctness requires retained campaign stdout/capture artifacts.

GPU isolation is a separate gate. `trace check -require isolation INPUT` needs
sampled clear observer evidence bound to the capture window, device and process
generation. Known contamination fails checks even without `-require isolation`.
Absent observer evidence remains unchecked in diagnostic views; a lock claim
or collector success does not establish sampled clear.

A comparison diagnostic can supply `-shape FILE` and optional `-campaign FILE`:

```
cudaprof trace summary -json -view comparison -shape shape.json comparison.tar
cudaprof trace summary -json -view comparison -shape shape.json -campaign campaign.json comparison.tar
```

The shape receipt must match both archive-bound capture hashes and is recomputed
from the retained original captures. Equivalent-operation distributions and
kernel-sum deltas use the existing comparison-summary analyzer. Selected and
full-capture stationarity are remeasured separately when an iteration marker is
retained. Campaign correspondence, stdout controls, decoded windows, selected
shape gates and the existing conditional block bootstrap are revalidated before
an interval is included. Campaign exchangeability remains an explicit assumption;
its interval is not assigned to an operation. Source artifact stamp claims and
ordinary wall-time measurements remain supplied campaign provenance.

Exported `correlations` and `unmatched` queries require the versioned native
receipts retained by `-submission-links`. They consume exact native results in
`-domain driver|runtime`, preserve unresolved activities, and distinguish zero
unmatched work from missing coverage. Duplicate source identities or inconsistent
coverage remain unavailable. Submission provenance establishes neither causal
dependencies nor collector completeness. Request delivery cadence remains
unavailable without explicit request identity.

## Compact native summary

```
cudaprof trace summary -json run.pprof
cudaprof trace summary -json -top 5 run.pprof
cudaprof trace summary -json -view detailed run.pprof
```

The default native overview uses `cudaprof.trace-summary.v2`. It bounds request,
operation, kernel-shape and device-window rows before encoding. `-top` defaults
to 10 and accepts 1–100. Rows report omitted totals; retained references are
limited to four per row, and long labels carry the complete label's SHA-256.
Raw-query pointers resolve the omitted evidence against the input capture hash.
`-view detailed` retains the exhaustive v1 overview; other views are unchanged.

Requests remain separate by journal, PID, request ID and role. Validated v2
lifecycles show wall and first-delivery latency, consumer spacing and interval
counts grouped by application-reported execution delta. A delta of two counts
one interval. Zero-delta intervals and trailing zero runs remain visible;
they do not prove zero GPU work. Median is the upper middle observed interval.
All-consumer spacing is reported separately from spacing within each delta group.
Invalid lifecycles retain reasons and null metrics. Native v1 endpoints cannot
supply request identity. No clock alignment across journals is inferred.

Collector, correctness, coverage, stationarity and isolation dispositions remain
independent. Application completion evidence does not replace unknown native
GPU completion. Kernel-duration sums and recorded device unions are separate
from request wall time and do not establish a critical path.

## Component execution contracts

```
cudaprof shape-compare -scope geometry base.pprof target.pprof
cudaprof shape-compare -scope component -request-pairs pairs.json base.pprof target.pprof
```

The default model scope and v1 receipt remain unchanged. Geometry scope emits
`cudaprof.kernel-geometry-comparison.v1`, with raw per-arm distributions from the
existing kernel-shapes query. That query groups journal/device/symbol/grid/block;
shared-memory grouping is unavailable. Geometry alone never qualifies application
work. Component scope emits `cudaprof.component-shape-comparison.v1` in a version2
envelope; existing model receipt consumers reject it.

Component NVTX messages use `cudaprof:v3` and require a component contract,
request ID/role, invocation, batch and root count. The contract declares bank and
root-set SHA-256, shape, element strides, dtype, math form, scalar policy,
execution policy and root policy. Signed strides are retained literally; this
metadata does not verify memory bounds. Token, cache, layer and lookahead fields
are forbidden. Version1/2 field and phase rules remain unchanged.

Use instantaneous `batch_start` and `batch_end` events with submission/completion
boundaries and cumulative successful `component_batch` counters. Each completed
batch increments once. Component completion is application evidence, not native
GPU completion. Explicit batch ranges must surround submission. Root ranges
must carry their own root ordinal and complete contract; fields do not inherit.
Shared aliases, shared inputs with distinct roots and distinct banks are separate
root policies. Construction scopes cannot identify kernels submitted by a later
batched Eval; without observed root submissions, root coverage stays unavailable.

Component matching uses exact journal/API correlation and same-process/thread
submission scopes. Asynchronous kernels may execute after host ranges end.
Repeated configurations compare as exact per-root multisets, including retained
dynamic shared memory; no individual occurrence pairing or independent timing
samples are inferred. Missing shared memory, malformed lifecycle/scope metadata,
duplicate or missing roots, unused request mappings, and counter discontinuities
fail qualification. Bank hashes never substitute for execution evidence.

A regex selecting several kernels per explicit batch produces a bounded
ambiguity diagnostic with counts and at most four sample identities, not inferred
component iterations. Explicit lifecycle intervals remain distinct from kernel
sums. Correctness, collector completeness, stationarity, native completion and
isolation remain separate; component analysis cannot promote those gates.

A complete batch submission range message (the example hashes are declarations,
not hashes of supplied fixture tensors):

```text
cudaprof:v3 {"component":{"name":"geglu","bank_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","root_set_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","shape":[256],"strides":[1],"dtype":"bfloat16","math_form":"direct-fused","scalar_policy":"weak-float","execution_policy":"synchronized_eval","root_policy":"distinct_banks"},"invocation":0,"batch":0,"root_count":45,"request_id":"geglu","role":"measured","boundary":"submission","execution_unit":"component_batch","executions":0,"phase":"evaluate","operation":"geglu"}
```

A corresponding request mapping, using the actual journal attachment names and
process IDs retained in each capture, is:

```json
[{"base_journal":"base.journal","base_process":101,"base_id":"geglu","target_journal":"target.journal","target_process":202,"target_id":"geglu","role":"measured"}]
```

For per-root attribution, native submission scopes must additionally carry
`root` ordinals0–44. Marking45 construction calls and later calling a single
synchronized batched Eval identifies neither45 root submissions nor their
individual kernels. That capture can retain batch/geometry diagnostics while
component root qualification remains unavailable.

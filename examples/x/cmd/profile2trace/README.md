# profile2trace

Convert an Nsight Systems SQLite export, ds4-min host sidecar, or Metal region
receipt to native Perfetto protobuf using `nvidia/x/perfetto`. The command runs
without a GPU. Nsight input requires `sqlite3` with `-json` support on PATH;
no Go SQLite driver or module changes are needed.

From `examples/`:

```sh
GOWORK=off go run ./x/cmd/profile2trace -nsys capture.sqlite \
  -nsys-release 2025.3 -nsys-build "$EXPORTER_BUILD" \
  -nsys-clock report-relative > capture.pftrace
GOWORK=off go run ./x/cmd/profile2trace -spans go.spans.jsonl > host.pftrace
GOWORK=off go run ./x/cmd/profile2trace -metal regions.json > metal.pftrace
```

Open the output with **Open trace file** at <https://ui.perfetto.dev>.
The command writes counts and diagnostics to stderr. Check its exit status
before using the output; a write failure can leave a partial trace.

## Nsight input

The consumer calls `nvidia/x/nsightsystems.Import` through a read-only SQLite
adapter. Admission uses the canonical full column schemas: release `2024.2`
requires schema `3.11.0`, and release `2025.3` requires schema `3.20.2`.
Supply the named release, actual exporter build, and clock with
`-nsys-release`, `-nsys-build`, and `-nsys-clock`. Builds are caller declarations;
matching a schema does not authenticate the exporter. Releases `2024.5` and
`2026.1` are refused until the canonical importer has reviewed schema pins.
This also applies to historical captures made with those releases.

`-nsys-clock` accepts `report-relative`, `unix-epoch`, or `unestablished`.
An unestablished clock is usable for a single-source view, but merging it with
other inputs is refused. Session UTC metadata does not establish timestamp
normalization. `-nsys-ts-shift-ns` records an exporter `--ts-shift` value; it is
not applied a second time. Clock declarations require capture evidence.

The input must be a quiescent, regular SQLite export without WAL, SHM, or
journal sidecars. The command copies it to a private read-only snapshot,
checks original identity and content during copying, and verifies the snapshot
again before returning intervals. It records the actual original SQLite and
snapshot SHA256 values. Optional `-nsys-sha256` checks an expected original
digest. These hashes identify the SQLite export, not an unprovided `.nsys-rep`
file. Snapshot checks assume private directory ownership; they do not provide
an atomic backup of a live database or attest the executable loaded by the OS.

Fixed queries use the installed `sqlite3` executable with `-readonly`, immutable
file URI, query-only mode, and startup files disabled. Report text never becomes
SQL identifiers, shell commands, or SQLite dot commands. Virtual tables and
views are refused. Per-call contexts cancel subprocesses. `-nsys-timeout`
(default 30 seconds, maximum 5 minutes) covers copying and import; regular file
I/O checks cancellation between bounded reads. Subprocess wait delay is bounded.

Bounds are checked before accumulating query output: `-nsys-input-bytes`
(default 1 GiB, maximum 64 GiB), `-nsys-bytes` (default 16 MiB, maximum 64 MiB),
and `-nsys-rows` (default 65536, maximum 262144). Query stdout has a cumulative
byte budget, including schema/count queries; stderr is limited to 64 KiB per
process. The row budget includes complete metadata/string tables and
whole-capture diagnostics. Over-limit input is refused, not silently truncated.

Kernel, memcpy, memset, CUDA runtime, NVTX, and OS runtime events admitted by the
canonical importer become intervals. Missing providers remain explicit. A
present `StringIds` table with a missing referenced ID is an error; an absent
string provider leaves names unavailable. Required schema/type/nullability,
release/schema disagreement, ambiguous IDs, and invalid intervals are refused.
NVTX marks and unfinished ranges with NULL ends are omitted and counted.
CPU sample tables and other unsupported providers are not imported.

Each interval retains its source table/rowid, exact projected row values,
original/snapshot digests, selected schema digest, declared exporter and clock,
and full/selected table counts. Serialized GlobalIds are not labeled as OS PIDs.
The log retains export metadata, provider status, and correlation diagnostics.
This evidence has basis `external:nsys`; it is not a local GPU measurement.

`DIAGNOSTIC_EVENT` and its optional severity enum are supplemental whole-capture
queries. Warnings, errors, and unknown severities are printed even when outside
the selected window; named Info/Verbose messages are suppressed. Missing or
empty diagnostics do not prove zero loss. Message text is not parsed as a
machine-readable completeness certificate. Diagnostic schema/query failures
refuse conversion. Check diagnostics and exit status before using the trace.

## Bounding a large capture

`-nsys-window START:END` selects intervals by **start** in the half-open range,
in source-clock nanoseconds. Either bound may be empty. The adapter imports a
selected population and reports original full counts and selected counts for
each supported table. This is an explicit window transform, not a complete
capture. Diagnostics and metadata are never windowed. Markers/open ranges and
absent providers remain visible in the log; windowing cannot repair capture loss.

```sh
GOWORK=off go run ./x/cmd/profile2trace \
  -nsys capture.sqlite -nsys-window 160300000000: \
  -nsys-release 2025.3 -nsys-build "$EXPORTER_BUILD" \
  -nsys-clock report-relative > tail.pftrace
```

`-max-slices` (default 500000) additionally limits selected kernel rows. The
smaller cumulative row/byte bounds still apply. Narrow the window or raise
bounds deliberately when a refusal names an exceeded limit.

## Combining clocks

An offset is **added** to the corresponding source timestamp. For a measured
pair `(source_ns, reference_ns)`, use `reference_ns - source_ns`. Nsight is the
reference when present; otherwise host spans are the reference. A single
source needs no offset. Multiple sources require explicit offsets:

```sh
GOWORK=off go run ./x/cmd/profile2trace \
  -nsys capture.sqlite -spans go.spans.jsonl \
  -nsys-release 2025.3 -nsys-build "$EXPORTER_BUILD" \
  -nsys-clock report-relative \
  -spans-offset-ns "$HOST_TO_NSYS_NS" > combined.pftrace

GOWORK=off go run ./x/cmd/profile2trace \
  -spans go.spans.jsonl -metal regions.json \
  -metal-offset-ns "$METAL_TO_HOST_NS" > combined.pftrace
```

Use zero only when the capture evidence establishes a shared clock. The
converter cannot establish calibration from unrelated start times, match runs
across files, or correct clock drift. In particular, a Metal test receipt and
a separate binary's host sidecar are separate executions; aligning their first
intervals would not make them a simultaneous capture.

After applying offsets, the earliest timestamp becomes zero. Each slice retains
its aligned timestamp, trace origin, and source clock domain as annotations.
The existing writer's slice API does not select a per-event clock domain, so
conversion applies the offsets before writing, rather than emitting an
uncorrelated clock snapshot. Nanoseconds remain integer timestamps; Metal's
seconds are rounded to nanoseconds.

Host sidecars must use `gputrace.host-event/v1`, `interval`, and
`cpu_uptime_ns`, with one nonempty run ID. They have no recorded thread IDs.
Metal `regions` contain command-buffer GPU intervals, not individual kernels.
Overlapping ranges receive additional tracks, preserving intervals without
inventing nesting or host thread identities. Input arrays and output event
ordering are held in memory; this is intended for bounded profiling captures.

## CPU checks

```sh
GOWORK=off go test ./x/cmd/profile2trace -count=1
```

Tests build real local SQLite databases from both canonical synthetic schema
fixtures, check both NVTX name forms and missing-string refusals, reject
missing/empty/invalid captures, validate sidecar schemas,
and independently decode the emitted protobuf to check durations and balanced
slices. Synthetic fixtures do not establish GPU parity or qualification of a
particular live Nsight export. Separate per-call fake-process controls exercise
output/row bounds, context and provider failures, and snapshot custody without
executing SQLite. Authentic export admission and independent trace-processor
qualification of this canonical-consumer revision remain open.

## Offline Perfetto queries

With the official [trace processor](https://perfetto.dev/docs/reference/trace-processor-cli)
installed, run the retained SQL without a browser:

```sh
trace_processor query -f x/cmd/profile2trace/check.sql combined.pftrace
sqlite3 capture.sqlite 'SELECT count(*) FROM CUPTI_ACTIVITY_KIND_KERNEL;'
```

The `cuda_kernel` count must equal the SQLite count. The query also reports
unfinished slices and nonzero parser error statistics. On the synthetic mixed
fixture, Perfetto UI v58.3 and offline trace processor v58.2 both returned two
CUDA slices, two NVTX slices, one host slice, and one Metal slice. Offline
parsing reported zero unfinished slices and no nonzero parser errors.

The preceding Perfetto results are historical output observations, not evidence
that a current canonical import or an authentic export passed this revision.
For windowed input, compare the kernel count with the same start predicate,
not the full source table count.

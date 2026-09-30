# cudaprof trace contracts v1

Track identity is its UUID. `Track.Write` emits the official TrackDescriptor
followed by one TrackEvent instant named `cudaprof.track.v1`, category
`cudaprof.semantic`, on that UUID. Metadata uses official debug annotations;
no custom protobuf extension or display-name selector is required. The writer
preserves distinct UUIDs even when display names match.

Annotations use underscore names, which the supported Trace Processor exposes
as `debug.cudaprof_*` keys in `args`:

| Wire annotation | Meaning |
| --- | --- |
| cudaprof_schema, cudaprof_schema_version | `cudaprof.track.v1`, version 1 |
| cudaprof_role | Machine-readable role independent of display name |
| cudaprof_arm, cudaprof_machine | Independent comparison arm and machine identity |
| cudaprof_capture_sha256, cudaprof_journal | Exact source capture and process generation |
| cudaprof_clock | Scoped exported clock identity; alignment does not imply clock synchronization |
| cudaprof_device | Optional device identity; present zero is device zero |
| cudaprof_scope_id, cudaprof_hierarchy_level | Explicit scope and hierarchy position |
| cudaprof_evidence_provenance | Observed, inferred, unsupported or unavailable provenance |
| cudaprof_aggregation | none, elapsed, host_union, gpu_union or kernel_sum |
| cudaprof_canonical | Sole counting representation for this scope and aggregation |

Emit metadata once per track. Canonical parent totals and noncanonical child
hierarchies remain separate. `ValidateTracks` rejects repeated canonical
arm/capture/journal/device/scope/aggregation representations and parent cycles.
A GPU union is not a kernel sum; neither is elapsed host time.

Verified SQL access through the official importer:

```sql
SELECT track_id,
  EXTRACT_ARG(arg_set_id, 'debug.cudaprof_role') AS role,
  EXTRACT_ARG(arg_set_id, 'debug.cudaprof_scope_id') AS scope_id,
  EXTRACT_ARG(arg_set_id, 'debug.cudaprof_aggregation') AS aggregation,
  EXTRACT_ARG(arg_set_id, 'debug.cudaprof_canonical') AS canonical
FROM slice
WHERE name = 'cudaprof.track.v1';
```

Join these rows by `track_id` to measurements on the same track. Exclude
`cudaprof.semantic` metadata instants from measurement rows. Count only canonical
representations for the requested aggregation; do not sum parent and child
work. Imported `slice.id` values identify loaded rows, not stable source records.

`Source.String` encodes compact JSON with capture_sha256, journal, explicit kind
and exactly one raw sequence or decoder index. A sequence is preferred; an
index is a documented fallback for older journals. Display labels are absent.

`Receipt.Write` emits `cudaprof.receipt.v1` with complete bounded JSON in
`cudaprof_receipt_json`. Collector, import health, workload coverage,
stationarity, shape and correctness gates are mandatory independent
passed/failed/unavailable/unchecked/unsupported dispositions. SQL can read
`debug.cudaprof_gate_correctness` and the corresponding keys directly. Successful
collector evidence does not promote correctness or workload coverage.

Comparison TAR archives store `.cudaprof-bindings.json` outside the official
Perfetto manifest's trace file list. The canonical index hashes exact bytes and
sizes of every artifact, including `perfetto_manifest.json`, `.alignment.json`
and each trace. Trace members also identify their source capture SHA256. The
index cannot hash itself; the analysis receipt separately hashes the exact
archive bytes parsed. `Bindings.Verify` rejects changed, missing and unbound
members. These hashes establish internal integrity, not signed authenticity.
Historical archives without bindings remain usable diagnostics with integrity
qualification unavailable; absence must not be silently treated as passed.

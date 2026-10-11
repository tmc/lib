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

## Extended trace receipts and backward reading

`Receipt` optionally carries granular evidence fields:
- `probe_coverage`: map of probe name to `{present, ratio}`
- `lost_records_by_family`: map of family name to `{dropped, total, unknown}`
- `capture_bounds`: `{start_ns, end_ns, clock}`
- `correlation_outcome`: `{matched, unmatched, ambiguous, missing, status}`
- `clock_domain`, `clock_provenance`, `clock_uncertainty_ns`, `uncertainty`

### Backward reading behavior
In older version 1 receipts where these fields were omitted:
- Nil `lost_records_by_family` indicates record loss was unassessed, not zero loss.
- Nil `probe_coverage` indicates probe presence was unassessed.
- Nil `capture_bounds` indicates bounds were unmeasured.
- Unspecified clock fields indicate clock domain was unstated.
- Global `partial: true` indicates incomplete collection overall, but does NOT universally invalidate independent findings whose specific evidence requirements are fully satisfied.

## Diagnostic finding admission v1

`Finding.Write` emits `cudaprof.finding.v1` on a dedicated findings track:

| Wire annotation | Meaning |
| --- | --- |
| cudaprof_schema, cudaprof_schema_version | `cudaprof.finding.v1`, version 1 |
| cudaprof_finding_id | Unique identifier for the finding |
| cudaprof_finding_rule | Diagnostic rule name producing the finding |
| cudaprof_finding_class | `observed`, `associated`, `experimentally_confirmed`, or `unavailable` |
| cudaprof_finding_admitted | Boolean admission status |
| cudaprof_finding_reason_code | Stable reason code (e.g. `admitted`, `missing_return`, `unknown_loss`) |
| cudaprof_finding_json | Complete deterministic JSON payload |

### Admission semantics
Admission is computed strictly from the evidence each finding actually requires:
- A missing call return blocks call duration (`missing_return`), but an independent launch count remains admissible.
- Lost stack records withhold stack-sampling profiles, but do not invalidate an independently complete launch count.
- Clock domain mismatches block cross-domain correlation (`clock_domain_mismatch`), leaving intra-domain findings intact.
- Unquantified record loss in a required family blocks conclusions with `unknown_loss`.
- Out-of-bounds intervals are withheld with `capture_bounds_exceeded`.
- `experimentally_confirmed` findings additionally require an explicit, verified intervention and control receipt.
- Downstream prose/LLM summaries cannot upgrade a finding's evidence class.

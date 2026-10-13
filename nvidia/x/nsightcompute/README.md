# Offline Nsight Compute import

This package reads project envelope version 1 around a documented NVIDIA
details CSV export. It does not parse `.ncu-rep`, invoke `ncu`, or collect GPU
data. The supported exporter is **2026.3.0.0**, with format profile
`ncu-details-csv-2026.3.0`. This is not an NVIDIA JSON schema. The exact real
header and omitted trailing empty fields are pinned by the bundled fixture.

The exporter must select details, CSV, base units, metric names, mangled kernel
names, and no summary. See [the fixture provenance](testdata/README.md) for the
actual command and original report/export digests.

## Envelope

The required JSON members are:

| Member | Meaning |
| --- | --- |
| `schema_version` | Project envelope version, exactly 1 |
| `tool_version` | Exporter release, exactly `2026.3.0.0` |
| `format` | Exact supported CSV header profile |
| `source_path` | Original report provenance; never opened by the importer |
| `report_digest` | Producer-supplied SHA256 of the original opaque report |
| `csv_digest` | SHA256 recomputed over the embedded exact CSV text |
| `csv` | Original CSV text, not reconstructed numeric rows |
| `bindings` | Optional explicit external launch-to-identity producer receipts |

The report digest is recorded, not independently authenticated from CSV. An
exporter version does not establish the original collection tool version.
These distinctions also apply to NVIDIA's bundled report fixture.

A binding contains `launch: {id, process, context, stream}` copied exactly from
the export, and the existing `cudaevidence.Identity` fields `artifact_digest`,
`entry`, `target`, `compiler_key`, and `workload_key`. Missing bindings stay
unjoined; CSV names and dimensions never fill their identity fields.

## Field mapping

| Details CSV fields | Imported representation |
| --- | --- |
| ID, Process ID, Context, Stream | Exact decimal string `LaunchKey` |
| Kernel Name | Exact mangled `Entry`; no demangling or fuzzy matching |
| Block Size, Grid Size | Three positive uint32 dimensions, checked product |
| CC | `sm_` target from major/minor; no feature suffix inference |
| Section Name | Raw section name |
| Metric Name, Metric Unit, Metric Value | Raw strings plus classified `Metric` |
| Rule fields, speedup fields | Preserved raw diagnostic row, unsupported |
| Process Name, Host Name, Device | Preserved raw fields, no ownership attestation |

Every row retains the original field vector. No rate or percent is converted
into a FLOPs, latency, efficiency, occupancy, or local CUPTI observation.

## Classification and joins

`Import` admits a numeric `cudaevidence.Measured` value only after an explicit
external identity matches an actual `Candidate` receipt created with the
existing artifact/workload/compiler identities. It then requires a native
CUBIN whose decoded target matches, a unique retained module generation, and
`cudacapture.VerifyModule` success for the exact entry. Conflicting capture
target/workload metadata vetoes admission. Candidate order resolves nothing.
The receipt verifies producer records against bytes, not a driver attestation
or the truth of an external measurement.

Available values use basis and producer **`external:ncu`** with a report/CSV/row
locator. The capture's local observations are not modified. Missing identity,
stale bytes/entry/target/compiler/workload, unavailable code roles, or ambiguous
generations remain unjoined and `Unread`.

The small admitted scalar table is explicit in `import.go`: duration sum,
elapsed cycle maximum, instruction sums, registers per thread, shared-memory
sizes, and thread count. Names and base units must match exactly. Unsupported
rates, percentages, averages, derived/instanced metrics, unknown denominators,
decimal/nonfinite values, rule speedups and units remain raw plus `Unread`.
Unsigned integers remain exact in `Metric.Unsigned`; values above 2^53 are not
rounded into the float64 evidence model. Numeric zero is available when its
identity and supported metric are admitted. No sample-loss denominator is
invented when the export does not supply one.

## CLI

```sh
cudaprof import-ncu -limit 16 envelope.json
cudaprof import-ncu -json -limit 128 envelope.json
cudaprof import-ncu -json -capture capture.pprof \
  -join-receipt receipt.json envelope.json
```

The separate project join receipt is
`{schema_version: 1, artifact: <actual cudakernel.Artifact>, workload:
<actual cudacache.WorkloadIdentity>, compiler_key: <original compiler key>}`.
Its artifact raw bytes/digest are validated; workload and compiler keys are
bound through the existing `cudaprof.BindEvidence`. Supplying this receipt and
capture alone cannot create a missing external launch binding.

The default summary contains 16 rows; the accepted range is 1..1024. JSON
summaries retain the selected rows' raw fields and exact integers, state total
counts and mark truncation. Text summaries clip individual fields to 128 bytes
and quote them. The importer report retains all rows and the complete export.

Inputs and output use a maximum 32 MiB byte budget. Capture decompression is
bounded separately before profile parsing. CLI files must be regular files,
not symlinks. JSON rejects duplicate, aliased, unknown and trailing fields.
Unsupported schema/exporter/header profiles and inconsistent launch metadata
are errors. Binding count/bytes, candidates, descriptor inventories, retained
code bytes, field lengths and rows are bounded. Receipts are verified once per
candidate, and joins once per compatible identity, not once per CSV row.

Tests and examples run offline. A real export without external identities is
an unjoined positive fixture, not proof that the bundled kernel matches a local
image. Positive numeric joins are explicitly synthetic producer controls backed
by real retained CUBIN fixtures. No runtime Nsight dependency is required.

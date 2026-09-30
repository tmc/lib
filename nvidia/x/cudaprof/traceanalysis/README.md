# Native summaries and queries

The CLI decodes each native input once and runs an immutable bounded analysis
session. Standard library output keeps explicit delivery endpoints, inferred
kernel-marker windows, materialization host scope unions, recorded GPU unions
and kernel duration sums separate.

```sh
cudaprof trace summary -json -domain runtime capture.pprof
cudaprof trace summary -html tokens.html -view tokens capture.pprof
cudaprof trace query -list -json
cudaprof trace query -json recorded-gaps capture.pprof
cudaprof trace query -json kernel-shapes capture.pprof
```

Named schemas list versions, backend support, units, required evidence and
aggregation. Native context uses the requested driver/runtime domain, without
fallback. Raw references carry process journal, decoded index and native record
sequence when present. A failed collector gate retains diagnostic inventory and
suppresses numerical views. Native correctness, workload coverage and importer
health do not follow from completeness. An explicit iteration marker invokes
the existing stationarity gate only in one journal/device scope; its failed
receipt remains visible separately.

Version 1 application markers do not retain request IDs or request-start events.
Delivery/EOS/token-limit observations are retained, but cross-token request
cadence and first-token latency remain unavailable. Reused phase/token numbers
cannot establish multiple-request boundaries. Native dependencies use only
retained ordering/event/graph receipts; coverage and delivery endpoints remain
partial unless independently qualified. Submission identity is not dependency
causality.

A comparison archive summary preserves both arm identities and selected/full
stationarity receipts. It verifies `.cudaprof-bindings.json` when present.
Hash consistency does not authenticate supplied receipts. No numerical operation
delta is produced without matched application shapes; no engine-run uncertainty
is substituted with per-launch distributions. Missing campaign receipts remain
unavailable.

## PerfettoSQL and structured summaries

The optional backend uses an existing local **Perfetto v58.2** executable. It
performs no download or installation. The following commands were qualified
against the local official v58.2 CLI:

```sh
TP=/path/to/trace_processor_shell
"$TP" --version
"$TP" --add-sql-package ./sql/cudaprof query run.pftrace \
  'INCLUDE PERFETTO MODULE cudaprof.views; SELECT * FROM cudaprof_device_totals_v1;'
"$TP" --add-sql-package ./sql/cudaprof summarize --metrics-v2 all \
  run.pftrace ./sql/summary.textproto
cudaprof trace query -json -perfetto "$TP" recorded-gaps run.pftrace
cudaprof trace summary -json -view gaps -perfetto "$TP" run.pftrace
```

Run module commands from this directory. SQL consumes `cudaprof.track.v1`
metadata instants, explicit `raw_device` role, and per-record
`cudaprof_activity_kind`. It never guesses roles from names. Annotation keys use underscores; Perfetto adds the `debug.` prefix (`debug.cudaprof_schema`). Summary/hierarchy
representations never join raw duration totals. Duplicate track metadata and
duplicate native sequence identity are excluded. Union partitions include exact
capture, arm, journal, device and clock. Window merging uses a running maximum
end, handling nested intervals. SQL interval endpoints are in the exported
clock; duration totals reconcile with native nanoseconds.

Old or external traces without these roles remain unavailable for role-sensitive
queries. Supplied exported annotations are provenance, not independently
measured collector/correctness evidence. External overview is a partial role
inventory; native token/operation/dependency views require their retained
metadata. The initial SQL delivery/correlation/unmatched modules return explicit
unavailable states until versioned request/submission receipts are supplied.
Native versions of those queries preserve the observations and matching reasons.

The CPU-only parity test emits a valid fixture with overlapping streams and a
repeated hierarchy representation. It compares native union/kernel counts and
sums, complement gaps, missing cadence states, and runs the structured spec:

```sh
CUDAPROF_TRACE_PROCESSOR="$TP" go test ./x/cudaprof/traceanalysis -run TestSQLParity
```

Run that test command from the nvidia module root. It is opt-in when the pinned
local executable is absent. [PerfettoSQL modules](https://perfetto.dev/docs/analysis/perfetto-sql-syntax)
and [TraceSummarySpec](https://perfetto.dev/docs/analysis/trace-summary) document
the native SQL/package and structured-summary interfaces used here.

The CLI exporter parity test also decodes native journal fixtures and runs the
actual raw plus overview exporters. It reconciles imported raw counts against
versioned export receipts and compares overlapping interval unions and kernel
shape totals against native analysis:

```sh
CUDAPROF_TRACE_PROCESSOR="$TP" go test ./x/cmd/cudaprof -run TestTraceExportSQLParity
```

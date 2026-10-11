# Activity findings

`AnalyzeActivityJSONL` consumes the existing `cuptiactivity.Record` JSON-lines
export and a `tracecontract.Receipt`. The receipt hashes the exact input bytes.
Caller metadata supplies the actual collection session, process generation,
run and CUPTI timestamp epoch; host timestamps never supply a device epoch.

The bounded parser rejects unknown record fields, multiple values per line,
missing final newline, changed raw hash and records exceeding its limits.
`AnalyzeActivities` also accepts an already decoded frozen batch. It retains
records, native/external IDs and identity outcomes in original record order.
Findings are sorted by raw-record digest for replay.

The result schema is version 1. The existing receipt schema remains version 1;
new intervention scope fields are optional when reading legacy receipts.
Legacy receipts lacking rule/workload/operation scope remain observations and
cannot confirm a workload-specific mechanism. A precise operation duration
requires explicit external correlation and complete native timestamps. A
missing identity withholds that row. It does not invalidate a different row.
Captured intervals are neither complete workload time nor a causal diagnosis.
`Observe` separately evaluates per-call return timing and device interval-union
coverage so unrelated missing evidence does not become a global refusal.

`testdata/activity-v1.jsonl` is a portable known-answer fixture, not a physical
GPU receipt. Its kernel maps to operation 9 through native correlation 7. Tests
mutate its identity, clock, raw hash and termination. Physical CUPTI layout and
correlation qualification remain distinct from portable parser controls.

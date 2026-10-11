# Application workload outcomes

`ReadTrainingContext` and `ReadServingContext` read bounded version 1 application
interchanges. The corresponding writers validate the same identities and bounds
before emitting any bytes. They do not parse vendor exporters. Attach an
existing `WorkloadSource` identifying the application's version and run, retain
its original export and SHA-256, and keep the referenced application artifacts.

Training updates use an optimizer step, attempt, rank and world size. Experiment
logging indices are not optimizer outcomes. Summaries require every expected
rank; resumed attempts remain separate. Successful updates with accepted quality
are counted separately from rejected or unknown quality, skipped updates,
failures and incomplete rank sets. Quality policy and evidence references are
caller assertions; a reference is not independently verified by the reader.
No checkpoint durability or training convergence is inferred.

Serving outcomes use supplied request, attempt, worker and worker restart epoch.
Summaries retain retries as separate attempts and distinguish accepted, failed,
cancelled, partial and unknown outcomes. An output policy is required for an
accepted attempt. Output token counts retain presence and exact uint64 values;
unknown counts do not become zero. Aggregate scrapes cannot produce these
records, and totals across attempts do not mean unique successful requests.

These context records have no time-to-kernel join or GPU allocation semantics.
They deliberately do not assign device time, counters, energy, cache capacity or
shared batch work to individual requests or training ranks. Existing
`cudaworkload` correlation readers and producer contracts own those joins.
Operator fusion membership, framework/native ID mappings, KV transfer identity,
clock calibration, physical checkpoint receipts and authentic vendor exports
still require independent artifacts and qualified consumer contracts.

Readers refuse duplicate JSON keys, unknown fields, unsupported versions/status,
trailing input, duplicate composite identities, inconsistent training world
sizes and quality policies, oversize data, invalid ranks and fractional or
negative integer counts. Writers check their encoded byte budget incrementally
and propagate errors, including short writes. Runnable public examples are in
`workload_context_test.go`.

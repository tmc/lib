// Package nvtelemetry collects NVIDIA telemetry through NVML, DCGM, and bounded
// documented exports, and records JSONL streams for replay and window summaries.
// It also provides explicit identity joins, health bundles, workload adapters,
// accounting models, and bounded publication and retention helpers.
//
// Readings are mapped to canonical units (percent, bytes, milliwatts,
// millijoules, Celsius, MHz, and clock event reasons) with explicit evidence
// classification and unavailable reasons. Fake and scripted samplers are
// provided for deterministic testing without physical hardware.
//
// Version 2 streams preserve per-query timing, device capabilities, MIG scope,
// and process-local clock identity. ReadStream also accepts version 1 streams.
// Missing sensor timestamps and clock uncertainty remain unknown: query bounds
// do not establish sensor freshness or synchronization with another process.
// Integer readings retain their uint64 value even when the accompanying float
// is rounded. Bitmask and event readings are excluded from numeric summaries;
// AggregateWindowStrict reports an error instead.
//
// AggregateHistogramWindow separately accumulates V5 delta distributions or
// differences compatible cumulative endpoints over declared source intervals.
// It requires completed canonical accounting, stable identity and known-zero
// declared clock bounds. V3/V4 temporal semantics, unknown bounds and straddling
// distributions are explicitly withheld. Query time never replaces source time.
// CDF counts remain exact; optional sums use finite float64 arithmetic and
// cumulative extrema are not derivable. Observer population coverage stays
// unknown when not assessed. A recovered reset between observations remains
// unobservable. Results describe recorded distributions, not individual latency,
// source authenticity, native/backend qualification or physical-time correlation.
// The legacy scalar window APIs return scalar summaries only; mixed histogram
// boundaries reduce scalar coverage and carry a reason. Replay is unchanged.
package nvtelemetry

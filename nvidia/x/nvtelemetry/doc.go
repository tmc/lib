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
package nvtelemetry

// Package cuptisass provides CUPTI SASS metrics data collection for NVIDIA GPUs.
//
// SASS metrics collection dynamically instruments kernel instructions at runtime
// to collect hardware metrics such as instruction execution counts, memory
// traffic, and branch behaviors at per-instruction PC granularity.
//
// SupportedMetrics queries the available SASS metrics for a target GPU chip.
// MetricProperties inspects details for a specific metric name.
// NewSession establishes an instrumented collection session on a CUDA context,
// enabling instruction patching across kernel launches and flushing decoded
// records per instruction.
//
// Range, PM sampling, and SASS metric sessions share a process-local device
// lease. This library policy avoids competing measurements and instrumentation;
// it is not a claim that all native profiler pairs are incompatible. Activity
// tracing is not excluded. Callers still coordinate PC sampling, raw CUPTI
// bindings, other libraries, and other processes when measurement integrity
// requires isolation. Native errors remain authoritative for external conflicts.
// Close must succeed before dropping a session or destroying its context. Native
// cleanup failures retain the lease and resources for an explicit retry.
//
// Collection is restricted to Linux amd64/arm64 and qualified CUPTI versions
// (24, 26, 28, 130001, 130101, 130200, 130202, 130301, 130400).
package cuptisass

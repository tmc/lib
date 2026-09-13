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
// Collection is restricted to Linux amd64/arm64 and qualified CUPTI versions
// (24, 26, 28, 130001, 130101, 130200, 130202, 130301, 130400).
package cuptisass

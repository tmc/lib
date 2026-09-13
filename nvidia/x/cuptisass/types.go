package cuptisass

import "github.com/tmc/lib/nvidia/cupti"

// Metric represents an enumerated hardware SASS metric supported on a chip.
type Metric struct {
	ID          uint64
	Name        string
	Description string
}

// OutputGranularity specifies the metric aggregation granularity across the GPU.
type OutputGranularity uint8

const (
	// GranularityGPU aggregates metrics across the entire GPU device.
	GranularityGPU OutputGranularity = OutputGranularity(cupti.SassMetricsOutputGranularityGpu)
	// GranularitySM aggregates metrics per Streaming Multiprocessor.
	GranularitySM OutputGranularity = OutputGranularity(cupti.SassMetricsOutputGranularitySm)
	// GranularitySMSP aggregates metrics per SM Sub-Partition.
	GranularitySMSP OutputGranularity = OutputGranularity(cupti.SassMetricsOutputGranularitySmsp)
)

func (g OutputGranularity) String() string {
	switch g {
	case GranularityGPU:
		return "GPU"
	case GranularitySM:
		return "SM"
	case GranularitySMSP:
		return "SMSP"
	default:
		return "Unknown"
	}
}

// MetricConfig specifies a metric and its output granularity for a session.
type MetricConfig struct {
	MetricID    uint64
	Granularity OutputGranularity
}

// Config specifies the configuration for SASS metric collection.
type Config struct {
	// DeviceIndex is the 0-based CUDA device ordinal.
	DeviceIndex uint32

	// Metrics is the list of metrics to instrument and record.
	Metrics []MetricConfig

	// LazyPatching enables on-demand instruction patching when each kernel
	// executes rather than eagerly patching all module functions on Enable.
	LazyPatching bool
}

// InstanceValue holds a single metric value instance (e.g. per-GPU, per-SM, or per-SMSP).
type InstanceValue struct {
	MetricID uint64
	Value    uint64
}

// Record holds the collected SASS metrics for a single instruction PC offset in a kernel.
type Record struct {
	CubinCRC      uint32
	FunctionIndex uint32
	FunctionName  string
	PCOffset      uint32
	Values        []InstanceValue
}

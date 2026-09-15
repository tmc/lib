package cuptibench

import (
	"github.com/tmc/lib/nvidia/x/cuptirange"
	"time"
)

// Metric is one native evaluated value. Unit is the native dimensional unit,
// or "unknown". Additive is true only for independently verified semantics;
// an integral value or a .sum suffix alone does not establish additivity.
type Metric struct {
	Name     string
	Unit     string
	Value    float64
	Additive bool
}

// Range identifies a collected operation, region, or kernel. Inclusive parents
// must not be added to their descendants. Metrics follow configuration order.
type Range struct {
	ID       string
	ParentID string
	Name     string
	// NativeName retains CUPTI's range name, which may be a launch ordinal.
	NativeName string
	Kind       string
	Inclusive  bool
	Metrics    []Metric
}

// Timing records a separate, uninstrumented execution, normalized by logical
// operations rather than replay passes or Go benchmark iterations.
type Timing struct {
	Operations int64
	Total      time.Duration
	Method     string
}

// Result retains one counter capture. On collection error only Quality and
// Metadata are diagnostic; Ranges is empty. Timing is nil for Collect.
// Reserved metadata keys include chip, device_uuid, cuda_driver_version,
// cupti_version, metrics, mode, preparation, capture_id, benchmark, and run_id.
type Result struct {
	// Aggregate is a separately identified whole-operation capture acquired by
	// Benchmark in Kernels mode. It is retained for aggregate columns, never added
	// to the individual kernel samples in a profile. Collect leaves it nil.
	Aggregate  *Result `json:",omitempty"`
	Operations int
	Ranges     []Range
	Timing     *Timing
	Quality    cuptirange.CollectionQuality
	Metadata   map[string]string
}

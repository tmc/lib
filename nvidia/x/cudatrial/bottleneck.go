package cudatrial

// BottleneckFamily identifies an exploration category, not a hardware verdict.
type BottleneckFamily string

const (
	MemoryBandwidthLimited   BottleneckFamily = "memory_bandwidth_limited"
	ComputeThroughputLimited BottleneckFamily = "compute_throughput_limited"
	LatencyOccupancyLimited  BottleneckFamily = "latency_occupancy_limited"
	MemoryLatencyBound       BottleneckFamily = "memory_latency_bound"
	MemoryBankConflicts      BottleneckFamily = "memory_bank_conflicts"
	CacheInefficient         BottleneckFamily = "cache_inefficient"
	HybridBound              BottleneckFamily = "hybrid_bound"
	APIOverheadDominant      BottleneckFamily = "api_overhead_dominant"
)

// Valid reports whether f is one of the eight supported categories.
func (f BottleneckFamily) Valid() bool {
	switch f {
	case MemoryBandwidthLimited, ComputeThroughputLimited, LatencyOccupancyLimited,
		MemoryLatencyBound, MemoryBankConflicts, CacheInefficient, HybridBound, APIOverheadDominant:
		return true
	}
	return false
}

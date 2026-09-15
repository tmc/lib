package cuptirange

import (
	"fmt"
	"unsafe"
)

// CachePolicy selects CUPTI's reset policy for profiling passes. It does not
// describe application cache contents, warmup, eviction, or timing executions.
// Comparing policies is an experiment, not a universal upper/lower bound.
type CachePolicy int

const (
	// CacheDefault leaves the native default in effect. Its documented behavior
	// on admitted versions is no CUPTI reset; physical behavior remains unmeasured.
	CacheDefault CachePolicy = iota
	// CachePreserve requests no CUPTI cache reset. Workload preparation and
	// preceding replay passes can still change cache contents.
	CachePreserve
	// CacheFlushL2 requests an L2 reset before each profiling pass, not each kernel.
	CacheFlushL2
	// CacheFlushAll requests all cache state supported by the native version.
	// CUPTI 13.4 documents that its supported set currently consists of L2.
	CacheFlushAll
)

// String returns the policy's stable metadata spelling.
func (p CachePolicy) String() string {
	switch p {
	case CacheDefault:
		return "default"
	case CachePreserve:
		return "preserve"
	case CacheFlushL2:
		return "flush-l2"
	case CacheFlushAll:
		return "flush-all"
	}
	return fmt.Sprintf("invalid(%d)", p)
}

// CachePolicySupport checks native-version and binding admission without
// executing workload code. Admission is an API contract, not hardware proof.
// Cache resets were added in CUPTI 13.4; NONE preserves earlier behavior:
// https://docs.nvidia.com/cupti/13.4.0/main/main.html#cache-state-reset
func CachePolicySupport(version uint32, policy CachePolicy) error {
	if policy < CacheDefault || policy > CacheFlushAll {
		return fmt.Errorf("cuptirange: invalid cache policy %d", policy)
	}
	if !qualifiedLayout(version) || unsafe.Sizeof(uintptr(0)) != 8 {
		return fmt.Errorf("cuptirange: unqualified cache policy version %d", version)
	}
	if policy == CacheFlushL2 || policy == CacheFlushAll {
		return fmt.Errorf("cuptirange: cache reset ABI not qualified for CUPTI %d", version)
	}
	return nil
}

package cudamem

import (
	"errors"
	"fmt"
	"strings"
	"unsafe"
)

// MemLocationType indicates the physical residency location type of a memory address.
type MemLocationType int

const (
	// LocationTypeInvalid indicates unknown or unmapped memory.
	LocationTypeInvalid MemLocationType = 0
	// LocationTypeDevice indicates the memory is resident on a GPU device.
	LocationTypeDevice MemLocationType = 1
	// LocationTypeHost indicates the memory is resident in host CPU memory.
	LocationTypeHost MemLocationType = 2
	// LocationTypeHostNUMA indicates the memory is resident on a specific host NUMA node.
	LocationTypeHostNUMA MemLocationType = 3
	// LocationTypeHostNUMACurrent indicates the memory is resident on the executing thread's NUMA node.
	LocationTypeHostNUMACurrent MemLocationType = 4
)

func (t MemLocationType) String() string {
	switch t {
	case LocationTypeDevice:
		return "device"
	case LocationTypeHost:
		return "host"
	case LocationTypeHostNUMA:
		return "host_numa"
	case LocationTypeHostNUMACurrent:
		return "host_numa_current"
	default:
		return "none"
	}
}

// MemLocation specifies the physical device or host NUMA node where a page is resident.
type MemLocation struct {
	Type MemLocationType
	ID   int // Device ordinal or NUMA node ID
}

// errLocationInfoUnavailable reports that the driver entry point backing
// [LocationInfo] is not bound.
var errLocationInfoUnavailable = fmt.Errorf("cudamem: page locality query requires cuMemGetLocationInfo, which package driver does not bind: %w", errors.ErrUnsupported)

// LocationInfo reports the physical residency of the pages backing
// [ptr, ptr+size), as one [MemLocation] per summaryGranularity-byte region.
// samplingGranularity is the intended sampling step within each region.
//
// The query is not implemented. It requires cuMemGetLocationInfo, which
// package driver does not bind, so LocationInfo validates its arguments and
// then returns an error wrapping [errors.ErrUnsupported]. It never reports a
// residency it has not measured.
func LocationInfo(ptr uintptr, size, summaryGranularity, samplingGranularity uint64) ([]MemLocation, error) {
	if ptr == 0 {
		return nil, errors.New("cudamem: nil pointer for location query")
	}
	if size == 0 {
		return nil, nil
	}
	if summaryGranularity == 0 || summaryGranularity > size {
		return nil, errors.New("cudamem: invalid summary granularity")
	}
	if samplingGranularity > summaryGranularity {
		return nil, errors.New("cudamem: sampling granularity exceeds summary granularity")
	}
	return nil, errLocationInfoUnavailable
}

// LocationInfo reports the page locality of the device slice across its address
// range. See [LocationInfo]: the query is not implemented and returns an error
// wrapping [errors.ErrUnsupported] for any non-empty slice.
func (s Slice[T]) LocationInfo(summaryGranularity, samplingGranularity uint64) ([]MemLocation, error) {
	if s.len == 0 {
		return nil, nil
	}
	totalBytes := uint64(s.len) * uint64(s.elemSize)
	return LocationInfo(uintptr(s.ptr), totalBytes, summaryGranularity, samplingGranularity)
}

// LocationInfo reports the page locality of the host-allocated page-locked
// memory. See [LocationInfo]: the query is not implemented and returns an error
// wrapping [errors.ErrUnsupported] for any non-empty allocation.
func (h HostAllocation[T]) LocationInfo(summaryGranularity, samplingGranularity uint64) ([]MemLocation, error) {
	if h.len == 0 {
		return nil, nil
	}
	var t T
	elemSize := unsafe.Sizeof(t)
	if elemSize == 0 {
		elemSize = 1
	}
	totalBytes := uint64(h.len) * uint64(elemSize)
	return LocationInfo(h.ptr, totalBytes, summaryGranularity, samplingGranularity)
}

// FormatLocationSummary returns a human-readable string summarizing the page locality distribution.
func FormatLocationSummary(locs []MemLocation) string {
	if len(locs) == 0 {
		return "empty"
	}
	counts := make(map[string]int)
	for _, l := range locs {
		key := fmt.Sprintf("%s:%d", l.Type, l.ID)
		counts[key]++
	}
	var b strings.Builder
	for k, count := range counts {
		if b.Len() > 0 {
			b.WriteString(", ")
		}
		pct := float64(count) / float64(len(locs)) * 100
		fmt.Fprintf(&b, "%s (%.1f%%)", k, pct)
	}
	return b.String()
}

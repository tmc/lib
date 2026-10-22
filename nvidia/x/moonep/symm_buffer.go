//go:build !windows

package moonep

import (
	"fmt"
	"sort"
	"strconv"
	"strings"
	"sync"
)

// SymmBufferKey describes the symmetric memory allocation requirement for a
// communication group.
type SymmBufferKey struct {
	GroupName  string
	MaxBytes   int64
	AlignBytes int64
}

// SymmViewSpec describes a zero-allocation sub-view within a communication
// group's pre-allocated symmetric buffer.
type SymmViewSpec struct {
	Shape      []int
	ElemBytes  int
	ByteOffset int64
}

// SymmBufferManager tracks high-water symmetric buffer requirements across
// communication groups and caches sub-view descriptors so CUDA Graph capture
// and replay reuse identical addresses without re-allocating.
type SymmBufferManager struct {
	mu        sync.Mutex
	groups    map[string]SymmBufferKey
	viewCache map[string]SymmViewSpec
}

// NewSymmBufferManager creates an initialized SymmBufferManager.
func NewSymmBufferManager() *SymmBufferManager {
	return &SymmBufferManager{
		groups:    make(map[string]SymmBufferKey),
		viewCache: make(map[string]SymmViewSpec),
	}
}

func alignBytesUp(n, align int64) int64 {
	if align <= 1 {
		return n
	}
	rem := n % align
	if rem == 0 {
		return n
	}
	return n + (align - rem)
}

// PlanAllocations records the high-water allocation requirements across
// communication groups and returns the consolidated allocations sorted
// largest-first by MaxBytes (breaking ties by GroupName).
func (m *SymmBufferManager) PlanAllocations(specs []SymmBufferKey) ([]SymmBufferKey, error) {
	if m == nil {
		return nil, fmt.Errorf("moonep: nil SymmBufferManager")
	}
	if len(specs) == 0 {
		return nil, fmt.Errorf("moonep: specs must not be empty")
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	if m.groups == nil {
		m.groups = make(map[string]SymmBufferKey)
	}
	if m.viewCache == nil {
		m.viewCache = make(map[string]SymmViewSpec)
	}

	for i, s := range specs {
		if s.GroupName == "" {
			return nil, fmt.Errorf("moonep: specs[%d] GroupName must not be empty", i)
		}
		if s.MaxBytes <= 0 {
			return nil, fmt.Errorf("moonep: specs[%d] (%s) MaxBytes %d must be positive", i, s.GroupName, s.MaxBytes)
		}
		if s.AlignBytes < 0 {
			return nil, fmt.Errorf("moonep: specs[%d] (%s) AlignBytes %d must be non-negative", i, s.GroupName, s.AlignBytes)
		}
		align := s.AlignBytes
		if align == 0 {
			align = 256
		}
		if align&(align-1) != 0 {
			return nil, fmt.Errorf("moonep: specs[%d] (%s) AlignBytes %d must be a power of two", i, s.GroupName, s.AlignBytes)
		}
		alignedMax := alignBytesUp(s.MaxBytes, align)
		prev, ok := m.groups[s.GroupName]
		if !ok {
			m.groups[s.GroupName] = SymmBufferKey{
				GroupName:  s.GroupName,
				MaxBytes:   alignedMax,
				AlignBytes: align,
			}
		} else {
			if align > prev.AlignBytes {
				prev.AlignBytes = align
			}
			alignedMax = alignBytesUp(alignedMax, prev.AlignBytes)
			if alignedMax > prev.MaxBytes {
				prev.MaxBytes = alignedMax
			} else {
				prev.MaxBytes = alignBytesUp(prev.MaxBytes, prev.AlignBytes)
			}
			m.groups[s.GroupName] = prev
		}
	}

	planned := make([]SymmBufferKey, 0, len(m.groups))
	for _, g := range m.groups {
		planned = append(planned, g)
	}
	sort.Slice(planned, func(i, j int) bool {
		if planned[i].MaxBytes != planned[j].MaxBytes {
			return planned[i].MaxBytes > planned[j].MaxBytes
		}
		return planned[i].GroupName < planned[j].GroupName
	})
	return planned, nil
}

func viewCacheKey(group string, shape []int, elemBytes int) string {
	var b strings.Builder
	b.WriteString(group)
	b.WriteByte('|')
	for i, d := range shape {
		if i > 0 {
			b.WriteByte('x')
		}
		b.WriteString(strconv.Itoa(d))
	}
	b.WriteByte('|')
	b.WriteString(strconv.Itoa(elemBytes))
	return b.String()
}

// GetSymmView returns a zero-allocation sub-view (ByteOffset: 0) over group's
// symmetric buffer after validating that TotalBytes <= MaxBytes, and caches the
// descriptor by (group, shape, elemBytes) for deterministic CUDA Graph replay.
func (m *SymmBufferManager) GetSymmView(group string, shape []int, elemBytes int) (SymmViewSpec, error) {
	if m == nil {
		return SymmViewSpec{}, fmt.Errorf("moonep: nil SymmBufferManager")
	}
	if group == "" {
		return SymmViewSpec{}, fmt.Errorf("moonep: group must not be empty")
	}
	if len(shape) == 0 {
		return SymmViewSpec{}, fmt.Errorf("moonep: shape must not be empty")
	}
	if elemBytes <= 0 {
		return SymmViewSpec{}, fmt.Errorf("moonep: elemBytes %d must be positive", elemBytes)
	}

	totalBytes := int64(elemBytes)
	for i, d := range shape {
		if d <= 0 {
			return SymmViewSpec{}, fmt.Errorf("moonep: shape[%d] = %d must be positive", i, d)
		}
		totalBytes *= int64(d)
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	g, ok := m.groups[group]
	if !ok {
		return SymmViewSpec{}, fmt.Errorf("moonep: unknown symmetric buffer group %q", group)
	}
	if totalBytes > g.MaxBytes {
		return SymmViewSpec{}, fmt.Errorf("moonep: requested view of %d bytes exceeds group %q MaxBytes %d", totalBytes, group, g.MaxBytes)
	}

	key := viewCacheKey(group, shape, elemBytes)
	if cached, found := m.viewCache[key]; found {
		shapeCopy := make([]int, len(cached.Shape))
		copy(shapeCopy, cached.Shape)
		cached.Shape = shapeCopy
		return cached, nil
	}

	storedShape := make([]int, len(shape))
	copy(storedShape, shape)
	spec := SymmViewSpec{
		Shape:      storedShape,
		ElemBytes:  elemBytes,
		ByteOffset: 0,
	}
	m.viewCache[key] = spec

	retShape := make([]int, len(shape))
	copy(retShape, shape)
	return SymmViewSpec{
		Shape:      retShape,
		ElemBytes:  elemBytes,
		ByteOffset: 0,
	}, nil
}

// PartitionSMBudget validates and returns the DeepEP communication vs compute
// SM budget split (defaulting commSMs to 24 SMs when commSMs == 0).
func PartitionSMBudget(totalSMs int, commSMs int) (comm int, compute int, err error) {
	if totalSMs < 4 {
		return 0, 0, fmt.Errorf("moonep: totalSMs %d must be at least 4", totalSMs)
	}
	if commSMs == 0 {
		commSMs = 24
	}
	if commSMs < 2 {
		return 0, 0, fmt.Errorf("moonep: commSMs %d must be at least 2", commSMs)
	}
	if commSMs%2 != 0 {
		return 0, 0, fmt.Errorf("moonep: commSMs %d must be an even CTA pair count", commSMs)
	}
	if commSMs >= totalSMs {
		return 0, 0, fmt.Errorf("moonep: commSMs %d must be less than totalSMs %d", commSMs, totalSMs)
	}
	return commSMs, totalSMs - commSMs, nil
}

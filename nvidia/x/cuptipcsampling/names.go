package cuptipcsampling

import (
	"errors"
	"fmt"
	"sort"
	"strings"
)

// samplingNames belongs to one coordinated sampling lifetime. The caller
// reserves enough entries for every PC in a drain before invoking CUPTI, then
// registers every returned name, even if copying an earlier name failed.
// Native names may be shared with Activity records: release is legal only after
// sampling is disabled and all sharing consumers have finished processing.
type samplingNames struct {
	entries                                       map[uintptr]samplingName
	maxEntries, maxNameBytes, maxBytes, usedBytes int
	read                                          func(uintptr, int) (string, error)
	free                                          func(uintptr)
	released                                      bool
}

type samplingName struct {
	value string
	err   error
}

func newNativeSamplingNames(maxEntries, maxNameBytes, maxBytes int) (*samplingNames, error) {
	if maxEntries < 1 || maxNameBytes < 1 || maxBytes < 1 {
		return nil, errors.New("cuptipcsampling: invalid native name limits")
	}
	free, err := samplingNameAllocator()
	if err != nil {
		return nil, err
	}
	return newSamplingNames(maxEntries, maxNameBytes, maxBytes, readSamplingName, free)
}

func newSamplingNames(maxEntries, maxNameBytes, maxBytes int, read func(uintptr, int) (string, error), free func(uintptr)) (*samplingNames, error) {
	if maxEntries < 1 || maxNameBytes < 1 || maxBytes < 1 || read == nil || free == nil {
		return nil, errors.New("cuptipcsampling: invalid native name limits or allocator")
	}
	return &samplingNames{entries: make(map[uintptr]samplingName), maxEntries: maxEntries, maxNameBytes: maxNameBytes, maxBytes: maxBytes, read: read, free: free}, nil
}

// reserve checks worst-case new pointers without making a native call. A
// rejected drain must not run: otherwise its names could exceed owned storage.
func (n *samplingNames) reserve(pcs int) error {
	if n.released || pcs < 0 || pcs > n.maxEntries-len(n.entries) {
		return errors.New("cuptipcsampling: native name entry budget exhausted")
	}
	return nil
}

func (n *samplingNames) copy(ptr uintptr) (string, error) {
	if n.released || ptr == 0 {
		return "", errors.New("cuptipcsampling: invalid native name access")
	}
	if old, ok := n.entries[ptr]; ok {
		return old.value, old.err
	}
	if err := n.reserve(1); err != nil {
		return "", err
	}
	// Acquire ownership before copying, so failure cannot lose this allocation.
	n.entries[ptr] = samplingName{}
	value, err := n.read(ptr, n.maxNameBytes)
	if err == nil && (value == "" || len(value) >= n.maxNameBytes) {
		err = errors.New("empty or overlong native name")
	}
	if err == nil && len(value) > n.maxBytes-n.usedBytes {
		err = errors.New("native name byte budget exhausted")
	}
	if err != nil {
		err = fmt.Errorf("cuptipcsampling: copy native name: %w", err)
		value = ""
	} else {
		value = strings.Clone(value)
		n.usedBytes += len(value)
	}
	n.entries[ptr] = samplingName{value: value, err: err}
	return value, err
}

func (n *samplingNames) release() {
	if n.released {
		return
	}
	ptrs := make([]uintptr, 0, len(n.entries))
	for ptr := range n.entries {
		ptrs = append(ptrs, ptr)
	}
	sort.Slice(ptrs, func(i, j int) bool { return ptrs[i] < ptrs[j] })
	for _, ptr := range ptrs {
		n.free(ptr)
	}
	n.released = true
	clear(n.entries)
}

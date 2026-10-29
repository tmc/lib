package cuptipcsampling

import (
	"errors"
	"fmt"
	"strings"
)

// samplingNames copies the native function names of one sampling session.
// The caller reserves enough entries for every PC in a drain before invoking
// CUPTI, then registers every returned name, even if copying an earlier name
// failed.
//
// The native strings are never freed. cupti_pcsampling.h allows free only
// after every record that may share the string, Activity records included,
// has been processed, and CUPTI returns one pointer per function that it hands
// out again to later sessions in the same process. No session can know it has
// seen the last such record, so freeing at Close would leave the next session
// reading and freeing a dangling pointer. What is retained is one string per
// distinct sampled function.
type samplingNames struct {
	entries                                       map[uintptr]samplingName
	maxEntries, maxNameBytes, maxBytes, usedBytes int
	read                                          func(uintptr, int) (string, error)
	released                                      bool
}

type samplingName struct {
	value string
	err   error
}

func newNativeSamplingNames(maxEntries, maxNameBytes, maxBytes int) (*samplingNames, error) {
	return newSamplingNames(maxEntries, maxNameBytes, maxBytes, readSamplingName)
}

func newSamplingNames(maxEntries, maxNameBytes, maxBytes int, read func(uintptr, int) (string, error)) (*samplingNames, error) {
	if maxEntries < 1 || maxNameBytes < 1 || maxBytes < 1 || read == nil {
		return nil, errors.New("cuptipcsampling: invalid native name limits or reader")
	}
	return &samplingNames{entries: make(map[uintptr]samplingName), maxEntries: maxEntries, maxNameBytes: maxNameBytes, maxBytes: maxBytes, read: read}, nil
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
	// Register before copying, so a failed copy still counts against the budget
	// and is not retried.
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

// release ends the session's access to native names. It frees nothing; see
// samplingNames.
func (n *samplingNames) release() {
	n.released = true
	clear(n.entries)
}

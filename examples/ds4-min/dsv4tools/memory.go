package dsv4tools

import (
	"container/list"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"sync"
)

var (
	ErrMemoryFull  = errors.New("tool memory has no capacity")
	ErrEmptyDSML   = errors.New("tool memory cannot store empty DSML")
	ErrNoToolCalls = errors.New("tool memory requires a tool call")
)

// Memory retains the exact sampled DSML for a bounded number of tool calls.
// It is safe for concurrent use. A lookup makes the entry most recently used.
// The zero value is not usable; use NewMemory.
type Memory struct {
	mu    sync.Mutex
	max   int
	items map[string]*memoryEntry
	order *list.List
}

type memoryEntry struct {
	id   string
	dsml string
	elem *list.Element
}

// NewMemory returns a bounded LRU tool-call memory. A non-positive capacity
// disables storage and causes Store to return ErrMemoryFull.
func NewMemory(maxIDs int) *Memory {
	return &Memory{
		max:   maxIDs,
		items: make(map[string]*memoryEntry),
		order: list.New(),
	}
}

// Store records dsml under a new unguessable call ID. The returned ID has the
// upstream-compatible call_ prefix and 32 hexadecimal random characters.
func (m *Memory) Store(dsml string) (string, error) {
	if dsml == "" {
		return "", ErrEmptyDSML
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.max <= 0 {
		return "", ErrMemoryFull
	}
	if m.items == nil || m.order == nil {
		return "", errors.New("tool memory is uninitialized")
	}
	return m.storeLocked(dsml)
}

// Record validates generated text as DSML and stores the exact text for each
// parsed call. The returned IDs are in source order. Recording is rejected if
// the text has no tool call or the configured capacity cannot hold all calls.
// The text is intentionally retained as a whole: replaying an ID preserves
// every byte of the sampled wrapper, including whitespace and sibling calls.
func (m *Memory) Record(dsml string) ([]string, error) {
	if len(dsml) == 0 {
		return nil, ErrEmptyDSML
	}
	_, calls, err := Parse(dsml)
	if err != nil {
		return nil, fmt.Errorf("validate tool DSML: %w", err)
	}
	if len(calls) == 0 {
		return nil, ErrNoToolCalls
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.max < len(calls) {
		return nil, ErrMemoryFull
	}
	if m.items == nil || m.order == nil {
		return nil, errors.New("tool memory is uninitialized")
	}
	ids := make([]string, 0, len(calls))
	for range calls {
		id, err := m.storeLocked(dsml)
		if err != nil {
			return nil, err
		}
		ids = append(ids, id)
	}
	return ids, nil
}

func (m *Memory) storeLocked(dsml string) (string, error) {
	for {
		var raw [16]byte
		if _, err := rand.Read(raw[:]); err != nil {
			return "", fmt.Errorf("generate tool call id: %w", err)
		}
		id := "call_" + hex.EncodeToString(raw[:])
		if _, exists := m.items[id]; exists {
			continue
		}
		entry := &memoryEntry{id: id, dsml: dsml}
		m.items[id] = entry
		entry.elem = m.order.PushFront(entry)
		if m.order.Len() > m.max {
			old := m.order.Back()
			delete(m.items, old.Value.(*memoryEntry).id)
			m.order.Remove(old)
		}
		return id, nil
	}
}

// Lookup returns the exact DSML stored for id and refreshes its LRU position.
func (m *Memory) Lookup(id string) (string, bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.items == nil {
		return "", false
	}
	entry, ok := m.items[id]
	if !ok {
		return "", false
	}
	m.order.MoveToFront(entry.elem)
	return entry.dsml, true
}

// Replay returns the exact DSML associated with id. It is equivalent to
// Lookup, but names the operation used by a renderer's replay path.
func (m *Memory) Replay(id string) (string, bool) {
	return m.Lookup(id)
}

// Delete removes id. It reports whether an entry was removed.
func (m *Memory) Delete(id string) bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	entry, ok := m.items[id]
	if !ok {
		return false
	}
	delete(m.items, id)
	m.order.Remove(entry.elem)
	return true
}

// Len reports the number of retained calls.
func (m *Memory) Len() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return len(m.items)
}

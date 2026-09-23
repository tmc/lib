package main

import (
	"sync"
	"time"
)

// ForwardMeasurement describes one completed model forward pass.
//
// GPUTime is reported by the backend when available. WallTime includes queueing
// and host synchronization and is retained separately for callers that need it.
type ForwardMeasurement struct {
	Position int           `json:"position"`
	GPUTime  time.Duration `json:"gpu_time"`
	WallTime time.Duration `json:"wall_time"`
	When     time.Time     `json:"when"`
}

// SpeculationNode records one token proposed by a draft engine and checked by
// a target engine. State is either "accepted" or "rejected".
type SpeculationNode struct {
	ID     string `json:"id"`
	Parent string `json:"parent,omitempty"`
	Token  int32  `json:"token"`
	Engine string `json:"engine"`
	State  string `json:"state"`
}

// SpeculationTrace records one paired draft and target comparison.
//
// A trace contains only tokens returned by the two engines. It does not
// represent probabilities or branches that were not evaluated.
type SpeculationTrace struct {
	When         time.Time         `json:"when"`
	DraftNodeID  string            `json:"draft_node_id"`
	TargetNodeID string            `json:"target_node_id"`
	Draft        []int32           `json:"draft"`
	Target       []int32           `json:"target"`
	Nodes        []SpeculationNode `json:"nodes"`
	Accepted     int               `json:"accepted"`
}

// RuntimeMetrics holds bounded measurements from the active execution engine.
type RuntimeMetrics struct {
	mu sync.RWMutex

	forwards []ForwardMeasurement
	limit    int

	kvCapacityBytes int64
	kvAllocated     bool
	kvTokens        int
	maxKVTokens     int
	traces          []SpeculationTrace
}

// RecordSpeculation records a comparison performed by paired engines.
func (m *RuntimeMetrics) RecordSpeculation(trace SpeculationTrace) {
	if m == nil {
		return
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	trace.Draft = append([]int32(nil), trace.Draft...)
	trace.Target = append([]int32(nil), trace.Target...)
	trace.Nodes = append([]SpeculationNode(nil), trace.Nodes...)
	m.traces = append(m.traces, trace)
	if len(m.traces) > m.limit {
		m.traces = append([]SpeculationTrace(nil), m.traces[len(m.traces)-m.limit:]...)
	}
}

// Speculation returns independent copies of paired-engine comparisons.
func (m *RuntimeMetrics) Speculation() []SpeculationTrace {
	if m == nil {
		return nil
	}
	m.mu.RLock()
	defer m.mu.RUnlock()
	traces := make([]SpeculationTrace, len(m.traces))
	for i, trace := range m.traces {
		traces[i] = trace
		traces[i].Draft = append([]int32(nil), trace.Draft...)
		traces[i].Target = append([]int32(nil), trace.Target...)
		traces[i].Nodes = append([]SpeculationNode(nil), trace.Nodes...)
	}
	return traces
}

// NewRuntimeMetrics returns a bounded runtime measurement store.
func NewRuntimeMetrics(limit int) *RuntimeMetrics {
	if limit <= 0 {
		limit = 256
	}
	return &RuntimeMetrics{limit: limit}
}

// RecordForward records a completed forward pass and its active KV position.
func (m *RuntimeMetrics) RecordForward(measurement ForwardMeasurement) {
	if m == nil {
		return
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	m.forwards = append(m.forwards, measurement)
	if len(m.forwards) > m.limit {
		m.forwards = append([]ForwardMeasurement(nil), m.forwards[len(m.forwards)-m.limit:]...)
	}
	if measurement.Position >= 0 && measurement.Position+1 > m.kvTokens {
		m.kvTokens = measurement.Position + 1
	}
	if m.maxKVTokens > 0 && m.kvTokens > m.maxKVTokens {
		m.kvTokens = m.maxKVTokens
	}
}

// SetKVAllocation records the cache storage allocated by the active engine.
func (m *RuntimeMetrics) SetKVAllocation(bytes int64, maxTokens int) {
	if m == nil {
		return
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	m.kvCapacityBytes = bytes
	m.kvAllocated = bytes > 0
	m.maxKVTokens = maxTokens
}

// Snapshot returns independent copies of the currently retained measurements.
func (m *RuntimeMetrics) Snapshot() (forwards []ForwardMeasurement, capacityBytes int64, allocated bool, tokens, maxTokens int) {
	if m == nil {
		return nil, 0, false, 0, 0
	}
	m.mu.RLock()
	defer m.mu.RUnlock()
	forwards = append([]ForwardMeasurement(nil), m.forwards...)
	return forwards, m.kvCapacityBytes, m.kvAllocated, m.kvTokens, m.maxKVTokens
}

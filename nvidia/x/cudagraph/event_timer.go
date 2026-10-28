package cudagraph

import (
	"errors"
	"fmt"
	"math"
	"sync"

	"github.com/tmc/lib/nvidia/driver"
)

// EventRecordNode wraps a CUDA graph node of type [driver.GraphNodeTypeEventRecord].
type EventRecordNode struct {
	raw driver.EventRecordNode
}

// NewEventRecordNode wraps raw as an [EventRecordNode].
func NewEventRecordNode(node driver.GraphNode) EventRecordNode {
	return EventRecordNode{raw: driver.EventRecordNode{GraphNode: node}}
}

// Node returns the underlying [driver.GraphNode] handle.
func (n EventRecordNode) Node() driver.GraphNode {
	return n.raw.GraphNode
}

// Raw returns the underlying [driver.EventRecordNode] handle.
func (n EventRecordNode) Raw() driver.EventRecordNode {
	return n.raw
}

var (
	eventRecordNodeGetEventFn = func(n driver.EventRecordNode) (driver.Event, error) {
		return n.Event()
	}
	eventRecordNodeSetEventFn = func(n driver.EventRecordNode, ev driver.Event) error {
		return n.SetEvent(ev)
	}
	graphAddEventRecordNodeFn = func(g driver.Graph, deps *driver.GraphNode, numDeps uint, ev driver.Event) (driver.EventRecordNode, error) {
		return g.AddEventRecordNode(deps, numDeps, ev)
	}
	graphNodesFn = func(g driver.Graph) ([]driver.GraphNode, error) {
		return g.Nodes()
	}
	graphNodeTypeFn = func(n driver.GraphNode) (driver.GraphNodeType, error) {
		return n.Type()
	}
	graphExecEventRecordNodeSetEventFn = func(exec driver.GraphExec, n driver.EventRecordNode, ev driver.Event) error {
		return exec.EventRecordNodeSetEvent(n, ev)
	}
	eventRecordWithFlagsFn = func(ev driver.Event, stream driver.Stream, flags driver.EventRecordFlags) error {
		return ev.RecordWithFlags(stream, flags)
	}
	eventElapsedMSFn = func(start, end driver.Event) (float64, error) {
		if err := start.Synchronize(); err != nil {
			return 0, fmt.Errorf("synchronize start event: %w", err)
		}
		if err := end.Synchronize(); err != nil {
			return 0, fmt.Errorf("synchronize end event: %w", err)
		}
		ms, err := start.ElapsedTime(end)
		if err != nil {
			return 0, fmt.Errorf("elapsed time: %w", err)
		}
		return float64(ms), nil
	}
)

// Event queries the [driver.Event] currently bound to this event-record node.
func (n EventRecordNode) Event() (driver.Event, error) {
	if n.raw.GraphNode == 0 {
		return 0, errors.New("cudagraph: zero event record node")
	}
	ev, err := eventRecordNodeGetEventFn(n.raw)
	if err != nil {
		return 0, fmt.Errorf("cudagraph: get event record node event: %w", err)
	}
	return ev, nil
}

// SetEvent updates the [driver.Event] bound to this uninstantiated event-record node.
func (n EventRecordNode) SetEvent(ev driver.Event) error {
	if n.raw.GraphNode == 0 {
		return errors.New("cudagraph: zero event record node")
	}
	if ev == 0 {
		return errors.New("cudagraph: zero event handle")
	}
	if err := eventRecordNodeSetEventFn(n.raw, ev); err != nil {
		return fmt.Errorf("cudagraph: set event record node event: %w", err)
	}
	return nil
}

// AddEventRecordNode adds an event-record node to g that records ev after deps complete.
func AddEventRecordNode(g driver.Graph, deps []driver.GraphNode, ev driver.Event) (EventRecordNode, error) {
	if g == 0 {
		return EventRecordNode{}, errors.New("cudagraph: zero graph handle")
	}
	if ev == 0 {
		return EventRecordNode{}, errors.New("cudagraph: zero event handle")
	}
	var depPtr *driver.GraphNode
	if len(deps) > 0 {
		for _, d := range deps {
			if d == 0 {
				return EventRecordNode{}, errors.New("cudagraph: zero dependency node in AddEventRecordNode")
			}
		}
		depCopy := make([]driver.GraphNode, len(deps))
		copy(depCopy, deps)
		depPtr = &depCopy[0]
	}
	raw, err := graphAddEventRecordNodeFn(g, depPtr, uint(len(deps)), ev)
	if err != nil {
		return EventRecordNode{}, fmt.Errorf("cudagraph: add event record node: %w", err)
	}
	return EventRecordNode{raw: raw}, nil
}

// EventRecordNodes enumerates all nodes in g and returns those of type
// [driver.GraphNodeTypeEventRecord] in the order reported by the graph.
func EventRecordNodes(g driver.Graph) ([]EventRecordNode, error) {
	if g == 0 {
		return nil, errors.New("cudagraph: zero graph handle")
	}
	nodes, err := graphNodesFn(g)
	if err != nil {
		return nil, fmt.Errorf("cudagraph: enumerate graph nodes: %w", err)
	}
	var out []EventRecordNode
	for _, n := range nodes {
		typ, err := graphNodeTypeFn(n)
		if err != nil {
			return nil, fmt.Errorf("cudagraph: query graph node type: %w", err)
		}
		if typ != driver.GraphNodeTypeEventRecord {
			continue
		}
		out = append(out, EventRecordNode{raw: driver.EventRecordNode{GraphNode: n}})
	}
	return out, nil
}

// SetEventRecordNodeEvent updates the event recorded by node in an instantiated
// executable graph without recapturing or reinstantiating the graph.
func SetEventRecordNodeEvent(exec driver.GraphExec, node EventRecordNode, ev driver.Event) error {
	if exec == 0 {
		return errors.New("cudagraph: zero executable graph handle")
	}
	if node.raw.GraphNode == 0 {
		return errors.New("cudagraph: zero event record node")
	}
	if ev == 0 {
		return errors.New("cudagraph: zero event handle")
	}
	if err := graphExecEventRecordNodeSetEventFn(exec, node.raw, ev); err != nil {
		return fmt.Errorf("cudagraph: set executable event record node event: %w", err)
	}
	return nil
}

// RecordExternalEvent records ev on stream with [driver.EventRecordExternal]
// (CU_EVENT_RECORD_EXTERNAL = 0x1) so stream capture records a
// [driver.GraphNodeTypeEventRecord] node inside the captured graph.
func RecordExternalEvent(stream driver.Stream, ev driver.Event) error {
	if ev == 0 {
		return errors.New("cudagraph: zero event handle")
	}
	if err := eventRecordWithFlagsFn(ev, stream, driver.EventRecordExternal); err != nil {
		return fmt.Errorf("cudagraph: record external event: %w", err)
	}
	return nil
}

// PingPongTimerConfig configures a [PingPongTimer] over a fixed number of
// event-record nodes in a captured CUDA graph.
type PingPongTimerConfig struct {
	// NumEventNodes is the number of EventRecordNode handles rebound before each replay.
	NumEventNodes int
	// HistoryCap bounds how many completed replay duration slices are retained.
	// When zero or negative, a default capacity of 16 is used.
	HistoryCap int
}

// PingPongTimer manages double-buffered (bank 0 and bank 1) event handles for
// event-record nodes inside an instantiated CUDA graph.
//
// Before each graph launch, [PingPongTimer.PrepareReplay] (or
// [PingPongTimer.RecordSimulatedReplay] in host-only simulation mode) swaps the
// active event bank (0 -> 1 -> 0), rebinds all event-record nodes to the newly
// selected bank, and harvests elapsed milliseconds from the previous bank once
// at least one prior replay has been launched. This avoids host synchronization
// on the first launch and prevents racing an in-flight graph replay.
type PingPongTimer struct {
	mu            sync.Mutex
	cfg           PingPongTimerConfig
	events        [2][]driver.Event
	pendingSim    [2][]float64
	hasPendingSim [2]bool
	nextBank      int
	activeBank    int
	replayCount   int
	lastDurations []float64
	history       [][]float64
}

// NewPingPongTimer constructs a [PingPongTimer] for cfg.
func NewPingPongTimer(cfg PingPongTimerConfig) (*PingPongTimer, error) {
	if cfg.NumEventNodes <= 0 {
		return nil, fmt.Errorf("cudagraph: invalid NumEventNodes %d: must be > 0", cfg.NumEventNodes)
	}
	if cfg.HistoryCap <= 0 {
		cfg.HistoryCap = 16
	}
	t := &PingPongTimer{
		cfg:        cfg,
		activeBank: -1,
	}
	for b := range 2 {
		t.events[b] = make([]driver.Event, cfg.NumEventNodes)
		for i := range cfg.NumEventNodes {
			// Assign deterministic synthetic non-zero handles by default so
			// PrepareReplay can be exercised with injected hooks; callers with
			// real CUDA events override via BindBankEvents.
			t.events[b][i] = driver.Event(uintptr((b+1)*1000 + i + 1))
		}
	}
	return t, nil
}

// BindBankEvents sets the [driver.Event] handles used for bank (0 or 1).
// The slice length must match [PingPongTimerConfig.NumEventNodes].
func (t *PingPongTimer) BindBankEvents(bank int, events []driver.Event) error {
	if t == nil {
		return errors.New("cudagraph: nil PingPongTimer")
	}
	if bank != 0 && bank != 1 {
		return fmt.Errorf("cudagraph: invalid event bank %d: want 0 or 1", bank)
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	if len(events) != t.cfg.NumEventNodes {
		return fmt.Errorf("cudagraph: bank %d event count %d does not match NumEventNodes %d", bank, len(events), t.cfg.NumEventNodes)
	}
	for i, ev := range events {
		if ev == 0 {
			return fmt.Errorf("cudagraph: zero event at index %d for bank %d", i, bank)
		}
	}
	copy(t.events[bank], events)
	return nil
}

// PrepareReplay swaps the active event bank (0 vs 1), rebinds every node in
// nodes to the new bank's events on exec, and reads elapsed times from the
// previous bank (prevBank = 1 - bank) if at least one prior replay occurred.
func (t *PingPongTimer) PrepareReplay(exec driver.GraphExec, nodes []EventRecordNode) (bank int, err error) {
	if t == nil {
		return 0, errors.New("cudagraph: nil PingPongTimer")
	}
	t.mu.Lock()
	defer t.mu.Unlock()

	if len(nodes) != t.cfg.NumEventNodes {
		return 0, fmt.Errorf("cudagraph: node count %d does not match NumEventNodes %d", len(nodes), t.cfg.NumEventNodes)
	}
	if exec == 0 {
		return 0, errors.New("cudagraph: zero executable graph handle")
	}
	for i, n := range nodes {
		if n.raw.GraphNode == 0 {
			return 0, fmt.Errorf("cudagraph: zero event record node at index %d", i)
		}
	}

	bank = t.nextBank
	prevBank := 1 - bank

	if t.replayCount > 0 {
		durations, err := t.collectBankDurationsLocked(prevBank)
		if err != nil {
			return 0, err
		}
		t.recordHarvestedLocked(durations)
	}

	for i, node := range nodes {
		if err := SetEventRecordNodeEvent(exec, node, t.events[bank][i]); err != nil {
			return 0, err
		}
	}

	t.activeBank = bank
	t.nextBank = prevBank
	t.replayCount++
	return bank, nil
}

func (t *PingPongTimer) collectBankDurationsLocked(bank int) ([]float64, error) {
	if t.hasPendingSim[bank] {
		out := make([]float64, len(t.pendingSim[bank]))
		copy(out, t.pendingSim[bank])
		t.hasPendingSim[bank] = false
		t.pendingSim[bank] = nil
		return out, nil
	}
	evs := t.events[bank]
	if len(evs) < 2 {
		return nil, nil
	}
	if len(evs)%2 == 0 {
		numPairs := len(evs) / 2
		out := make([]float64, numPairs)
		for i := range numPairs {
			ms, err := eventElapsedMSFn(evs[2*i], evs[2*i+1])
			if err != nil {
				return nil, fmt.Errorf("cudagraph: bank %d pair %d: %w", bank, i, err)
			}
			out[i] = ms
		}
		return out, nil
	}
	numSpans := len(evs) - 1
	out := make([]float64, numSpans)
	for i := range numSpans {
		ms, err := eventElapsedMSFn(evs[i], evs[i+1])
		if err != nil {
			return nil, fmt.Errorf("cudagraph: bank %d span %d: %w", bank, i, err)
		}
		out[i] = ms
	}
	return out, nil
}

// RecordSimulatedReplay advances the ping-pong state machine in pure-Go
// simulation mode without a CUDA driver context.
//
// On the first replay (ReplayCount == 1), bank 0 is armed with durationsMS and
// no prior bank is read, so [PingPongTimer.LastDurationsMS] remains nil (zero
// host-blocking on the first launch). On replay k >= 2, the active bank flips
// (0 -> 1 -> 0) and [PingPongTimer.LastDurationsMS] returns the durations
// recorded by replay k-1 in prevBank = 1 - bank.
func (t *PingPongTimer) RecordSimulatedReplay(durationsMS []float64) error {
	if t == nil {
		return errors.New("cudagraph: nil PingPongTimer")
	}
	if len(durationsMS) == 0 {
		return errors.New("cudagraph: empty simulated durations slice")
	}
	for i, d := range durationsMS {
		if math.IsNaN(d) || math.IsInf(d, 0) || d < 0 {
			return fmt.Errorf("cudagraph: invalid simulated duration at index %d: %v", i, d)
		}
	}

	t.mu.Lock()
	defer t.mu.Unlock()

	bank := t.nextBank
	prevBank := 1 - bank

	if t.replayCount > 0 && t.hasPendingSim[prevBank] {
		harvested := make([]float64, len(t.pendingSim[prevBank]))
		copy(harvested, t.pendingSim[prevBank])
		t.recordHarvestedLocked(harvested)
		t.hasPendingSim[prevBank] = false
		t.pendingSim[prevBank] = nil
	}

	staged := make([]float64, len(durationsMS))
	copy(staged, durationsMS)
	t.pendingSim[bank] = staged
	t.hasPendingSim[bank] = true
	t.activeBank = bank
	t.nextBank = prevBank
	t.replayCount++
	return nil
}

func (t *PingPongTimer) recordHarvestedLocked(durations []float64) {
	cp := make([]float64, len(durations))
	copy(cp, durations)
	t.lastDurations = cp
	if t.cfg.HistoryCap > 0 {
		if len(t.history) >= t.cfg.HistoryCap {
			copy(t.history, t.history[1:])
			t.history[len(t.history)-1] = cp
		} else {
			t.history = append(t.history, cp)
		}
	}
}

// LastDurationsMS returns a copy of the most recently harvested previous-bank
// replay durations in milliseconds, or nil if fewer than two replays have been
// prepared/recorded.
func (t *PingPongTimer) LastDurationsMS() []float64 {
	if t == nil {
		return nil
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.lastDurations == nil {
		return nil
	}
	out := make([]float64, len(t.lastDurations))
	copy(out, t.lastDurations)
	return out
}

// HistoryDurationsMS returns a copy of the bounded history of harvested replay
// duration slices in chronological order.
func (t *PingPongTimer) HistoryDurationsMS() [][]float64 {
	if t == nil {
		return nil
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	if len(t.history) == 0 {
		return nil
	}
	out := make([][]float64, len(t.history))
	for i, s := range t.history {
		cp := make([]float64, len(s))
		copy(cp, s)
		out[i] = cp
	}
	return out
}

// ReplayCount returns the total number of replays prepared or simulated so far.
func (t *PingPongTimer) ReplayCount() int {
	if t == nil {
		return 0
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.replayCount
}

// ActiveBank returns the bank index (0 or 1) armed by the most recent replay,
// or -1 before the first replay.
func (t *PingPongTimer) ActiveBank() int {
	if t == nil {
		return -1
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.activeBank
}

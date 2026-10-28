package cudagraph

import (
	"errors"
	"math"
	"slices"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
)

func TestEventRecordNodeOperations(t *testing.T) {
	origGetEvent := eventRecordNodeGetEventFn
	origSetEvent := eventRecordNodeSetEventFn
	origAddNode := graphAddEventRecordNodeFn
	origNodes := graphNodesFn
	origNodeType := graphNodeTypeFn
	origExecSetEvent := graphExecEventRecordNodeSetEventFn
	origRecordFlags := eventRecordWithFlagsFn
	t.Cleanup(func() {
		eventRecordNodeGetEventFn = origGetEvent
		eventRecordNodeSetEventFn = origSetEvent
		graphAddEventRecordNodeFn = origAddNode
		graphNodesFn = origNodes
		graphNodeTypeFn = origNodeType
		graphExecEventRecordNodeSetEventFn = origExecSetEvent
		eventRecordWithFlagsFn = origRecordFlags
	})

	boundEvents := map[driver.GraphNode]driver.Event{}
	execBoundEvents := map[driver.GraphNode]driver.Event{}
	nodeTypes := map[driver.GraphNode]driver.GraphNodeType{
		10: driver.GraphNodeTypeKernel,
		11: driver.GraphNodeTypeEventRecord,
		12: driver.GraphNodeTypeMemcpy,
		13: driver.GraphNodeTypeEventRecord,
		14: driver.GraphNodeTypeEmpty,
	}

	eventRecordNodeGetEventFn = func(n driver.EventRecordNode) (driver.Event, error) {
		return boundEvents[n.GraphNode], nil
	}
	eventRecordNodeSetEventFn = func(n driver.EventRecordNode, ev driver.Event) error {
		boundEvents[n.GraphNode] = ev
		return nil
	}
	graphAddEventRecordNodeFn = func(g driver.Graph, deps *driver.GraphNode, numDeps uint, ev driver.Event) (driver.EventRecordNode, error) {
		if g != 1 {
			return driver.EventRecordNode{}, errors.New("unexpected graph handle")
		}
		if numDeps == 1 && deps == nil {
			return driver.EventRecordNode{}, errors.New("missing dependency pointer")
		}
		node := driver.GraphNode(20)
		boundEvents[node] = ev
		return driver.EventRecordNode{GraphNode: node}, nil
	}
	graphNodesFn = func(g driver.Graph) ([]driver.GraphNode, error) {
		return []driver.GraphNode{10, 11, 12, 13, 14}, nil
	}
	graphNodeTypeFn = func(n driver.GraphNode) (driver.GraphNodeType, error) {
		return nodeTypes[n], nil
	}
	graphExecEventRecordNodeSetEventFn = func(exec driver.GraphExec, n driver.EventRecordNode, ev driver.Event) error {
		if exec != 99 {
			return errors.New("unexpected exec handle")
		}
		execBoundEvents[n.GraphNode] = ev
		return nil
	}

	var recordedStream driver.Stream
	var recordedEvent driver.Event
	var recordedFlags driver.EventRecordFlags
	eventRecordWithFlagsFn = func(ev driver.Event, stream driver.Stream, flags driver.EventRecordFlags) error {
		recordedEvent = ev
		recordedStream = stream
		recordedFlags = flags
		return nil
	}

	// Zero-handle validation checks.
	var zeroNode EventRecordNode
	if _, err := zeroNode.Event(); err == nil {
		t.Fatal("zeroNode.Event() succeeded, want error")
	}
	if err := zeroNode.SetEvent(1); err == nil {
		t.Fatal("zeroNode.SetEvent() succeeded, want error")
	}
	if _, err := AddEventRecordNode(0, nil, 1); err == nil {
		t.Fatal("AddEventRecordNode with zero graph succeeded, want error")
	}
	if _, err := AddEventRecordNode(1, nil, 0); err == nil {
		t.Fatal("AddEventRecordNode with zero event succeeded, want error")
	}
	if _, err := AddEventRecordNode(1, []driver.GraphNode{0}, 1); err == nil {
		t.Fatal("AddEventRecordNode with zero dep succeeded, want error")
	}
	if _, err := EventRecordNodes(0); err == nil {
		t.Fatal("EventRecordNodes with zero graph succeeded, want error")
	}
	if err := SetEventRecordNodeEvent(0, NewEventRecordNode(11), 1); err == nil {
		t.Fatal("SetEventRecordNodeEvent with zero exec succeeded, want error")
	}
	if err := SetEventRecordNodeEvent(99, zeroNode, 1); err == nil {
		t.Fatal("SetEventRecordNodeEvent with zero node succeeded, want error")
	}
	if err := SetEventRecordNodeEvent(99, NewEventRecordNode(11), 0); err == nil {
		t.Fatal("SetEventRecordNodeEvent with zero event succeeded, want error")
	}
	if err := RecordExternalEvent(7, 0); err == nil {
		t.Fatal("RecordExternalEvent with zero event succeeded, want error")
	}

	// Positive AddEventRecordNode + Event + SetEvent check.
	added, err := AddEventRecordNode(1, []driver.GraphNode{10}, 500)
	if err != nil {
		t.Fatalf("AddEventRecordNode: %v", err)
	}
	if added.Node() != 20 || added.Raw().GraphNode != 20 {
		t.Fatalf("added.Node() = %v, want 20", added.Node())
	}
	if err := added.SetEvent(0); err == nil {
		t.Fatal("added.SetEvent(0) succeeded, want error")
	}
	ev, err := added.Event()
	if err != nil {
		t.Fatalf("added.Event(): %v", err)
	}
	if ev != 500 {
		t.Fatalf("added.Event() = %v, want 500", ev)
	}
	if err := added.SetEvent(501); err != nil {
		t.Fatalf("added.SetEvent(501): %v", err)
	}
	ev, err = added.Event()
	if err != nil || ev != 501 {
		t.Fatalf("added.Event() after SetEvent = (%v, %v), want (501, nil)", ev, err)
	}

	// Positive EventRecordNodes topological filtering check.
	evNodes, err := EventRecordNodes(1)
	if err != nil {
		t.Fatalf("EventRecordNodes: %v", err)
	}
	if len(evNodes) != 2 || evNodes[0].Node() != 11 || evNodes[1].Node() != 13 {
		t.Fatalf("EventRecordNodes = %+v, want nodes [11, 13]", evNodes)
	}

	// Positive SetEventRecordNodeEvent check.
	if err := SetEventRecordNodeEvent(99, evNodes[0], 701); err != nil {
		t.Fatalf("SetEventRecordNodeEvent: %v", err)
	}
	if execBoundEvents[11] != 701 {
		t.Fatalf("execBoundEvents[11] = %v, want 701", execBoundEvents[11])
	}

	// Positive RecordExternalEvent check: must pass driver.EventRecordExternal (0x1).
	if err := RecordExternalEvent(42, 888); err != nil {
		t.Fatalf("RecordExternalEvent: %v", err)
	}
	if recordedStream != 42 || recordedEvent != 888 || recordedFlags != driver.EventRecordExternal {
		t.Fatalf("RecordExternalEvent recorded (stream=%v, event=%v, flags=%v), want (42, 888, %v)",
			recordedStream, recordedEvent, recordedFlags, driver.EventRecordExternal)
	}
}

func TestPingPongTimerStateMachine(t *testing.T) {
	if _, err := NewPingPongTimer(PingPongTimerConfig{NumEventNodes: 0}); err == nil {
		t.Fatal("NewPingPongTimer with NumEventNodes=0 succeeded, want error")
	}

	timer, err := NewPingPongTimer(PingPongTimerConfig{
		NumEventNodes: 4,
		HistoryCap:    2,
	})
	if err != nil {
		t.Fatalf("NewPingPongTimer: %v", err)
	}
	if timer.ReplayCount() != 0 || timer.ActiveBank() != -1 || timer.LastDurationsMS() != nil {
		t.Fatalf("initial state = (count=%d, bank=%d, last=%v), want (0, -1, nil)",
			timer.ReplayCount(), timer.ActiveBank(), timer.LastDurationsMS())
	}

	// Invalid simulated inputs must be rejected.
	for _, bad := range [][]float64{
		nil,
		{},
		{-0.1},
		{math.NaN()},
		{math.Inf(1)},
	} {
		if err := timer.RecordSimulatedReplay(bad); err == nil {
			t.Fatalf("RecordSimulatedReplay(%v) succeeded, want error", bad)
		}
	}

	// Replay 1: arms bank 0, does not block or harvest a previous bank.
	if err := timer.RecordSimulatedReplay([]float64{1.25, 2.50}); err != nil {
		t.Fatalf("RecordSimulatedReplay #1: %v", err)
	}
	if got := timer.ActiveBank(); got != 0 {
		t.Fatalf("after replay #1 ActiveBank = %d, want 0", got)
	}
	if got := timer.ReplayCount(); got != 1 {
		t.Fatalf("after replay #1 ReplayCount = %d, want 1", got)
	}
	if got := timer.LastDurationsMS(); got != nil {
		t.Fatalf("after replay #1 LastDurationsMS = %v, want nil (zero host blocking on first launch)", got)
	}

	// Replay 2: arms bank 1, harvests bank 0 ([1.25, 2.50]).
	if err := timer.RecordSimulatedReplay([]float64{3.75, 4.00}); err != nil {
		t.Fatalf("RecordSimulatedReplay #2: %v", err)
	}
	if got := timer.ActiveBank(); got != 1 {
		t.Fatalf("after replay #2 ActiveBank = %d, want 1", got)
	}
	if got := timer.ReplayCount(); got != 2 {
		t.Fatalf("after replay #2 ReplayCount = %d, want 2", got)
	}
	if got, want := timer.LastDurationsMS(), []float64{1.25, 2.50}; !slices.Equal(got, want) {
		t.Fatalf("after replay #2 LastDurationsMS = %v, want %v", got, want)
	}

	// Replay 3: arms bank 0, harvests bank 1 ([3.75, 4.00]).
	if err := timer.RecordSimulatedReplay([]float64{5.50, 6.25}); err != nil {
		t.Fatalf("RecordSimulatedReplay #3: %v", err)
	}
	if got := timer.ActiveBank(); got != 0 {
		t.Fatalf("after replay #3 ActiveBank = %d, want 0", got)
	}
	if got, want := timer.LastDurationsMS(), []float64{3.75, 4.00}; !slices.Equal(got, want) {
		t.Fatalf("after replay #3 LastDurationsMS = %v, want %v", got, want)
	}

	// Replay 4: arms bank 1, harvests bank 0 ([5.50, 6.25]), history capped at 2.
	if err := timer.RecordSimulatedReplay([]float64{7.00, 8.00}); err != nil {
		t.Fatalf("RecordSimulatedReplay #4: %v", err)
	}
	if got := timer.ActiveBank(); got != 1 {
		t.Fatalf("after replay #4 ActiveBank = %d, want 1", got)
	}
	if got, want := timer.LastDurationsMS(), []float64{5.50, 6.25}; !slices.Equal(got, want) {
		t.Fatalf("after replay #4 LastDurationsMS = %v, want %v", got, want)
	}
	hist := timer.HistoryDurationsMS()
	if len(hist) != 2 || !slices.Equal(hist[0], []float64{3.75, 4.00}) || !slices.Equal(hist[1], []float64{5.50, 6.25}) {
		t.Fatalf("HistoryDurationsMS = %v, want [[3.75 4] [5.5 6.25]]", hist)
	}
}

func TestPingPongTimerPrepareReplayWithDriverHooks(t *testing.T) {
	origExecSetEvent := graphExecEventRecordNodeSetEventFn
	origElapsed := eventElapsedMSFn
	t.Cleanup(func() {
		graphExecEventRecordNodeSetEventFn = origExecSetEvent
		eventElapsedMSFn = origElapsed
	})

	rebound := map[driver.GraphNode]driver.Event{}
	graphExecEventRecordNodeSetEventFn = func(exec driver.GraphExec, n driver.EventRecordNode, ev driver.Event) error {
		rebound[n.GraphNode] = ev
		return nil
	}

	elapsedPairs := map[[2]driver.Event]float64{
		{101, 102}: 1.5,
		{103, 104}: 2.5,
		{201, 202}: 3.5,
		{203, 204}: 4.5,
	}
	elapsedCalls := 0
	eventElapsedMSFn = func(start, end driver.Event) (float64, error) {
		elapsedCalls++
		v, ok := elapsedPairs[[2]driver.Event{start, end}]
		if !ok {
			return 0, errors.New("unexpected event pair")
		}
		return v, nil
	}

	timer, err := NewPingPongTimer(PingPongTimerConfig{NumEventNodes: 4, HistoryCap: 4})
	if err != nil {
		t.Fatalf("NewPingPongTimer: %v", err)
	}
	if err := timer.BindBankEvents(0, []driver.Event{101, 102, 103, 104}); err != nil {
		t.Fatalf("BindBankEvents(0): %v", err)
	}
	if err := timer.BindBankEvents(1, []driver.Event{201, 202, 203, 204}); err != nil {
		t.Fatalf("BindBankEvents(1): %v", err)
	}
	if err := timer.BindBankEvents(2, []driver.Event{1, 2, 3, 4}); err == nil || !strings.Contains(err.Error(), "invalid event bank") {
		t.Fatalf("BindBankEvents(2) = %v, want invalid event bank error", err)
	}

	nodes := []EventRecordNode{
		NewEventRecordNode(11),
		NewEventRecordNode(12),
		NewEventRecordNode(13),
		NewEventRecordNode(14),
	}

	// Launch 1: binds bank 0 events, makes 0 elapsed queries.
	bank, err := timer.PrepareReplay(77, nodes)
	if err != nil {
		t.Fatalf("PrepareReplay #1: %v", err)
	}
	if bank != 0 || elapsedCalls != 0 || timer.LastDurationsMS() != nil {
		t.Fatalf("after PrepareReplay #1: bank=%d elapsedCalls=%d last=%v", bank, elapsedCalls, timer.LastDurationsMS())
	}
	if rebound[11] != 101 || rebound[14] != 104 {
		t.Fatalf("rebound after #1 = %v, want bank 0 events", rebound)
	}

	// Launch 2: harvests bank 0 ([1.5, 2.5]) and binds bank 1 events.
	bank, err = timer.PrepareReplay(77, nodes)
	if err != nil {
		t.Fatalf("PrepareReplay #2: %v", err)
	}
	if bank != 1 || elapsedCalls != 2 {
		t.Fatalf("after PrepareReplay #2: bank=%d elapsedCalls=%d, want bank=1 elapsedCalls=2", bank, elapsedCalls)
	}
	if got, want := timer.LastDurationsMS(), []float64{1.5, 2.5}; !slices.Equal(got, want) {
		t.Fatalf("after PrepareReplay #2 LastDurationsMS = %v, want %v", got, want)
	}
	if rebound[11] != 201 || rebound[14] != 204 {
		t.Fatalf("rebound after #2 = %v, want bank 1 events", rebound)
	}

	// Launch 3: harvests bank 1 ([3.5, 4.5]) and binds bank 0 events.
	bank, err = timer.PrepareReplay(77, nodes)
	if err != nil {
		t.Fatalf("PrepareReplay #3: %v", err)
	}
	if bank != 0 || elapsedCalls != 4 {
		t.Fatalf("after PrepareReplay #3: bank=%d elapsedCalls=%d, want bank=0 elapsedCalls=4", bank, elapsedCalls)
	}
	if got, want := timer.LastDurationsMS(), []float64{3.5, 4.5}; !slices.Equal(got, want) {
		t.Fatalf("after PrepareReplay #3 LastDurationsMS = %v, want %v", got, want)
	}
}

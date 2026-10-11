package cudaworkload

import (
	"context"
	"errors"
	"fmt"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/cupti"
)

func curGoroutineID() uint64 {
	var buf [64]byte
	n := runtime.Stack(buf[:], false)
	s := strings.TrimPrefix(string(buf[:n]), "goroutine ")
	idx := strings.IndexByte(s, ' ')
	if idx > 0 {
		id, _ := strconv.ParseUint(s[:idx], 10, 64)
		return id
	}
	return 0
}

type mockHook struct {
	mu          sync.Mutex
	stacks      map[uint64][]uint64
	totalPushes int
	totalPops   int
}

func (m *mockHook) Push(kind cupti.ExternalCorrelationKind, id uint64) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	gid := curGoroutineID()
	if m.stacks == nil {
		m.stacks = make(map[uint64][]uint64)
	}
	m.stacks[gid] = append(m.stacks[gid], id)
	m.totalPushes++
	return nil
}

func (m *mockHook) Pop(kind cupti.ExternalCorrelationKind) (uint64, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	gid := curGoroutineID()
	s := m.stacks[gid]
	if len(s) == 0 {
		return 0, errors.New("pop on empty stack")
	}
	top := s[len(s)-1]
	m.stacks[gid] = s[:len(s)-1]
	if len(m.stacks[gid]) == 0 {
		delete(m.stacks, gid)
	}
	m.totalPops++
	return top, nil
}

func (m *mockHook) Depth() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	gid := curGoroutineID()
	return len(m.stacks[gid])
}

func (m *mockHook) ActiveStacks() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return len(m.stacks)
}

func TestJoin_ProcessGenIsolation(t *testing.T) {
	joiner := NewJoiner(JoinConfig{MaxPendingRecords: 100})

	// Process PID 42 in Generation 1
	procGen1 := ProcessGenID{PID: 42, Generation: 1}
	sub1 := HostSubmission{
		Process:             procGen1,
		OpID:                OpID(101),
		NativeCorrelationID: 500,
		Context:             ContextStreamID{Context: 1, Stream: 1},
		TimestampNs:         1000,
	}
	joiner.SubmitHost(sub1)

	// Process PID 42 reused in Generation 2
	procGen2 := ProcessGenID{PID: 42, Generation: 2}
	act2 := DeviceActivity{
		Process:             procGen2,
		NativeCorrelationID: 500,
		Context:             ContextStreamID{Context: 1, Stream: 1},
		StartNs:             2000,
		EndNs:               2500,
	}

	results := joiner.SubmitDevice(act2)
	for _, res := range results {
		if res.Outcome == OutcomeExact {
			t.Fatalf("unexpected exact join across process generations: %+v", res)
		}
	}

	flushed := joiner.Flush()
	for _, res := range flushed {
		if res.Outcome == OutcomeExact {
			t.Fatalf("unexpected exact join after flush across process generations: %+v", res)
		}
	}

	metrics := joiner.Metrics()
	if metrics.ExactJoins != 0 {
		t.Errorf("expected 0 exact joins across generations, got %d", metrics.ExactJoins)
	}
}

func TestJoin_IdenticalOperationNames(t *testing.T) {
	proc := ProcessGenID{PID: 100, Generation: 1}
	annotator := NewAnnotator(proc, "test-service")

	ctx := context.Background()
	_, op1 := annotator.Begin(ctx, "matmul")
	_, op2 := annotator.Begin(ctx, "matmul")

	if op1.ID() == op2.ID() {
		t.Fatalf("operations with identical names received identical OpIDs: %v", op1.ID())
	}

	joiner := NewJoiner(JoinConfig{})
	sub1 := op1.NewSubmission(10, ContextStreamID{1, 1}, GraphReplayGeneration{}, "kernel_a", "ref1")
	sub2 := op2.NewSubmission(20, ContextStreamID{1, 1}, GraphReplayGeneration{}, "kernel_b", "ref2")

	joiner.SubmitHost(sub1)
	joiner.SubmitHost(sub2)

	act1 := DeviceActivity{
		Process:               proc,
		NativeCorrelationID:   10,
		ExternalCorrelationID: uint64(op1.ID()),
		Kind:                  "kernel",
		Name:                  "kernel_a",
		Context:               ContextStreamID{1, 1},
		StartNs:               100,
		EndNs:                 200,
	}
	act2 := DeviceActivity{
		Process:               proc,
		NativeCorrelationID:   20,
		ExternalCorrelationID: uint64(op2.ID()),
		Kind:                  "kernel",
		Name:                  "kernel_b",
		Context:               ContextStreamID{1, 1},
		StartNs:               300,
		EndNs:                 400,
	}

	res1 := joiner.SubmitDevice(act1)
	if len(res1) != 1 || res1[0].Outcome != OutcomeExact || res1[0].OpID != op1.ID() {
		t.Fatalf("act1 did not join to op1: %+v", res1)
	}

	res2 := joiner.SubmitDevice(act2)
	if len(res2) != 1 || res2[0].Outcome != OutcomeExact || res2[0].OpID != op2.ID() {
		t.Fatalf("act2 did not join to op2: %+v", res2)
	}
}

func TestJoin_NestedAnnotations(t *testing.T) {
	proc := ProcessGenID{PID: 200, Generation: 1}
	annotator := NewAnnotator(proc, "training")

	ctx := context.Background()
	ctxOuter, outer := annotator.Begin(ctx, "train_step")
	_, inner := annotator.Begin(ctxOuter, "backward")

	if inner.ParentID() != outer.ID() {
		t.Fatalf("inner op parent ID %v != outer op ID %v", inner.ParentID(), outer.ID())
	}

	joiner := NewJoiner(JoinConfig{})
	sub := inner.NewSubmission(55, ContextStreamID{1, 1}, GraphReplayGeneration{}, "backward_kernel", "ev")
	joiner.SubmitHost(sub)

	act := DeviceActivity{
		Process:               proc,
		NativeCorrelationID:   55,
		ExternalCorrelationID: uint64(inner.ID()),
		Kind:                  "kernel",
		Name:                  "backward_kernel",
		Context:               ContextStreamID{1, 1},
		StartNs:               500,
		EndNs:                 600,
	}

	results := joiner.SubmitDevice(act)
	if len(results) != 1 {
		t.Fatalf("expected 1 result, got %d", len(results))
	}
	res := results[0]
	if res.Outcome != OutcomeExact {
		t.Fatalf("expected OutcomeExact, got %s", res.Outcome)
	}
	if res.OpID != inner.ID() {
		t.Fatalf("expected OpID %v, got %v", inner.ID(), res.OpID)
	}
	if res.ParentOpID != outer.ID() {
		t.Fatalf("expected ParentOpID %v, got %v", outer.ID(), res.ParentOpID)
	}
}

func TestJoin_MultipleStreams(t *testing.T) {
	proc := ProcessGenID{PID: 300, Generation: 1}
	annotator := NewAnnotator(proc, "multistream")
	joiner := NewJoiner(JoinConfig{})

	ctx := context.Background()
	_, opA := annotator.Begin(ctx, "streamA_op")
	_, opB := annotator.Begin(ctx, "streamB_op")

	subA := opA.NewSubmission(101, ContextStreamID{Context: 1, Stream: 10}, GraphReplayGeneration{}, "k_a", "")
	subB := opB.NewSubmission(102, ContextStreamID{Context: 1, Stream: 20}, GraphReplayGeneration{}, "k_b", "")

	joiner.SubmitHost(subA)
	joiner.SubmitHost(subB)

	actB := DeviceActivity{
		Process:             proc,
		NativeCorrelationID: 102,
		Context:             ContextStreamID{Context: 1, Stream: 20},
		StartNs:             100,
		EndNs:               200,
	}
	actA := DeviceActivity{
		Process:             proc,
		NativeCorrelationID: 101,
		Context:             ContextStreamID{Context: 1, Stream: 10},
		StartNs:             150,
		EndNs:               250,
	}

	resB := joiner.SubmitDevice(actB)
	if len(resB) != 1 || resB[0].Outcome != OutcomeExact || resB[0].OpID != opB.ID() {
		t.Fatalf("stream B activity misjoined: %+v", resB)
	}

	resA := joiner.SubmitDevice(actA)
	if len(resA) != 1 || resA[0].Outcome != OutcomeExact || resA[0].OpID != opA.ID() {
		t.Fatalf("stream A activity misjoined: %+v", resA)
	}
}

func TestJoin_OverlappingActivities(t *testing.T) {
	proc := ProcessGenID{PID: 400, Generation: 1}
	joiner := NewJoiner(JoinConfig{})

	sub1 := HostSubmission{
		Process:             proc,
		OpID:                OpID(1),
		NativeCorrelationID: 11,
		Context:             ContextStreamID{1, 1},
	}
	sub2 := HostSubmission{
		Process:             proc,
		OpID:                OpID(2),
		NativeCorrelationID: 22,
		Context:             ContextStreamID{1, 2},
	}

	joiner.SubmitHost(sub1)
	joiner.SubmitHost(sub2)

	// Activities overlap in time: act1 from 100 to 300, act2 from 150 to 250
	act1 := DeviceActivity{
		Process:             proc,
		NativeCorrelationID: 11,
		Context:             ContextStreamID{1, 1},
		StartNs:             100,
		EndNs:               300,
	}
	act2 := DeviceActivity{
		Process:             proc,
		NativeCorrelationID: 22,
		Context:             ContextStreamID{1, 2},
		StartNs:             150,
		EndNs:               250,
	}

	r2 := joiner.SubmitDevice(act2)
	r1 := joiner.SubmitDevice(act1)

	if len(r1) != 1 || r1[0].Outcome != OutcomeExact || r1[0].OpID != OpID(1) {
		t.Fatalf("act1 did not join correctly: %+v", r1)
	}
	if len(r2) != 1 || r2[0].Outcome != OutcomeExact || r2[0].OpID != OpID(2) {
		t.Fatalf("act2 did not join correctly: %+v", r2)
	}
}

func TestJoin_GraphReplayGenerations(t *testing.T) {
	proc := ProcessGenID{PID: 500, Generation: 1}
	joiner := NewJoiner(JoinConfig{})

	graphGen1 := GraphReplayGeneration{GraphID: 0xAAAA, ExecGraphID: 0xBBBB, ReplayGeneration: 1}
	graphGen2 := GraphReplayGeneration{GraphID: 0xAAAA, ExecGraphID: 0xBBBB, ReplayGeneration: 2}

	sub1 := HostSubmission{
		Process:             proc,
		OpID:                OpID(10),
		NativeCorrelationID: 99,
		GraphReplay:         graphGen1,
	}
	sub2 := HostSubmission{
		Process:             proc,
		OpID:                OpID(20),
		NativeCorrelationID: 99, // Same node correlation ID inside graph!
		GraphReplay:         graphGen2,
	}

	joiner.SubmitHost(sub1)
	joiner.SubmitHost(sub2)

	actGen2 := DeviceActivity{
		Process:             proc,
		NativeCorrelationID: 99,
		GraphReplay:         graphGen2,
	}

	res := joiner.SubmitDevice(actGen2)
	if len(res) != 1 {
		t.Fatalf("expected 1 result, got %d", len(res))
	}
	if res[0].Outcome != OutcomeExact {
		t.Fatalf("expected OutcomeExact, got %s", res[0].Outcome)
	}
	if res[0].OpID != OpID(20) {
		t.Fatalf("expected activity to join with replay gen 2 (OpID 20), got OpID %v", res[0].OpID)
	}
	if res[0].Reason != ReasonGraphReplayMatch {
		t.Fatalf("expected ReasonGraphReplayMatch, got %s", res[0].Reason)
	}
}

func TestJoin_LateAttach(t *testing.T) {
	proc := ProcessGenID{PID: 600, Generation: 1}
	joiner := NewJoiner(JoinConfig{LateAttachCutoffNs: 5000})

	// Device activity occurred before attachment window cutoff
	act := DeviceActivity{
		Process:             proc,
		NativeCorrelationID: 777,
		StartNs:             1000,
		EndNs:               1200,
	}

	results := joiner.SubmitDevice(act)
	if len(results) != 1 {
		t.Fatalf("expected 1 result, got %d", len(results))
	}
	if results[0].Outcome != OutcomeMissing {
		t.Fatalf("expected OutcomeMissing for late attach, got %s", results[0].Outcome)
	}
	if results[0].Reason != ReasonLateAttach {
		t.Fatalf("expected ReasonLateAttach, got %s", results[0].Reason)
	}
}

func TestJoin_ZeroFalseExactJoins(t *testing.T) {
	proc := ProcessGenID{PID: 700, Generation: 1}
	joiner := NewJoiner(JoinConfig{})

	// 1. Conflicting external correlation IDs
	subConf := HostSubmission{
		Process:             proc,
		OpID:                OpID(88),
		NativeCorrelationID: 123,
	}
	joiner.SubmitHost(subConf)

	actConf := DeviceActivity{
		Process:               proc,
		NativeCorrelationID:   123,
		ExternalCorrelationID: 9999, // Mismatched external ID!
	}
	resConf := joiner.SubmitDevice(actConf)
	if len(resConf) != 1 || resConf[0].Outcome == OutcomeExact {
		t.Fatalf("expected non-exact join for conflicting external IDs, got: %+v", resConf)
	}
	if resConf[0].Outcome != OutcomeAmbiguous {
		t.Errorf("expected OutcomeAmbiguous, got %s", resConf[0].Outcome)
	}

	// 2. Proximity-only (stream context active without correlation IDs)
	subSingle := HostSubmission{
		Process: proc,
		OpID:    OpID(30),
		Context: ContextStreamID{1, 9},
	}
	joiner.SubmitHost(subSingle)
	actSingle := DeviceActivity{
		Process: proc,
		Context: ContextStreamID{1, 9},
		StartNs: 3000,
	}
	resSingle := joiner.SubmitDevice(actSingle)
	for _, r := range resSingle {
		if r.Outcome == OutcomeExact {
			t.Fatalf("stream proximity without correlation promoted to OutcomeExact! %+v", r)
		}
	}

	// 3. Ambiguous stream context (multiple active submissions on same stream)
	subStream1 := HostSubmission{
		Process: proc,
		OpID:    OpID(10),
		Context: ContextStreamID{1, 5},
	}
	subStream2 := HostSubmission{
		Process: proc,
		OpID:    OpID(20),
		Context: ContextStreamID{1, 5},
	}
	joiner.SubmitHost(subStream1)
	joiner.SubmitHost(subStream2)

	actNoCorr := DeviceActivity{
		Process: proc,
		Context: ContextStreamID{1, 5},
		StartNs: 2000,
	}
	resProx := joiner.SubmitDevice(actNoCorr)
	for _, r := range resProx {
		if r.Outcome == OutcomeExact {
			t.Fatalf("nearby timestamp or ambiguous stream promoted to OutcomeExact! %+v", r)
		}
	}

	m := joiner.Metrics()
	if m.FalseExactJoins != 0 {
		t.Errorf("expected 0 false exact joins, got %d", m.FalseExactJoins)
	}
}

func TestJoin_MetricsCalculation(t *testing.T) {
	m := JoinMetrics{
		TotalActivities:     10,
		MissingActivities:   1,
		AmbiguousActivities: 1,
		ExactJoins:          6,
		AssociatedJoins:     2,
		AmbiguousJoins:      1,
		MissingJoins:        1,
		FalseExactJoins:     0,
	}

	if p := m.Precision(); p != 1.0 {
		t.Errorf("expected precision 1.0, got %f", p)
	}
	if r := m.Recall(); r != 0.6 {
		t.Errorf("expected recall 0.6, got %f", r)
	}
	if mf := m.MissingFraction(); mf != 0.1 {
		t.Errorf("expected missing fraction 0.1, got %f", mf)
	}
	if af := m.AmbiguousFraction(); af != 0.1 {
		t.Errorf("expected ambiguous fraction 0.1, got %f", af)
	}

	// Test precision degradation if false exact join occurred
	m.FalseExactJoins = 2
	if p := m.Precision(); p != 6.0/8.0 {
		t.Errorf("expected precision 0.75, got %f", p)
	}
}

func TestJoin_BoundedStateMemory(t *testing.T) {
	limit := 10
	joiner := NewJoiner(JoinConfig{MaxPendingRecords: limit})

	proc := ProcessGenID{PID: 800, Generation: 1}
	var expiredCount int

	// Submit 50 unmatched host submissions
	for i := 1; i <= 50; i++ {
		sub := HostSubmission{
			Process:             proc,
			OpID:                OpID(i),
			NativeCorrelationID: uint32(i * 10),
		}
		res := joiner.SubmitHost(sub)
		for _, r := range res {
			if r.Outcome == OutcomeMissing && r.Reason == ReasonExpiredFIFO {
				expiredCount++
			}
		}
	}

	if expiredCount != 40 {
		t.Errorf("expected 40 FIFO expired records, got %d", expiredCount)
	}

	if len(joiner.pendingSubs) > limit {
		t.Errorf("pending submissions count %d exceeds limit %d", len(joiner.pendingSubs), limit)
	}

	// Submit 30 unmatched device activities
	expiredActCount := 0
	for i := 1; i <= 30; i++ {
		act := DeviceActivity{
			Process:             proc,
			NativeCorrelationID: uint32(1000 + i),
		}
		res := joiner.SubmitDevice(act)
		for _, r := range res {
			if r.Outcome == OutcomeMissing && r.Reason == ReasonExpiredFIFO {
				expiredActCount++
			}
		}
	}

	if expiredActCount != 20 {
		t.Errorf("expected 20 FIFO expired device activities, got %d", expiredActCount)
	}

	if len(joiner.pendingActs) > limit {
		t.Errorf("pending activities count %d exceeds limit %d", len(joiner.pendingActs), limit)
	}
}

func TestBridge_LockOSThreadAndBalance(t *testing.T) {
	hook := &mockHook{}
	proc := ProcessGenID{PID: 900, Generation: 1}
	annotator := NewAnnotator(proc, "test", WithCorrelationHook(hook))

	ctx := context.Background()
	_, op := annotator.Begin(ctx, "scoped_op")

	// 1. Normal execution
	err := op.NativeCall(func() error {
		return nil
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if hook.Depth() != 0 {
		t.Errorf("expected hook depth 0 after call, got %d", hook.Depth())
	}
	if hook.totalPushes != 1 || hook.totalPops != 1 {
		t.Errorf("expected 1 push and 1 pop, got %d and %d", hook.totalPushes, hook.totalPops)
	}

	// 2. Error in call
	expectedErr := errors.New("native failure")
	err = op.NativeCall(func() error {
		return expectedErr
	})
	if !errors.Is(err, expectedErr) {
		t.Errorf("expected %v, got %v", expectedErr, err)
	}
	if hook.Depth() != 0 {
		t.Errorf("expected hook depth 0 after error, got %d", hook.Depth())
	}

	// 3. Panic in call
	func() {
		defer func() {
			_ = recover()
		}()
		_ = op.NativeCall(func() error {
			panic("simulated panic")
		})
	}()
	if hook.Depth() != 0 {
		t.Errorf("expected hook depth 0 after panic recovery, got %d", hook.Depth())
	}

	// 4. Nested calls
	_, inner := annotator.Begin(ctx, "inner_op")
	err = op.NativeCall(func() error {
		return inner.NativeCall(func() error {
			if hook.Depth() != 2 {
				return fmt.Errorf("expected hook depth 2 inside nested call, got %d", hook.Depth())
			}
			return nil
		})
	})
	if err != nil {
		t.Fatalf("nested call failed: %v", err)
	}
	if hook.Depth() != 0 {
		t.Errorf("expected hook depth 0 after nested calls, got %d", hook.Depth())
	}
}

func TestBridge_CancellationCleanup(t *testing.T) {
	hook := &mockHook{}
	proc := ProcessGenID{PID: 950, Generation: 1}
	annotator := NewAnnotator(proc, "test-cancel", WithCorrelationHook(hook))

	ctx, cancel := context.WithCancel(context.Background())
	ctxOp, op := annotator.Begin(ctx, "cancel_op")

	// Cancel context prior to native call
	cancel()

	err := op.NativeCall(func() error {
		if ctxOp.Err() != nil {
			return ctxOp.Err()
		}
		return nil
	})

	if !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context.Canceled, got %v", err)
	}
	if hook.Depth() != 0 {
		t.Errorf("expected hook depth 0 after cancellation, got %d", hook.Depth())
	}
}

func TestBridge_ConcurrentGoroutines(t *testing.T) {
	hook := &mockHook{}
	proc := ProcessGenID{PID: 999, Generation: 1}
	annotator := NewAnnotator(proc, "concurrent", WithCorrelationHook(hook))

	var wg sync.WaitGroup
	var completed atomic.Uint64
	numGoroutines := 30

	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			ctx := context.Background()
			_, op := annotator.Begin(ctx, fmt.Sprintf("worker-%d", idx))
			defer op.End()

			time.Sleep(time.Duration(idx%3) * time.Millisecond)

			err := op.NativeCall(func() error {
				time.Sleep(time.Millisecond)
				return nil
			})
			if err == nil {
				completed.Add(1)
			}
		}(i)
	}

	wg.Wait()

	if completed.Load() != uint64(numGoroutines) {
		t.Fatalf("expected %d completed goroutines, got %d", numGoroutines, completed.Load())
	}
	if hook.ActiveStacks() != 0 {
		t.Errorf("expected 0 active stacks remaining, got %d", hook.ActiveStacks())
	}
}

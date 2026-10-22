package cudacollective

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nccl"
)

func TestQuiescenceCommunicatorBeforePlanRetainsFailure(t *testing.T) {
	submitErr, completionErr := errors.New("submission failed"), errors.New("completion failed")
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	calls := &ncclCalls{
		info:     func(nccl.Comm) (int, int, error) { return 1, 0, nil },
		invoke:   func(ncclInvocation) error { cancel(); return submitErr },
		complete: func(driver.Stream) error { return completionErr },
	}
	_, comm := testNCCL(t, 1, 0, calls)
	pipe, err := NewPipelineManager(comm, DoubleBuffer{Slots: [2]driver.DevicePtr{0x1000, 0x2000}, Bytes: 8}, 2)
	if err != nil {
		t.Fatal(err)
	}
	plan, err := NewPlan(comm, pipe)
	if err != nil {
		t.Fatal(err)
	}
	if err := plan.AddCollectiveStep("error", OperationDescriptor{Kind: KindAllReduce, Count: 1, Reduction: ReductionSum}, 0x1000, 0x1004); err != nil {
		t.Fatal(err)
	}
	executeErr := plan.ExecuteSteps(ctx)
	wantCompletion := completionErr
	completionErr = nil
	if err := comm.Close(); err != nil {
		t.Fatal(err)
	}
	tr := pipe.inFlight[0]
	select {
	case <-tr.Done:
	default:
		t.Fatal("close did not retire transfer")
	}
	for _, want := range []error{submitErr, wantCompletion, context.Canceled} {
		if !errors.Is(executeErr, want) || !errors.Is(tr.Err, want) {
			t.Fatalf("close order lost %v: execute=%v retired=%v", want, executeErr, tr.Err)
		}
	}
	if err := plan.Close(); err != nil {
		t.Fatal(err)
	}
	if err := plan.Close(); err != nil || pipe.pending != 0 {
		t.Fatalf("duplicate retirement: %v %d", err, pipe.pending)
	}
	if err := pipe.Drain(context.Background()); !errors.Is(err, wantCompletion) {
		t.Fatalf("drain lost failure: %v", err)
	}
}

func TestQuiescenceFailureRetainsPipeline(t *testing.T) {
	failure := errors.New("completion not established")
	completionErr := failure
	completions := 0
	calls := &ncclCalls{
		info:     func(nccl.Comm) (int, int, error) { return 1, 0, nil },
		invoke:   func(ncclInvocation) error { return nil },
		complete: func(driver.Stream) error { completions++; return completionErr },
	}
	_, comm := testNCCL(t, 1, 0, calls)
	pipe, err := NewPipelineManager(comm, DoubleBuffer{Slots: [2]driver.DevicePtr{0x1000, 0x2000}, Bytes: 8}, 2)
	if err != nil {
		t.Fatal(err)
	}
	plan, err := NewPlan(comm, pipe)
	if err != nil {
		t.Fatal(err)
	}
	op := OperationDescriptor{Kind: KindAllReduce, Count: 1, Reduction: ReductionSum}
	if err := plan.AddCollectiveStep("pending", op, 0x1000, 0x1004); err != nil {
		t.Fatal(err)
	}
	if err := plan.ExecuteSteps(context.Background()); !errors.Is(err, failure) {
		t.Fatal(err)
	}
	tr := pipe.inFlight[0]
	if tr == nil {
		t.Fatal("no reservation")
	}
	select {
	case <-tr.Done:
		t.Fatal("completion failure released pipeline buffer")
	default:
	}
	pipe.CompleteStage(tr, nil)
	select {
	case <-tr.Done:
		t.Fatal("nil completion hid pending ownership")
	default:
	}
	if err := comm.Close(); !errors.Is(err, failure) {
		t.Fatalf("close hid incomplete work: %v", err)
	}
	select {
	case <-tr.Done:
		t.Fatal("communicator close released unquiesced buffer")
	default:
	}
	if err := plan.Close(); !errors.Is(err, failure) {
		t.Fatalf("plan close lost pending owner: %v", err)
	}
	select {
	case <-tr.Done:
		t.Fatal("failed recovery released pipeline")
	default:
	}
	completionErr = nil
	if err := plan.Close(); err != nil {
		t.Fatal(err)
	}
	select {
	case <-tr.Done:
	default:
		t.Fatal("successful recovery did not finalize transfer")
	}
	if !errors.Is(tr.Err, failure) || pipe.pending != 0 {
		t.Fatalf("lost original error or slot count: %v %d", tr.Err, pipe.pending)
	}
	count := completions
	if err := plan.Close(); err != nil || completions != count {
		t.Fatalf("duplicate plan recovery: %v %d", err, completions)
	}
	if err := comm.Close(); err != nil {
		t.Fatal(err)
	}
	if err := comm.Close(); err != nil || completions != count {
		t.Fatalf("duplicate transport recovery: %v %d", err, completions)
	}
}

func TestQuiescenceErrorRetryAndIdentity(t *testing.T) {
	submitErr, completeErr := errors.New("partial submission"), errors.New("unknown completion")
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	invokes, completes := 0, 0
	calls := &ncclCalls{
		info:   func(nccl.Comm) (int, int, error) { return 1, 0, nil },
		invoke: func(ncclInvocation) error { invokes++; cancel(); return submitErr },
		complete: func(driver.Stream) error {
			completes++
			if completes < 3 {
				return completeErr
			}
			return nil
		},
	}
	transport, comm := testNCCL(t, 1, 0, calls)
	op := OperationDescriptor{Kind: KindAllReduce, Count: 1, Reduction: ReductionSum}
	err := transport.Execute(ctx, comm, op, 0x1000, 0x2000)
	for _, want := range []error{submitErr, completeErr, context.Canceled, ErrCompletionPending} {
		if !errors.Is(err, want) {
			t.Fatalf("lost %v: %v", want, err)
		}
	}
	var pending *CompletionError
	if !errors.As(err, &pending) || !pending.Pending() {
		t.Fatalf("no retryable owner: %v", err)
	}
	if transport.Available(comm.Membership()) {
		t.Fatal("pending transport is available")
	}
	if nextErr := transport.Execute(context.Background(), comm, op, 0x3000, 0x4000); nextErr != ErrCompletionPending || invokes != 1 {
		t.Fatalf("new work submitted while incomplete: %v invokes=%d", nextErr, invokes)
	}
	if retryErr := pending.Close(); !errors.Is(retryErr, submitErr) || !errors.Is(retryErr, completeErr) || !errors.Is(retryErr, context.Canceled) {
		t.Fatal(retryErr)
	}
	if !pending.Pending() {
		t.Fatal("failed retry dropped owner")
	}
	if err := pending.Close(); err != nil || pending.Pending() {
		t.Fatalf("retry: %v pending=%v", err, pending.Pending())
	}
	if err := pending.Close(); err != nil || completes != 3 || invokes != 1 {
		t.Fatalf("duplicate completion/resubmission: %v %d %d", err, completes, invokes)
	}
	if !transport.Available(comm.Membership()) || transport.closed {
		t.Fatal("per-operation recovery closed borrowed transport")
	}
	if err := transport.Close(); err != nil || completes != 3 {
		t.Fatalf("close: %v %d", err, completes)
	}
}

func TestQuiescencePlanCloseOnlyOwnOperation(t *testing.T) {
	completeErr := errors.New("still pending")
	calls := &ncclCalls{info: func(nccl.Comm) (int, int, error) { return 1, 0, nil }, invoke: func(ncclInvocation) error { return nil }, complete: func(driver.Stream) error { return completeErr }}
	transport, comm := testNCCL(t, 1, 0, calls)
	plan, _ := NewPlan(comm, nil)
	op := OperationDescriptor{Kind: KindAllReduce, Count: 1, Reduction: ReductionSum}
	if err := plan.AddCollectiveStep("direct", op, 0x1000, 0x2000); err != nil {
		t.Fatal(err)
	}
	if err := plan.ExecuteSteps(context.Background()); !errors.Is(err, completeErr) {
		t.Fatal(err)
	}
	if err := plan.Close(); !errors.Is(err, completeErr) {
		t.Fatal(err)
	}
	completeErr = nil
	if err := plan.Close(); err != nil {
		t.Fatal(err)
	}
	if transport.closed || comm.closed {
		t.Fatal("plan closed borrowed execution owner")
	}
	if err := plan.ExecuteSteps(context.Background()); !errors.Is(err, ErrCommunicatorClosed) {
		t.Fatal(err)
	}
	if err := comm.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestQuiescenceRecoveryContext(t *testing.T) {
	current := driver.Context(7)
	contextErr, failure := error(nil), errors.New("completion failure")
	invokes, completions := 0, 0
	calls := &ncclCalls{
		info:   func(nccl.Comm) (int, int, error) { return 1, 0, nil },
		invoke: func(ncclInvocation) error { invokes++; return nil },
		complete: func(driver.Stream) error {
			completions++
			if completions == 1 {
				return failure
			}
			return nil
		},
		currentContext: func() (driver.Context, error) { return current, contextErr },
	}
	transport, comm := testNCCL(t, 1, 0, calls)
	transport.Streams[0] = 0 // Wrong-context default-stream sync could otherwise succeed.
	op := OperationDescriptor{Kind: KindAllReduce, Count: 1, Reduction: ReductionSum}
	err := transport.Execute(context.Background(), comm, op, 0x1000, 0x2000)
	var owner *CompletionError
	if !errors.As(err, &owner) {
		t.Fatal(err)
	}
	current = 8
	if err := owner.Close(); !errors.Is(err, failure) || !owner.Pending() || completions != 1 {
		t.Fatalf("wrong context retired work: %v %d", err, completions)
	}
	contextErr = errors.New("context query failed")
	if err := owner.Close(); !errors.Is(err, contextErr) || !owner.Pending() || completions != 1 {
		t.Fatalf("context error lost: %v", err)
	}
	current, contextErr = 7, nil
	if err := owner.Close(); err != nil || owner.Pending() || completions != 2 || invokes != 1 {
		t.Fatalf("recovery: %v %d %d", err, completions, invokes)
	}
	if err := transport.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestQuiescenceAdmissionDoesNotTransferForeignOwner(t *testing.T) {
	failure := errors.New("rank zero still pending")
	completeCalls := 0
	calls := &ncclCalls{
		info:     func(c nccl.Comm) (int, int, error) { return 2, int(c) - 1, nil },
		invoke:   func(ncclInvocation) error { return nil },
		complete: func(driver.Stream) error { completeCalls++; return failure },
	}
	transport, firstComm := testNCCL(t, 2, 0, calls)
	secondComm, err := NewCommunicator(CommunicatorConfig{Rank: 1, Membership: firstComm.Membership(), Generation: 1, Transport: transport})
	if err != nil {
		t.Fatal(err)
	}
	op := OperationDescriptor{Kind: KindAllReduce, Count: 1, Reduction: ReductionSum}
	firstPlan, _ := NewPlan(firstComm, nil)
	secondPlan, _ := NewPlan(secondComm, nil)
	for _, p := range []*Plan{firstPlan, secondPlan} {
		if err := p.AddCollectiveStep("rank", op, 0x1000, 0x2000); err != nil {
			t.Fatal(err)
		}
	}
	firstErr := firstPlan.ExecuteSteps(context.Background())
	if !errors.Is(firstErr, failure) {
		t.Fatal(firstErr)
	}
	err = secondPlan.ExecuteSteps(context.Background())
	var foreign *CompletionError
	if !errors.Is(err, ErrCompletionPending) || errors.As(err, &foreign) {
		t.Fatalf("foreign recovery authority leaked: %v", err)
	}
	if err := secondPlan.Close(); err != nil || completeCalls != 1 {
		t.Fatalf("second plan recovered another rank: %v %d", err, completeCalls)
	}
	if err := firstPlan.Close(); !errors.Is(err, failure) || completeCalls != 2 {
		t.Fatalf("first plan lost recovery: %v %d", err, completeCalls)
	}
}

type quiescenceCloseTransport struct {
	mu                       sync.Mutex
	calls, active, maxActive int
	failure                  error
	entered                  [2]chan struct{}
	release                  [2]chan struct{}
}

func (*quiescenceCloseTransport) Type() TransportType       { return TransportTypeControlPlaneTest }
func (*quiescenceCloseTransport) Available(Membership) bool { return true }
func (*quiescenceCloseTransport) Execute(context.Context, *Communicator, OperationDescriptor, driver.DevicePtr, driver.DevicePtr) error {
	return nil
}
func (x *quiescenceCloseTransport) Close() error {
	x.mu.Lock()
	i := x.calls
	x.calls++
	x.active++
	if x.active > x.maxActive {
		x.maxActive = x.active
	}
	x.mu.Unlock()
	defer func() { x.mu.Lock(); x.active--; x.mu.Unlock() }()
	if i >= len(x.entered) {
		return fmt.Errorf("unexpected close attempt %d", i)
	}
	close(x.entered[i])
	<-x.release[i]
	if i == 0 {
		return x.failure
	}
	return nil
}

func TestQuiescenceCommunicatorCloseRetryConcurrent(t *testing.T) {
	failure := errors.New("transport remains active")
	x := &quiescenceCloseTransport{failure: failure}
	for i := range x.entered {
		x.entered[i], x.release[i] = make(chan struct{}), make(chan struct{})
	}
	m, _ := NewMembership([]Rank{{ID: 0}})
	c, err := NewCommunicator(CommunicatorConfig{Rank: 0, Membership: m, Generation: 1, Transport: x})
	if err != nil {
		t.Fatal(err)
	}
	tr, err := c.AcquireTransferBuffer(0x1000, 8, 0)
	if err != nil {
		t.Fatal(err)
	}
	first, second := make(chan error, 1), make(chan error, 1)
	go func() { first <- c.Close() }()
	<-x.entered[0]
	go func() { second <- c.Close() }()
	close(x.release[0])
	if err := <-first; !errors.Is(err, failure) {
		t.Fatal(err)
	}
	<-x.entered[1]
	select {
	case <-tr.Done:
		t.Fatal("failed transport close released buffer")
	default:
	}
	if _, err := c.AcquireTransferBuffer(0x2000, 8, 1); !errors.Is(err, ErrCommunicatorClosed) {
		t.Fatal(err)
	}
	close(x.release[1])
	if err := <-second; err != nil {
		t.Fatal(err)
	}
	select {
	case <-tr.Done:
	default:
		t.Fatal("quiescent close did not release")
	}
	if err := c.Close(); err != nil {
		t.Fatal(err)
	}
	x.mu.Lock()
	defer x.mu.Unlock()
	if x.calls != 2 || x.maxActive != 1 {
		t.Fatalf("calls=%d concurrent=%d", x.calls, x.maxActive)
	}
}

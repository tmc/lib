package cudacollective

import (
	"context"
	"errors"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nccl"
	"github.com/tmc/lib/nvidia/x/cudair"
	"testing"
	"time"
)

func testNCCL(t *testing.T, world, rank int, calls *ncclCalls) (*NCCLTransport, *Communicator) {
	t.Helper()
	ranks := make([]Rank, world)
	handles := make([]nccl.Comm, world)
	streams := make([]driver.Stream, world)
	for i := range ranks {
		ranks[i] = Rank{ID: i}
		handles[i] = nccl.Comm(i + 1)
		streams[i] = driver.Stream(i + 11)
	}
	m, err := NewMembership(ranks)
	if err != nil {
		t.Fatal(err)
	}
	tr := &NCCLTransport{Comms: handles, Streams: streams, calls: calls}
	c, err := NewCommunicator(CommunicatorConfig{Rank: rank, Membership: m, Generation: 1, Transport: tr})
	if err != nil {
		t.Fatal(err)
	}
	return tr, c
}
func TestNCCLInvocation(t *testing.T) {
	for _, kind := range []Kind{KindAllReduce, KindAllGather, KindBroadcast, KindReduceScatter, KindReduce} {
		t.Run(kind.String(), func(t *testing.T) {
			complete := false
			calls := &ncclCalls{info: func(nccl.Comm) (int, int, error) { return 2, 1, nil }, invoke: func(q ncclInvocation) error {
				if q.comm != 2 || q.stream != 12 || q.op.Count != 3 || q.op.Root != 1 || q.dtype != nccl.Float16 || q.send != 0x1000 || q.recv != 0x2000 {
					t.Fatalf("invocation: %+v", q)
				}
				send, recv := uint64(6), uint64(6)
				if kind == KindAllGather {
					recv = 12
				}
				if kind == KindReduceScatter {
					send = 12
				}
				if q.sendBytes != send || q.recvBytes != recv {
					t.Fatalf("bytes: %d %d", q.sendBytes, q.recvBytes)
				}
				if kind != KindAllGather && kind != KindBroadcast && q.reduction != nccl.Max {
					t.Fatal(q.reduction)
				}
				return nil
			}, complete: func(s driver.Stream) error { complete = true; return nil }}
			tr, c := testNCCL(t, 2, 1, calls)
			err := tr.Execute(context.Background(), c, OperationDescriptor{Kind: kind, DataType: cudair.Float16, Count: 3, Root: 1, Reduction: ReductionMax}, 0x1000, 0x2000)
			if err != nil || !complete {
				t.Fatalf("error=%v complete=%v", err, complete)
			}
		})
	}
}
func TestNCCLRefusals(t *testing.T) {
	calls := &ncclCalls{info: func(nccl.Comm) (int, int, error) { return 1, 0, nil }, invoke: func(ncclInvocation) error { t.Fatal("unexpected submission"); return nil }, complete: func(driver.Stream) error { t.Fatal("unexpected completion"); return nil }}
	tr, c := testNCCL(t, 1, 0, calls)
	base := OperationDescriptor{Kind: KindAllReduce, Count: 2, Reduction: ReductionSum}
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	if err := tr.Execute(cancelled, c, base, 0x1000, 0x2000); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	for _, edit := range []func(*OperationDescriptor){func(o *OperationDescriptor) { o.Count = 0 }, func(o *OperationDescriptor) { o.Kind = KindP2P }, func(o *OperationDescriptor) { o.DataType = cudair.DataType(100) }, func(o *OperationDescriptor) { o.Reduction = 0 }, func(o *OperationDescriptor) { o.Root = 1 }, func(o *OperationDescriptor) { o.Topology = TopologyRDMA }, func(o *OperationDescriptor) { o.Count = int(^uint(0) >> 1) }} {
		op := base
		edit(&op)
		if err := tr.Execute(context.Background(), c, op, 0x1000, 0x2000); err == nil {
			t.Fatalf("accepted %+v", op)
		}
	}
	if err := tr.Execute(context.Background(), c, base, 0x1000, 0x1001); !errors.Is(err, ErrBufferOverlap) {
		t.Fatal(err)
	}
	tr.Comms[0] = 0
	if err := tr.Execute(context.Background(), c, base, 0x1000, 0x2000); !errors.Is(err, ErrExecutionUnavailable) {
		t.Fatal(err)
	}
	tr.Comms[0] = 1
	calls.info = func(nccl.Comm) (int, int, error) { return 2, 1, nil }
	if err := tr.Execute(context.Background(), c, base, 0x1000, 0x2000); !errors.Is(err, ErrRankOrderMismatch) {
		t.Fatal(err)
	}
	if err := tr.Close(); err != nil {
		t.Fatal(err)
	}
	if err := tr.Execute(context.Background(), c, base, 0x1000, 0x2000); !errors.Is(err, ErrCommunicatorClosed) {
		t.Fatal(err)
	}
}
func TestNCCLCancelAfterSubmissionDrains(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	entered, release := make(chan struct{}), make(chan struct{})
	calls := &ncclCalls{info: func(nccl.Comm) (int, int, error) { return 1, 0, nil }, invoke: func(ncclInvocation) error { cancel(); return nil }, complete: func(driver.Stream) error { close(entered); <-release; return nil }}
	tr, c := testNCCL(t, 1, 0, calls)
	done := make(chan error, 1)
	go func() {
		done <- tr.Execute(ctx, c, OperationDescriptor{Kind: KindAllReduce, Count: 1, Reduction: ReductionSum}, 0x1000, 0x2000)
	}()
	<-entered
	closed := make(chan struct{})
	go func() { tr.Close(); close(closed) }()
	select {
	case <-done:
		t.Fatal("returned before completion")
	case <-closed:
		t.Fatal("closed before completion")
	case <-time.After(20 * time.Millisecond):
	}
	close(release)
	if err := <-done; !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	<-closed
	if tr.Comms[0] != 1 {
		t.Fatal("borrowed communicator destroyed")
	}
}
func TestNCCLErrorsStillComplete(t *testing.T) {
	submitErr, finishErr := errors.New("submit"), errors.New("complete")
	complete := false
	calls := &ncclCalls{info: func(nccl.Comm) (int, int, error) { return 1, 0, nil }, invoke: func(ncclInvocation) error { return submitErr }, complete: func(driver.Stream) error { complete = true; return finishErr }}
	tr, c := testNCCL(t, 1, 0, calls)
	err := tr.Execute(context.Background(), c, OperationDescriptor{Kind: KindAllReduce, Count: 1, Reduction: ReductionSum}, 0x1000, 0x2000)
	if !errors.Is(err, submitErr) || !errors.Is(err, finishErr) || !complete {
		t.Fatal(err, complete)
	}
}
func TestNCCLInPlaceOffsets(t *testing.T) {
	for _, kind := range []Kind{KindAllGather, KindReduceScatter} {
		op := OperationDescriptor{Kind: kind, Count: 4, Reduction: ReductionSum, InPlace: true}
		send, recv := driver.DevicePtr(0x1000), driver.DevicePtr(0x1010)
		if kind == KindAllGather {
			send, recv = recv, send
		}
		if _, err := ncclRequest(op, 1, 2, send, recv); err != nil {
			t.Fatal(err)
		}
		if _, err := ncclRequest(op, 1, 2, send+1, recv); !errors.Is(err, ErrBufferOverlap) {
			t.Fatal(err)
		}
	}
}

func TestNCCLRanksSubmitConcurrently(t *testing.T) {
	entered := make(chan struct{}, 2)
	release := make(chan struct{})
	calls := &ncclCalls{info: func(c nccl.Comm) (int, int, error) { return 2, int(c) - 1, nil }, invoke: func(ncclInvocation) error { entered <- struct{}{}; <-release; return nil }, complete: func(driver.Stream) error { return nil }}
	tr, c0 := testNCCL(t, 2, 0, calls)
	c1, err := NewCommunicator(CommunicatorConfig{Rank: 1, Membership: c0.Membership(), Generation: 1, Transport: tr})
	if err != nil {
		t.Fatal(err)
	}
	done := make(chan error, 2)
	op := OperationDescriptor{Kind: KindAllReduce, Count: 1, Reduction: ReductionSum}
	for _, c := range []*Communicator{c0, c1} {
		go func(c *Communicator) { done <- tr.Execute(context.Background(), c, op, 0x1000, 0x2000) }(c)
	}
	for range 2 {
		select {
		case <-entered:
		case <-time.After(time.Second):
			close(release)
			t.Fatal("rank submissions serialized across collective")
		}
	}
	close(release)
	for range 2 {
		if err := <-done; err != nil {
			t.Fatal(err)
		}
	}
}

func TestNCCLScalarAndReductionMappings(t *testing.T) {
	types := []struct {
		ir     cudair.DataType
		native nccl.DataType
		bytes  uint64
	}{{cudair.Float32, nccl.Float32, 4}, {cudair.Float16, nccl.Float16, 2}, {cudair.BFloat16, nccl.Bfloat16, 2}, {cudair.Int32, nccl.Int32, 4}, {cudair.Int64, nccl.Int64, 8}, {cudair.Int8, nccl.Int8, 1}, {cudair.Uint8, nccl.Uint8, 1}}
	reductions := []struct {
		ir     Reduction
		native nccl.RedOp
	}{{ReductionSum, nccl.Sum}, {ReductionProd, nccl.Prod}, {ReductionMin, nccl.Min}, {ReductionMax, nccl.Max}, {ReductionAvg, nccl.Avg}}
	for _, dt := range types {
		for _, r := range reductions {
			q, err := ncclRequest(OperationDescriptor{Kind: KindAllReduce, Count: 3, DataType: dt.ir, Reduction: r.ir}, 0, 1, 0x1000, 0x2000)
			if err != nil || q.dtype != dt.native || q.reduction != r.native || q.sendBytes != 3*dt.bytes {
				t.Fatalf("mapping %v %v: %+v %v", dt.ir, r.ir, q, err)
			}
		}
	}
}

func TestNCCLCommunicatorCloseWaitsForPipeline(t *testing.T) {
	entered, release := make(chan struct{}), make(chan struct{})
	calls := &ncclCalls{info: func(nccl.Comm) (int, int, error) { return 1, 0, nil }, invoke: func(ncclInvocation) error { return nil }, complete: func(driver.Stream) error { close(entered); <-release; return nil }}
	tr, c := testNCCL(t, 1, 0, calls)
	pipe, err := NewPipelineManager(c, DoubleBuffer{Slots: [2]driver.DevicePtr{0x1000, 0x2000}, Bytes: 4}, 2)
	if err != nil {
		t.Fatal(err)
	}
	op := OperationDescriptor{Kind: KindAllReduce, Count: 1, Reduction: ReductionSum}
	transfer, err := pipe.SubmitStage(context.Background(), op)
	if err != nil {
		t.Fatal(err)
	}
	done := make(chan error, 1)
	go func() {
		err := tr.Execute(context.Background(), c, op, 0x1000, 0x3000)
		pipe.CompleteStage(transfer, err)
		done <- err
	}()
	<-entered
	closed := make(chan error, 2)
	for range 2 {
		go func() { closed <- c.Close() }()
	}
	deadline := time.Now().Add(time.Second)
	for c.CheckAlive() == nil {
		if time.Now().After(deadline) {
			t.Fatal("close never stopped admission")
		}
		time.Sleep(time.Millisecond)
	}
	if _, err := pipe.SubmitStage(context.Background(), op); !errors.Is(err, ErrCommunicatorClosed) {
		t.Fatal(err)
	}
	select {
	case <-transfer.Done:
		t.Fatal("marked complete before GPU completion")
	case <-closed:
		t.Fatal("close returned early")
	case <-time.After(20 * time.Millisecond):
	}
	close(release)
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	for range 2 {
		if err := <-closed; err != nil {
			t.Fatal(err)
		}
	}
	// Late and repeated completion must neither panic nor decrement twice.
	pipe.CompleteStage(transfer, nil)
	pipe.CompleteStage(transfer, nil)
	if pipe.pending != 0 {
		t.Fatal("pending", pipe.pending)
	}
}

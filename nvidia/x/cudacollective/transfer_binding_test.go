package cudacollective

import (
	"context"
	"errors"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudair"
)

type bindingCheckTransport struct {
	worldSize int
	calls     int
	onExecute func(comm *Communicator, op OperationDescriptor, send, recv driver.DevicePtr) error
}

func (t *bindingCheckTransport) Type() TransportType { return TransportTypeControlPlaneTest }

func (t *bindingCheckTransport) Available(m Membership) bool {
	return m.WorldSize() == t.worldSize
}

func (t *bindingCheckTransport) Execute(ctx context.Context, comm *Communicator, op OperationDescriptor, send, recv driver.DevicePtr) error {
	t.calls++
	if t.onExecute != nil {
		return t.onExecute(comm, op, send, recv)
	}
	return nil
}

func (t *bindingCheckTransport) Close() error { return nil }

func newBindingFixture(t *testing.T, tr Transport, buf DoubleBuffer, maxQueue int) (*Communicator, *PipelineManager, *Plan) {
	t.Helper()
	m, err := NewMembership([]Rank{
		{ID: 0, HostID: "hostA", DeviceID: 0},
		{ID: 1, HostID: "hostA", DeviceID: 1},
	})
	if err != nil {
		t.Fatal(err)
	}
	if tr == nil {
		tr = NewDeterministicTestTransport(2)
	}
	t.Cleanup(func() { tr.Close() })
	comm, err := NewCommunicator(CommunicatorConfig{
		Rank:       0,
		Membership: m,
		Generation: 1,
		Transport:  tr,
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { comm.Close() })
	pipeline, err := NewPipelineManager(comm, buf, maxQueue)
	if err != nil {
		t.Fatal(err)
	}
	plan, err := NewPlan(comm, pipeline)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { plan.Close() })
	return comm, pipeline, plan
}

func TestPlanBoundBufferExecution(t *testing.T) {
	buf := DoubleBuffer{
		Slots: [2]driver.DevicePtr{0x10000, 0x20000},
		Bytes: 64,
	}
	tr := &bindingCheckTransport{worldSize: 2}
	comm, pipeline, plan := newBindingFixture(t, tr, buf, 2)

	tr.onExecute = func(c *Communicator, op OperationDescriptor, send, recv driver.DevicePtr) error {
		sendBytes, recvBytes, err := op.transferBytes(c.Membership().WorldSize())
		if err != nil {
			t.Fatalf("transferBytes in Execute: %v", err)
		}
		// While Execute runs, both send and recv subranges must be held in-flight on comm.
		if _, err := c.AcquireTransferBuffer(uintptr(send), sendBytes, op.PipelineSlot); !errors.Is(err, ErrBufferInUse) {
			t.Fatalf("send subrange [0x%x, +%d) not held in flight during Execute: %v", send, sendBytes, err)
		}
		if _, err := c.AcquireTransferBuffer(uintptr(recv), recvBytes, op.PipelineSlot); !errors.Is(err, ErrBufferInUse) {
			t.Fatalf("recv subrange [0x%x, +%d) not held in flight during Execute: %v", recv, recvBytes, err)
		}
		return nil
	}

	step0 := OperationDescriptor{
		Kind:         KindAllReduce,
		DataType:     cudair.Float32,
		Count:        4, // 16 bytes send, 16 bytes recv
		Reduction:    ReductionSum,
		PipelineSlot: 0,
	}
	step1 := OperationDescriptor{
		Kind:         KindAllGather,
		DataType:     cudair.Float32,
		Count:        4, // 16 bytes send, 32 bytes recv (worldSize=2)
		PipelineSlot: 1,
	}
	step2 := OperationDescriptor{
		Kind:         KindAllReduce,
		DataType:     cudair.Float32,
		Count:        8, // 32 bytes in-place
		Reduction:    ReductionMax,
		InPlace:      true,
		PipelineSlot: 0,
	}

	if err := plan.AddCollectiveStep("step0-out-of-place", step0, buf.Slots[0], buf.Slots[0]+16); err != nil {
		t.Fatalf("AddCollectiveStep step0: %v", err)
	}
	if err := plan.AddCollectiveStep("step1-allgather", step1, buf.Slots[1], buf.Slots[1]+16); err != nil {
		t.Fatalf("AddCollectiveStep step1: %v", err)
	}
	if err := plan.AddCollectiveStep("step2-inplace", step2, buf.Slots[0], buf.Slots[0]); err != nil {
		t.Fatalf("AddCollectiveStep step2: %v", err)
	}

	ctx := context.Background()
	if err := plan.ExecuteSteps(ctx); err != nil {
		t.Fatalf("ExecuteSteps: %v", err)
	}
	if tr.calls != 3 {
		t.Fatalf("transport calls = %d; want 3", tr.calls)
	}
	if err := pipeline.Drain(ctx); err != nil {
		t.Fatalf("Drain: %v", err)
	}
	if len(comm.inFlight) != 0 || pipeline.pending != 0 {
		t.Fatalf("leaked in-flight state after ExecuteSteps: comm.inFlight=%v pending=%d", comm.inFlight, pipeline.pending)
	}
}

func TestPlanRefusesMismatchedStageBuffers(t *testing.T) {
	buf := DoubleBuffer{
		Slots: [2]driver.DevicePtr{0x10000, 0x20000},
		Bytes: 64,
	}
	tr := &bindingCheckTransport{worldSize: 2}
	comm, pipeline, plan := newBindingFixture(t, tr, buf, 1)

	baseOp := OperationDescriptor{
		Kind:         KindAllReduce,
		DataType:     cudair.Float32,
		Count:        4, // 16 bytes
		Reduction:    ReductionSum,
		PipelineSlot: 0,
	}

	tests := []struct {
		name    string
		op      OperationDescriptor
		send    driver.DevicePtr
		recv    driver.DevicePtr
		wantErr error
	}{
		{
			name:    "unrelated send and recv",
			op:      baseOp,
			send:    0x90000,
			recv:    0xa0000,
			wantErr: ErrBufferUnbound,
		},
		{
			name:    "unrelated recv only",
			op:      baseOp,
			send:    buf.Slots[0],
			recv:    0x90000,
			wantErr: ErrBufferUnbound,
		},
		{
			name:    "slot 0 step using slot 1 buffers",
			op:      baseOp,
			send:    buf.Slots[1],
			recv:    buf.Slots[1] + 16,
			wantErr: ErrBufferUnbound,
		},
		{
			name: "slot 1 step using slot 0 buffers",
			op: func() OperationDescriptor {
				op := baseOp
				op.PipelineSlot = 1
				return op
			}(),
			send:    buf.Slots[0],
			recv:    buf.Slots[0] + 16,
			wantErr: ErrBufferUnbound,
		},
		{
			name:    "split across slot 0 and slot 1",
			op:      baseOp,
			send:    buf.Slots[0],
			recv:    buf.Slots[1],
			wantErr: ErrBufferUnbound,
		},
		{
			name: "in-place with distinct buffers",
			op: func() OperationDescriptor {
				op := baseOp
				op.InPlace = true
				return op
			}(),
			send:    buf.Slots[0],
			recv:    buf.Slots[0] + 16,
			wantErr: ErrBufferOverlap,
		},
	}

	ctx := context.Background()
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := plan.AddCollectiveStep(tt.name, tt.op, tt.send, tt.recv)
			if !errors.Is(err, tt.wantErr) {
				t.Fatalf("AddCollectiveStep(%s) = %v; want %v", tt.name, err, tt.wantErr)
			}
			if len(plan.steps) != 0 || tr.calls != 0 || pipeline.pending != 0 || len(comm.inFlight) != 0 {
				t.Fatalf("unexpected state after rejected AddCollectiveStep: steps=%d calls=%d pending=%d inFlight=%d",
					len(plan.steps), tr.calls, pipeline.pending, len(comm.inFlight))
			}

			// Mutated step in plan.steps must also be refused by ExecuteSteps before transport execution.
			plan.steps = []CollectiveStep{{
				StepID:     0,
				Name:       tt.name,
				Descriptor: tt.op,
				SendBuffer: tt.send,
				RecvBuffer: tt.recv,
				Slot:       tt.op.PipelineSlot,
			}}
			err = plan.ExecuteSteps(ctx)
			plan.steps = nil
			if !errors.Is(err, tt.wantErr) {
				t.Fatalf("ExecuteSteps(%s) = %v; want %v", tt.name, err, tt.wantErr)
			}
			if tr.calls != 0 || pipeline.pending != 0 || len(comm.inFlight) != 0 {
				t.Fatalf("unexpected state after rejected ExecuteSteps: calls=%d pending=%d inFlight=%d",
					tr.calls, pipeline.pending, len(comm.inFlight))
			}
		})
	}
}

func TestPlanRefusesExtentOverrun(t *testing.T) {
	const top = ^driver.DevicePtr(0)
	buf := DoubleBuffer{
		Slots: [2]driver.DevicePtr{0x10000, 0x20000},
		Bytes: 32,
	}
	tr := &bindingCheckTransport{worldSize: 2}
	comm, pipeline, plan := newBindingFixture(t, tr, buf, 1)

	tests := []struct {
		name string
		op   OperationDescriptor
		send driver.DevicePtr
		recv driver.DevicePtr
	}{
		{
			name: "in-place count exceeds slot bytes",
			op: OperationDescriptor{
				Kind:         KindAllReduce,
				DataType:     cudair.Float32,
				Count:        16, // 64 bytes > 32-byte slot
				Reduction:    ReductionSum,
				InPlace:      true,
				PipelineSlot: 0,
			},
			send: buf.Slots[0],
			recv: buf.Slots[0],
		},
		{
			name: "subrange tail overruns slot end",
			op: OperationDescriptor{
				Kind:         KindAllReduce,
				DataType:     cudair.Float32,
				Count:        4, // 16 bytes
				Reduction:    ReductionSum,
				PipelineSlot: 0,
			},
			send: buf.Slots[0],
			recv: buf.Slots[0] + 20, // [0x10014, 0x10024) overruns [0x10000, 0x10020)
		},
		{
			name: "allgather rank-multiplied recv overruns slot",
			op: OperationDescriptor{
				Kind:         KindAllGather,
				DataType:     cudair.Float32,
				Count:        4, // send=16B, recv=32B; at offset 16 [0x10010, 0x10030) overruns 32B slot
				PipelineSlot: 0,
			},
			send: buf.Slots[0],
			recv: buf.Slots[0] + 16,
		},
		{
			name: "reducescatter rank-multiplied send overruns slot",
			op: OperationDescriptor{
				Kind:         KindReduceScatter,
				DataType:     cudair.Float32,
				Count:        4, // send=32B, recv=16B; send+recv need 48B > 32B slot
				Reduction:    ReductionSum,
				PipelineSlot: 0,
			},
			send: buf.Slots[0],
			recv: buf.Slots[0] + 32,
		},
		{
			name: "send address wraps 64-bit space",
			op: OperationDescriptor{
				Kind:         KindAllReduce,
				DataType:     cudair.Float32,
				Count:        4, // 16 bytes
				Reduction:    ReductionSum,
				PipelineSlot: 0,
			},
			send: top - 7,
			recv: buf.Slots[0],
		},
	}

	ctx := context.Background()
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := plan.AddCollectiveStep(tt.name, tt.op, tt.send, tt.recv)
			if !errors.Is(err, ErrBufferLengthMismatch) {
				t.Fatalf("AddCollectiveStep(%s) = %v; want ErrBufferLengthMismatch", tt.name, err)
			}
			if len(plan.steps) != 0 || tr.calls != 0 || pipeline.pending != 0 || len(comm.inFlight) != 0 {
				t.Fatalf("state mutated after rejected overrun: steps=%d calls=%d pending=%d inFlight=%d",
					len(plan.steps), tr.calls, pipeline.pending, len(comm.inFlight))
			}

			plan.steps = []CollectiveStep{{
				StepID:     0,
				Name:       tt.name,
				Descriptor: tt.op,
				SendBuffer: tt.send,
				RecvBuffer: tt.recv,
				Slot:       tt.op.PipelineSlot,
			}}
			err = plan.ExecuteSteps(ctx)
			plan.steps = nil
			if !errors.Is(err, ErrBufferLengthMismatch) {
				t.Fatalf("ExecuteSteps(%s) = %v; want ErrBufferLengthMismatch", tt.name, err)
			}
		})
	}
}

func TestCommunicatorAndPlanRefusePartialOverlap(t *testing.T) {
	buf := DoubleBuffer{
		Slots: [2]driver.DevicePtr{0x10000, 0x20000},
		Bytes: 64,
	}
	tr := &bindingCheckTransport{worldSize: 2}
	comm, pipeline, plan := newBindingFixture(t, tr, buf, 2)

	// 1. Step with partially overlapping out-of-place send and recv ranges.
	op := OperationDescriptor{
		Kind:         KindAllReduce,
		DataType:     cudair.Float32,
		Count:        4, // 16 bytes
		Reduction:    ReductionSum,
		PipelineSlot: 0,
	}
	if err := plan.AddCollectiveStep("partial-send-recv", op, buf.Slots[0], buf.Slots[0]+8); !errors.Is(err, ErrBufferOverlap) {
		t.Fatalf("overlapping send/recv: got %v; want ErrBufferOverlap", err)
	}
	if err := plan.AddCollectiveStep("identical-out-of-place", op, buf.Slots[0], buf.Slots[0]); !errors.Is(err, ErrBufferOverlap) {
		t.Fatalf("identical out-of-place send/recv: got %v; want ErrBufferOverlap", err)
	}
	if err := plan.AddCollectiveStep("straddles-slot-start", op, buf.Slots[0]-8, buf.Slots[0]+16); !errors.Is(err, ErrBufferOverlap) {
		t.Fatalf("straddling slot start: got %v; want ErrBufferOverlap", err)
	}

	// 2. Communicator.AcquireTransferBuffer partial-overlap refusal against active [0x10000, 0x10020).
	active, err := comm.AcquireTransferBuffer(0x10000, 32, 0)
	if err != nil {
		t.Fatalf("initial AcquireTransferBuffer: %v", err)
	}

	overlapCases := []struct {
		name  string
		addr  uintptr
		bytes int
	}{
		{"exact match", 0x10000, 32},
		{"inner subrange", 0x10008, 8},
		{"tail overlap", 0x10018, 16},
		{"head overlap", 0x0fff8, 16},
		{"last byte overlap", 0x1001f, 8},
		{"first byte overlap", 0x0fff9, 8},
		{"enclosing superset", 0x0fff0, 64},
	}
	for _, tc := range overlapCases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := comm.AcquireTransferBuffer(tc.addr, tc.bytes, 1)
			if got != nil || !errors.Is(err, ErrBufferInUse) {
				t.Fatalf("AcquireTransferBuffer(%#x, %d) = %v, %v; want ErrBufferInUse", tc.addr, tc.bytes, got, err)
			}
			if tc.addr != 0x10000 || tc.bytes != 32 {
				if !errors.Is(err, ErrBufferOverlap) {
					t.Fatalf("partial overlap AcquireTransferBuffer(%#x, %d) = %v; want ErrBufferOverlap", tc.addr, tc.bytes, err)
				}
			}
		})
	}

	// Adjacent non-overlapping ranges [0x0ffe0, 0x10000) and [0x10020, 0x10040) must succeed.
	before, err := comm.AcquireTransferBuffer(0x0ffe0, 32, 1)
	if err != nil {
		t.Fatalf("adjacent preceding range refused: %v", err)
	}
	comm.ReleaseTransferBuffer(before, nil)

	after, err := comm.AcquireTransferBuffer(0x10020, 32, 1)
	if err != nil {
		t.Fatalf("adjacent following range refused: %v", err)
	}
	comm.ReleaseTransferBuffer(after, nil)

	comm.ReleaseTransferBuffer(active, nil)

	// 3. Plan execution refuses when an overlapping subrange inside the target pipeline slot is already held in flight.
	subHold, err := comm.AcquireTransferBuffer(uintptr(buf.Slots[0]+8), 8, 1)
	if err != nil {
		t.Fatalf("AcquireTransferBuffer subrange: %v", err)
	}
	if err := plan.AddCollectiveStep("bound-step", op, buf.Slots[0], buf.Slots[0]+16); err != nil {
		t.Fatalf("AddCollectiveStep bound-step: %v", err)
	}
	err = plan.ExecuteSteps(context.Background())
	if !errors.Is(err, ErrBufferInUse) || !errors.Is(err, ErrBufferOverlap) {
		t.Fatalf("ExecuteSteps with held subrange = %v; want ErrBufferInUse and ErrBufferOverlap", err)
	}
	if tr.calls != 0 || pipeline.pending != 0 {
		t.Fatalf("transport executed or pipeline pending leaked on partial overlap: calls=%d pending=%d", tr.calls, pipeline.pending)
	}
	comm.ReleaseTransferBuffer(subHold, nil)

	// Once the overlapping subrange is released, the bound plan step executes cleanly.
	if err := plan.ExecuteSteps(context.Background()); err != nil {
		t.Fatalf("ExecuteSteps after releasing overlapping subrange: %v", err)
	}
	if tr.calls != 1 {
		t.Fatalf("transport calls = %d; want 1", tr.calls)
	}
}

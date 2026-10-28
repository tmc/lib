package cudacollective_test

import (
	"context"
	"errors"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudacollective"
	"github.com/tmc/lib/nvidia/x/cudair"
)

func newAdmissionComm(t *testing.T) *cudacollective.Communicator {
	t.Helper()
	m, err := cudacollective.NewMembership([]cudacollective.Rank{
		{ID: 0, HostID: "hostA", DeviceID: 0},
		{ID: 1, HostID: "hostA", DeviceID: 1},
	})
	if err != nil {
		t.Fatal(err)
	}
	transport := cudacollective.NewDeterministicTestTransport(2)
	t.Cleanup(func() { transport.Close() })
	comm, err := cudacollective.NewCommunicator(cudacollective.CommunicatorConfig{
		Rank:       0,
		Membership: m,
		Generation: 1,
		Transport:  transport,
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { comm.Close() })
	return comm
}

func TestPipelineSlotRangeAdmission(t *testing.T) {
	const top = ^driver.DevicePtr(0)
	tests := []struct {
		name  string
		slots [2]driver.DevicePtr
		bytes int
		want  error
	}{
		{"disjoint", [2]driver.DevicePtr{0x1000, 0x2000}, 16, nil},
		{"adjacent", [2]driver.DevicePtr{0x1000, 0x1010}, 16, nil},
		{"adjacent reversed", [2]driver.DevicePtr{0x1010, 0x1000}, 16, nil},
		{"equal", [2]driver.DevicePtr{0x1000, 0x1000}, 16, cudacollective.ErrBufferOverlap},
		{"overlap", [2]driver.DevicePtr{0x1000, 0x1004}, 16, cudacollective.ErrBufferOverlap},
		{"overlap reversed", [2]driver.DevicePtr{0x1004, 0x1000}, 16, cudacollective.ErrBufferOverlap},
		{"overlap last byte", [2]driver.DevicePtr{0x1000, 0x100f}, 16, cudacollective.ErrBufferOverlap},
		{"ends below top", [2]driver.DevicePtr{0x1000, top - 16}, 16, nil},
		{"ends past top", [2]driver.DevicePtr{0x1000, top - 15}, 16, cudacollective.ErrBufferLengthMismatch},
		{"wraps", [2]driver.DevicePtr{0x1000, top - 7}, 16, cudacollective.ErrBufferLengthMismatch},
		{"wraps onto other slot", [2]driver.DevicePtr{0x8, top - 7}, 32, cudacollective.ErrBufferLengthMismatch},
		{"first ends past top", [2]driver.DevicePtr{top - 15, 0x1000}, 16, cudacollective.ErrBufferLengthMismatch},
		{"first wraps", [2]driver.DevicePtr{top - 7, 0x1000}, 16, cudacollective.ErrBufferLengthMismatch},
		{"first wraps onto other slot", [2]driver.DevicePtr{top - 7, 0x8}, 32, cudacollective.ErrBufferLengthMismatch},
		{"zero bytes", [2]driver.DevicePtr{0x1000, 0x2000}, 0, cudacollective.ErrBufferLengthMismatch},
		{"nil slot", [2]driver.DevicePtr{0, 0x2000}, 16, cudacollective.ErrNilBuffer},
	}
	comm := newAdmissionComm(t)
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			buf := cudacollective.DoubleBuffer{Slots: tt.slots, Bytes: tt.bytes}
			p, err := cudacollective.NewPipelineManager(comm, buf, 2)
			if tt.want == nil {
				if err != nil || p == nil {
					t.Fatalf("NewPipelineManager(%#x, %d) = %v, %v; want pipeline", tt.slots, tt.bytes, p, err)
				}
				return
			}
			if !errors.Is(err, tt.want) || p != nil {
				t.Fatalf("NewPipelineManager(%#x, %d) = %v, %v; want %v", tt.slots, tt.bytes, p, err, tt.want)
			}
		})
	}
}

func TestPlanRefusesForeignPipeline(t *testing.T) {
	a, b := newAdmissionComm(t), newAdmissionComm(t)
	buf := cudacollective.DoubleBuffer{Slots: [2]driver.DevicePtr{0x1000, 0x2000}, Bytes: 16}
	pb, err := cudacollective.NewPipelineManager(b, buf, 2)
	if err != nil {
		t.Fatal(err)
	}
	plan, err := cudacollective.NewPlan(a, pb)
	if !errors.Is(err, cudacollective.ErrPipelineCommunicator) || plan != nil {
		t.Fatalf("NewPlan(a, pipeline of b) = %v, %v; want ErrPipelineCommunicator", plan, err)
	}
	if plan, err := cudacollective.NewPlan(a, nil); err != nil || plan == nil {
		t.Fatalf("NewPlan(a, nil) = %v, %v; want plan", plan, err)
	}
}

func TestPlanMatchedPipelineExecutes(t *testing.T) {
	comm := newAdmissionComm(t)
	buf := cudacollective.DoubleBuffer{Slots: [2]driver.DevicePtr{0x1000, 0x1010}, Bytes: 16}
	pipeline, err := cudacollective.NewPipelineManager(comm, buf, 1)
	if err != nil {
		t.Fatal(err)
	}
	plan, err := cudacollective.NewPlan(comm, pipeline)
	if err != nil {
		t.Fatal(err)
	}
	op := cudacollective.OperationDescriptor{
		Kind:      cudacollective.KindAllReduce,
		DataType:  cudair.Float32,
		Count:     2,
		Reduction: cudacollective.ReductionSum,
	}
	for i, name := range []string{"s0", "s1", "s2"} {
		stepOp := op
		stepOp.PipelineSlot = i % 2
		base := buf.Slots[stepOp.PipelineSlot]
		if err := plan.AddCollectiveStep(name, stepOp, base, base+8); err != nil {
			t.Fatal(err)
		}
	}
	ctx := context.Background()
	for i := 0; i < 2; i++ {
		if err := plan.ExecuteSteps(ctx); err != nil {
			t.Fatalf("ExecuteSteps run %d: %v", i, err)
		}
	}
	if err := pipeline.Drain(ctx); err != nil {
		t.Fatalf("Drain: %v", err)
	}
	if err := comm.CheckAlive(); err != nil {
		t.Fatalf("CheckAlive after plan: %v", err)
	}

	// Backpressure with queue size 1 is unchanged by plan admission.
	tr, err := pipeline.SubmitStage(ctx, op)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := pipeline.SubmitStage(ctx, op); !errors.Is(err, cudacollective.ErrBackpressureExceeded) {
		t.Fatalf("second SubmitStage = %v; want ErrBackpressureExceeded", err)
	}
	pipeline.CompleteStage(tr, nil)
	tr, err = pipeline.SubmitStage(ctx, op)
	if err != nil {
		t.Fatalf("SubmitStage after completion: %v", err)
	}
	pipeline.CompleteStage(tr, nil)

	if err := plan.Close(); err != nil {
		t.Fatal(err)
	}
	if err := plan.ExecuteSteps(ctx); !errors.Is(err, cudacollective.ErrCommunicatorClosed) {
		t.Fatalf("ExecuteSteps after Close = %v; want ErrCommunicatorClosed", err)
	}
	// Close leaves the caller-owned pipeline usable.
	tr, err = pipeline.SubmitStage(ctx, op)
	if err != nil {
		t.Fatalf("SubmitStage after plan Close: %v", err)
	}
	pipeline.CompleteStage(tr, nil)
}

func TestPipelineSubmitStageOperationAdmission(t *testing.T) {
	const maxInt = int(^uint(0) >> 1)

	comm := newAdmissionComm(t)
	buf := cudacollective.DoubleBuffer{
		Slots: [2]driver.DevicePtr{0x1000, 0x2000},
		Bytes: 64,
	}
	pipeline, err := cudacollective.NewPipelineManager(comm, buf, 1)
	if err != nil {
		t.Fatal(err)
	}

	validOp := cudacollective.OperationDescriptor{
		Kind:         cudacollective.KindAllReduce,
		DataType:     cudair.Float32,
		Count:        4,
		Reduction:    cudacollective.ReductionSum,
		Topology:     cudacollective.TopologyAny,
		PipelineSlot: 0,
	}

	invalidCases := []struct {
		name    string
		mutate  func(*cudacollective.OperationDescriptor)
		wantErr error
	}{
		{"zero kind", func(op *cudacollective.OperationDescriptor) { op.Kind = 0 }, nil},
		{"negative kind", func(op *cudacollective.OperationDescriptor) { op.Kind = cudacollective.Kind(-1) }, nil},
		{"unknown kind above range", func(op *cudacollective.OperationDescriptor) { op.Kind = cudacollective.KindP2P + 1 }, nil},
		{"negative dtype", func(op *cudacollective.OperationDescriptor) { op.DataType = cudair.DataType(-1) }, nil},
		{"unknown dtype above range", func(op *cudacollective.OperationDescriptor) { op.DataType = cudair.Int8 + 1 }, nil},
		{"omitted reduction on AllReduce", func(op *cudacollective.OperationDescriptor) { op.Reduction = 0 }, nil},
		{"omitted reduction on ReduceScatter", func(op *cudacollective.OperationDescriptor) {
			op.Kind = cudacollective.KindReduceScatter
			op.Reduction = 0
		}, nil},
		{"omitted reduction on Reduce", func(op *cudacollective.OperationDescriptor) {
			op.Kind = cudacollective.KindReduce
			op.Reduction = 0
		}, nil},
		{"invalid reduction on AllReduce", func(op *cudacollective.OperationDescriptor) {
			op.Reduction = cudacollective.ReductionAvg + 1
		}, nil},
		{"invalid reduction on AllGather", func(op *cudacollective.OperationDescriptor) {
			op.Kind = cudacollective.KindAllGather
			op.Reduction = cudacollective.Reduction(-1)
		}, nil},
		{"negative topology", func(op *cudacollective.OperationDescriptor) {
			op.Topology = cudacollective.TopologyRequirement(-1)
		}, cudacollective.ErrTopologyMismatch},
		{"unknown topology above range", func(op *cudacollective.OperationDescriptor) {
			op.Topology = cudacollective.TopologyRDMA + 1
		}, cudacollective.ErrTopologyMismatch},
		{"zero count", func(op *cudacollective.OperationDescriptor) {
			op.Count = 0
		}, cudacollective.ErrCountMismatch},
		{"negative count", func(op *cudacollective.OperationDescriptor) {
			op.Count = -4
		}, cudacollective.ErrCountMismatch},
		{"invalid root", func(op *cudacollective.OperationDescriptor) {
			op.Root = 2
		}, cudacollective.ErrInvalidRank},
		{"invalid pipeline slot", func(op *cudacollective.OperationDescriptor) {
			op.PipelineSlot = 2
		}, nil},
		{"element byte overflow", func(op *cudacollective.OperationDescriptor) {
			op.Count = maxInt/4 + 1
		}, cudacollective.ErrBufferLengthMismatch},
		{"rank multiplier overflow AllGather", func(op *cudacollective.OperationDescriptor) {
			op.Kind = cudacollective.KindAllGather
			op.Reduction = 0
			op.Count = maxInt / 4
		}, cudacollective.ErrBufferLengthMismatch},
		{"rank multiplier overflow ReduceScatter", func(op *cudacollective.OperationDescriptor) {
			op.Kind = cudacollective.KindReduceScatter
			op.Count = maxInt / 4
		}, cudacollective.ErrBufferLengthMismatch},
		{"rank multiplier overflow AllToAll", func(op *cudacollective.OperationDescriptor) {
			op.Kind = cudacollective.KindAllToAll
			op.Reduction = 0
			op.Count = maxInt / 4
		}, cudacollective.ErrBufferLengthMismatch},
	}

	ctx := context.Background()
	for _, tc := range invalidCases {
		t.Run(tc.name, func(t *testing.T) {
			badOp := validOp
			tc.mutate(&badOp)

			tr, err := pipeline.SubmitStage(ctx, badOp)
			if err == nil || tr != nil {
				t.Fatalf("SubmitStage(%s) = (%v, %v); want (nil, error)", tc.name, tr, err)
			}
			if tc.wantErr != nil && !errors.Is(err, tc.wantErr) {
				t.Fatalf("SubmitStage(%s) error = %v; want %v", tc.name, err, tc.wantErr)
			}

			// Both slot ranges in the communicator must remain unreserved after refusal.
			for slot := 0; slot < 2; slot++ {
				probe, err := comm.AcquireTransferBuffer(uintptr(buf.Slots[slot]), buf.Bytes, slot)
				if err != nil {
					t.Fatalf("slot %d unexpectedly reserved after rejected SubmitStage(%s): %v", slot, tc.name, err)
				}
				comm.ReleaseTransferBuffer(probe, nil)
			}

			// Valid positive submission must still succeed in slot 0 (proving neither
			// pending queue capacity nor ping-pong activeSlot was mutated by refusal),
			// and cycle through slot 1 back to slot 0 for the next case.
			pos0, err := pipeline.SubmitStage(ctx, validOp)
			if err != nil {
				t.Fatalf("valid SubmitStage slot 0 after %s failed: %v", tc.name, err)
			}
			if pos0.Slot != 0 || pos0.BufferAddr != uintptr(buf.Slots[0]) {
				t.Fatalf("valid SubmitStage after %s bound slot=%d addr=0x%x; want slot=0 addr=0x%x",
					tc.name, pos0.Slot, pos0.BufferAddr, buf.Slots[0])
			}
			pipeline.CompleteStage(pos0, nil)

			pos1, err := pipeline.SubmitStage(ctx, validOp)
			if err != nil {
				t.Fatalf("valid SubmitStage slot 1 after %s failed: %v", tc.name, err)
			}
			if pos1.Slot != 1 || pos1.BufferAddr != uintptr(buf.Slots[1]) {
				t.Fatalf("second valid SubmitStage after %s bound slot=%d addr=0x%x; want slot=1 addr=0x%x",
					tc.name, pos1.Slot, pos1.BufferAddr, buf.Slots[1])
			}
			pipeline.CompleteStage(pos1, nil)
		})
	}
}

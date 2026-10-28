package cudacollective

import (
	"context"
	"errors"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudair"
)

func TestOperationDescriptorAdmission(t *testing.T) {
	const maxInt = int(^uint(0) >> 1)

	validBase := OperationDescriptor{
		Kind:         KindAllReduce,
		DataType:     cudair.Float32,
		Count:        4,
		Root:         0,
		Reduction:    ReductionSum,
		Topology:     TopologyAny,
		PipelineSlot: 0,
	}

	for _, dt := range []cudair.DataType{
		cudair.Float32,
		cudair.Float16,
		cudair.BFloat16,
		cudair.Int32,
		cudair.Int64,
		cudair.Uint8,
		cudair.Int8,
	} {
		op := validBase
		op.DataType = dt
		if err := op.Validate(2); err != nil {
			t.Fatalf("valid dtype %v rejected: %v", dt, err)
		}
	}

	for _, red := range []Reduction{
		ReductionSum,
		ReductionProd,
		ReductionMin,
		ReductionMax,
		ReductionAvg,
	} {
		for _, kind := range []Kind{KindAllReduce, KindReduceScatter, KindReduce} {
			op := validBase
			op.Kind = kind
			op.Reduction = red
			if err := op.Validate(2); err != nil {
				t.Fatalf("valid (%v, %v) rejected: %v", kind, red, err)
			}
		}
	}

	for _, kind := range []Kind{KindAllGather, KindBroadcast, KindAllToAll, KindScatter, KindGather, KindP2P} {
		for _, red := range []Reduction{0, ReductionSum} {
			op := validBase
			op.Kind = kind
			op.Reduction = red
			if err := op.Validate(2); err != nil {
				t.Fatalf("valid non-reduction (%v, red=%v) rejected: %v", kind, red, err)
			}
		}
	}

	for _, topo := range []TopologyRequirement{
		TopologyAny,
		TopologyNVLink,
		TopologyNVSwitchMulticast,
		TopologyRDMA,
	} {
		op := validBase
		op.Topology = topo
		if err := op.Validate(2); err != nil {
			t.Fatalf("valid topology %v rejected: %v", topo, err)
		}
	}

	tests := []struct {
		name      string
		worldSize int
		mutate    func(*OperationDescriptor)
		wantErr   error
	}{
		{
			name:      "zero world size",
			worldSize: 0,
			mutate:    func(*OperationDescriptor) {},
			wantErr:   ErrInvalidWorldSize,
		},
		{
			name:      "zero kind",
			worldSize: 2,
			mutate:    func(op *OperationDescriptor) { op.Kind = 0 },
		},
		{
			name:      "negative kind",
			worldSize: 2,
			mutate:    func(op *OperationDescriptor) { op.Kind = Kind(-1) },
		},
		{
			name:      "unknown kind above range",
			worldSize: 2,
			mutate:    func(op *OperationDescriptor) { op.Kind = KindP2P + 1 },
		},
		{
			name:      "negative dtype",
			worldSize: 2,
			mutate:    func(op *OperationDescriptor) { op.DataType = cudair.DataType(-1) },
		},
		{
			name:      "unknown dtype above range",
			worldSize: 2,
			mutate:    func(op *OperationDescriptor) { op.DataType = cudair.Int8 + 1 },
		},
		{
			name:      "unknown dtype 100",
			worldSize: 2,
			mutate:    func(op *OperationDescriptor) { op.DataType = cudair.DataType(100) },
		},
		{
			name:      "omitted reduction on AllReduce",
			worldSize: 2,
			mutate:    func(op *OperationDescriptor) { op.Kind = KindAllReduce; op.Reduction = 0 },
		},
		{
			name:      "omitted reduction on ReduceScatter",
			worldSize: 2,
			mutate:    func(op *OperationDescriptor) { op.Kind = KindReduceScatter; op.Reduction = 0 },
		},
		{
			name:      "omitted reduction on Reduce",
			worldSize: 2,
			mutate:    func(op *OperationDescriptor) { op.Kind = KindReduce; op.Reduction = 0 },
		},
		{
			name:      "negative reduction on AllReduce",
			worldSize: 2,
			mutate:    func(op *OperationDescriptor) { op.Reduction = Reduction(-1) },
		},
		{
			name:      "unknown reduction above range on AllReduce",
			worldSize: 2,
			mutate:    func(op *OperationDescriptor) { op.Reduction = ReductionAvg + 1 },
		},
		{
			name:      "invalid reduction on AllGather",
			worldSize: 2,
			mutate:    func(op *OperationDescriptor) { op.Kind = KindAllGather; op.Reduction = ReductionAvg + 1 },
		},
		{
			name:      "negative topology",
			worldSize: 2,
			mutate:    func(op *OperationDescriptor) { op.Topology = TopologyRequirement(-1) },
			wantErr:   ErrTopologyMismatch,
		},
		{
			name:      "unknown topology above range",
			worldSize: 2,
			mutate:    func(op *OperationDescriptor) { op.Topology = TopologyRDMA + 1 },
			wantErr:   ErrTopologyMismatch,
		},
		{
			name:      "count overflow float32",
			worldSize: 2,
			mutate:    func(op *OperationDescriptor) { op.DataType = cudair.Float32; op.Count = maxInt/4 + 1 },
			wantErr:   ErrBufferLengthMismatch,
		},
		{
			name:      "count overflow int64",
			worldSize: 2,
			mutate:    func(op *OperationDescriptor) { op.DataType = cudair.Int64; op.Count = maxInt/8 + 1 },
			wantErr:   ErrBufferLengthMismatch,
		},
		{
			name:      "count overflow float16",
			worldSize: 2,
			mutate:    func(op *OperationDescriptor) { op.DataType = cudair.Float16; op.Count = maxInt/2 + 1 },
			wantErr:   ErrBufferLengthMismatch,
		},
		{
			name:      "rank-multiplied overflow AllGather",
			worldSize: 4,
			mutate: func(op *OperationDescriptor) {
				op.Kind = KindAllGather
				op.DataType = cudair.Float32
				op.Count = maxInt / 8
			},
			wantErr: ErrBufferLengthMismatch,
		},
		{
			name:      "rank-multiplied overflow ReduceScatter",
			worldSize: 4,
			mutate: func(op *OperationDescriptor) {
				op.Kind = KindReduceScatter
				op.DataType = cudair.Float32
				op.Count = maxInt / 8
			},
			wantErr: ErrBufferLengthMismatch,
		},
		{
			name:      "rank-multiplied overflow AllToAll",
			worldSize: 4,
			mutate: func(op *OperationDescriptor) {
				op.Kind = KindAllToAll
				op.DataType = cudair.Float32
				op.Count = maxInt / 8
			},
			wantErr: ErrBufferLengthMismatch,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			op := validBase
			tt.mutate(&op)
			err := op.Validate(tt.worldSize)
			if err == nil {
				t.Fatalf("Validate(%d) accepted invalid descriptor %+v", tt.worldSize, op)
			}
			if tt.wantErr != nil && !errors.Is(err, tt.wantErr) {
				t.Fatalf("Validate(%d) = %v; want %v", tt.worldSize, err, tt.wantErr)
			}
		})
	}
}

func TestExpectedBytesCheckedArithmetic(t *testing.T) {
	const maxInt = int(^uint(0) >> 1)

	tests := []struct {
		name string
		op   OperationDescriptor
		want int
	}{
		{
			name: "valid float32",
			op:   OperationDescriptor{Kind: KindAllReduce, DataType: cudair.Float32, Count: 16, Reduction: ReductionSum},
			want: 64,
		},
		{
			name: "valid int64",
			op:   OperationDescriptor{Kind: KindAllReduce, DataType: cudair.Int64, Count: 5, Reduction: ReductionSum},
			want: 40,
		},
		{
			name: "valid uint8 limit",
			op:   OperationDescriptor{Kind: KindAllReduce, DataType: cudair.Uint8, Count: maxInt, Reduction: ReductionSum},
			want: maxInt,
		},
		{
			name: "unknown dtype positive",
			op:   OperationDescriptor{Kind: KindAllReduce, DataType: cudair.DataType(99), Count: 16, Reduction: ReductionSum},
			want: 0,
		},
		{
			name: "unknown dtype negative",
			op:   OperationDescriptor{Kind: KindAllReduce, DataType: cudair.DataType(-1), Count: 16, Reduction: ReductionSum},
			want: 0,
		},
		{
			name: "zero count",
			op:   OperationDescriptor{Kind: KindAllReduce, DataType: cudair.Float32, Count: 0, Reduction: ReductionSum},
			want: 0,
		},
		{
			name: "negative count",
			op:   OperationDescriptor{Kind: KindAllReduce, DataType: cudair.Float32, Count: -4, Reduction: ReductionSum},
			want: 0,
		},
		{
			name: "overflow float32",
			op:   OperationDescriptor{Kind: KindAllReduce, DataType: cudair.Float32, Count: maxInt/4 + 1, Reduction: ReductionSum},
			want: 0,
		},
		{
			name: "overflow int64",
			op:   OperationDescriptor{Kind: KindAllReduce, DataType: cudair.Int64, Count: maxInt/8 + 1, Reduction: ReductionSum},
			want: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.op.ExpectedBytes(); got != tt.want {
				t.Fatalf("ExpectedBytes(%+v) = %d; want %d", tt.op, got, tt.want)
			}
		})
	}
}

func TestPlanRefusesInvalidDescriptorBeforeReservation(t *testing.T) {
	const maxInt = int(^uint(0) >> 1)

	m, err := NewMembership([]Rank{
		{ID: 0, HostID: "hostA", DeviceID: 0},
		{ID: 1, HostID: "hostA", DeviceID: 1},
	})
	if err != nil {
		t.Fatal(err)
	}
	transport := NewDeterministicTestTransport(2)
	defer transport.Close()

	comm, err := NewCommunicator(CommunicatorConfig{
		Rank:       0,
		Membership: m,
		Generation: 1,
		Transport:  transport,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer comm.Close()

	buf := DoubleBuffer{
		Slots: [2]driver.DevicePtr{0x10000, 0x20000},
		Bytes: 64,
	}
	pipeline, err := NewPipelineManager(comm, buf, 1)
	if err != nil {
		t.Fatal(err)
	}
	plan, err := NewPlan(comm, pipeline)
	if err != nil {
		t.Fatal(err)
	}
	defer plan.Close()

	validOp := OperationDescriptor{
		Kind:         KindAllReduce,
		DataType:     cudair.Float32,
		Count:        4,
		Reduction:    ReductionSum,
		Topology:     TopologyAny,
		PipelineSlot: 0,
	}

	invalidCases := []struct {
		name   string
		mutate func(*OperationDescriptor)
	}{
		{"unknown kind", func(op *OperationDescriptor) { op.Kind = KindP2P + 1 }},
		{"unknown dtype", func(op *OperationDescriptor) { op.DataType = cudair.DataType(77) }},
		{"omitted reduction", func(op *OperationDescriptor) { op.Reduction = 0 }},
		{"invalid reduction", func(op *OperationDescriptor) { op.Reduction = ReductionAvg + 5 }},
		{"invalid topology", func(op *OperationDescriptor) { op.Topology = TopologyRDMA + 1 }},
		{"element byte overflow", func(op *OperationDescriptor) { op.Count = maxInt/4 + 1 }},
		{"rank multiplier overflow", func(op *OperationDescriptor) { op.Kind = KindAllGather; op.Count = maxInt / 4 }},
	}

	ctx := context.Background()
	for _, tc := range invalidCases {
		t.Run(tc.name, func(t *testing.T) {
			badOp := validOp
			tc.mutate(&badOp)

			if tr, err := pipeline.SubmitStage(ctx, badOp); err == nil || tr != nil {
				t.Fatalf("SubmitStage accepted %s: tr=%v err=%v", tc.name, tr, err)
			}
			if pipeline.pending != 0 || pipeline.inFlight[0] != nil || pipeline.inFlight[1] != nil || pipeline.activeSlot != 0 {
				t.Fatalf("pipeline state mutated on rejected SubmitStage: pending=%d activeSlot=%d inFlight=%v",
					pipeline.pending, pipeline.activeSlot, pipeline.inFlight)
			}
			if len(comm.inFlight) != 0 {
				t.Fatalf("communicator inFlight mutated on rejected SubmitStage: %v", comm.inFlight)
			}

			if err := plan.AddCollectiveStep(tc.name, badOp, buf.Slots[0], buf.Slots[0]+16); err == nil {
				t.Fatalf("AddCollectiveStep accepted %s: %+v", tc.name, badOp)
			}
			if len(plan.steps) != 0 {
				t.Fatalf("rejected step recorded in plan: len(steps) = %d", len(plan.steps))
			}
			if pipeline.pending != 0 || pipeline.inFlight[0] != nil || pipeline.inFlight[1] != nil || pipeline.activeSlot != 0 {
				t.Fatalf("pipeline state mutated on rejected AddCollectiveStep: pending=%d activeSlot=%d inFlight=%v",
					pipeline.pending, pipeline.activeSlot, pipeline.inFlight)
			}
			if len(comm.inFlight) != 0 {
				t.Fatalf("communicator inFlight mutated on rejected AddCollectiveStep: %v", comm.inFlight)
			}

			// Also verify ExecuteSteps refuses before reserving a pipeline slot if a step descriptor is mutated prior to execution.
			plan.steps = []CollectiveStep{{
				StepID:     0,
				Name:       tc.name,
				Descriptor: badOp,
				SendBuffer: buf.Slots[0],
				RecvBuffer: buf.Slots[0] + 16,
				Slot:       badOp.PipelineSlot,
			}}
			err := plan.ExecuteSteps(ctx)
			plan.steps = nil
			if err == nil {
				t.Fatalf("ExecuteSteps accepted %s: %+v", tc.name, badOp)
			}
			if pipeline.pending != 0 || pipeline.inFlight[0] != nil || pipeline.inFlight[1] != nil || pipeline.activeSlot != 0 {
				t.Fatalf("pipeline slot consumed on rejected ExecuteSteps: pending=%d activeSlot=%d inFlight=%v",
					pipeline.pending, pipeline.activeSlot, pipeline.inFlight)
			}
			if len(comm.inFlight) != 0 {
				t.Fatalf("communicator buffer acquired on rejected ExecuteSteps: %v", comm.inFlight)
			}
		})
	}

	// Verify the single-slot queue and slot 0 remain unconsumed and immediately usable.
	if err := plan.AddCollectiveStep("valid-after-refusals", validOp, buf.Slots[0], buf.Slots[0]+16); err != nil {
		t.Fatalf("AddCollectiveStep valid failed: %v", err)
	}
	if err := plan.ExecuteSteps(ctx); err != nil {
		t.Fatalf("ExecuteSteps valid failed: %v", err)
	}
}

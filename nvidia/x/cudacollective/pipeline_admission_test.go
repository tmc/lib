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
		Kind:     cudacollective.KindAllReduce,
		DataType: cudair.Float32,
		Count:    4,
	}
	for _, name := range []string{"s0", "s1", "s2"} {
		if err := plan.AddCollectiveStep(name, op, 0x9000, 0xa000); err != nil {
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

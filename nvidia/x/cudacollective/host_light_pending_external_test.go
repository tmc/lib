package cudacollective_test

import (
	"context"
	"errors"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nccl"
	"github.com/tmc/lib/nvidia/x/cudacollective"
	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudakv"
	"github.com/tmc/lib/nvidia/x/cudaplan"
)

// TestHostLightPendingCompletion drives the cudaplan host-light registry
// through the public ring and consumer into NCCLTransport.Execute, with every
// vendor call replaced by CPU test-only info, invoke and complete functions.
// The communicator and stream handles and the device addresses are CPU
// test-only values that nothing dereferences. Completion first fails, so the
// transport returns a real pending CompletionError. The test checks the
// registry, Plan and CompletionError owner state and the exact call counts;
// it makes no data-movement claim.
func TestHostLightPendingCompletion(t *testing.T) {
	errCompletion := errors.New("synthetic unknown completion")
	completion := errCompletion
	var infos, invokes, completes, uploads int
	tr := cudacollective.ExportTestNCCLTransport(
		[]nccl.Comm{1}, []driver.Stream{11},
		func(nccl.Comm) (int, int, error) { infos++; return 1, 0, nil },
		func(op cudacollective.OperationDescriptor, send, recv driver.DevicePtr) error {
			invokes++
			if op.Kind != cudacollective.KindAllReduce || op.DataType != cudair.Int32 || op.Count != 2 || send != 0x1000 || recv != 0x2000 {
				t.Errorf("submitted %+v send %#x recv %#x", op, send, recv)
			}
			return nil
		},
		func(driver.Stream) error { completes++; return completion },
	)
	m, err := cudacollective.NewMembership([]cudacollective.Rank{{ID: 0}})
	if err != nil {
		t.Fatal(err)
	}
	q, err := cudacollective.NewClique(m)
	if err != nil {
		t.Fatal(err)
	}
	comm, err := cudacollective.NewCommunicator(cudacollective.CommunicatorConfig{Clique: q, Generation: 1, Transport: tr})
	if err != nil {
		t.Fatal(err)
	}
	op := cudacollective.OperationDescriptor{Kind: cudacollective.KindAllReduce, DataType: cudair.Int32, Count: 2, Reduction: cudacollective.ReductionSum}
	reg, err := cudaplan.NewHostLightCollective(cudaplan.HostLightCollectiveConfig{
		Communicator: comm,
		Bindings: []cudaplan.HostLightCollectiveBinding{
			{SeqID: 1, Generation: 7, World: 1, CommunicatorGeneration: 1, Op: op, Send: 0x1000, Recv: 0x2000},
			{SeqID: 2, Generation: 7, World: 1, CommunicatorGeneration: 1, Op: op, Send: 0x3000, Recv: 0x4000},
		},
		Prepare: func(context.Context, cudaplan.HostLightCollectiveCall) error { uploads++; return nil },
	})
	if err != nil {
		t.Fatal(err)
	}
	ring, err := cudakv.NewDeviceCommandRing(2, 7)
	if err != nil {
		t.Fatal(err)
	}
	consumer, err := cudakv.NewHostCommandConsumer(ring, cudakv.HostCommandConfig{
		Mode:             "host-finite",
		Generation:       7,
		VocabSize:        1,
		MaxEntries:       2,
		MaxRegistryWords: 4,
		MaxBatch:         1,
		MaxBatchWords:    2,
	}, []cudakv.HostCommandInput{{SeqID: 1, Words: []uint32{1, 2}}, {SeqID: 2, Words: []uint32{3, 4}}})
	if err != nil {
		t.Fatal(err)
	}
	for _, seq := range []uint64{1, 2} {
		if err := ring.Publish(cudakv.Descriptor{SeqID: seq, Generation: 7}); err != nil {
			t.Fatal(err)
		}
	}
	counts := func(step string, wantUploads, wantInfos, wantInvokes, wantCompletes int) {
		t.Helper()
		if uploads != wantUploads || infos != wantInfos || invokes != wantInvokes || completes != wantCompletes {
			t.Fatalf("%s: uploads %d info %d invoke %d complete %d; want %d %d %d %d", step, uploads, infos, invokes, completes, wantUploads, wantInfos, wantInvokes, wantCompletes)
		}
	}

	// The first command uploads once and submits once; completion fails.
	batch, runErr := consumer.ConsumeBatch(context.Background(), 1, reg.Run)
	var pending *cudacollective.CompletionError
	if !errors.As(runErr, &pending) || !pending.Pending() || !errors.Is(runErr, errCompletion) || !errors.Is(runErr, cudacollective.ErrCompletionPending) {
		t.Fatalf("first command: %v, want a pending CompletionError", runErr)
	}
	if batch != (cudakv.HostCommandBatch{Consumed: 1, Words: 2}) || ring.Len() != 1 {
		t.Fatalf("first batch = %+v, ring %d", batch, ring.Len())
	}
	counts("first command", 1, 1, 1, 1)
	if tr.Available(comm.Membership()) {
		t.Fatal("transport available while completion is unproven")
	}

	// The next command is refused by the registry's pending owner, ahead of
	// the communicator liveness check, before any upload or submission.
	batch, err = consumer.ConsumeBatch(context.Background(), 1, reg.Run)
	if !errors.Is(err, cudacollective.ErrCompletionPending) || errors.Is(err, cudacollective.ErrCommunicatorGeneration) || errors.Is(err, errCompletion) {
		t.Fatalf("second command: %v, want the registry's ErrCompletionPending refusal", err)
	}
	if batch != (cudakv.HostCommandBatch{Consumed: 1, Words: 2}) || ring.Len() != 0 {
		t.Fatalf("second batch = %+v, ring %d", batch, ring.Len())
	}
	counts("second command", 1, 1, 1, 1)

	// The first Close retries completion once, fails, and keeps the owner.
	closeErr := reg.Close()
	var got *cudacollective.CompletionError
	if !errors.As(closeErr, &got) || got != pending || !errors.Is(closeErr, errCompletion) || !pending.Pending() {
		t.Fatalf("first Close: %v (owner %p, want %p, pending %v)", closeErr, got, pending, pending.Pending())
	}
	counts("first Close", 1, 1, 1, 2)
	if err := reg.Run(context.Background(), cudakv.Descriptor{SeqID: 1, Generation: 7}, []uint32{1, 2}); !errors.Is(err, cudaplan.ErrHostLightCollectiveClosed) {
		t.Fatalf("Run after failed Close: %v", err)
	}
	counts("Run after failed Close", 1, 1, 1, 2)

	// Once completion succeeds, an explicit Close retires the owner without
	// resubmitting, and a further Close is a no-op.
	completion = nil
	if err := reg.Close(); err != nil || pending.Pending() {
		t.Fatalf("second Close: %v, pending %v", err, pending.Pending())
	}
	counts("second Close", 1, 1, 1, 3)
	if err := reg.Close(); err != nil {
		t.Fatalf("third Close: %v", err)
	}
	counts("third Close", 1, 1, 1, 3)

	// The historical identities stay reachable after recovery.
	if !errors.As(runErr, &got) || got != pending || !errors.Is(runErr, errCompletion) || !errors.Is(runErr, cudacollective.ErrCompletionPending) {
		t.Fatalf("first command error after recovery: %v", runErr)
	}
	if err := comm.CheckAlive(); !errors.Is(err, cudacollective.ErrCommunicatorGeneration) || !errors.Is(err, errCompletion) {
		t.Fatalf("communicator after recovery: %v", err)
	}

	// Only now, after quiescence, are the borrowed transport and
	// communicator released.
	if !tr.Available(comm.Membership()) {
		t.Fatal("transport unavailable after quiescence")
	}
	if err := comm.Close(); err != nil {
		t.Fatalf("communicator Close: %v", err)
	}
	counts("communicator Close", 1, 1, 1, 3)
	if tr.Available(comm.Membership()) {
		t.Fatal("transport available after communicator Close")
	}
}

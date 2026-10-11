package cudacollective_test

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudacollective"
	"github.com/tmc/lib/nvidia/x/cudair"
)

// TestCollectiveRankOrder tests rank ordering and membership integrity.
// Negative controls: out of order ranks, negative ranks, duplicate ranks.
func TestCollectiveRankOrder(t *testing.T) {
	// Valid contiguous ranks
	validRanks := []cudacollective.Rank{
		{ID: 0, HostID: "hostA", DeviceID: 0},
		{ID: 1, HostID: "hostA", DeviceID: 1},
		{ID: 2, HostID: "hostB", DeviceID: 0},
	}
	m, err := cudacollective.NewMembership(validRanks)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if m.WorldSize() != 3 {
		t.Fatalf("world size = %d, want 3", m.WorldSize())
	}
	if m.SameHost() {
		t.Fatalf("expected different hosts, got SameHost=true")
	}

	// Negative control: out of order ranks
	outOfOrder := []cudacollective.Rank{
		{ID: 1, HostID: "hostA", DeviceID: 1},
		{ID: 0, HostID: "hostA", DeviceID: 0},
	}
	if _, err := cudacollective.NewMembership(outOfOrder); !errors.Is(err, cudacollective.ErrRankOrderMismatch) {
		t.Fatalf("out-of-order ranks: got %v, want ErrRankOrderMismatch", err)
	}

	// Negative control: duplicate ranks
	dupRanks := []cudacollective.Rank{
		{ID: 0, HostID: "hostA", DeviceID: 0},
		{ID: 0, HostID: "hostA", DeviceID: 1},
	}
	if _, err := cudacollective.NewMembership(dupRanks); err == nil {
		t.Fatalf("duplicate ranks: want error, got nil")
	}

	// Negative control: empty ranks
	if _, err := cudacollective.NewMembership(nil); !errors.Is(err, cudacollective.ErrInvalidWorldSize) {
		t.Fatalf("empty ranks: got %v, want ErrInvalidWorldSize", err)
	}
}

// TestCommunicatorGeneration tests generation creation, failure marking,
// and prohibition of reusing a failed communicator generation.
func TestCommunicatorGeneration(t *testing.T) {
	ranks := []cudacollective.Rank{
		{ID: 0, HostID: "hostA", DeviceID: 0},
		{ID: 1, HostID: "hostA", DeviceID: 1},
	}
	m, err := cudacollective.NewMembership(ranks)
	if err != nil {
		t.Fatal(err)
	}

	transport := cudacollective.NewDeterministicTestTransport(2)
	defer transport.Close()

	// Invalid generation 0
	_, err = cudacollective.NewCommunicator(cudacollective.CommunicatorConfig{
		Rank:       0,
		Membership: m,
		Generation: 0,
		Transport:  transport,
	})
	if !errors.Is(err, cudacollective.ErrCommunicatorGeneration) {
		t.Fatalf("generation 0: got %v, want ErrCommunicatorGeneration", err)
	}

	// Valid generation 1
	comm, err := cudacollective.NewCommunicator(cudacollective.CommunicatorConfig{
		Rank:       0,
		Membership: m,
		Generation: 1,
		Transport:  transport,
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := comm.CheckAlive(); err != nil {
		t.Fatalf("expected alive: %v", err)
	}

	// Mark failed and verify subsequent operations fail
	injectedErr := errors.New("simulated hardware communication failure")
	comm.MarkFailed(injectedErr)

	if err := comm.CheckAlive(); !errors.Is(err, cudacollective.ErrCommunicatorGeneration) {
		t.Fatalf("after failure: got %v, want ErrCommunicatorGeneration", err)
	}

	// Attempt to acquire buffer on failed communicator
	_, err = comm.AcquireTransferBuffer(0x1000, 1024, 0)
	if !errors.Is(err, cudacollective.ErrCommunicatorGeneration) {
		t.Fatalf("acquire on failed comm: got %v, want ErrCommunicatorGeneration", err)
	}
}

// TestCollectiveBufferLifetime tests buffer acquisition, in-flight tracking,
// and negative control attempting to overwrite a transfer buffer still in use.
func TestCollectiveBufferLifetime(t *testing.T) {
	ranks := []cudacollective.Rank{
		{ID: 0, HostID: "hostA", DeviceID: 0},
		{ID: 1, HostID: "hostA", DeviceID: 1},
	}
	m, _ := cudacollective.NewMembership(ranks)
	transport := cudacollective.NewDeterministicTestTransport(2)
	defer transport.Close()

	comm, err := cudacollective.NewCommunicator(cudacollective.CommunicatorConfig{
		Rank:       0,
		Membership: m,
		Generation: 1,
		Transport:  transport,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer comm.Close()

	bufAddr := uintptr(0x2000)
	bytes := 4096

	// First acquisition succeeds
	tr, err := comm.AcquireTransferBuffer(bufAddr, bytes, 0)
	if err != nil {
		t.Fatalf("first acquire failed: %v", err)
	}

	// Negative control: attempt to overwrite transfer buffer while still in use
	_, err = comm.AcquireTransferBuffer(bufAddr, bytes, 1)
	if !errors.Is(err, cudacollective.ErrBufferInUse) {
		t.Fatalf("overwrite in-flight buffer: got %v, want ErrBufferInUse", err)
	}

	// Negative control: nil buffer address
	_, err = comm.AcquireTransferBuffer(0, bytes, 0)
	if !errors.Is(err, cudacollective.ErrNilBuffer) {
		t.Fatalf("nil buffer: got %v, want ErrNilBuffer", err)
	}

	// Negative control: negative/zero bytes
	_, err = comm.AcquireTransferBuffer(bufAddr+0x100, 0, 0)
	if !errors.Is(err, cudacollective.ErrBufferLengthMismatch) {
		t.Fatalf("zero bytes: got %v, want ErrBufferLengthMismatch", err)
	}

	// Release transfer buffer
	comm.ReleaseTransferBuffer(tr, nil)

	// Now acquisition should succeed again
	tr2, err := comm.AcquireTransferBuffer(bufAddr, bytes, 1)
	if err != nil {
		t.Fatalf("re-acquire after release failed: %v", err)
	}
	comm.ReleaseTransferBuffer(tr2, nil)
}

// TestDistributedBackpressure tests double-buffered pipeline transfers and
// backpressure threshold enforcement.
func TestDistributedBackpressure(t *testing.T) {
	ranks := []cudacollective.Rank{
		{ID: 0, HostID: "hostA", DeviceID: 0},
		{ID: 1, HostID: "hostA", DeviceID: 1},
	}
	m, _ := cudacollective.NewMembership(ranks)
	transport := cudacollective.NewDeterministicTestTransport(2)
	defer transport.Close()

	comm, _ := cudacollective.NewCommunicator(cudacollective.CommunicatorConfig{
		Rank:       0,
		Membership: m,
		Generation: 1,
		Transport:  transport,
	})
	defer comm.Close()

	// Double buffer setup
	dbuf := cudacollective.DoubleBuffer{
		Slots: [2]driver.DevicePtr{0x10000, 0x20000},
		Bytes: 8192,
	}

	// Queue limit 2
	pipeline, err := cudacollective.NewPipelineManager(comm, dbuf, 2)
	if err != nil {
		t.Fatal(err)
	}

	ctx := context.Background()
	op := cudacollective.OperationDescriptor{
		Kind:     cudacollective.KindAllReduce,
		DataType: cudair.Float32,
		Count:    2048,
	}

	// Stage 1: slot 0
	tr1, err := pipeline.SubmitStage(ctx, op)
	if err != nil {
		t.Fatalf("stage 1 failed: %v", err)
	}

	// Stage 2: slot 1
	tr2, err := pipeline.SubmitStage(ctx, op)
	if err != nil {
		t.Fatalf("stage 2 failed: %v", err)
	}

	// Stage 3: Backpressure exceeded (queue full: 2 in flight)
	_, err = pipeline.SubmitStage(ctx, op)
	if !errors.Is(err, cudacollective.ErrBackpressureExceeded) {
		t.Fatalf("stage 3 (queue full): got %v, want ErrBackpressureExceeded", err)
	}

	// Complete stage 1
	pipeline.CompleteStage(tr1, nil)

	// Now stage 3 can be submitted (fills slot 0 again)
	tr3, err := pipeline.SubmitStage(ctx, op)
	if err != nil {
		t.Fatalf("stage 3 after drain failed: %v", err)
	}

	// Cleanup
	pipeline.CompleteStage(tr2, nil)
	pipeline.CompleteStage(tr3, nil)

	if err := pipeline.Drain(ctx); err != nil {
		t.Fatalf("drain failed: %v", err)
	}
}

// TestRankFailure tests the negative control where a participant is lost mid-job.
func TestRankFailure(t *testing.T) {
	ranks := []cudacollective.Rank{
		{ID: 0, HostID: "hostA", DeviceID: 0},
		{ID: 1, HostID: "hostA", DeviceID: 1},
		{ID: 2, HostID: "hostA", DeviceID: 2},
	}
	m, _ := cudacollective.NewMembership(ranks)
	transport := cudacollective.NewDeterministicTestTransport(3)
	defer transport.Close()

	comm, _ := cudacollective.NewCommunicator(cudacollective.CommunicatorConfig{
		Rank:       0,
		Membership: m,
		Generation: 1,
		Transport:  transport,
	})
	defer comm.Close()

	if err := comm.CheckAlive(); err != nil {
		t.Fatalf("expected healthy, got: %v", err)
	}

	// Simulate rank 2 dropping out / losing heartbeat
	comm.ReportLostParticipant(2)

	if err := comm.CheckAlive(); !errors.Is(err, cudacollective.ErrLostParticipant) {
		t.Fatalf("after lost participant: got %v, want ErrLostParticipant", err)
	}

	// Further operations fail closed
	_, err := comm.AcquireTransferBuffer(0x3000, 1024, 0)
	if !errors.Is(err, cudacollective.ErrCommunicatorGeneration) {
		t.Fatalf("buffer acquire on lost rank: got %v, want ErrCommunicatorGeneration", err)
	}
}

// TestUnsupportedDeviceCollective tests resolution failure when required topology
// or device features (such as NVSwitch multicast or NVLink) are missing.
func TestUnsupportedDeviceCollective(t *testing.T) {
	ranks := []cudacollective.Rank{
		{ID: 0, HostID: "hostA", DeviceID: 0},
		{ID: 1, HostID: "hostB", DeviceID: 0}, // Cross-host
	}
	m, _ := cudacollective.NewMembership(ranks)

	// Cross-host asking for NVSwitch multicast must fail with ErrUnsupportedDevice.
	// Only NVLSHM and MoonEP can satisfy it and both refuse a cross-host
	// membership, so the default chain answers the same way on every host.
	_, err := cudacollective.NewResolver().Resolve(m, cudacollective.TopologyNVSwitchMulticast)
	if !errors.Is(err, cudacollective.ErrUnsupportedDevice) {
		t.Fatalf("cross-host NVSwitch multicast: got %v, want ErrUnsupportedDevice", err)
	}

	// TopologyAny falls through the chain to the socket fallback. The chain is
	// given explicitly: the default one includes NCCL, whose Available reports
	// only whether libnccl loads, so on a host with NCCL installed this would
	// stop at nccltest and the assertion would be testing the host rather than
	// the fall-through.
	resolver := cudacollective.NewResolver(
		&cudacollective.NVLSHMTransport{},
		&cudacollective.MoonEPTransport{},
		&cudacollective.TCPSocketTransport{},
	)
	tr, err := resolver.Resolve(m, cudacollective.TopologyAny)
	if err != nil {
		t.Fatalf("topology any should resolve fallback: %v", err)
	}
	if tr.Type() != cudacollective.TransportTypeTCPSocket {
		t.Fatalf("got transport %v, want TransportTypeTCPSocket", tr.Type())
	}
}

// TestUninitializedNCCLTransportRefusesExecution checks host-side admission.
func TestUninitializedNCCLTransportRefusesExecution(t *testing.T) {
	ranks := []cudacollective.Rank{
		{ID: 0, HostID: "hostA", DeviceID: 0},
		{ID: 1, HostID: "hostA", DeviceID: 1},
	}
	m, _ := cudacollective.NewMembership(ranks)

	ncclTr := &cudacollective.NCCLTransport{}
	comm, _ := cudacollective.NewCommunicator(cudacollective.CommunicatorConfig{
		Rank:       0,
		Membership: m,
		Generation: 1,
		Transport:  ncclTr,
	})
	defer comm.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	op := cudacollective.OperationDescriptor{
		Kind:     cudacollective.KindAllReduce,
		DataType: cudair.Float32,
		Count:    1024,
	}

	// Attempting native execution in unit test environment must return ErrExecutionUnavailable
	err := ncclTr.Execute(ctx, comm, op, 0x1000, 0x2000)
	if !errors.Is(err, cudacollective.ErrExecutionUnavailable) {
		t.Fatalf("native execute without hardware: got %v, want ErrExecutionUnavailable", err)
	}
}

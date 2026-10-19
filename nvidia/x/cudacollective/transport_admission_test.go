package cudacollective_test

import (
	"context"
	"errors"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudacollective"
	"github.com/tmc/lib/nvidia/x/cudair"
)

func transportMembership(t *testing.T) cudacollective.Membership {
	t.Helper()
	m, err := cudacollective.NewMembership([]cudacollective.Rank{
		{ID: 0, HostID: "a"}, {ID: 1, HostID: "b"},
	})
	if err != nil {
		t.Fatal(err)
	}
	return m
}

func transportPlan(t *testing.T, tr cudacollective.Transport) *cudacollective.Plan {
	t.Helper()
	comm, err := cudacollective.NewCommunicator(cudacollective.CommunicatorConfig{
		Rank: 0, Membership: transportMembership(t), Generation: 1, Transport: tr,
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { comm.Close() })
	plan, err := cudacollective.NewPlan(comm, nil)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { plan.Close() })
	op := cudacollective.OperationDescriptor{
		Kind: cudacollective.KindAllReduce, DataType: cudair.Float32,
		Count: 4, Reduction: cudacollective.ReductionSum,
	}
	// These opaque addresses must never be dereferenced by a host control test.
	if err := plan.AddCollectiveStep("sum", op, 0x1000, 0x2000); err != nil {
		t.Fatal(err)
	}
	return plan
}

func TestTCPSocketTransportUnavailable(t *testing.T) {
	for _, addr := range []string{"", "127.0.0.1:1"} {
		t.Run(addr, func(t *testing.T) {
			tr := &cudacollective.TCPSocketTransport{HubAddr: addr}
			if tr.Available(transportMembership(t)) {
				t.Fatal("unimplemented TCP transport reports available")
			}
			plan := transportPlan(t, tr)
			if err := plan.ExecuteSteps(context.Background()); !errors.Is(err, cudacollective.ErrExecutionUnavailable) {
				t.Fatalf("TCP plan: got %v, want ErrExecutionUnavailable", err)
			}
		})
	}
}

func TestTCPSocketDirectExecutionRefuses(t *testing.T) {
	tr := &cudacollective.TCPSocketTransport{}
	comm, err := cudacollective.NewCommunicator(cudacollective.CommunicatorConfig{
		Rank: 0, Membership: transportMembership(t), Generation: 1, Transport: tr,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer comm.Close()
	op := cudacollective.OperationDescriptor{Kind: cudacollective.KindAllReduce, DataType: cudair.Float32, Count: 4, Reduction: cudacollective.ReductionSum}
	if err := tr.Execute(context.Background(), comm, op, 0x1000, 0x2000); !errors.Is(err, cudacollective.ErrExecutionUnavailable) {
		t.Fatalf("direct TCP execution: got %v, want ErrExecutionUnavailable", err)
	}
}

func TestDefaultResolverRefusesUnavailableTransport(t *testing.T) {
	tr, err := cudacollective.NewResolver().Resolve(transportMembership(t), cudacollective.TopologyAny)
	if tr != nil || !errors.Is(err, cudacollective.ErrExecutionUnavailable) {
		t.Fatalf("default resolver: transport %v, error %v", tr, err)
	}
}

func TestResolverSkipsSocketForControlTransport(t *testing.T) {
	control := cudacollective.NewDeterministicTestTransport(2)
	t.Cleanup(func() { control.Close() })
	resolver := cudacollective.NewResolver(&cudacollective.TCPSocketTransport{}, control)
	tr, err := resolver.Resolve(transportMembership(t), cudacollective.TopologyAny)
	if err != nil || tr != control {
		t.Fatalf("injected control transport: got %v, error %v", tr, err)
	}
	// Success establishes metadata execution only, not a numeric collective.
	if err := transportPlan(t, tr).ExecuteSteps(context.Background()); err != nil {
		t.Fatal(err)
	}
}

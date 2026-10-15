package cudacollective_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudacollective"
	"github.com/tmc/lib/nvidia/x/cudair"
)

// Example demonstrates constructing a rank membership, resolving a transport,
// creating a communicator, and configuring an operation descriptor.
func Example() {
	ranks := []cudacollective.Rank{
		{ID: 0, HostID: "node-0", DeviceID: 0},
		{ID: 1, HostID: "node-0", DeviceID: 1},
	}

	membership, err := cudacollective.NewMembership(ranks)
	if err != nil {
		fmt.Printf("membership error: %v\n", err)
		return
	}

	// This explicit control-plane transport checks metadata without moving data.
	resolver := cudacollective.NewResolver(cudacollective.NewDeterministicTestTransport(2))
	transport, err := resolver.Resolve(membership, cudacollective.TopologyAny)
	if err != nil {
		fmt.Printf("resolve error: %v\n", err)
		return
	}

	comm, err := cudacollective.NewCommunicator(cudacollective.CommunicatorConfig{
		Rank:       0,
		Membership: membership,
		Generation: 1,
		Transport:  transport,
	})
	if err != nil {
		fmt.Printf("communicator error: %v\n", err)
		return
	}
	defer comm.Close()

	op := cudacollective.OperationDescriptor{
		Kind:      cudacollective.KindAllReduce,
		DataType:  cudair.Float32,
		Count:     1024,
		Reduction: cudacollective.ReductionSum,
	}

	fmt.Printf("Rank: %d, WorldSize: %d, Transport: %s, Op: %s\n",
		comm.Rank(), comm.Membership().WorldSize(), comm.Transport().Type(), op.Kind)

	// Output:
	// Rank: 0, WorldSize: 2, Transport: controlplanetest, Op: AllReduce
}

func ExampleNewMembership() {
	membership, err := cudacollective.NewMembership([]cudacollective.Rank{
		{ID: 0, HostID: "hostA", DeviceID: 0},
		{ID: 1, HostID: "hostA", DeviceID: 1},
	})
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Printf("world size: %d, same host: %v\n", membership.WorldSize(), membership.SameHost())
	// Output:
	// world size: 2, same host: true
}

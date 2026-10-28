package cudacollective

import (
	"context"
	"errors"
	"fmt"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nccl"
)

func ExampleCompletionError() {
	// Controlled host calls only: no NCCL/CUDA library or device is used.
	failure := errors.New("completion unavailable")
	t := &NCCLTransport{Comms: []nccl.Comm{1}, Streams: []driver.Stream{0}, calls: &ncclCalls{
		info:     func(nccl.Comm) (int, int, error) { return 1, 0, nil },
		invoke:   func(ncclInvocation) error { return nil },
		complete: func(driver.Stream) error { return failure },
	}}
	m, _ := NewMembership([]Rank{{ID: 0}})
	c, _ := NewCommunicator(CommunicatorConfig{Rank: 0, Membership: m, Generation: 1, Transport: t})
	err := t.Execute(context.Background(), c, OperationDescriptor{Kind: KindAllReduce, Count: 1, Reduction: ReductionSum}, 0x1000, 0x2000)
	var pending *CompletionError
	if errors.As(err, &pending) {
		fmt.Println(errors.Is(err, failure), pending.Pending())
		failure = nil
		fmt.Println(pending.Close(), pending.Pending())
	}
	_ = c.Close()
	// Output:
	// true true
	// <nil> false
}

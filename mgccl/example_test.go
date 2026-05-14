package mgccl_test

import (
	"context"
	"fmt"

	"github.com/tmc/lib/gopccl"
	"github.com/tmc/lib/mgccl"
)

func ExampleRunAcceptPump() {
	err := mgccl.RunAcceptPump(context.Background(), exampleIsland{}, mgccl.PumpConfig{
		Grid: mgccl.Grid{
			PCCLRanks: 1,
			FSDPRanks: 1,
			Ranks:     []mgccl.Rank{{PCCL: 0, FSDP: 0}},
		},
		Revision: 1,
		Strategy: gopccl.EnforcePopular,
	})
	fmt.Println(err == nil)
	// Output: true
}

type exampleIsland struct{}

func (exampleIsland) PeersPending(context.Context) (bool, error) { return false, nil }
func (exampleIsland) UpdateTopology(context.Context) error       { return nil }
func (exampleIsland) OptimizeTopology(context.Context) error     { return nil }
func (exampleIsland) SynchronizeState(context.Context, uint64, []gopccl.StateTensor, gopccl.SyncStrategy) (gopccl.SyncInfo, error) {
	return gopccl.SyncInfo{}, nil
}

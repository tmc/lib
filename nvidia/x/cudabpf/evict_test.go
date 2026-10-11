package cudabpf_test

import (
	"fmt"
	"github.com/tmc/lib/nvidia/x/cudabpf"
	"testing"
)

type holders []int

func (h holders) Holders(block int) (int, error) {
	if block < 0 || block >= len(h) {
		return 0, fmt.Errorf("unknown block %d", block)
	}
	return h[block], nil
}

// Residency ownership stays in the consumer. This control checks that owner
// errors are not interpreted as an unreferenced block or a migration request.
func TestHolderPolicyUsesOwnerState(t *testing.T) {
	h := holders{0, 2, 1}
	p := &cudabpf.HolderPolicy{State: h, Block: func(s cudabpf.PMMSnapshot) (int, bool) { return int(s.RootID), true }}
	for block, count := range h {
		got, ok, err := p.Decide(cudabpf.PMMSnapshot{RootID: uint64(block)}, cudabpf.NewBudget(8))
		if err != nil || !ok {
			t.Fatalf("block %d: %v %v", block, ok, err)
		}
		want := cudabpf.DestinationUsed
		if count == 0 {
			want = cudabpf.DestinationUnused
		}
		if got.Destination != want {
			t.Errorf("block %d: got %v want %v", block, got.Destination, want)
		}
	}
	h[1] = 0
	got, _, err := p.Decide(cudabpf.PMMSnapshot{RootID: 1}, cudabpf.NewBudget(8))
	if err != nil || got.Destination != cudabpf.DestinationUnused {
		t.Fatalf("owner release: %v %v", got, err)
	}
	if _, _, err := p.Decide(cudabpf.PMMSnapshot{RootID: 4}, cudabpf.NewBudget(8)); err == nil {
		t.Fatal("unknown block admitted")
	}
}

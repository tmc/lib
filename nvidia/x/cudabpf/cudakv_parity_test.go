package cudabpf_test

import (
	"errors"
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudabpf"
	"github.com/tmc/lib/nvidia/x/cudakv"
)

// These are real allocator bookkeeping transitions, not device-memory tier
// migrations. Policy decisions remain advisory and do not actuate a driver.
func TestBlockEvictionDecisionParityWithCudakv(t *testing.T) {
	allocator := cudakv.NewBlockAllocator(4, 3)
	block := 0
	policy := &cudabpf.HolderPolicy{
		State: allocator,
		Block: func(s cudabpf.PMMSnapshot) (int, bool) {
			return block, s.RootID == 42
		},
	}
	for _, step := range []struct {
		name    string
		change  func() error
		holders int
		free    int
	}{
		{"initially free", func() error { return nil }, 0, 3},
		{"allocated", func() error {
			if err := allocator.AllocSequence(1, nil, 4); err != nil {
				return err
			}
			table := allocator.Table(1)
			if len(table) != 1 {
				return fmt.Errorf("allocated table = %v; want one physical block", table)
			}
			block = table[0]
			return nil
		}, 1, 2},
		{"shared prefix", func() error { return allocator.AllocSequence(2, allocator.Table(1), 4) }, 2, 2},
		{"retained prefix", func() error { return allocator.Retain([]int{block}) }, 3, 2},
		{"first sequence freed", func() error { allocator.FreeSequence(1); return nil }, 2, 2},
		{"last sequence freed with retainer", func() error { allocator.FreeSequence(2); return nil }, 1, 2},
		{"retainer released", func() error { return allocator.Release([]int{block}) }, 0, 3},
	} {
		if !t.Run(step.name, func(t *testing.T) {
			if err := step.change(); err != nil {
				t.Fatal(err)
			}
			holders, err := allocator.Holders(block)
			if err != nil || holders != step.holders || allocator.NumFreeBlocks() != step.free {
				t.Fatalf("allocator state = holders%d, free%d, %v; want holders%d, free%d", holders, allocator.NumFreeBlocks(), err, step.holders, step.free)
			}
			want := cudabpf.Reorder{Destination: cudabpf.DestinationUsed, Position: cudabpf.PositionTail}
			if step.holders == 0 {
				want = cudabpf.Reorder{Destination: cudabpf.DestinationUnused, Position: cudabpf.PositionHead}
			}
			before := readKVParityState(t, allocator)
			decision, ok, err := policy.Decide(cudabpf.PMMSnapshot{RootID: 42}, cudabpf.NewBudget(1))
			if err != nil || !ok || decision != want {
				t.Fatalf("policy decision = %v, opinion%t, %v; want %v from real allocator", decision, ok, err, want)
			}
			for _, call := range []struct {
				name string
				fn   func(*cudabpf.PMMDecision, *cudabpf.Budget) (cudabpf.Reorder, bool, error)
			}{{"access", policy.BlockAccess}, {"activate", policy.BlockActivate}} {
				decision, ok, err := call.fn(&cudabpf.PMMDecision{Observed: cudabpf.PMMSnapshot{RootID: 42}}, cudabpf.NewBudget(1))
				if err != nil || !ok || decision != want {
					t.Fatalf("%s decision = %v, opinion%t, %v; want %v", call.name, decision, ok, err, want)
				}
			}
			if after := readKVParityState(t, allocator); !reflect.DeepEqual(before, after) {
				t.Fatalf("policy changed allocator authority: before%+v, after%+v", before, after)
			}
		}) {
			return
		}
	}
}

type kvParityState struct {
	Free       int
	Holders    []int
	First      []int
	Second     []int
	Generation [2]uint64
}

func readKVParityState(t *testing.T, allocator *cudakv.BlockAllocator) kvParityState {
	t.Helper()
	state := kvParityState{Free: allocator.NumFreeBlocks(), First: allocator.Table(1), Second: allocator.Table(2)}
	for block := 0; block < allocator.TotalBlocks(); block++ {
		n, err := allocator.Holders(block)
		if err != nil {
			t.Fatal(err)
		}
		state.Holders = append(state.Holders, n)
	}
	state.Generation[0], _ = allocator.AllocationGeneration(1)
	state.Generation[1], _ = allocator.AllocationGeneration(2)
	return state
}

func TestHolderPolicyRealCudakvOwnerErrors(t *testing.T) {
	allocator := cudakv.NewBlockAllocator(4, 2)
	policy := &cudabpf.HolderPolicy{
		State: allocator,
		Block: func(s cudabpf.PMMSnapshot) (int, bool) {
			return allocator.TotalBlocks(), s.RootID == 42
		},
	}
	before := readKVParityState(t, allocator)
	_, ownerErr := allocator.Holders(allocator.TotalBlocks())
	if ownerErr == nil {
		t.Fatal("fixture needs a real allocator owner error")
	}
	decision, ok, err := policy.Decide(cudabpf.PMMSnapshot{RootID: 42}, cudabpf.NewBudget(1))
	if err == nil || !strings.Contains(err.Error(), ownerErr.Error()) || ok || decision != (cudabpf.Reorder{}) {
		t.Fatalf("owner error decision = %v, opinion%t, %v; want refusal containing %q", decision, ok, err, ownerErr)
	}
	// An unmapped driver chunk has no opinion, rather than querying a guessed block.
	decision, ok, err = policy.Decide(cudabpf.PMMSnapshot{RootID: 99}, cudabpf.NewBudget(1))
	if err != nil || ok || decision != (cudabpf.Reorder{}) {
		t.Fatalf("unmapped decision = %v, opinion%t, %v; want no opinion", decision, ok, err)
	}
	decision, ok, err = policy.Decide(cudabpf.PMMSnapshot{RootID: 42}, cudabpf.NewBudget(0))
	if !errors.Is(err, cudabpf.ErrBudgetExceeded) || ok || decision != (cudabpf.Reorder{}) {
		t.Fatalf("exhausted budget = %v, opinion%t, %v; want budget refusal", decision, ok, err)
	}
	if after := readKVParityState(t, allocator); !reflect.DeepEqual(before, after) {
		t.Fatalf("refusal changed allocator authority: before%+v, after%+v", before, after)
	}
}

package mgccl

import (
	"context"
	"errors"
	"reflect"
	"testing"
	"time"

	"github.com/tmc/lib/gopccl"
)

func TestGridValidateRejectsIncompleteMatrix(t *testing.T) {
	err := (Grid{
		PCCLRanks: 2,
		FSDPRanks: 2,
		Ranks:     []Rank{{PCCL: 0, FSDP: 0}, {PCCL: 0, FSDP: 1}, {PCCL: 1, FSDP: 0}},
	}).Validate()
	if !errors.Is(err, ErrIncompleteGrid) {
		t.Fatalf("Validate err = %v, want ErrIncompleteGrid", err)
	}
}

func TestGridValidateRejectsDuplicateCell(t *testing.T) {
	err := (Grid{
		PCCLRanks: 1,
		FSDPRanks: 2,
		Ranks:     []Rank{{PCCL: 0, FSDP: 0}, {PCCL: 0, FSDP: 0}},
	}).Validate()
	if !errors.Is(err, ErrInvalidGrid) {
		t.Fatalf("Validate err = %v, want ErrInvalidGrid", err)
	}
}

func TestRunAcceptPumpSequence(t *testing.T) {
	island := &fakeIsland{pending: []bool{true, true, false}}
	var ready bool
	err := RunAcceptPump(context.Background(), island, PumpConfig{
		Grid:     fullGrid(),
		Revision: 3,
		Strategy: gopccl.EnforcePopular,
		Ready: func(context.Context) error {
			ready = true
			return nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	want := []string{
		"peers",
		"update",
		"optimize",
		"peers",
		"update",
		"optimize",
		"peers",
		"sync",
	}
	if !reflect.DeepEqual(island.calls, want) {
		t.Fatalf("calls = %v, want %v", island.calls, want)
	}
	if !ready {
		t.Fatal("ready callback was not invoked")
	}
}

func TestRunAcceptPumpDoesNotCallReadyForIncompleteGrid(t *testing.T) {
	island := &fakeIsland{}
	called := false
	err := RunAcceptPump(context.Background(), island, PumpConfig{
		Grid: Grid{
			PCCLRanks: 2,
			FSDPRanks: 1,
			Ranks:     []Rank{{PCCL: 0, FSDP: 0}},
		},
		Revision: 1,
		Ready: func(context.Context) error {
			called = true
			return nil
		},
	})
	if !errors.Is(err, ErrIncompleteGrid) {
		t.Fatalf("RunAcceptPump err = %v, want ErrIncompleteGrid", err)
	}
	if called {
		t.Fatal("ready callback invoked for incomplete grid")
	}
	if len(island.calls) != 0 {
		t.Fatalf("island calls = %v, want none", island.calls)
	}
}

func TestPumpRejectsStaleRevision(t *testing.T) {
	var p Pump
	island := &fakeIsland{pending: []bool{false, false}}
	cfg := PumpConfig{Grid: fullGrid(), Revision: 2}
	if err := p.Run(context.Background(), island, cfg); err != nil {
		t.Fatal(err)
	}
	err := p.Run(context.Background(), island, cfg)
	if !errors.Is(err, ErrStaleRevision) {
		t.Fatalf("Run err = %v, want ErrStaleRevision", err)
	}
}

func TestRunAcceptPumpHonorsContextCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	island := &fakeIsland{pending: []bool{true}}
	island.afterPeers = cancel
	err := RunAcceptPump(ctx, island, PumpConfig{
		Grid:          fullGrid(),
		Revision:      1,
		PollInterval:  time.Hour,
		MaxIterations: 2,
	})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("RunAcceptPump err = %v, want context.Canceled", err)
	}
}

func TestRunAcceptPumpIterationLimit(t *testing.T) {
	err := RunAcceptPump(context.Background(), &fakeIsland{pending: []bool{true, true}}, PumpConfig{
		Grid:          fullGrid(),
		Revision:      1,
		MaxIterations: 1,
	})
	if !errors.Is(err, ErrIterationLimit) {
		t.Fatalf("RunAcceptPump err = %v, want ErrIterationLimit", err)
	}
}

func fullGrid() Grid {
	return Grid{
		PCCLRanks: 2,
		FSDPRanks: 2,
		Ranks: []Rank{
			{PCCL: 0, FSDP: 0},
			{PCCL: 0, FSDP: 1},
			{PCCL: 1, FSDP: 0},
			{PCCL: 1, FSDP: 1},
		},
	}
}

type fakeIsland struct {
	pending    []bool
	calls      []string
	afterPeers func()
}

func (f *fakeIsland) PeersPending(context.Context) (bool, error) {
	f.calls = append(f.calls, "peers")
	pending := false
	if len(f.pending) > 0 {
		pending = f.pending[0]
		f.pending = f.pending[1:]
	}
	if f.afterPeers != nil {
		f.afterPeers()
		f.afterPeers = nil
	}
	return pending, nil
}

func (f *fakeIsland) UpdateTopology(context.Context) error {
	f.calls = append(f.calls, "update")
	return nil
}

func (f *fakeIsland) OptimizeTopology(context.Context) error {
	f.calls = append(f.calls, "optimize")
	return nil
}

func (f *fakeIsland) SynchronizeState(context.Context, uint64, []gopccl.StateTensor, gopccl.SyncStrategy) (gopccl.SyncInfo, error) {
	f.calls = append(f.calls, "sync")
	return gopccl.SyncInfo{}, nil
}

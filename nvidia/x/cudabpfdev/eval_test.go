package cudabpfdev

import (
	"context"
	"errors"
	"sync"
	"testing"
)

func TestWarpLeaderAndPartialMask(t *testing.T) {
	p := program(Instruction{Op: Lane, Dst: 1}, Instruction{Op: Leader, Dst: 2, A: 1}, Instruction{Op: Const, Dst: 3, Imm: 1}, Instruction{Op: MapAdd, A: 2, B: 3}, Instruction{Op: Repeat, Target: 3, Imm: 2}, Instruction{Op: Exit, A: 1})
	for _, mask := range []uint32{0, 1, 1<<3 | 1<<7, ^uint32(0)} {
		r, err := Evaluate(context.Background(), p, budget(), mask)
		if err != nil {
			t.Fatal(err)
		}
		if mask == 0 {
			if r.Steps != 0 || r.Misses != 0 {
				t.Fatal("inactive warp executed")
			}
			continue
		}
		key := 0
		if mask == 1<<3|1<<7 {
			key = 3
		}
		if r.Map[key] != 2 {
			t.Fatalf("mask=%x map=%v", mask, r.Map)
		}
		var sum uint32
		for _, v := range r.Map {
			sum += v
		}
		if sum != 2 {
			t.Fatalf("side effect was per-lane: %v", r.Map)
		}
		for lane, v := range r.Values {
			if mask&(uint32(1)<<lane) != 0 && v != uint32(lane) {
				t.Fatalf("lane=%d value=%d", lane, v)
			}
		}
	}
}

func TestBallotAndMapBounds(t *testing.T) {
	p := program(Instruction{Op: Lane, Dst: 1}, Instruction{Op: Ballot, Dst: 2, A: 1}, Instruction{Op: MapAdd, A: 2}, Instruction{Op: Exit, A: 2})
	r, err := Evaluate(context.Background(), p, budget(), 0x89)
	if err != nil {
		t.Fatal(err)
	}
	if r.Values[0] != 0x88 || r.Values[3] != 0x88 || r.Values[7] != 0x88 || r.Misses != 1 {
		t.Fatalf("result=%+v", r)
	}
	for _, v := range r.Map {
		if v != 0 {
			t.Fatal("out-of-range key wrote map")
		}
	}
}

func TestInvocationSnapshotsAreIndependent(t *testing.T) {
	p := program(Instruction{Op: Const, Dst: 1, Imm: 1}, Instruction{Op: MapAdd, B: 1}, Instruction{Op: Exit})
	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			r, err := Evaluate(context.Background(), p, budget(), ^uint32(0))
			if err != nil {
				t.Error(err)
				return
			}
			if r.Map[0] != 1 {
				t.Errorf("map=%v", r.Map)
			}
			r.Map[0] = 99
		}()
	}
	wg.Wait()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	r, err := Evaluate(ctx, p, budget(), 1)
	if !errors.Is(err, context.Canceled) || r.Map != nil {
		t.Fatalf("cancellation published snapshot: %+v %v", r, err)
	}
}

func TestNestedLoopsMatchVerifiedBound(t *testing.T) {
	p := program(Instruction{Op: Const, Dst: 1, Imm: 1}, Instruction{Op: MapAdd, B: 1}, Instruction{Op: Repeat, Target: 1, Imm: 3}, Instruction{Op: Repeat, Target: 1, Imm: 2}, Instruction{Op: Exit})
	a, err := Verify(p, budget())
	if err != nil {
		t.Fatal(err)
	}
	r, err := Evaluate(context.Background(), p, budget(), 1)
	if err != nil {
		t.Fatal(err)
	}
	if r.Map[0] != 6 || r.Steps != a.MaxSteps {
		t.Fatalf("result=%+v analysis=%+v", r, a)
	}
}

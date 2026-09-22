package cudasearch

import (
	"context"
	"errors"
	"fmt"
	"math"
	"reflect"
	"testing"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudatrial"
	"github.com/tmc/lib/nvidia/x/egraph"
)

func TestPrefillCompilerBinding(t *testing.T) {
	a, b := make([]uint16, 256*16), make([]uint16, 16*16)
	c, err := NewPrefillCompiler(&cuda.Device{}, "sm_121a", 16, 16, a, b, func() bool { return false })
	if err != nil {
		t.Fatal(err)
	}
	key := c.Bindings().Key()
	a[0] = 0x3c00
	b[0] = 0x4000
	if c.a[0] != 0 || c.b[0] != 0 || c.Bindings().Key() != key {
		t.Fatal("constructor retained mutable inputs")
	}
	for _, op := range []string{"prefill_scalar", "prefill_wmma"} {
		expr := &egraph.Expr{Op: op}
		if err := c.admit(expr, c.Bindings()); err != nil {
			t.Fatal(err)
		}
		if _, err := c.Compile(context.Background(), expr, c.Bindings()); !errors.Is(err, cudatrial.ErrHardwareLockRequired) {
			t.Fatalf("lease refusal=%v", err)
		}
	}
	for _, change := range []func(*WorkloadBindings){
		func(b *WorkloadBindings) { b.InputIdentities["a"] = "different" },
		func(b *WorkloadBindings) { b.WeightIdentities["b"] = "different" },
		func(b *WorkloadBindings) { b.InputNames[0] = "missing" },
		func(b *WorkloadBindings) { b.Layout = "column_major" },
	} {
		binding := c.Bindings()
		change(&binding)
		if c.admit(&egraph.Expr{Op: "prefill_wmma"}, binding) == nil {
			t.Fatal("accepted different workload")
		}
		if c.Bindings().Key() != key {
			t.Fatal("bindings alias compiler state")
		}
	}
	for _, expr := range []*egraph.Expr{nil, {Op: "matmul"}, {Op: "prefill_wmma", Data: 256}, {Op: "prefill_wmma", Children: []*egraph.Expr{{Op: "x"}}}} {
		if c.admit(expr, c.Bindings()) == nil {
			t.Fatal("accepted unsupported expression")
		}
	}
}

func TestPrefillExecutionFailures(t *testing.T) {
	stages := []string{"poison", "record0", "launch", "record1", "wait", "elapsed"}
	for failureAt := 0; failureAt <= len(stages); failureAt++ {
		t.Run(fmt.Sprint(failureAt), func(t *testing.T) {
			x := &prefillExecutable{identity: "fixture", checkLease: func() bool { return true }}
			failure := errors.New("native operation failed")
			var calls []string
			call := func(name string) error {
				calls = append(calls, name)
				if len(calls)-1 == failureAt {
					return failure
				}
				return nil
			}
			ops := prefillRunOps{
				poison: func() error { return call("poison") }, record: func(i int) error { return call(fmt.Sprintf("record%d", i)) },
				launch: func() error { return call("launch") }, wait: func(context.Context) error { return call("wait") },
				elapsed: func() (float32, error) { return 0.5, call("elapsed") },
			}
			r, err := x.runWith(context.Background(), true, ops)
			if failureAt < len(stages) {
				if !errors.Is(err, failure) || r.Complete || !x.closed {
					t.Fatalf("receipt=%+v err=%v closed=%v", r, err, x.closed)
				}
				if !reflect.DeepEqual(calls, stages[:failureAt+1]) {
					t.Fatalf("calls=%v", calls)
				}
				if _, err := x.runWith(context.Background(), true, ops); err == nil {
					t.Fatal("accepted reuse after failed operation")
				}
				if len(calls) != failureAt+1 {
					t.Fatal("submitted after failure")
				}
			} else if err != nil || !r.Complete || r.SummedKernel != 500000 || x.closed {
				t.Fatalf("receipt=%+v err=%v", r, err)
			}
		})
	}
}

func TestPrefillVerificationDoesNotTime(t *testing.T) {
	x := &prefillExecutable{checkLease: func() bool { return true }}
	var calls []string
	ops := prefillRunOps{
		poison:  func() error { calls = append(calls, "poison"); return nil },
		launch:  func() error { calls = append(calls, "launch"); return nil },
		wait:    func(context.Context) error { calls = append(calls, "wait"); return nil },
		record:  func(int) error { t.Fatal("verification recorded timing event"); return nil },
		elapsed: func() (float32, error) { t.Fatal("verification read timing"); return 0, nil },
	}
	r, err := x.runWith(context.Background(), false, ops)
	if err != nil || !r.Complete || r.HasTiming(cudatrial.TimingSummedKernel) || !reflect.DeepEqual(calls, []string{"poison", "launch", "wait"}) {
		t.Fatalf("receipt=%+v calls=%v err=%v", r, calls, err)
	}
	for _, ms := range []float32{0, -1, float32(math.NaN()), float32(math.Inf(1))} {
		x.closed = false
		ops.record = func(int) error { return nil }
		ops.elapsed = func() (float32, error) { return ms, nil }
		if _, err := x.runWith(context.Background(), true, ops); !errors.Is(err, cudatrial.ErrTimingUnavailable) {
			t.Fatalf("duration %v: %v", ms, err)
		}
	}
}

func TestPrefillCompilerAdmission(t *testing.T) {
	a, b := make([]uint16, 256*16), make([]uint16, 16*16)
	for _, bad := range []uint16{0x7c00, 0xfc00, 0x7e00} {
		a[0] = bad
		if _, err := NewPrefillCompiler(&cuda.Device{}, "sm_121a", 16, 16, a, b, func() bool { return false }); err == nil {
			t.Fatal("accepted nonfinite half")
		}
	}
	a[0] = 0
	if _, err := NewPrefillCompiler(&cuda.Device{}, "sm_121a", 16, 16, a[:len(a)-1], b, func() bool { return false }); err == nil {
		t.Fatal("accepted truncated input")
	}
	if _, err := NewPrefillCompiler(&cuda.Device{}, "sm_90", 16, 16, a, b, func() bool { return false }); err == nil {
		t.Fatal("accepted unqualified target")
	}
}

func TestWorkloadIdentitySections(t *testing.T) {
	a := WorkloadBindings{InputNames: []string{"a", "b", "c"}}
	b := WorkloadBindings{InputNames: []string{"a"}, InputIdentities: map[string]string{"b": "c"}}
	if a.Key() == b.Key() {
		t.Fatal("identity omitted field boundaries")
	}
	a = WorkloadBindings{InputIdentities: map[string]string{"a\x00b": "c"}}
	b = WorkloadBindings{InputIdentities: map[string]string{"a": "b\x00c"}}
	if a.Key() == b.Key() {
		t.Fatal("identity omitted string lengths")
	}
}

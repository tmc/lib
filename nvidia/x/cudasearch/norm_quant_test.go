package cudasearch

import (
	"context"
	"errors"
	"fmt"
	"math"
	"slices"
	"testing"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudatrial"
	"github.com/tmc/lib/nvidia/x/egraph"
)

func normSearchWorkload() NormQuantWorkload {
	w := NormQuantWorkload{Rows: 2, Cols: 64, Input: make([]float32, 128), Weight: make([]float32, 64), Epsilon: 3}
	for i := range w.Input {
		w.Input[i] = 1
	}
	for i := range w.Weight {
		w.Weight[i] = 1
	}
	return w
}

func TestNormQuantWorkload(t *testing.T) {
	w := normSearchWorkload()
	c, err := NewNormQuantCompiler(&cuda.Device{}, "sm_121a", w, func() bool { return false })
	if err != nil {
		t.Fatal(err)
	}
	w.Input[0] = 99
	if c.w.Input[0] != 1 {
		t.Fatal("retained caller input")
	}
	b := c.Bindings()
	key := b.Key()
	b.InputNames[0] = "changed"
	b.InputIdentities["input"] = "changed"
	if c.Bindings().Key() != key {
		t.Fatal("bindings alias compiler")
	}
	ref := c.References()["q8_bytes"]
	for block := 0; block < 4; block++ {
		want := []float32{8, 28, 0, 76} // fp16 d=0x1c08, s=0x4c00.
		if !slices.Equal(ref[block*36:block*36+4], want) {
			t.Fatalf("header=%v", ref[block*36:block*36+4])
		}
		for _, q := range ref[block*36+4 : (block+1)*36] {
			if q != 127 {
				t.Fatalf("quant=%g", q)
			}
		}
	}
	ref[0] = 99
	if c.References()["q8_bytes"][0] != 8 {
		t.Fatal("references alias compiler")
	}
	for _, op := range []string{"norm_quant_fused", "norm_quant_unfused"} {
		expr := &egraph.Expr{Op: op}
		if _, err := c.admit(expr, c.Bindings()); err != nil {
			t.Fatal(err)
		}
		if x, err := c.Compile(context.Background(), expr, c.Bindings()); x != nil || !errors.Is(err, cudatrial.ErrHardwareLockRequired) {
			t.Fatalf("lease: %v %v", x, err)
		}
	}
	w = normSearchWorkload()
	w.MaterializeNorm = true
	keep, err := NewNormQuantCompiler(&cuda.Device{}, "sm_121a", w, func() bool { return false })
	if err != nil {
		t.Fatal(err)
	}
	if _, err := keep.admit(&egraph.Expr{Op: "norm_quant_fused"}, keep.Bindings()); err == nil {
		t.Fatal("dropped secondary consumer")
	}
	for _, v := range keep.References()["norm"] {
		if v != 0.5 {
			t.Fatalf("secondary=%g", v)
		}
	}
	if keep.Bindings().Key() == key {
		t.Fatal("output consumers absent from identity")
	}
}

func TestNormQuantIdentityAndAdmission(t *testing.T) {
	base, err := NewNormQuantCompiler(&cuda.Device{}, "sm_121a", normSearchWorkload(), func() bool { return false })
	if err != nil {
		t.Fatal(err)
	}
	for _, edit := range []func(*NormQuantWorkload){
		func(w *NormQuantWorkload) { w.Epsilon = 1 }, func(w *NormQuantWorkload) { w.Weighted = true },
		func(w *NormQuantWorkload) { w.Input[0] = 2 }, func(w *NormQuantWorkload) { w.MaterializeNorm = true },
	} {
		w := normSearchWorkload()
		edit(&w)
		c, err := NewNormQuantCompiler(&cuda.Device{}, "sm_121a", w, func() bool { return false })
		if err != nil {
			t.Fatal(err)
		}
		if c.Bindings().Key() == base.Bindings().Key() {
			t.Fatal("identity collision")
		}
	}
	for _, edit := range []func(*NormQuantWorkload){
		func(w *NormQuantWorkload) { w.Epsilon = -1 }, func(w *NormQuantWorkload) { w.Input[0] = float32(math.NaN()) },
		func(w *NormQuantWorkload) { w.Weighted = true; w.Weight = nil }, func(w *NormQuantWorkload) { w.Rows = 0 },
		func(w *NormQuantWorkload) { w.Epsilon = 0; clear(w.Input) },
	} {
		w := normSearchWorkload()
		edit(&w)
		if _, err := NewNormQuantCompiler(&cuda.Device{}, "sm_121a", w, func() bool { return false }); err == nil {
			t.Fatal("accepted invalid workload")
		}
	}
	for _, expr := range []*egraph.Expr{nil, {Op: "norm_quant_unknown"}, {Op: "norm_quant_fused", Data: 1}, {Op: "norm_quant_fused", Children: []*egraph.Expr{{Op: "x"}}}} {
		if _, err := base.admit(expr, base.Bindings()); err == nil {
			t.Fatal("accepted malformed expression")
		}
	}
	b := base.Bindings()
	b.NumericalPolicy = NumericalPolicyFP32
	if _, err := base.admit(&egraph.Expr{Op: "norm_quant_fused"}, b); err == nil {
		t.Fatal("accepted changed policy")
	}
}

func TestNormQuantExecutionOrder(t *testing.T) {
	order := []string{"poison", "record0", "launch0", "launch1", "record1", "wait", "elapsed"}
	for fail := -1; fail < len(order); fail++ {
		t.Run(fmt.Sprint(fail), func(t *testing.T) {
			x := &sequenceExecutable{checkLease: func() bool { return true }, stages: make([]sequenceStage, 2)}
			var got []string
			failure := errors.New("injected")
			step := func(name string) error {
				got = append(got, name)
				if len(got)-1 == fail {
					return failure
				}
				return nil
			}
			ops := sequenceRunOps{poison: func() error { return step("poison") }, record: func(i int) error { return step(fmt.Sprint("record", i)) }, launch: func(i int) error { return step(fmt.Sprint("launch", i)) }, wait: func(context.Context) error { return step("wait") }, elapsed: func() (float32, error) { return 0.5, step("elapsed") }}
			r, err := x.runWith(context.Background(), true, ops)
			if fail < 0 {
				if err != nil || !r.Complete || r.TimingKind != cudatrial.TimingCUDAEvent || r.Timings[cudatrial.TimingCUDAEvent] != 500000 || r.HasTiming(cudatrial.TimingGPUBusy) || !slices.Equal(got, order) {
					t.Fatalf("success: %+v %v %v", r, err, got)
				}
			} else {
				if !errors.Is(err, failure) || r.Complete || !x.closed || !slices.Equal(got, order[:fail+1]) {
					t.Fatalf("failure: %+v %v %v", r, err, got)
				}
				if _, err := x.runWith(context.Background(), true, ops); err == nil || len(got) != fail+1 {
					t.Fatal("reused failed executable")
				}
			}
		})
	}
	x := &sequenceExecutable{checkLease: func() bool { return true }, stages: make([]sequenceStage, 2)}
	var got []int
	r, err := x.runWith(context.Background(), false, sequenceRunOps{poison: func() error { return nil }, launch: func(i int) error { got = append(got, i); return nil }, wait: func(context.Context) error { return nil }})
	if err != nil || !r.Complete || !slices.Equal(got, []int{0, 1}) {
		t.Fatalf("untimed: %+v %v %v", r, err, got)
	}
}

func TestNormQuantExecutionAdmission(t *testing.T) {
	for _, ms := range []float32{0, -1, float32(math.NaN()), float32(math.Inf(1))} {
		x := &sequenceExecutable{checkLease: func() bool { return true }, stages: make([]sequenceStage, 1)}
		ops := sequenceRunOps{poison: func() error { return nil }, record: func(int) error { return nil }, launch: func(int) error { return nil }, wait: func(context.Context) error { return nil }, elapsed: func() (float32, error) { return ms, nil }}
		r, err := x.runWith(context.Background(), true, ops)
		if !errors.Is(err, cudatrial.ErrTimingUnavailable) || r.Complete || !x.closed {
			t.Fatalf("ms=%g receipt=%+v err=%v", ms, r, err)
		}
	}
	x := &sequenceExecutable{checkLease: func() bool { return false }}
	if _, err := x.runWith(context.Background(), true, sequenceRunOps{}); !errors.Is(err, cudatrial.ErrHardwareLockRequired) {
		t.Fatalf("missing lease: %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := x.runWith(ctx, true, sequenceRunOps{}); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled: %v", err)
	}
	for i := 0; i < 2; i++ {
		if err := x.Close(); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := x.runWith(context.Background(), false, sequenceRunOps{}); err == nil {
		t.Fatal("accepted closed executable")
	}
}

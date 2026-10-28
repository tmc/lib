package cudasearch_test

import (
	"context"
	"errors"
	"fmt"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudasearch"
	"github.com/tmc/lib/nvidia/x/cudashape"
	"github.com/tmc/lib/nvidia/x/cudatrial"
	"github.com/tmc/lib/nvidia/x/egraph"
)

func boundWorkload(t *testing.T) cudasearch.WorkloadBindings {
	t.Helper()
	shape, err := cudashape.New(2, 4)
	if err != nil {
		t.Fatal(err)
	}
	return cudasearch.WorkloadBindings{
		Name: "identity", Shape: shape, DType: cudair.Float32,
		Layout: "row_major", NumericalPolicy: cudasearch.NumericalPolicyFP32,
		Objective:        cudasearch.ObjectiveMinLatency,
		InputNames:       []string{"x", "w"},
		InputIdentities:  map[string]string{"x": "input-v1"},
		WeightIdentities: map[string]string{"w": "weight-v1"},
	}
}

func TestWorkloadIdentityCompleteBindings(t *testing.T) {
	base := boundWorkload(t)
	want := base.WorkloadIdentity()
	if want.BindingsKey != base.Key() || want.ProgramID != base.Name || want.NumericalMode != string(base.NumericalPolicy) {
		t.Fatalf("identity lost bindings: %+v", want)
	}
	for _, test := range []struct {
		name   string
		change func(*cudasearch.WorkloadBindings)
	}{
		{"dtype", func(b *cudasearch.WorkloadBindings) { b.DType = cudair.Float16 }},
		{"layout", func(b *cudasearch.WorkloadBindings) { b.Layout = "col_major" }},
		{"objective", func(b *cudasearch.WorkloadBindings) { b.Objective = cudasearch.ObjectiveMaxThroughput }},
		{"input", func(b *cudasearch.WorkloadBindings) { b.InputIdentities["x"] = "input-v2" }},
		{"weight", func(b *cudasearch.WorkloadBindings) { b.WeightIdentities["w"] = "weight-v2" }},
		{"input_order", func(b *cudasearch.WorkloadBindings) {
			b.InputNames[0], b.InputNames[1] = b.InputNames[1], b.InputNames[0]
		}},
		{"input_name", func(b *cudasearch.WorkloadBindings) {
			b.InputNames[0] = "y"
			b.InputIdentities = map[string]string{"y": "input-v1"}
		}},
	} {
		t.Run(test.name, func(t *testing.T) {
			changed := boundWorkload(t)
			test.change(&changed)
			if err := changed.Validate(); err != nil {
				t.Fatal(err)
			}
			got := changed.WorkloadIdentity()
			if changed.Key() == base.Key() || got.BindingsKey == want.BindingsKey || got.Key() == want.Key() {
				t.Fatal("different bindings share cache workload identity")
			}
		})
	}
}

func TestWorkloadIdentityDetachedAndCanonical(t *testing.T) {
	b := boundWorkload(t)
	b.InputIdentities["aux"] = "aux-v1"
	identity := b.WorkloadIdentity()
	key := identity.Key()
	identity.ShapeDims[0] = 99
	if b.Shape.Dim(0) != 2 {
		t.Fatal("identity dimensions alias binding shape")
	}
	identity = b.WorkloadIdentity()
	b.InputIdentities = map[string]string{"aux": "aux-v1", "x": "input-v1"}
	if b.WorkloadIdentity().Key() != key {
		t.Fatal("map insertion order changed identity")
	}
	b.InputIdentities["x"] = "changed"
	if identity.Key() != key || b.WorkloadIdentity().Key() == key {
		t.Fatal("identity is not a detached binding snapshot")
	}
}

func TestWorkloadBindingEnums(t *testing.T) {
	for _, dtype := range []cudair.DataType{cudair.Float32, cudair.Float16, cudair.BFloat16, cudair.Int32, cudair.Int64, cudair.Uint8, cudair.Int8} {
		b := boundWorkload(t)
		b.DType = dtype
		if err := b.Validate(); err != nil {
			t.Fatalf("dtype %d: %v", dtype, err)
		}
	}
	for _, layout := range []string{"", "row_major", "col_major", "strided"} {
		for _, objective := range []cudasearch.Objective{"", cudasearch.ObjectiveMinLatency, cudasearch.ObjectiveMaxThroughput, cudasearch.ObjectiveProfileCost} {
			b := boundWorkload(t)
			b.Layout, b.Objective = layout, objective
			if err := b.Validate(); err != nil {
				t.Fatalf("layout %q objective %q: %v", layout, objective, err)
			}
		}
	}
	for _, test := range []struct {
		name   string
		change func(*cudasearch.WorkloadBindings)
	}{
		{"dtype", func(b *cudasearch.WorkloadBindings) { b.DType = cudair.DataType(100) }},
		{"negative_dtype", func(b *cudasearch.WorkloadBindings) { b.DType = cudair.DataType(-1) }},
		{"layout", func(b *cudasearch.WorkloadBindings) { b.Layout = "row-majro" }},
		{"layout_case", func(b *cudasearch.WorkloadBindings) { b.Layout = "ROW_MAJOR" }},
		{"objective", func(b *cudasearch.WorkloadBindings) { b.Objective = "minimum_latency" }},
		{"custom_objective", func(b *cudasearch.WorkloadBindings) { b.Objective = "custom" }},
	} {
		t.Run(test.name, func(t *testing.T) {
			b := boundWorkload(t)
			test.change(&b)
			if err := b.Validate(); !errors.Is(err, cudasearch.ErrMissingBinding) {
				t.Fatalf("Validate() = %v, want ErrMissingBinding", err)
			}
		})
	}
}

func TestWorkloadBindingUnknownDTypesHaveDistinctKeys(t *testing.T) {
	a, b := boundWorkload(t), boundWorkload(t)
	a.DType, b.DType = cudair.DataType(100), cudair.DataType(101)
	if a.DType.String() != b.DType.String() {
		t.Fatal("fixture no longer exercises the predecessor's lossy encoding")
	}
	if !errors.Is(a.Validate(), cudasearch.ErrMissingBinding) || !errors.Is(b.Validate(), cudasearch.ErrMissingBinding) {
		t.Fatal("unknown dtypes admitted")
	}
	if a.Key() == b.Key() || a.WorkloadIdentity().Key() == b.WorkloadIdentity().Key() {
		t.Fatal("unsupported dtypes alias before validation")
	}
}

type bindingExecutable struct{ calls *int }

func (x bindingExecutable) Identity() string { return "binding-fixture" }
func (x bindingExecutable) Outputs(context.Context) (map[string][]float32, error) {
	*x.calls++
	return map[string][]float32{"out": {1}}, nil
}
func (x bindingExecutable) Run(context.Context) (cudatrial.Receipt, error) {
	*x.calls++
	return cudatrial.Receipt{}, nil
}
func (x bindingExecutable) Close() error { *x.calls++; return nil }

func TestWorkloadBindingEnumsBeforeExecutableWork(t *testing.T) {
	for _, field := range []string{"dtype", "layout", "objective"} {
		t.Run(field, func(t *testing.T) {
			b := boundWorkload(t)
			if field == "dtype" {
				b.DType = cudair.DataType(100)
			} else if field == "layout" {
				b.Layout = "invalid"
			} else {
				b.Objective = "invalid"
			}
			calls := 0
			b.ResetState = func() error { calls++; return nil }
			e := &cudasearch.ExecutableEvaluator{
				Workload: b, Baseline: bindingExecutable{&calls}, Rounds: 2,
				References: map[string][]float32{"out": {1}}, TimingKind: cudatrial.TimingSummedKernel,
				Compile: func(context.Context, *egraph.Expr, cudasearch.WorkloadBindings) (cudasearch.Executable, error) {
					calls++
					return bindingExecutable{&calls}, nil
				},
				HardwareLockChecker: func() bool { calls++; return true },
			}
			g := egraph.New()
			root := g.Add(egraph.ENode{Op: "input"})
			state := egraph.NewSearchState(g, map[string]egraph.Id{"out": root})
			candidate, err := e.Evaluate(context.Background(), state, root)
			if candidate != nil || !errors.Is(err, cudasearch.ErrMissingBinding) || calls != 0 {
				t.Fatalf("candidate=%v err=%v executable calls=%d", candidate, err, calls)
			}
		})
	}
}

func ExampleWorkloadBindings_WorkloadIdentity() {
	b := cudasearch.WorkloadBindings{Name: "sum", InputNames: []string{"x"},
		InputIdentities: map[string]string{"x": "input-v1"},
		NumericalPolicy: cudasearch.NumericalPolicyExact}
	id := b.WorkloadIdentity()
	fmt.Println(id.ProgramID, id.BindingsKey == b.Key())
	// Output:
	// sum true
}

package cudaref_test

import (
	"math"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudaref"
)

func TestReferenceDtypeAdmission(t *testing.T) {
	for _, dtype := range []cudair.DataType{cudair.Float16, cudair.BFloat16, cudair.Int32, cudair.Int64, cudair.Uint8, cudair.Int8} {
		for _, data := range []struct {
			name string
			a, b []float32
		}{
			{"finite", []float32{3}, []float32{2}},
			{"empty", nil, nil},
			{"NaN", []float32{float32(math.NaN())}, []float32{2}},
		} {
			t.Run(dtype.String()+"/"+data.name, func(t *testing.T) {
				g := cudair.NewGraph()
				a := g.AddInput("a", []int{len(data.a)}, dtype)
				b := g.AddInput("b", []int{len(data.b)}, dtype)
				out := g.AddBinary("div", a, b)
				if err := g.SetOutputs(out); err != nil {
					t.Fatal(err)
				}
				inputs := map[string][]float32{"a": data.a, "b": data.b}
				checkDtypeRejection(t, g, inputs, dtype.String())
			})
		}
	}
}

func checkDtypeRejection(t *testing.T, g *cudair.Graph, inputs map[string][]float32, dtype string) {
	t.Helper()
	snap, err := g.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	for _, evaluate := range []struct {
		name string
		run  func() (bool, error)
	}{
		{"Evaluate", func() (bool, error) {
			out, err := cudaref.Evaluate(g, inputs)
			return out == nil, err
		}},
		{"EvaluateSnapshot", func() (bool, error) {
			out, err := cudaref.EvaluateSnapshot(snap, inputs)
			return out == nil, err
		}},
		{"EvaluateSnapshotAll", func() (bool, error) {
			out, err := cudaref.EvaluateSnapshotAll(snap, inputs)
			return out == nil, err
		}},
	} {
		t.Run(evaluate.name, func(t *testing.T) {
			nilOutput, err := evaluate.run()
			if !nilOutput || err == nil || !strings.Contains(err.Error(), "unsupported data type "+dtype) {
				t.Fatalf("nil output = %v, error = %v; want unsupported %s", nilOutput, err, dtype)
			}
		})
	}
}

func TestReferenceDtypePreflightBeforeInputs(t *testing.T) {
	for _, output := range []bool{false, true} {
		g := cudair.NewGraph()
		x := g.AddInput("x", []int{1}, cudair.Float32)
		y := g.AddUnary("neg", x)
		constant := g.AddConstant("typed", []int{1}, cudair.Int32, []float32{3})
		outputs := []*cudair.Node{y}
		if output {
			outputs = append(outputs, constant)
		}
		if err := g.SetOutputs(outputs...); err != nil {
			t.Fatal(err)
		}
		// Even an unsupported constant outside the output path is executed by
		// EvaluateSnapshotAll. Admission must precede missing or malformed input.
		for _, inputs := range []map[string][]float32{nil, {"x": {1, 2}}} {
			checkDtypeRejection(t, g, inputs, "int32")
		}
	}
}

func TestReferenceDtypeFloat32Control(t *testing.T) {
	g := cudair.NewGraph()
	a := g.AddInput("a", []int{1}, cudair.Float32)
	b := g.AddInput("b", []int{1}, cudair.Float32)
	y := g.AddBinary("div", a, b)
	if err := g.SetOutputs(y); err != nil {
		t.Fatal(err)
	}
	snap, err := g.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	inputs := map[string][]float32{"a": {3}, "b": {2}}
	outputs, err := cudaref.Evaluate(g, inputs)
	if err != nil || len(outputs[y.Name]) != 1 || outputs[y.Name][0] != 1.5 {
		t.Fatalf("Evaluate = %v, %v", outputs, err)
	}
	outputs, err = cudaref.EvaluateSnapshot(snap, inputs)
	if err != nil || len(outputs[y.Name]) != 1 || outputs[y.Name][0] != 1.5 {
		t.Fatalf("EvaluateSnapshot = %v, %v", outputs, err)
	}
	values, err := cudaref.EvaluateSnapshotAll(snap, inputs)
	if err != nil || len(values[y.ID]) != 1 || values[y.ID][0] != 1.5 {
		t.Fatalf("EvaluateSnapshotAll = %v, %v", values, err)
	}
	if inputs["a"][0] != 3 || inputs["b"][0] != 2 {
		t.Fatal("input values changed")
	}
	if _, err := cudaref.EvaluateSnapshotAll(nil, nil); err == nil || err.Error() != "cudaref: nil snapshot" {
		t.Fatalf("nil snapshot error = %v", err)
	}
	if _, err := cudaref.EvaluateSnapshotAll(snap, nil); err == nil || !strings.Contains(err.Error(), "missing input buffer") {
		t.Fatalf("missing FP32 input error = %v", err)
	}
	if _, err := cudaref.EvaluateSnapshotAll(snap, map[string][]float32{"a": {1, 2}, "b": {2}}); err == nil || !strings.Contains(err.Error(), "buffer length") {
		t.Fatalf("malformed FP32 input error = %v", err)
	}
}

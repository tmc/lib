package cudaref_test

import (
	"math"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudaref"
)

func TestElementwiseUnaryRejectsUnsupportedOpsBeforeInspectingValues(t *testing.T) {
	nan := float32(math.NaN())
	for _, op := range []string{"unknown", "rmsnorm", "rmsnorm_eps=1e-05"} {
		for _, input := range [][]float32{nil, {nan, nan}, {nan, 1}} {
			got, err := cudaref.ElementwiseUnary(input, op)
			if err == nil || !strings.Contains(err.Error(), "unknown unary op "+op) {
				t.Errorf("ElementwiseUnary(%q, %v) = %v, %v; want unknown-op error", op, input, got, err)
			}
		}
	}
}

func TestElementwiseBinaryRejectsUnsupportedOpsBeforeInspectingValues(t *testing.T) {
	nan := float32(math.NaN())
	for _, op := range []string{"unknown", "rope", "rope_theta=10000"} {
		for _, input := range [][]float32{nil, {nan, nan}, {nan, 1}} {
			got, err := cudaref.ElementwiseBinary(input, input, op)
			if err == nil || !strings.Contains(err.Error(), "unknown binary op "+op) {
				t.Errorf("ElementwiseBinary(%q, %v) = %v, %v; want unknown-op error", op, input, got, err)
			}
		}
	}
	if _, err := cudaref.ElementwiseBinary([]float32{1}, nil, "unknown"); err == nil || !strings.Contains(err.Error(), "length mismatch") {
		t.Errorf("length mismatch precedence = %v, want length mismatch", err)
	}
	got, err := cudaref.ElementwiseBinary([]float32{2}, []float32{3}, "multiply")
	if err != nil || len(got) != 1 || got[0] != 6 {
		t.Errorf("multiply alias = %v, %v; want [6], nil", got, err)
	}
}

func TestEvaluateSnapshotRefusesUnsupportedPrefixOpsForEveryInputContents(t *testing.T) {
	nan := float32(math.NaN())
	tests := []struct {
		name string
		make func(*cudair.Graph, *cudair.Node) *cudair.Node
		want string
	}{
		{"rmsnorm", func(g *cudair.Graph, x *cudair.Node) *cudair.Node { return g.AddUnary("rmsnorm_eps=1e-05", x) }, "unknown unary op rmsnorm_eps=1e-05"},
		{"rope", func(g *cudair.Graph, x *cudair.Node) *cudair.Node {
			return g.AddBinary("rope_theta=10000", x, g.AddConstant("scale", nil, cudair.Float32, []float32{2}))
		}, "unknown binary op rope_theta=10000"},
	}
	inputs := []struct {
		name string
		data []float32
	}{
		{"finite", []float32{1, 4}},
		{"empty", nil},
		{"all NaN", []float32{nan, nan}},
		{"mixed", []float32{nan, 1}},
	}
	for _, tt := range tests {
		for _, in := range inputs {
			t.Run(tt.name+"/"+in.name, func(t *testing.T) {
				g := cudair.NewGraph()
				x := g.AddInput("x", []int{len(in.data)}, cudair.Float32)
				y := tt.make(g, x)
				if err := g.SetOutputs(y); err != nil {
					t.Fatal(err)
				}
				snap, err := g.Snapshot()
				if err != nil {
					t.Fatalf("Snapshot() = %v", err)
				}
				if _, err := cudaref.EvaluateSnapshot(snap, map[string][]float32{"x": in.data}); err == nil || !strings.Contains(err.Error(), tt.want) {
					t.Errorf("EvaluateSnapshot() error = %v, want %q", err, tt.want)
				}
			})
		}
	}
}

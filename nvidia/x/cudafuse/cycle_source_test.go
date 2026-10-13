package cudafuse

import (
	"github.com/tmc/lib/nvidia/x/cudair"
	"slices"
	"strings"
	"testing"
)

func TestCycleSourceMapping(t *testing.T) {
	for _, tt := range []struct {
		name          string
		input, target []int
		values, want  []float32
	}{
		{"repeat", []int{3}, []int{2, 3}, []float32{2, 5, 9}, []float32{2, 5, 9, 2, 5, 9}},
		{"truncate", []int{3}, []int{2}, []float32{2, 5, 9}, []float32{2, 5}},
		{"scalar", nil, []int{4}, []float32{7}, []float32{7, 7, 7, 7}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			ctx := NewLazyContext()
			a := NewLazyArray(ctx, tt.input, cudair.Float32)
			cycle := a.Cycle(tt.target)
			f, err := FuseLazy([]*LazyNode{a.Node}, []*LazyNode{cycle.Node}, "periodic")
			if err != nil {
				t.Fatal(err)
			}
			if len(tt.values) > 1 && !strings.Contains(f.Source, "% 3ULL") {
				t.Fatal("no source-count modulo", f.Source)
			}
			got, err := shapeSourceModel(f, [][]float32{tt.values}, len(tt.want))
			if err != nil || !slices.Equal(got[0], tt.want) {
				t.Fatal(got, err, tt.want)
			}
			p, err := cycle.Materialize()
			if err != nil {
				t.Fatal(err)
			}
			if p.RootNode != cycle.Node || p.AllocatedBytes != int64(len(tt.want)*4) {
				t.Fatal("wrong planner output")
			}
			found := false
			for _, r := range p.Regions {
				if r.Boundary == "cycle" {
					found = r.Kernel != nil && r.Reason != ""
				}
			}
			if !found {
				t.Fatal("missing owned compiled Cycle boundary")
			}
			for _, eager := range []bool{false, true} {
				plan, err := snapshotExecution(p, eager)
				if err != nil {
					t.Fatal(err)
				}
				last := plan.steps[len(plan.steps)-1]
				if last.count != len(tt.want) {
					t.Fatal("snapshot target count", last.count)
				}
				source, err := last.source("dispatch_cycle")
				if err != nil {
					t.Fatal(err)
				}
				values, err := shapeSourceModel(source, [][]float32{tt.values}, last.count)
				if err != nil || !slices.Equal(values[0], tt.want) {
					t.Fatal("dispatch", values, err)
				}
			}
		})
	}
}
func TestCycleHandBuiltRefusal(t *testing.T) {
	for _, mutate := range []func(*LazyNode){
		func(n *LazyNode) { n.Attrs["target_shape"] = []int{99} }, func(n *LazyNode) { n.Shape = nil; n.Attrs["target_shape"] = []int(nil) }, func(n *LazyNode) { n.Attrs["extra"] = true }, func(n *LazyNode) { n.Inputs = nil },
	} {
		ctx := NewLazyContext()
		a := NewLazyArray(ctx, []int{3}, cudair.Float32)
		n := a.Cycle([]int{6}).Node
		mutate(n)
		if _, err := CompileDeferredGraph(ctx, n); err == nil {
			t.Fatal("bad graph admitted")
		}
		if _, err := FuseLazy([]*LazyNode{a.Node}, []*LazyNode{n}, "bad"); err == nil {
			t.Fatal("bad indexed graph admitted")
		}
	}
}

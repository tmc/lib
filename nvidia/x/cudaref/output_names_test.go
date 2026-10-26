package cudaref_test

import (
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudaref"
)

// outputGraph returns a graph whose explicit outputs are two distinct
// constants named first and second, holding 1 and 2.
func outputGraph(t *testing.T, first, second string) (*cudair.Graph, *cudair.Node, *cudair.Node) {
	t.Helper()
	g := cudair.NewGraph()
	a := g.AddConstant(first, nil, cudair.Float32, []float32{1})
	b := g.AddConstant(second, nil, cudair.Float32, []float32{2})
	if err := g.SetOutputs(a, b); err != nil {
		t.Fatal(err)
	}
	return g, a, b
}

func TestEvaluateSnapshotOutputNames(t *testing.T) {
	g, a, b := outputGraph(t, "first", "second")
	s, err := g.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	got, err := cudaref.EvaluateSnapshot(s, nil)
	if err != nil {
		t.Fatal(err)
	}
	want := map[string][]float32{a.Name: {1}, b.Name: {2}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("EvaluateSnapshot = %v, want %v", got, want)
	}
}

func TestEvaluateSnapshotRefusesSharedOutputName(t *testing.T) {
	for _, name := range []string{"first", ""} {
		t.Run(fmt.Sprintf("%q", name), func(t *testing.T) {
			g, a, b := outputGraph(t, name, name)
			s, err := g.Snapshot()
			if err != nil {
				t.Fatal(err)
			}
			if n := len(s.Outputs()); n != 2 {
				t.Fatalf("snapshot has %d outputs, want 2", n)
			}

			all, err := cudaref.EvaluateSnapshotAll(s, nil)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(all[a.ID], []float32{1}) || !reflect.DeepEqual(all[b.ID], []float32{2}) {
				t.Fatalf("EvaluateSnapshotAll = %v, want node %d = [1] and node %d = [2]", all, a.ID, b.ID)
			}

			got, err := cudaref.EvaluateSnapshot(s, nil)
			if err == nil {
				t.Fatalf("EvaluateSnapshot = %v, nil; want an error for the shared output name", got)
			}
			if got != nil {
				t.Errorf("EvaluateSnapshot returned %v with error; want nil map", got)
			}
			want := fmt.Sprintf("output name %q is shared by nodes %d and %d", name, a.ID, b.ID)
			if !strings.Contains(err.Error(), want) {
				t.Errorf("EvaluateSnapshot error = %q, want it to contain %q", err, want)
			}

			got, err = cudaref.Evaluate(g, nil)
			if err == nil || !strings.Contains(err.Error(), want) {
				t.Errorf("Evaluate = %v, %v; want error containing %q", got, err, want)
			}
		})
	}
}

func TestEvaluateSnapshotRepeatedOutputNode(t *testing.T) {
	g := cudair.NewGraph()
	a := g.AddConstant("first", nil, cudair.Float32, []float32{1})
	if err := g.SetOutputs(a, a); err != nil {
		t.Fatal(err)
	}
	s, err := g.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	if n := len(s.Outputs()); n != 2 {
		t.Fatalf("snapshot has %d outputs, want 2", n)
	}
	got, err := cudaref.EvaluateSnapshot(s, nil)
	if err != nil {
		t.Fatal(err)
	}
	want := map[string][]float32{"first": {1}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("EvaluateSnapshot = %v, want %v", got, want)
	}
}

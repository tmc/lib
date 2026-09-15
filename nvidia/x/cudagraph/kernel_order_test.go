package cudagraph

import (
	"fmt"
	"github.com/tmc/lib/nvidia/driver"
	"reflect"
	"strings"
	"testing"
)

func TestKernelOrderTopology(t *testing.T) {
	for _, tt := range []struct {
		name            string
		nodes, from, to []driver.GraphNode
		want            []string
		err             string
	}{
		{"scrambled native enumeration", []driver.GraphNode{3, 1, 2}, []driver.GraphNode{1, 2}, []driver.GraphNode{2, 3}, []string{"a", "b", "c"}, ""},
		{"reversed dependency mutant", []driver.GraphNode{3, 1, 2}, []driver.GraphNode{3, 2}, []driver.GraphNode{2, 1}, []string{"c", "b", "a"}, ""},
		{"fork", []driver.GraphNode{1, 2, 3}, []driver.GraphNode{1, 1}, []driver.GraphNode{2, 3}, nil, "ambiguous"},
		{"missing dependency mutant", []driver.GraphNode{1, 2, 3}, []driver.GraphNode{1}, []driver.GraphNode{2}, nil, "ambiguous"},
		{"cycle", []driver.GraphNode{1, 2}, []driver.GraphNode{1, 2}, []driver.GraphNode{2, 1}, nil, "cycle"},
		{"duplicate edge", []driver.GraphNode{1, 2}, []driver.GraphNode{1, 1}, []driver.GraphNode{2, 2}, nil, "duplicate"},
		{"unknown endpoint", []driver.GraphNode{1}, []driver.GraphNode{1}, []driver.GraphNode{2}, nil, "unknown"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			got, err := orderKernels(tt.nodes, tt.from, tt.to, map[driver.GraphNode]string{1: "a", 2: "b", 3: "c"})
			if tt.err != "" {
				if err == nil || !strings.Contains(err.Error(), tt.err) {
					t.Fatalf("got %v; want %s", err, tt.err)
				}
				return
			}
			if err != nil || !reflect.DeepEqual(got, tt.want) {
				t.Fatalf("got %v,%v; want %v", got, err, tt.want)
			}
		})
	}
}
func TestKernelOrderSnapshotOwned(t *testing.T) {
	e := &Exec{raw: 7, kernelOrder: []string{"a", "b"}}
	raw, names, err := e.KernelOrder()
	if err != nil || raw != 7 {
		t.Fatal(raw, err)
	}
	names[0] = "mutated"
	_, again, _ := e.KernelOrder()
	if again[0] != "a" {
		t.Fatal("snapshot exposed mutable storage")
	}
	e.kernelOrder = nil
	if _, _, err := e.KernelOrder(); err == nil {
		t.Fatal("invalidated evidence accepted")
	}
}
func ExampleExec_KernelOrder() {
	var exec *Exec
	_, _, err := exec.KernelOrder()
	fmt.Println(err)
	// Output: cudagraph: executable graph is closed
}

func TestKernelOrderNonkernelDependencies(t *testing.T) {
	// Nodes 1 and 2 represent allocation actions, 5 and 6 free actions.
	// Only 3 and 4 have kernel symbols; enumeration order is immaterial.
	nodes := []driver.GraphNode{6, 4, 2, 5, 1, 3}
	symbols := map[driver.GraphNode]string{3: "project", 4: "decode"}
	for _, tt := range []struct {
		name     string
		from, to []driver.GraphNode
		want     []string
	}{
		{"independent allocations and frees", []driver.GraphNode{1, 2, 3, 4, 4}, []driver.GraphNode{3, 3, 4, 5, 6}, []string{"project", "decode"}},
		{"reverse kernel dependencies", []driver.GraphNode{1, 2, 4, 3, 3}, []driver.GraphNode{4, 4, 3, 5, 6}, []string{"decode", "project"}},
		{"kernel released through nonkernel branch", []driver.GraphNode{1, 2, 3, 4}, []driver.GraphNode{3, 4, 5, 6}, nil},
		{"cycle outside ordered kernels", []driver.GraphNode{1, 2, 3, 5, 6}, []driver.GraphNode{3, 3, 4, 6, 5}, nil},
	} {
		t.Run(tt.name, func(t *testing.T) {
			got, err := orderKernels(nodes, tt.from, tt.to, symbols)
			if tt.want == nil {
				if err == nil {
					t.Fatalf("ambiguous or cyclic graph accepted: %v", got)
				}
				return
			}
			if err != nil || !reflect.DeepEqual(got, tt.want) {
				t.Fatalf("got=%v err=%v want=%v", got, err, tt.want)
			}
		})
	}
}

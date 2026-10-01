package overview

import (
	"reflect"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaprof/marker"
	"github.com/tmc/lib/nvidia/x/cudaprof/timeline"
)

func TestComponentHierarchy(t *testing.T) {
	var records timeline.Records
	zero, one := 0, 1
	contract := &marker.ComponentContract{Name: "gelu", BankSHA256: strings.Repeat("a", 64), RootSetSHA256: strings.Repeat("b", 64), Shape: []int{256}, Strides: []int{1}, DType: "bfloat16", MathForm: "compiled", ScalarPolicy: "weak", ExecutionPolicy: "synchronized_eval", RootPolicy: "distinct_banks"}
	for _, pid := range []uint32{1, 2} {
		for batch := 0; batch < 2; batch++ {
			meta := marker.Metadata{Version: 3, RequestID: "same", Role: "measured", Phase: "evaluate", Boundary: "submission", ExecutionUnit: "component_batch", Executions: &zero, Invocation: &zero, Batch: &batch, RootCount: &one, Root: &zero, Operation: "gelu", Component: contract}
			text, err := marker.Encode(meta)
			if err != nil {
				t.Fatal(err)
			}
			records.Ranges = append(records.Ranges, timeline.Range{Journal: "p", Process: pid, Thread: 1, Start: int64(batch*10 + 1), End: int64(batch*10 + 5), Name: text})
		}
	}
	report := Analyze(records, "driver", nil)
	if len(report.Tokens) > 0 || len(report.Inferred) > 0 {
		t.Fatal("invented model tokens")
	}
	counts := map[string]int{}
	var visit func(*Node)
	visit = func(n *Node) {
		counts[n.Kind]++
		if n.Kind == "token" || n.Kind == "layer" {
			t.Fatalf("model hierarchy for component: %+v", n)
		}
		for _, c := range n.Children {
			visit(c)
		}
	}
	visit(report.Hierarchy)
	if counts["request"] != 2 || counts["component"] != 2 || counts["component_batch"] != 4 || counts["component_root"] != 4 {
		t.Fatalf("merged PID/batch identities: %v", counts)
	}
	if !reflect.DeepEqual(report, Analyze(records, "driver", nil)) {
		t.Fatal("pointer addresses or unstable IDs in hierarchy")
	}
}

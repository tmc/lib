package timeline

import (
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaprof/marker"
)

func componentRangeMetadata() marker.Metadata {
	zero, one := 0, 1
	return marker.Metadata{Version: 3, RequestID: "r", Role: "measured", Phase: "evaluate", Boundary: "submission", ExecutionUnit: "component_batch", Executions: &zero, Invocation: &zero, Batch: &zero, RootCount: &one, Operation: "gelu", Component: &marker.ComponentContract{Name: "gelu", BankSHA256: strings.Repeat("a", 64), RootSetSHA256: strings.Repeat("b", 64), Shape: []int{256}, Strides: []int{1}, DType: "bfloat16", MathForm: "compiled", ScalarPolicy: "weak", ExecutionPolicy: "asynchronous_submission", RootPolicy: "distinct_banks"}}
}
func TestComponentCorrelation(t *testing.T) {
	for _, tt := range []struct {
		name     string
		alter    func(*marker.Metadata)
		expected string
	}{
		{"same contract", func(m *marker.Metadata) {}, "matched"},
		{"different bank", func(m *marker.Metadata) { m.Component.BankSHA256 = strings.Repeat("c", 64) }, "ambiguous"},
		{"different phase", func(m *marker.Metadata) { m.Phase = "submit" }, "ambiguous"},
		{"construction phase", func(m *marker.Metadata) { m.Phase = "construct" }, "ambiguous"},
		{"different execution policy", func(m *marker.Metadata) { m.Component.ExecutionPolicy = "synchronized_eval" }, "ambiguous"},
		{"different batch", func(m *marker.Metadata) { n := 1; m.Batch = &n }, "ambiguous"},
		{"different invocation", func(m *marker.Metadata) { n := 1; m.Invocation = &n }, "ambiguous"},
		{"different root policy", func(m *marker.Metadata) { m.Component.RootPolicy = "shared_aliases" }, "ambiguous"},
		{"different shape", func(m *marker.Metadata) { m.Component.Shape = []int{512} }, "ambiguous"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			batch := componentRangeMetadata()
			root := componentRangeMetadata()
			zero := 0
			root.Root = &zero
			tt.alter(&root)
			outer, err := marker.Encode(batch)
			if err != nil {
				t.Fatal(err)
			}
			inner, err := marker.Encode(root)
			if err != nil {
				t.Fatal(err)
			}
			records := Records{Activities: []Activity{{Journal: "p", Name: "kernel", Correlation: 1, Start: 100, End: 110}}, Calls: []Call{{Journal: "p", Process: 1, Thread: 2, Domain: "driver", Correlation: 1, Start: 12, End: 13}}, Ranges: []Range{{Journal: "p", Process: 1, Thread: 2, Name: outer, Start: 1, End: 20}, {Journal: "p", Process: 1, Thread: 2, Name: inner, Start: 10, End: 15}}}
			got := Correlations(records, "driver")[0]
			if got.MetadataStatus != tt.expected {
				t.Fatalf("%+v", got)
			}
			if tt.expected == "matched" && (got.Metadata.Root == nil || *got.Metadata.Root != 0) {
				t.Fatal("root was not observed under submitting range")
			}
			// Moving construction before the submit call leaves only batch attribution.
			records.Ranges[1].Start = 2
			records.Ranges[1].End = 8
			got = Correlations(records, "driver")[0]
			if got.MetadataStatus != "matched" || got.Metadata.Root != nil {
				t.Fatalf("nearest construction promoted root: %+v", got)
			}
		})
	}
}

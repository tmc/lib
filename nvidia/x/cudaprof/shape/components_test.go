package shape

import (
	"fmt"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaprof/marker"
	"github.com/tmc/lib/nvidia/x/cudaprof/timeline"
)

func componentCapture(t *testing.T, batches, roots int) timeline.Records {
	t.Helper()
	var records timeline.Records
	contract := &marker.ComponentContract{Name: "GeGLU", BankSHA256: strings.Repeat("a", 64), RootSetSHA256: strings.Repeat("b", 64), Shape: []int{256}, Strides: []int{1}, DType: "bfloat16", MathForm: "direct-fused", ScalarPolicy: "weak-float", ExecutionPolicy: "synchronized_eval", RootPolicy: "distinct_banks"}
	invocation := 0
	for batch := 0; batch < batches; batch++ {
		before, after := batch, batch+1
		m := marker.Metadata{Version: 3, RequestID: "activation", Role: "measured", Phase: "evaluate", Boundary: "submission", ExecutionUnit: "component_batch", Executions: &before, Invocation: &invocation, Batch: &batch, RootCount: &roots, Operation: "GeGLU", Component: contract}
		text, err := marker.Encode(m)
		if err != nil {
			t.Fatal(err)
		}
		start := int64(batch*10000 + 100)
		records.Ranges = append(records.Ranges, timeline.Range{Journal: "capture", Process: 1, Thread: 2, Name: text, Start: start + 1, End: start + 5000})
		m.Event = "batch_start"
		text, err = marker.Encode(m)
		if err != nil {
			t.Fatal(err)
		}
		records.Markers = append(records.Markers, timeline.Marker{Journal: "capture", Process: 1, Thread: 2, Name: text, Time: start, Flags: timeline.MarkerInstantaneous})
		for root := 0; root < roots; root++ {
			m.Event = ""
			m.Root = &root
			text, err = marker.Encode(m)
			if err != nil {
				t.Fatal(err)
			}
			rs := start + 10 + int64(root*10)
			records.Ranges = append(records.Ranges, timeline.Range{Journal: "capture", Process: 1, Thread: 2, Name: text, Start: rs, End: rs + 8})
			for kernel := 0; kernel < 2; kernel++ {
				correlation := uint32(len(records.Activities) + 1)
				when := rs + 1 + int64(kernel*2)
				shared := uint32(0)
				records.Calls = append(records.Calls, timeline.Call{Journal: "capture", Process: 1, Thread: 2, Domain: "driver", Correlation: correlation, Start: when, End: when + 1})
				records.Activities = append(records.Activities, timeline.Activity{Journal: "capture", Subtype: "kernel", Name: fmt.Sprintf("kernel%d", kernel), Correlation: correlation, Start: start + 6000 + int64(root*4+kernel*2), End: start + 6001 + int64(root*4+kernel*2), Grid: [3]int{1, 1, 1}, Block: [3]int{32, 1, 1}, SharedMemory: &shared})
			}
		}
		m.Event = "batch_end"
		m.Root = nil
		m.Executions = &after
		m.Boundary = "completion"
		text, err = marker.Encode(m)
		if err != nil {
			t.Fatal(err)
		}
		records.Markers = append(records.Markers, timeline.Marker{Journal: "capture", Process: 1, Thread: 2, Name: text, Time: start + 8000, Flags: timeline.MarkerInstantaneous})
	}
	return records
}
func componentPairs() []RequestPair {
	return []RequestPair{{BaseJournal: "capture", BaseProcess: 1, BaseID: "activation", TargetJournal: "capture", TargetProcess: 1, TargetID: "activation", Role: "measured"}}
}
func TestComponentCompare(t *testing.T) {
	base, target := componentCapture(t, 2, 3), componentCapture(t, 2, 3)
	r := CompareComponents(base, target, "driver", componentPairs())
	if !r.Pass || r.MatchedRoots != 6 || r.Arms[0].ExplicitIterations != 2 || r.Arms[0].Kernels != 12 {
		t.Fatalf("%+v", r)
	}
	if r.Qualifications["native_completion"] != "unknown" || r.Qualifications["correctness"] != "unverified" {
		t.Fatal("promoted independent gate")
	}
}
func TestComponentFailures(t *testing.T) {
	for _, tt := range []struct {
		name  string
		alter func(*timeline.Records)
		pairs func() []RequestPair
	}{
		{"deferred Eval", func(r *timeline.Records) { r.Ranges = r.Ranges[:1] }, componentPairs},
		{"partial CSE", func(r *timeline.Records) { r.Activities = r.Activities[:2] }, componentPairs},
		{"duplicate root", func(r *timeline.Records) { r.Ranges = append(r.Ranges, r.Ranges[1]) }, componentPairs},
		{"missing lifecycle", func(r *timeline.Records) { r.Markers = r.Markers[1:] }, componentPairs},
		{"malformed lifecycle", func(r *timeline.Records) {
			r.Markers = append(r.Markers, timeline.Marker{Journal: "capture", Process: 1, Name: `cudaprof:v3 {"bad":1}`, Flags: timeline.MarkerInstantaneous})
		}, componentPairs},
		{"missing shared memory", func(r *timeline.Records) { r.Activities[0].SharedMemory = nil }, componentPairs},
		{"shared memory mismatch", func(r *timeline.Records) { n := uint32(16); r.Activities[0].SharedMemory = &n }, componentPairs},
		{"wrong PID", func(r *timeline.Records) { r.Calls[0].Process = 9 }, componentPairs},
		{"wrong journal", func(r *timeline.Records) { r.Activities[0].Journal = "other" }, componentPairs},
		{"ambiguous API", func(r *timeline.Records) { r.Calls = append(r.Calls, r.Calls[0]) }, componentPairs},
		{"absent extra mapping", func(r *timeline.Records) {}, func() []RequestPair {
			return append(componentPairs(), RequestPair{BaseJournal: "capture", BaseProcess: 1, BaseID: "absent", TargetJournal: "capture", TargetProcess: 1, TargetID: "absent", Role: "measured"})
		}},
		{"unknown mapping", func(r *timeline.Records) {}, func() []RequestPair { return nil }},
	} {
		t.Run(tt.name, func(t *testing.T) {
			base, target := componentCapture(t, 1, 3), componentCapture(t, 1, 3)
			tt.alter(&target)
			r := CompareComponents(base, target, "driver", tt.pairs())
			if r.Pass {
				t.Fatalf("accepted %+v", r)
			}
		})
	}
}
func TestComponentFullFixture(t *testing.T) {
	if testing.Short() {
		t.Skip("full18000-kernel provenance fixture")
	}
	base, target := componentCapture(t, 200, 45), componentCapture(t, 200, 45)
	r := CompareComponents(base, target, "driver", componentPairs())
	if !r.Pass || r.MatchedRoots != 9000 || r.Arms[0].ExplicitIterations != 200 || r.Arms[1].ExplicitIterations != 200 || r.Arms[0].Kernels != 18000 || r.Arms[1].Kernels != 18000 {
		t.Fatalf("pass=%t roots=%d iterations=%d/%d kernels=%d/%d failures=%v", r.Pass, r.MatchedRoots, r.Arms[0].ExplicitIterations, r.Arms[1].ExplicitIterations, r.Arms[0].Kernels, r.Arms[1].Kernels, r.Failures)
	}
}
func ExampleCompareComponents() {
	r := CompareComponents(timeline.Records{}, timeline.Records{}, "driver", nil)
	fmt.Println(r.Pass, r.Qualifications["native_completion"])
	// Output: false unknown
}

func TestComponentRootContractChanges(t *testing.T) {
	for _, tt := range []struct {
		name  string
		alter func(*marker.Metadata)
	}{
		{"shared aliases", func(m *marker.Metadata) { m.Component.RootPolicy = "shared_aliases" }},
		{"different bank", func(m *marker.Metadata) { m.Component.BankSHA256 = strings.Repeat("c", 64) }},
		{"different scalar policy", func(m *marker.Metadata) { m.Component.ScalarPolicy = "bf16-precast" }},
		{"different math form", func(m *marker.Metadata) { m.Component.MathForm = "separate-unary" }},
		{"different strides", func(m *marker.Metadata) { m.Component.Strides = []int{2} }},
	} {
		t.Run(tt.name, func(t *testing.T) {
			base, target := componentCapture(t, 1, 3), componentCapture(t, 1, 3)
			for i := range target.Ranges {
				m, err := marker.ParseRange(target.Ranges[i].Name)
				if err != nil {
					t.Fatal(err)
				}
				tt.alter(&m)
				target.Ranges[i].Name, err = marker.Encode(m)
				if err != nil {
					t.Fatal(err)
				}
			}
			for i := range target.Markers {
				m, err := marker.ParseRange(target.Markers[i].Name)
				if err != nil {
					t.Fatal(err)
				}
				tt.alter(&m)
				target.Markers[i].Name, err = marker.Encode(m)
				if err != nil {
					t.Fatal(err)
				}
			}
			if r := CompareComponents(base, target, "driver", componentPairs()); r.Pass {
				t.Fatalf("accepted %s", tt.name)
			}
		})
	}
}

func TestComponentCaptureIdentity(t *testing.T) {
	makeArm := func() timeline.Records {
		first, second := componentCapture(t, 1, 3), componentCapture(t, 1, 3)
		for i := range second.Activities {
			second.Activities[i].Journal = "other"
		}
		for i := range second.Calls {
			second.Calls[i].Journal = "other"
			second.Calls[i].Process = 2
		}
		for i := range second.Ranges {
			second.Ranges[i].Journal = "other"
			second.Ranges[i].Process = 2
		}
		for i := range second.Markers {
			second.Markers[i].Journal = "other"
			second.Markers[i].Process = 2
		}
		first.Activities = append(first.Activities, second.Activities...)
		first.Calls = append(first.Calls, second.Calls...)
		first.Ranges = append(first.Ranges, second.Ranges...)
		first.Markers = append(first.Markers, second.Markers...)
		return first
	}
	pairs := append(componentPairs(), RequestPair{BaseJournal: "other", BaseProcess: 2, BaseID: "activation", TargetJournal: "other", TargetProcess: 2, TargetID: "activation", Role: "measured"})
	r := CompareComponents(makeArm(), makeArm(), "driver", pairs)
	if !r.Pass || r.MatchedRoots != 6 || len(r.Arms[0].Batches) != 2 {
		t.Fatalf("merged journal/PID/request identity: %+v", r)
	}
	pairs[1].TargetJournal = "capture"
	pairs[1].TargetProcess = 1
	if r := CompareComponents(makeArm(), makeArm(), "driver", pairs); r.Pass {
		t.Fatal("reused capture-local request mapping")
	}
}

func ExampleComponentConfiguration() {
	c := ComponentConfiguration{Symbol: "kernel", Grid: [3]int{1, 1, 1}, Block: [3]int{32, 1, 1}, SharedMemory: 0}
	fmt.Println(c.Symbol, c.Block, c.SharedMemory)
	// Output: kernel [32 1 1] 0
}
func ExampleComponentRoot() {
	root := ComponentRoot{Ordinal: 0, Operation: "gelu", Kernels: 2}
	fmt.Println(root.Ordinal, root.Operation, root.Kernels)
	// Output: 0 gelu 2
}
func ExampleComponentBatch() {
	b := ComponentBatch{Ordinal: 3, RootCount: 45}
	fmt.Println(b.Ordinal, b.RootCount, len(b.Roots))
	// Output: 3 45 0
}
func ExampleComponentArm() {
	arm := ComponentArm{Kernels: 90, UnattributedKernels: 90}
	fmt.Println(arm.Kernels, arm.UnattributedKernels)
	// Output: 90 90
}
func ExampleComponentReport() {
	r := CompareComponents(timeline.Records{}, timeline.Records{}, "driver", nil)
	fmt.Println(r.Pass, r.MatchedRoots, r.Qualifications["correctness"])
	// Output: false 0 unverified
}

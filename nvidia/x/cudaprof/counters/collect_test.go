package counters

import (
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaprof/shape"
	"github.com/tmc/lib/nvidia/x/cuptirange"
)

func testCollectOptions() CollectOptions {
	id := shape.Identity{Phase: "decode", Operation: "gemv", DType: "f32", Grid: [3]int{2, 1, 1}, Block: [3]int{128, 1, 1}}
	launch := shape.Launch{Identity: &id, Activity: 0, Journal: "generation", RawSymbol: "gemv", Start: 1, Duration: 2, APIStatus: "matched", MetadataStatus: "matched"}
	return CollectOptions{Selection: Selection{Identity: id, Base: launch, Target: launch}, Arm: "base", CaptureSHA256: strings.Repeat("a", 64), ExpectedOutputSHA256: strings.Repeat("b", 64), ToolVersion: "test", Metrics: []string{"dram__bytes.sum"}, MaxPasses: 4, LockDeclared: true}
}
func TestCollectRefusesBeforeNative(t *testing.T) {
	tests := []struct {
		name string
		edit func(*CollectOptions)
	}{
		{"missing lock", func(o *CollectOptions) { o.LockDeclared = false }},
		{"missing capture", func(o *CollectOptions) { o.CaptureSHA256 = "" }},
		{"missing output", func(o *CollectOptions) { o.ExpectedOutputSHA256 = "" }},
		{"unknown arm", func(o *CollectOptions) { o.Arm = "guess" }},
		{"unqualified", func(o *CollectOptions) { o.Selection.Base.MetadataStatus = "missing" }},
		{"duplicate metrics", func(o *CollectOptions) { o.Metrics = append(o.Metrics, o.Metrics[0]) }},
		{"unbounded replay", func(o *CollectOptions) { o.MaxPasses = 65 }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			o := testCollectOptions()
			tt.edit(&o)
			r, err := Collect(nil, o, Replay{})
			if err == nil || r.Failure == "" || len(r.Witnesses) != 0 {
				t.Fatalf("report=%+v error=%v", r, err)
			}
		})
	}
}
func TestReplayWitness(t *testing.T) {
	o := testCollectOptions()
	w := Witness{Identity: o.Selection.Identity, OutputSHA256: o.ExpectedOutputSHA256}
	if err := validateReplayWitness(o, w); err != nil {
		t.Fatal(err)
	}
	w.Identity.CacheLength++
	if err := validateReplayWitness(o, w); err == nil {
		t.Fatal("accepted different replay identity")
	}
	w.Identity = o.Selection.Identity
	w.OutputSHA256 = strings.Repeat("c", 64)
	if err := validateReplayWitness(o, w); err == nil {
		t.Fatal("accepted wrong completed output")
	}
}
func TestNativeReplayRange(t *testing.T) {
	o := testCollectOptions()
	r := cuptirange.RangeInfo{Kind: "kernel", Name: "gemv", Grid: [3]uint32{2, 1, 1}, Block: [3]uint32{128, 1, 1}, GeometryAvailable: true}
	if err := validateNativeRange(o, o.Selection.Base, r); err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		name string
		edit func(*cuptirange.RangeInfo)
	}{
		{"wrong symbol", func(r *cuptirange.RangeInfo) { r.Name = "gemv_short" }},
		{"missing geometry", func(r *cuptirange.RangeInfo) { r.GeometryAvailable = false }},
		{"wrong grid", func(r *cuptirange.RangeInfo) { r.Grid[0]++ }},
		{"wrong block", func(r *cuptirange.RangeInfo) { r.Block[0]++ }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			bad := r
			tt.edit(&bad)
			if err := validateNativeRange(o, o.Selection.Base, bad); err == nil {
				t.Fatal("accepted incompatible native range")
			}
		})
	}
	zero := uint32(0)
	o.Selection.Identity.SharedMemory = &zero
	r.SharedMemory = 4
	if err := validateNativeRange(o, o.Selection.Base, r); err == nil {
		t.Fatal("accepted different shared memory")
	}
}

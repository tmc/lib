//go:build linux

package main

import (
	"strings"
	"testing"
)

func TestDSparkGroupedCompleteGuard(t *testing.T) {
	fresh := func() *cudaDSparkFeatures {
		r := &cudaRuntime{pos: 130}
		s := &cudaRequestState{runtime: r, generation: 4}
		f := &cudaDSparkFeatures{r: r, request: s, targets: []int{40, 41, 42}, history: &cudaDSparkHistory{}, grouped: &cudaDSparkGroupedCapture{start: 128, tokens: []int{10, 20}, generation: 4, captured: []bool{true, true, true, true, true, true}}}
		s.dspark, r.dspark = f, f
		return f
	}
	if err := fresh().validateGroupedComplete(); err != nil {
		t.Fatal(err)
	}
	for _, test := range []struct {
		name string
		edit func(*cudaDSparkFeatures)
		want string
	}{
		{"wrong-owner", func(f *cudaDSparkFeatures) { f.request.runtime = &cudaRuntime{} }, "incomplete target Eval"},
		{"inactive", func(f *cudaDSparkFeatures) { f.r.dspark = nil }, "incomplete target Eval"},
		{"wrong-frontier", func(f *cudaDSparkFeatures) { f.r.pos-- }, "incomplete target Eval"},
		{"wrong-generation", func(f *cudaDSparkFeatures) { f.request.generation++ }, "incomplete target Eval"},
		{"map-shape", func(f *cudaDSparkFeatures) { f.grouped.captured = []bool{true} }, "malformed capture map"},
		{"second-row-hole", func(f *cudaDSparkFeatures) { f.grouped.captured[4] = false }, "selected row missing"},
	} {
		t.Run(test.name, func(t *testing.T) {
			f := fresh()
			test.edit(f)
			if err := f.validateGroupedComplete(); err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("error=%v, want %q", err, test.want)
			}
		})
	}
}

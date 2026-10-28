package cudamegacpu

import (
	"reflect"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudamega"
	"github.com/tmc/lib/nvidia/x/cudataskgraph"
)

func extentProgram(t *testing.T, extent uint64, maximum uint32) *cudamega.Program {
	t.Helper()
	g, err := cudataskgraph.New([]cudataskgraph.Task{
		{ID: "add", Scope: cudataskgraph.ScopeDevice, Extent: extent},
		{ID: "mul", Rank: 1, Deps: []string{"add"}, Scope: cudataskgraph.ScopeDevice, Extent: extent},
	})
	if err != nil {
		t.Fatal(err)
	}
	p, err := cudamega.NewProgram(g, map[string]cudamega.Operation{"add": {Kind: cudamega.Add, Scalar: 3}, "mul": {Kind: cudamega.Multiply, Scalar: 2}}, cudamega.Grid{CTAs: 1, ThreadsPerCTA: 32, MaxElements: maximum}, cudamega.SM80)
	if err != nil {
		t.Fatal(err)
	}
	return p
}

func TestExtentAdmissionBeforeWorkers(t *testing.T) {
	e, err := New(cudamega.Admission{ResidentCTAs: 1, WorkerThreads: 32, SharedBytes: 1})
	if err != nil {
		t.Fatal(err)
	}
	var wraps int
	e.wrap = func(b cudamega.Backend) cudamega.Backend { wraps++; return b }
	for _, tc := range []struct {
		name     string
		declared uint64
		maximum  uint32
		input    []uint32
		want     string
	}{
		{"above maximum", 0, 1, []uint32{7, 11}, "exceeds maximum"},
		{"declared too long", 3, 3, []uint32{7, 11}, "requires 3 input elements"},
		{"empty", 0, 1, nil, "invalid finite program extent"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p := extentProgram(t, tc.declared, tc.maximum)
			wraps = 0
			got, runErr := e.Run(p, tc.input, 1, 0)
			ref, refErr := Reference(p, tc.input)
			if runErr == nil || !strings.Contains(runErr.Error(), tc.want) || wraps != 0 || got.Scratch != nil || got.StartOrder != nil {
				t.Fatalf("Run: error %v, wraps %d, result %+v", runErr, wraps, got)
			}
			if refErr == nil || !strings.Contains(refErr.Error(), tc.want) || ref != nil {
				t.Fatalf("Reference: error %v, output %v", refErr, ref)
			}
		})
	}
}

func TestFixedPrefixExtent(t *testing.T) {
	e, err := New(cudamega.Admission{ResidentCTAs: 1, WorkerThreads: 32, SharedBytes: 1})
	if err != nil {
		t.Fatal(err)
	}
	p := extentProgram(t, 1, 2)
	got, err := e.Run(p, []uint32{7, 11}, 1, 0)
	if err != nil || !reflect.DeepEqual(got.Scratch, []uint32{10, 20}) {
		t.Fatalf("fixed output extent = %v, %v", got.Scratch, err)
	}
	ref, err := Reference(p, []uint32{7, 11})
	if err != nil || !reflect.DeepEqual(ref, []uint32{10, 20}) {
		t.Fatalf("fixed ordinary output extent = %v, %v", ref, err)
	}
}

func TestExtentAdmissionNumericalParity(t *testing.T) {
	e, err := New(cudamega.Admission{ResidentCTAs: 1, WorkerThreads: 32, SharedBytes: 1})
	if err != nil {
		t.Fatal(err)
	}
	in := []uint32{0, 1, 0xffffffff}
	want := []uint32{3, 4, 2, 6, 8, 4}
	for _, tc := range []struct {
		name     string
		declared uint64
		maximum  uint32
	}{
		{"runtime", 0, 3},
		{"at maximum", 0, 3},
		{"below maximum", 0, 4},
		{"exact declaration", 3, 3},
		{"declaration and maximum", 3, 3},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p := extentProgram(t, tc.declared, tc.maximum)
			got, runErr := e.Run(p, in, 1, 0)
			ref, refErr := Reference(p, in)
			if runErr != nil || refErr != nil || got.Abort != 0 || !reflect.DeepEqual(got.Scratch, want) || !reflect.DeepEqual(ref, want) {
				t.Fatalf("Run %+v, %v; Reference %v, %v; want %v", got, runErr, ref, refErr, want)
			}
		})
	}
}

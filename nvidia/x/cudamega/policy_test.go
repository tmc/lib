package cudamega_test

import (
	"math"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudamega"
	"github.com/tmc/lib/nvidia/x/cudataskgraph"
)

func TestPolicyFor(t *testing.T) {
	dev := cudataskgraph.ScopeDevice
	tests := []struct {
		name  string
		tasks []cudataskgraph.Task
		want  cudamega.Policy
	}{
		{"one-task", []cudataskgraph.Task{{ID: "a", Scope: dev}}, cudamega.WavePolicy},
		{"chain", []cudataskgraph.Task{
			{ID: "a", Scope: dev},
			{ID: "b", Deps: []string{"a"}, Rank: 1, Scope: dev},
			{ID: "c", Deps: []string{"b"}, Rank: 2, Scope: dev},
		}, cudamega.WavePolicy},
		// Rank gaps do not widen a rank.
		{"sparse-ranks", []cudataskgraph.Task{
			{ID: "a", Scope: dev},
			{ID: "b", Deps: []string{"a"}, Rank: 5, Scope: dev},
		}, cudamega.WavePolicy},
		{"two-sources", []cudataskgraph.Task{
			{ID: "a", Scope: dev},
			{ID: "b", Scope: dev},
		}, cudamega.CountedPolicy},
		{"fan-out-fan-in", []cudataskgraph.Task{
			{ID: "a", Scope: dev},
			{ID: "b", Deps: []string{"a"}, Rank: 1, Scope: dev},
			{ID: "c", Deps: []string{"a"}, Rank: 1, Scope: dev},
			{ID: "d", Deps: []string{"b", "c"}, Rank: 2, Scope: dev},
		}, cudamega.CountedPolicy},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g, err := cudataskgraph.New(tt.tasks)
			if err != nil {
				t.Fatal(err)
			}
			got, err := cudamega.PolicyFor(g)
			if err != nil {
				t.Fatalf("PolicyFor: %v", err)
			}
			if got != tt.want {
				t.Fatalf("PolicyFor = %s, want %s", got, tt.want)
			}
		})
	}
}

// TestPolicyForEmptyGraph checks that a graph with no shape is refused
// rather than given the zero Policy, which is WavePolicy.
func TestPolicyForEmptyGraph(t *testing.T) {
	for name, g := range map[string]cudataskgraph.Graph{
		"zero": {},
		"new":  mustGraph(t, nil),
	} {
		if p, err := cudamega.PolicyFor(g); err == nil {
			t.Errorf("%s: PolicyFor(empty) = %s, want error", name, p)
		}
	}
}

func mustGraph(t *testing.T, tasks []cudataskgraph.Task) cudataskgraph.Graph {
	t.Helper()
	g, err := cudataskgraph.New(tasks)
	if err != nil {
		t.Fatal(err)
	}
	return g
}

func TestAdmissionFromOccupancy(t *testing.T) {
	ok := cudamega.Occupancy{BlocksPerMultiprocessor: 6, Multiprocessors: 48, WorkerThreads: 256, SharedBytes: 4096}
	with := func(f func(*cudamega.Occupancy)) cudamega.Occupancy {
		o := ok
		f(&o)
		return o
	}
	tests := []struct {
		name string
		in   cudamega.Occupancy
		want cudamega.Admission
		err  string // substring of the refusal; empty means admitted
	}{
		{name: "measured", in: ok, want: cudamega.Admission{ResidentCTAs: 288, WorkerThreads: 256, SharedBytes: 4096}},
		{name: "one-block", in: cudamega.Occupancy{BlocksPerMultiprocessor: 1, Multiprocessors: 1, WorkerThreads: 32, SharedBytes: 1},
			want: cudamega.Admission{ResidentCTAs: 1, WorkerThreads: 32, SharedBytes: 1}},
		// 65535 * 65537 is exactly 2^32-1, the largest product that fits.
		{name: "largest", in: with(func(o *cudamega.Occupancy) { o.BlocksPerMultiprocessor, o.Multiprocessors = 65535, 65537 }),
			want: cudamega.Admission{ResidentCTAs: math.MaxUint32, WorkerThreads: 256, SharedBytes: 4096}},
		{name: "overflow", in: with(func(o *cudamega.Occupancy) { o.BlocksPerMultiprocessor, o.Multiprocessors = 65536, 65536 }), err: "overflows"},
		{name: "no-blocks", in: with(func(o *cudamega.Occupancy) { o.BlocksPerMultiprocessor = 0 }), err: "blocks per multiprocessor is zero"},
		{name: "no-multiprocessors", in: with(func(o *cudamega.Occupancy) { o.Multiprocessors = 0 }), err: "multiprocessor count is zero"},
		{name: "no-threads", in: with(func(o *cudamega.Occupancy) { o.WorkerThreads = 0 }), err: "worker threads is zero"},
		{name: "no-shared", in: with(func(o *cudamega.Occupancy) { o.SharedBytes = 0 }), err: "shared bytes is zero"},
		{name: "too-many-threads", in: with(func(o *cudamega.Occupancy) { o.WorkerThreads = 2048 }), err: "block thread limit"},
		{name: "unmeasured", in: cudamega.Occupancy{}, err: "is zero"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := cudamega.AdmissionFromOccupancy(tt.in)
			switch {
			case tt.err == "" && err != nil:
				t.Fatalf("AdmissionFromOccupancy: %v", err)
			case tt.err != "" && err == nil:
				t.Fatalf("AdmissionFromOccupancy(%+v) = %+v, want refusal", tt.in, got)
			case tt.err != "" && !strings.Contains(err.Error(), tt.err):
				t.Fatalf("AdmissionFromOccupancy: %v, want error containing %q", err, tt.err)
			case tt.err != "" && got != (cudamega.Admission{}):
				t.Fatalf("refusal returned admission %+v, want zero", got)
			case tt.err == "" && got != tt.want:
				t.Fatalf("AdmissionFromOccupancy = %+v, want %+v", got, tt.want)
			}
		})
	}
}

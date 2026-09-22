package cuptipcsampling_test

import (
	"slices"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaprof"
	"github.com/tmc/lib/nvidia/x/cuptipcsampling"
)

// TestStallSampleTypes checks the published set against cudaprof's own
// declaration rather than against a second copy of the list. A list and a test
// of that list written together agree by construction and cannot notice a type
// neither mentions, which is how stall_other came to be counted here and
// absent from every capture the examples produced.
//
// The comparison runs in both directions: a stall type cudaprof carries and
// this set omits is a column callers would never select, and one this set
// names and cudaprof does not carry is a name SetSampleTypes rejects.
func TestStallSampleTypes(t *testing.T) {
	got := cuptipcsampling.StallSampleTypes()
	var want []string
	for _, name := range cudaprof.SampleTypeNames() {
		if strings.HasPrefix(name, "stall_") {
			want = append(want, name)
		}
	}
	if !slices.Equal(got, want) {
		t.Errorf("StallSampleTypes() = %v, want %v", got, want)
	}
	// The set publishes what the classifier fills and nothing else. The values
	// RecordBatch writes beside the stalls are named in the doc comment, not
	// returned, because a caller that records launches adds types this package
	// knows nothing about and one that does not must not declare them.
	for _, unwanted := range []string{"samples", "pc_samples", "inst_executed", "gpu_time", "modeled_cost"} {
		if slices.Contains(got, unwanted) {
			t.Errorf("StallSampleTypes() = %v, want no %q", got, unwanted)
		}
	}
}

// TestStallSampleTypesAreSelectable pins that the set is usable as it stands:
// SetSampleTypes rejects an unknown or duplicate name, so this fails if the
// set drifts from cudaprof's vocabulary in either way.
func TestStallSampleTypesAreSelectable(t *testing.T) {
	prof := cudaprof.NewSession(nil)
	if err := prof.SetSampleTypes(append(cuptipcsampling.StallSampleTypes(), "pc_samples", "inst_executed")...); err != nil {
		t.Fatalf("SetSampleTypes: %v", err)
	}
}

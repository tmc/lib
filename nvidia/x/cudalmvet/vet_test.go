package cudalmvet_test

import (
	"testing"

	"github.com/tmc/lib/nvidia/x/cudalmvet"
)

func TestVetLayer(t *testing.T) {
	got := []float32{1.0, 2.0, 3.0, 4.0}
	want := []float32{1.0001, 2.0001, 2.9999, 4.0}

	diff, err := cudalmvet.VetLayer("layer0_norm", got, want, 1e-3)
	if err != nil {
		t.Fatalf("VetLayer failed: %v", err)
	}

	if !diff.Passed {
		t.Errorf("expected layer to pass with tol 1e-3, got max diff %g", diff.MaxAbsDiff)
	}
	if diff.CosineSim < 0.9999 {
		t.Errorf("expected cosine similarity close to 1.0, got %g", diff.CosineSim)
	}

	report := cudalmvet.NewReport(*diff)
	if report.TotalPassed != 1 || report.TotalFailed != 0 {
		t.Errorf("unexpected report counts: %+v", report)
	}
}

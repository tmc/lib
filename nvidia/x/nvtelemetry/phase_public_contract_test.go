package nvtelemetry_test

import (
	"errors"
	"testing"

	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

func TestPhasePublicHistoryWitness(t *testing.T) {
	c := nvtelemetry.NewCollector("public-phase-bound", nvtelemetry.NewFakeSampler(), nvtelemetry.NewFakeClock(1, 1), nvtelemetry.CollectorConfig{MaxRecordBudget: 2}, nil)
	for _, name := range []string{"a", "b"} {
		if err := c.SetPhase(name); err != nil {
			t.Fatal(err)
		}
	}
	if err := c.SetPhase("c"); !errors.Is(err, nvtelemetry.ErrBudgetExceeded) {
		t.Fatalf("public phase cap escaped: %v", err)
	}
	m, err := c.Stop()
	if err != nil || !m.IsComplete || len(m.Phases) != 2 || m.Phases[0].Name != "a" || m.Phases[1].Name != "b" {
		t.Fatalf("public phase cap boundary: %+v %v", m, err)
	}
}

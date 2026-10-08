package cudasanitizer

import (
	"context"
	"testing"
)

func TestGatedRunner_BypassesSanitizerOnDuplicateTrace(t *testing.T) {
	cfg := Config{
		Tool:        Memcheck,
		MissingTool: MissingToolSkip,
		Gating:      GatingCoverageIncrease,
	}
	runner, err := NewGatedRunner(cfg, 32, 100)
	if err != nil {
		t.Fatal(err)
	}

	dispatches := 0
	trial := GatedTrial{
		Name:  "trial1",
		Trace: []byte{0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		SanitizeRun: func(ctx context.Context) (*Result, error) {
			dispatches++
			return &Result{Outcome: OutcomeClean}, nil
		},
	}

	// First execution: trace is novel, sanitizer must be dispatched.
	res1, err := runner.RunTrial(context.Background(), trial)
	if err != nil {
		t.Fatal(err)
	}
	if !res1.Novel || !res1.SanitizerDispatched || dispatches != 1 {
		t.Fatalf("first run should dispatch sanitizer: %+v, dispatches=%d", res1, dispatches)
	}

	// Second execution with duplicate trace: sanitizer must be BYPASSED.
	res2, err := runner.RunTrial(context.Background(), trial)
	if err != nil {
		t.Fatal(err)
	}
	if res2.Novel || res2.SanitizerDispatched || dispatches != 1 {
		t.Fatalf("duplicate trace must bypass sanitizer: %+v, dispatches=%d", res2, dispatches)
	}
}

func TestGatedRunner_DispatchesSanitizerOnNewEdge(t *testing.T) {
	cfg := Config{
		Tool:        Memcheck,
		MissingTool: MissingToolSkip,
		Gating:      GatingCoverageIncrease,
	}
	runner, err := NewGatedRunner(cfg, 32, 100)
	if err != nil {
		t.Fatal(err)
	}

	dispatches := 0
	makeTrial := func(name string, edgeIndex int) GatedTrial {
		trace := make([]byte, 32)
		trace[edgeIndex] = 1
		return GatedTrial{
			Name:  name,
			Trace: trace,
			SanitizeRun: func(ctx context.Context) (*Result, error) {
				dispatches++
				return &Result{Outcome: OutcomeClean}, nil
			},
		}
	}

	// Trial 1: edge 1 -> dispatched
	res1, err := runner.RunTrial(context.Background(), makeTrial("trial1", 1))
	if err != nil || !res1.SanitizerDispatched || dispatches != 1 {
		t.Fatalf("trial 1 failed: %+v, dispatches=%d", res1, dispatches)
	}

	// Trial 2: edge 1 (duplicate) -> bypassed
	res2, err := runner.RunTrial(context.Background(), makeTrial("trial2", 1))
	if err != nil || res2.SanitizerDispatched || dispatches != 1 {
		t.Fatalf("trial 2 should be bypassed: %+v, dispatches=%d", res2, dispatches)
	}

	// Trial 3: edge 2 (new edge) -> dispatched!
	res3, err := runner.RunTrial(context.Background(), makeTrial("trial3", 2))
	if err != nil || !res3.SanitizerDispatched || dispatches != 2 {
		t.Fatalf("trial 3 with new edge must dispatch: %+v, dispatches=%d", res3, dispatches)
	}
}

func TestHardwareStatus_ReportsPendingWithoutDevice(t *testing.T) {
	// StatusPendingHardware is reported when no physical GPU is present.
	status := CheckHardwareStatus()
	if status != StatusPendingHardware && status != StatusHardwareAvailable {
		t.Fatalf("unexpected hardware status: %v", status)
	}
}

func TestControl_FakedHardwareCoverageRejected(t *testing.T) {
	fakeCoverage := []byte{1, 2, 3, 4}
	err := ValidateHardwareCoverage(fakeCoverage, StatusPendingHardware)
	if err == nil {
		t.Fatal("expected error validating hardware coverage with StatusPendingHardware")
	}
}

func TestControl_MismatchedMapSizePanics(t *testing.T) {
	g, err := NewTraceNovelty(GatingUniqueTrace, 32, 10)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		r := recover()
		if r == nil {
			t.Fatal("expected panic on mismatched map size, got nil")
		}
	}()
	g.MustEvaluate(make([]byte, 64))
}


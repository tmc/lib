package cudadiagdemo

import (
	"context"
	"fmt"
	"math"
	"sync"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudaflightrecorder"
	"github.com/tmc/lib/nvidia/x/cudaworkload"
)

func newTestFixture(t *testing.T) *ServiceFixture {
	t.Helper()
	proc := cudaworkload.ProcessGenID{PID: 10001, Generation: 1}
	mockHook := NewCorrelationHookMock()
	annotator := cudaworkload.NewAnnotator(proc, "test-workload",
		cudaworkload.WithRunID("test-run"),
		cudaworkload.WithCorrelationHook(mockHook),
	)

	rec, err := cudaflightrecorder.New(cudaflightrecorder.Config{
		Window:   5 * time.Second,
		Capacity: 4096,
	})
	if err != nil {
		t.Fatalf("new recorder: %v", err)
	}
	if err := rec.Start(); err != nil {
		t.Fatalf("start recorder: %v", err)
	}
	t.Cleanup(func() { rec.Stop() })

	budgets := cudaflightrecorder.QualificationBudgets()
	coord, err := cudaflightrecorder.NewEscalationCoordinator(rec, budgets)
	if err != nil {
		t.Fatalf("new coordinator: %v", err)
	}

	joiner := cudaworkload.NewJoiner(cudaworkload.JoinConfig{MaxPendingRecords: 1024})

	sf, err := NewServiceFixture(annotator, rec, coord, joiner, WithProcess(proc))
	if err != nil {
		t.Fatalf("new fixture: %v", err)
	}
	return sf
}

func TestServiceFixture_NumericalCorrectness(t *testing.T) {
	input := []float64{1.0, 2.5, -3.0, 4.25, 0.0}
	weight := 2.0
	bias := 1.0

	// Analytical oracle: y = 2*x + 1
	expected := []float64{3.0, 6.0, -5.0, 9.5, 1.0}
	actual := ComputeReferenceOutput(input, weight, bias)
	response, err := newTestFixture(t).Process(context.Background(), Request{ID: "oracle", Data: input, Mode: ModeBaseline})
	if err != nil {
		t.Fatal(err)
	}
	if err := CheckNumericalCorrectness(expected, response.Output, 1e-6); err != nil {
		t.Fatal(err)
	}

	if err := CheckNumericalCorrectness(expected, actual, 1e-6); err != nil {
		t.Fatalf("numerical correctness failed: %v", err)
	}

	// Negative control / Mutant check: mutant with subtraction instead of addition
	mutant := []float64{1.0, 4.0, -7.0, 7.5, -1.0} // y = 2*x - 1
	if err := CheckNumericalCorrectness(expected, mutant, 1e-6); err == nil {
		t.Fatal("expected mutant output to fail numerical correctness check, but it passed")
	}

	// NaN check
	nanCand := []float64{3.0, math.NaN(), -5.0, 9.5, 1.0}
	if err := CheckNumericalCorrectness(expected, nanCand, 1e-6); err == nil {
		t.Fatal("expected NaN to fail numerical correctness check")
	}

	// Length mismatch
	if err := CheckNumericalCorrectness(expected, actual[:3], 1e-6); err == nil {
		t.Fatal("expected length mismatch to fail")
	}
}

func TestServiceFixture_HostDelayIsolation(t *testing.T) {
	sf := newTestFixture(t)
	ctx := context.Background()

	injectedDelay := 20 * time.Millisecond
	req := Request{
		ID:           "req-delay-1",
		WorkloadName: "test-isolation",
		Data:         []float64{1.0, 2.0, 3.0},
		Mode:         ModeHostSubmissionDelay,
		Delay:        injectedDelay,
	}

	start := time.Now()
	resp, err := sf.Process(ctx, req)
	elapsed := time.Since(start)

	if err != nil {
		t.Fatalf("process failed: %v", err)
	}
	if elapsed < injectedDelay {
		t.Fatalf("expected elapsed %v >= injected delay %v", elapsed, injectedDelay)
	}

	gt := resp.GroundTruth
	observation := resp.Observation
	if gt.HostPrepDelay != time.Duration(observation.PreEntry.Duration()) || gt.HostAPIDuration != time.Duration(observation.API.Span.Duration()) || observation.PreEntry.End != observation.API.Span.Start || observation.Service.End < observation.API.Span.End {
		t.Fatal("observed phases not retained separately")
	}
	// Wall durations include scheduler pauses; do not use host-load-dependent
	// upper bounds as a numerical or isolation oracle.
	if gt.HostPrepDelay+gt.HostAPIDuration+gt.KernelDuration > gt.TotalDuration {
		t.Fatal("phase durations exceed total duration")
	}

}

func TestServiceFixture_DeviceBottleneck(t *testing.T) {
	sf := newTestFixture(t)
	ctx := context.Background()

	deviceDelay := 300 * time.Microsecond
	req := Request{
		ID:           "req-device-1",
		WorkloadName: "test-device",
		Data:         []float64{1.0, 2.0, 3.0},
		Mode:         ModeDeviceBottleneck,
		Delay:        deviceDelay,
	}

	resp, err := sf.Process(ctx, req)
	if err != nil {
		t.Fatalf("process failed: %v", err)
	}

	gt := resp.GroundTruth
	if gt.KernelDuration < deviceDelay {
		t.Fatalf("expected kernel duration >= %v, got %v", deviceDelay, gt.KernelDuration)
	}

}

func TestServiceFixture_ConcurrentRequests(t *testing.T) {
	sf := newTestFixture(t)
	ctx := context.Background()

	numReqs := 24
	concurrency := 4
	reqs := make([]Request, numReqs)
	for i := 0; i < numReqs; i++ {
		// All requests share the same workload name to test disambiguation under concurrent same-name requests
		reqs[i] = Request{
			ID:           fmt.Sprintf("req-same-name-%03d", i),
			WorkloadName: "concurrent-batch-gemm",
			Data:         []float64{float64(i), float64(i) * 2},
			Mode:         ModeBaseline,
		}
	}

	resps, err := sf.BatchProcess(ctx, reqs, concurrency)
	if err != nil {
		t.Fatalf("batch process failed: %v", err)
	}
	if len(resps) != numReqs {
		t.Fatalf("expected %d responses, got %d", numReqs, len(resps))
	}

	seenOpIDs := make(map[cudaworkload.OpID]bool)
	for i, r := range resps {
		if r.Error != nil {
			t.Fatalf("request %d failed: %v", i, r.Error)
		}
		if r.OpID == 0 {
			t.Fatalf("request %d has unassigned OpID", i)
		}
		if seenOpIDs[r.OpID] {
			t.Fatalf("duplicate OpID %d encountered across concurrent requests", r.OpID)
		}
		seenOpIDs[r.OpID] = true

		expected := ComputeReferenceOutput(reqs[i].Data, sf.weight, sf.bias)
		if err := CheckNumericalCorrectness(expected, r.Output, 1e-6); err != nil {
			t.Fatalf("numerical check failed for request %d: %v", i, err)
		}
	}

	// Verify Joiner metrics: no ambiguous or false exact joins
	metrics := sf.joiner.Metrics()
	if metrics.ExactJoins == 0 {
		t.Fatalf("expected exact joins, got %d", metrics.ExactJoins)
	}
	if metrics.AmbiguousJoins > 0 {
		t.Fatalf("expected 0 ambiguous joins, got %d", metrics.AmbiguousJoins)
	}
	if metrics.Precision() < 1.0 {
		t.Fatalf("expected precision 1.0, got %v", metrics.Precision())
	}
}

func TestCorrelationHookMock(t *testing.T) {
	hook := &CorrelationHookMock{}
	var wg sync.WaitGroup

	for i := 0; i < 5; i++ {
		wg.Add(1)
		go func(id uint64) {
			defer wg.Done()
			_ = hook.Push(0, id)
			_, _ = hook.Pop(0)
		}(uint64(i + 1))
	}
	wg.Wait()
}

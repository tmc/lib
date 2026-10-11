package nvtelemetry_test

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

type mockSampler struct {
	mu       sync.Mutex
	devices  []nvtelemetry.DeviceMetadata
	batches  [][]nvtelemetry.Reading
	step     int
	queryDur time.Duration
	clock    *nvtelemetry.FakeClock
}

func (m *mockSampler) Devices() []nvtelemetry.DeviceMetadata {
	return m.devices
}

func (m *mockSampler) Sample(ctx context.Context) ([]nvtelemetry.Reading, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.clock != nil && m.queryDur > 0 {
		m.clock.Advance(m.queryDur)
	}
	if m.step < len(m.batches) {
		b := m.batches[m.step]
		m.step++
		return b, nil
	}
	return nil, nil
}

func TestCollectorLifecycleWithFakeClock(t *testing.T) {
	clock := nvtelemetry.NewFakeClock(1_000_000_000, 1_700_000_000_000_000_000)
	dev := nvtelemetry.DeviceMetadata{UUID: "GPU-1", Model: "L4"}
	sampler := &mockSampler{
		devices: []nvtelemetry.DeviceMetadata{dev},
		clock:   clock,
		batches: [][]nvtelemetry.Reading{
			{{Name: nvtelemetry.MetricGPUUtilization, Unit: nvtelemetry.UnitPercent, DeviceUUID: "GPU-1", Value: 10, Evidence: nvtelemetry.EvidenceMeasured}},
			{{Name: nvtelemetry.MetricGPUUtilization, Unit: nvtelemetry.UnitPercent, DeviceUUID: "GPU-1", Value: 20, Evidence: nvtelemetry.EvidenceMeasured}},
			{{Name: nvtelemetry.MetricGPUUtilization, Unit: nvtelemetry.UnitPercent, DeviceUUID: "GPU-1", Value: 30, Evidence: nvtelemetry.EvidenceMeasured}},
		},
	}

	var buf bytes.Buffer
	cfg := nvtelemetry.CollectorConfig{
		IntervalNanos: int64(100 * time.Millisecond),
	}
	w := nvtelemetry.NewWriter(&buf, cfg)
	c := nvtelemetry.NewCollector("run-1", sampler, clock, cfg, w)

	ctx := context.Background()

	if err := c.InitStream(); err != nil {
		t.Fatalf("InitStream failed: %v", err)
	}

	// Initial step in warmup
	c.SetPhase(nvtelemetry.PhaseWarmup)
	if err := c.Step(ctx); err != nil {
		t.Fatalf("Step 1 failed: %v", err)
	}
	clock.Advance(100 * time.Millisecond)

	// Step in measurement
	c.SetPhase(nvtelemetry.PhaseMeasurement)
	if err := c.Step(ctx); err != nil {
		t.Fatalf("Step 2 failed: %v", err)
	}
	clock.Advance(100 * time.Millisecond)

	// Step in cooldown
	c.SetPhase(nvtelemetry.PhaseCooldown)
	if err := c.Step(ctx); err != nil {
		t.Fatalf("Step 3 failed: %v", err)
	}
	clock.Advance(100 * time.Millisecond)

	samples := c.Samples()
	if len(samples) != 3 {
		t.Fatalf("len(samples) = %d, want 3", len(samples))
	}

	manifest, phases, readSamples, err := nvtelemetry.ReadStream(&buf, 0)
	if err != nil {
		t.Fatalf("ReadStream failed: %v", err)
	}

	if manifest.RunID != "run-1" {
		t.Errorf("manifest.RunID = %s, want run-1", manifest.RunID)
	}
	if len(phases) < 2 {
		t.Errorf("len(phases) = %d, want >= 2", len(phases))
	}
	if len(readSamples) != 3 {
		t.Errorf("len(readSamples) = %d, want 3", len(readSamples))
	}
}

func TestCollectorMissedTicks(t *testing.T) {
	clock := nvtelemetry.NewFakeClock(1_000_000_000, 1_700_000_000_000_000_000)
	dev := nvtelemetry.DeviceMetadata{UUID: "GPU-1"}
	sampler := &mockSampler{
		devices:  []nvtelemetry.DeviceMetadata{dev},
		clock:    clock,
		queryDur: 350 * time.Millisecond, // 3.5 intervals
		batches: [][]nvtelemetry.Reading{
			{{Name: nvtelemetry.MetricGPUUtilization, Value: 50, Evidence: nvtelemetry.EvidenceMeasured}},
		},
	}

	cfg := nvtelemetry.CollectorConfig{
		IntervalNanos: int64(100 * time.Millisecond),
	}
	c := nvtelemetry.NewCollector("run-missed", sampler, clock, cfg, nil)

	if err := c.Step(context.Background()); err != nil {
		t.Fatalf("Step failed: %v", err)
	}

	samples := c.Samples()
	if len(samples) != 1 {
		t.Fatalf("len(samples) = %d, want 1", len(samples))
	}
	if samples[0].LagNanos != int64(350*time.Millisecond) {
		t.Errorf("LagNanos = %d, want %d", samples[0].LagNanos, 350*time.Millisecond)
	}
}

func TestCollectorBudgetExceededFail(t *testing.T) {
	var buf bytes.Buffer
	cfg := nvtelemetry.CollectorConfig{
		MaxRecordBudget: 2,
		DropOnFull:      false,
	}
	w := nvtelemetry.NewWriter(&buf, cfg)

	s1 := &nvtelemetry.Sample{Seq: 1}
	s2 := &nvtelemetry.Sample{Seq: 2}
	s3 := &nvtelemetry.Sample{Seq: 3}

	if err := w.WriteSample(s1); err != nil {
		t.Fatalf("WriteSample 1 failed: %v", err)
	}
	if err := w.WriteSample(s2); err != nil {
		t.Fatalf("WriteSample 2 failed: %v", err)
	}
	if err := w.WriteSample(s3); !errors.Is(err, nvtelemetry.ErrBudgetExceeded) {
		t.Fatalf("WriteSample 3 err = %v, want ErrBudgetExceeded", err)
	}
}

func TestCollectorBudgetExceededDrop(t *testing.T) {
	var buf bytes.Buffer
	cfg := nvtelemetry.CollectorConfig{
		MaxRecordBudget: 2,
		DropOnFull:      true,
	}
	w := nvtelemetry.NewWriter(&buf, cfg)

	s1 := &nvtelemetry.Sample{Seq: 1}
	s2 := &nvtelemetry.Sample{Seq: 2}
	s3 := &nvtelemetry.Sample{Seq: 3}

	_ = w.WriteSample(s1)
	_ = w.WriteSample(s2)
	_ = w.WriteSample(s3)

	if w.LostRecords() != 1 {
		t.Errorf("LostRecords() = %d, want 1", w.LostRecords())
	}
}

func TestCollectorCompletionManifestPopulatesEndFields(t *testing.T) {
	clock := nvtelemetry.NewFakeClock(1_000_000_000, 1_700_000_000_000_000_000)
	dev := nvtelemetry.DeviceMetadata{UUID: "GPU-1"}
	sampler := &mockSampler{
		devices: []nvtelemetry.DeviceMetadata{dev},
		clock:   clock,
		batches: [][]nvtelemetry.Reading{
			{{Name: nvtelemetry.MetricGPUUtilization, Unit: nvtelemetry.UnitPercent, DeviceUUID: "GPU-1", Value: 10, Evidence: nvtelemetry.EvidenceMeasured}},
		},
	}

	var buf bytes.Buffer
	cfg := nvtelemetry.CollectorConfig{
		IntervalNanos: int64(100 * time.Millisecond),
	}
	w := nvtelemetry.NewWriter(&buf, cfg)
	c := nvtelemetry.NewCollector("run-complete", sampler, clock, cfg, w)

	if err := c.InitStream(); err != nil {
		t.Fatal(err)
	}
	_ = c.SetPhase(nvtelemetry.PhaseWarmup)
	_ = c.Step(context.Background())
	clock.Advance(50 * time.Millisecond)
	_ = c.SetPhase(nvtelemetry.PhaseMeasurement)
	clock.Advance(100 * time.Millisecond)

	m, err := c.Stop()
	if err != nil {
		t.Fatalf("Stop failed: %v", err)
	}

	if m.EndTimeMonotonic <= m.StartTimeMonotonic {
		t.Errorf("EndTimeMonotonic %d <= StartTimeMonotonic %d", m.EndTimeMonotonic, m.StartTimeMonotonic)
	}
	if m.EndTimeUnix <= 0 {
		t.Errorf("EndTimeUnix = %d, want > 0", m.EndTimeUnix)
	}
	if len(m.Phases) != 2 {
		t.Fatalf("len(m.Phases) = %d, want 2", len(m.Phases))
	}
	finalPhase := m.Phases[1]
	if finalPhase.Name != nvtelemetry.PhaseMeasurement {
		t.Errorf("final phase name = %s, want measurement", finalPhase.Name)
	}
	if finalPhase.EndMonotonicNanos <= finalPhase.StartMonotonicNanos {
		t.Errorf("final phase end %d <= start %d", finalPhase.EndMonotonicNanos, finalPhase.StartMonotonicNanos)
	}

	// Verify stream read sees the final updated completeness manifest
	streamManifest, phases, _, err := nvtelemetry.ReadStream(&buf, 0)
	if err != nil {
		t.Fatalf("ReadStream error: %v", err)
	}
	if streamManifest.EndTimeMonotonic <= 0 {
		t.Errorf("streamManifest.EndTimeMonotonic = %d, want > 0", streamManifest.EndTimeMonotonic)
	}
	if len(phases) != 2 {
		t.Errorf("len(phases) in stream = %d, want 2", len(phases))
	}
}

func TestCollectorBoundedBufferOnDrop(t *testing.T) {
	clock := nvtelemetry.NewFakeClock(1_000_000_000, 1_700_000_000_000_000_000)
	sampler := &mockSampler{
		devices: []nvtelemetry.DeviceMetadata{{UUID: "GPU-1"}},
		batches: [][]nvtelemetry.Reading{
			{{Name: nvtelemetry.MetricGPUUtilization, Value: 10, Evidence: nvtelemetry.EvidenceMeasured}},
			{{Name: nvtelemetry.MetricGPUUtilization, Value: 20, Evidence: nvtelemetry.EvidenceMeasured}},
			{{Name: nvtelemetry.MetricGPUUtilization, Value: 30, Evidence: nvtelemetry.EvidenceMeasured}},
			{{Name: nvtelemetry.MetricGPUUtilization, Value: 40, Evidence: nvtelemetry.EvidenceMeasured}},
			{{Name: nvtelemetry.MetricGPUUtilization, Value: 50, Evidence: nvtelemetry.EvidenceMeasured}},
		},
	}

	cfg := nvtelemetry.CollectorConfig{
		MaxBufferedRecords: 2,
		BudgetPolicy:       nvtelemetry.BudgetExceededDrop,
		DropOnFull:         true,
	}
	c := nvtelemetry.NewCollector("run-bounded", sampler, clock, cfg, nil)

	ctx := context.Background()
	for i := 0; i < 5; i++ {
		_ = c.Step(ctx)
		clock.Advance(10 * time.Millisecond)
	}

	samples := c.Samples()
	if len(samples) != 2 {
		t.Fatalf("len(samples) = %d, want exactly 2 (buffer must remain bounded)", len(samples))
	}
	m := c.Manifest()
	if m.LostRecords != 3 {
		t.Errorf("m.LostRecords = %d, want 3", m.LostRecords)
	}
}

type failWriter struct{}

func (failWriter) Write(p []byte) (int, error) {
	return 0, errors.New("write failure")
}

func TestCollectorPhaseWriteFailureDetected(t *testing.T) {
	clock := nvtelemetry.NewFakeClock(1_000_000_000, 1_700_000_000_000_000_000)
	w := nvtelemetry.NewWriter(failWriter{}, nvtelemetry.CollectorConfig{})
	c := nvtelemetry.NewCollector("run-err", &mockSampler{}, clock, nvtelemetry.CollectorConfig{}, w)

	_ = c.SetPhase("phase-1")
	clock.Advance(100)
	err := c.SetPhase("phase-2")
	if err == nil {
		t.Fatalf("expected error from SetPhase when writer fails, got nil")
	}
}

func TestCollectorInitStreamFailureDoesNotDeadlockStop(t *testing.T) {
	clock := nvtelemetry.NewFakeClock(1_000_000_000, 1_700_000_000_000_000_000)
	w := nvtelemetry.NewWriter(failWriter{}, nvtelemetry.CollectorConfig{})
	c := nvtelemetry.NewCollector("run-deadlock-test", &mockSampler{}, clock, nvtelemetry.CollectorConfig{}, w)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	_, err := c.Run(ctx)
	if err == nil {
		t.Fatalf("expected Run to fail on InitStream with failWriter, got nil")
	}

	// Must NOT deadlock!
	done := make(chan struct{})
	go func() {
		_, _ = c.Stop()
		close(done)
	}()

	select {
	case <-done:
		// Success! Stop did not deadlock.
	case <-time.After(2 * time.Second):
		t.Fatalf("Stop deadlocked after InitStream failure")
	}
}

type failAfterNWriter struct {
	mu    sync.Mutex
	count int
	n     int
}

func (f *failAfterNWriter) Write(p []byte) (int, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.count++
	if f.count > f.n {
		return 0, errors.New("failing write")
	}
	return len(p), nil
}

func TestCollectorBufferDropPropagatesWriterError(t *testing.T) {
	clock := nvtelemetry.NewFakeClock(1_000_000_000, 1_700_000_000_000_000_000)
	sampler := &mockSampler{
		devices: []nvtelemetry.DeviceMetadata{{UUID: "GPU-1"}},
		batches: [][]nvtelemetry.Reading{
			{{Name: nvtelemetry.MetricGPUUtilization, Unit: nvtelemetry.UnitPercent, DeviceUUID: "GPU-1", Value: 10, Evidence: nvtelemetry.EvidenceMeasured}},
			{{Name: nvtelemetry.MetricGPUUtilization, Unit: nvtelemetry.UnitPercent, DeviceUUID: "GPU-1", Value: 20, Evidence: nvtelemetry.EvidenceMeasured}},
		},
	}

	// Allows 1 write (the first sample), then fails on the second write during buffer drop
	fw := &failAfterNWriter{n: 1}
	w := nvtelemetry.NewWriter(fw, nvtelemetry.CollectorConfig{})
	cfg := nvtelemetry.CollectorConfig{
		MaxBufferedRecords: 1, // Fills on first sample
		DropOnFull:         true,
	}
	c := nvtelemetry.NewCollector("run-drop-err", sampler, clock, cfg, w)

	ctx := context.Background()
	if err := c.Step(ctx); err != nil {
		t.Fatalf("Step 1 failed: %v", err)
	}

	// Step 2 triggers buffer drop path. Writer should fail and error must be returned!
	err := c.Step(ctx)
	if err == nil {
		t.Fatalf("expected error from Step 2 during buffer-drop write failure, got nil")
	}
}

func TestCollectorCancellationCapturesFinishError(t *testing.T) {
	clock := nvtelemetry.NewFakeClock(1_000_000_000, 1_700_000_000_000_000_000)
	sampler := &mockSampler{
		devices: []nvtelemetry.DeviceMetadata{{UUID: "GPU-1"}},
		batches: [][]nvtelemetry.Reading{
			{{Name: nvtelemetry.MetricGPUUtilization, Unit: nvtelemetry.UnitPercent, DeviceUUID: "GPU-1", Value: 10, Evidence: nvtelemetry.EvidenceMeasured}},
		},
	}

	// Allows 2 writes (initial manifest, 1 sample), fails on finish manifest
	fw := &failAfterNWriter{n: 2}
	w := nvtelemetry.NewWriter(fw, nvtelemetry.CollectorConfig{})
	c := nvtelemetry.NewCollector("run-finish-err", sampler, clock, nvtelemetry.CollectorConfig{IntervalNanos: int64(time.Second)}, w)

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // pre-cancel context

	_, err := c.Run(ctx)
	if err == nil {
		t.Fatalf("expected error from Run, got nil")
	}
	if !strings.Contains(err.Error(), "finish") && !strings.Contains(err.Error(), "failing write") {
		t.Errorf("expected finish write error in error message, got: %v", err)
	}
}

func TestCollectorStopIdempotent(t *testing.T) {
	clock := nvtelemetry.NewFakeClock(1_000_000_000, 1_700_000_000_000_000_000)
	sampler := &mockSampler{
		devices: []nvtelemetry.DeviceMetadata{{UUID: "GPU-1"}},
		batches: [][]nvtelemetry.Reading{
			{{Name: nvtelemetry.MetricGPUUtilization, Unit: nvtelemetry.UnitPercent, DeviceUUID: "GPU-1", Value: 10, Evidence: nvtelemetry.EvidenceMeasured}},
		},
	}

	var buf bytes.Buffer
	w := nvtelemetry.NewWriter(&buf, nvtelemetry.CollectorConfig{})
	c := nvtelemetry.NewCollector("run-idempotent", sampler, clock, nvtelemetry.CollectorConfig{}, w)

	_ = c.SetPhase("measurement")
	_ = c.Step(context.Background())

	m1, err1 := c.Stop()
	if err1 != nil {
		t.Fatalf("Stop 1 failed: %v", err1)
	}

	m2, err2 := c.Stop()
	if err2 != nil {
		t.Fatalf("Stop 2 failed: %v", err2)
	}

	if len(m1.Phases) != len(m2.Phases) {
		t.Errorf("phase count changed on second Stop: %d vs %d", len(m1.Phases), len(m2.Phases))
	}
	if m1.TotalSamples != m2.TotalSamples {
		t.Errorf("sample count changed on second Stop: %d vs %d", m1.TotalSamples, m2.TotalSamples)
	}
}

func TestCollectorConcurrentSetPhaseAndStep(t *testing.T) {
	clock := nvtelemetry.NewFakeClock(1_000_000_000, 1_700_000_000_000_000_000)
	sampler := &mockSampler{
		devices: []nvtelemetry.DeviceMetadata{{UUID: "GPU-1"}},
		batches: [][]nvtelemetry.Reading{
			{{Name: nvtelemetry.MetricGPUUtilization, Unit: nvtelemetry.UnitPercent, DeviceUUID: "GPU-1", Value: 10, Evidence: nvtelemetry.EvidenceMeasured}},
		},
	}

	var buf bytes.Buffer
	w := nvtelemetry.NewWriter(&buf, nvtelemetry.CollectorConfig{})
	c := nvtelemetry.NewCollector("run-concurrent", sampler, clock, nvtelemetry.CollectorConfig{}, w)

	var wg sync.WaitGroup
	ctx := context.Background()

	// 5 concurrent workers setting phases
	for i := 0; i < 5; i++ {
		wg.Add(1)
		phaseName := fmt.Sprintf("phase-%d", i)
		go func() {
			defer wg.Done()
			for j := 0; j < 20; j++ {
				_ = c.SetPhase(phaseName)
				clock.Advance(10 * time.Millisecond)
			}
		}()
	}

	// 5 concurrent workers stepping
	for i := 0; i < 5; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 20; j++ {
				_ = c.Step(ctx)
				clock.Advance(10 * time.Millisecond)
			}
		}()
	}

	wg.Wait()
	m, err := c.Stop()
	if err != nil {
		t.Fatalf("Stop failed: %v", err)
	}
	if m.TotalSamples == 0 {
		t.Errorf("TotalSamples = 0, want > 0")
	}
}

func TestCollectorRunRejectsRestart(t *testing.T) {
	clock := nvtelemetry.NewFakeClock(1_000_000_000, 1_700_000_000_000_000_000)
	dev := nvtelemetry.DeviceMetadata{UUID: "GPU-1", Model: "L4"}
	sampler := &mockSampler{
		devices: []nvtelemetry.DeviceMetadata{dev},
		batches: [][]nvtelemetry.Reading{
			{{Name: nvtelemetry.MetricGPUUtilization, Value: 10, Evidence: nvtelemetry.EvidenceMeasured, DeviceUUID: "GPU-1"}},
		},
	}
	var buf bytes.Buffer
	w := nvtelemetry.NewWriter(&buf, nvtelemetry.CollectorConfig{})
	c := nvtelemetry.NewCollector("run-restart", sampler, clock, nvtelemetry.CollectorConfig{}, w)

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // cancel immediately

	_, _ = c.Run(ctx)
	_, _ = c.Stop()

	// Calling Run again must be rejected, not panic or close doneCh twice
	_, err := c.Run(context.Background())
	if err == nil {
		t.Fatal("expected Run to fail on finished collector, got nil")
	}
	if !strings.Contains(err.Error(), "already finished") {
		t.Fatalf("expected error mentioning already finished, got: %v", err)
	}
}

func TestCollectorTerminalErrorRecordedInManifest(t *testing.T) {
	clock := nvtelemetry.NewFakeClock(1_000_000_000, 1_700_000_000_000_000_000)
	dev := nvtelemetry.DeviceMetadata{UUID: "GPU-1", Model: "L4"}
	expectedErr := errors.New("simulated sampler failure")
	sampler := &mockSampler{
		devices: []nvtelemetry.DeviceMetadata{dev},
		batches: [][]nvtelemetry.Reading{}, // no batches, will return nil
	}
	var buf bytes.Buffer
	w := nvtelemetry.NewWriter(&buf, nvtelemetry.CollectorConfig{
		MaxBufferedRecords: 1,
		BudgetPolicy:       nvtelemetry.BudgetPolicyFail,
	})
	c := nvtelemetry.NewCollector("run-terminal-err", sampler, clock, nvtelemetry.CollectorConfig{
		MaxBufferedRecords: 1,
		BudgetPolicy:       nvtelemetry.BudgetPolicyFail,
	}, w)

	// Step once
	_ = c.Step(context.Background())
	// Step second time causes buffer overflow with BudgetPolicyFail
	err := c.Step(context.Background())
	if !errors.Is(err, nvtelemetry.ErrBudgetExceeded) {
		t.Fatalf("expected ErrBudgetExceeded, got: %v", err)
	}

	m, stopErr := c.Stop()
	if !errors.Is(stopErr, nvtelemetry.ErrBudgetExceeded) {
		t.Fatalf("expected Stop to return terminal ErrBudgetExceeded, got: %v", stopErr)
	}
	if m.IsComplete {
		t.Error("expected IsComplete = false when terminal error occurs")
	}
	if !strings.Contains(m.IncompleteReason, "terminal_error") {
		t.Errorf("expected IncompleteReason to contain terminal_error, got: %q", m.IncompleteReason)
	}
	_ = expectedErr
}

func TestCollectorRejectsMutationAfterStop(t *testing.T) {
	c := nvtelemetry.NewCollector("run", &mockSampler{}, nvtelemetry.NewFakeClock(1, 1), nvtelemetry.CollectorConfig{}, nil)
	if _, err := c.Stop(); err != nil {
		t.Fatal(err)
	}
	if err := c.SetPhase("later"); err == nil {
		t.Fatal("phase change accepted after stop")
	}
	if err := c.Step(context.Background()); err == nil {
		t.Fatal("sample accepted after stop")
	}
}

package cuptipcsampling

import (
	"context"
	"errors"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
	"unsafe"

	"github.com/tmc/lib/nvidia/cupti"
)

func testPhaseConfig() PhaseConfig {
	return PhaseConfig{
		Config: testPublicConfig(),
		Limits: PhaseLimits{
			MaxBatches:   2,
			DrainTimeout: 500 * time.Millisecond,
		},
	}
}

// TestPhaseCaptureValidation verifies pre-flight validation of context, limits, and configuration.
func TestPhaseCaptureValidation(t *testing.T) {
	validWorkload := func(context.Context) error { return nil }
	validSync := func() error { return nil }

	tests := []struct {
		name     string
		ctx      cupti.Context
		cfg      func(*PhaseConfig)
		syncFn   func() error
		workload func(context.Context) error
		wantErr  string
	}{
		{
			name:     "zero context",
			ctx:      0,
			cfg:      func(*PhaseConfig) {},
			syncFn:   validSync,
			workload: validWorkload,
			wantErr:  "invalid context",
		},
		{
			name: "zero max batches",
			ctx:  5,
			cfg: func(c *PhaseConfig) {
				c.Limits.MaxBatches = 0
			},
			syncFn:   validSync,
			workload: validWorkload,
			wantErr:  "invalid phase limits",
		},
		{
			name: "zero drain timeout",
			ctx:  5,
			cfg: func(c *PhaseConfig) {
				c.Limits.DrainTimeout = 0
			},
			syncFn:   validSync,
			workload: validWorkload,
			wantErr:  "invalid phase limits",
		},
		{
			name:     "nil workload",
			ctx:      5,
			cfg:      func(*PhaseConfig) {},
			syncFn:   validSync,
			workload: nil,
			wantErr:  "nil workload function",
		},
		{
			name:     "nil sync",
			ctx:      5,
			cfg:      func(*PhaseConfig) {},
			syncFn:   nil,
			workload: validWorkload,
			wantErr:  "nil synchronize function",
		},
		{
			name: "max batches exceeds max reads",
			ctx:  5,
			cfg: func(c *PhaseConfig) {
				c.Config.MaxReads = 2
				c.Limits.MaxBatches = 5
			},
			syncFn:   validSync,
			workload: validWorkload,
			wantErr:  "phase max batches exceeds session max reads",
		},
		{
			name: "invalid pc sampling config",
			ctx:  5,
			cfg: func(c *PhaseConfig) {
				c.Config.Period = 35 // invalid period
			},
			syncFn:   validSync,
			workload: validWorkload,
			wantErr:  "invalid sampling configuration or limits",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := testPhaseConfig()
			tt.cfg(&cfg)
			_, err := capturePhase(context.Background(), tt.ctx, cfg, nil, tt.syncFn, tt.workload)
			if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("expected error containing %q, got %v", tt.wantErr, err)
			}
		})
	}
}

// TestPhaseCaptureLifecycle tests normal lifecycle execution:
// start -> workload -> sync -> stop -> drain -> close.
func TestPhaseCaptureLifecycle(t *testing.T) {
	var callOrder []string
	var mu sync.Mutex
	record := func(op string) {
		mu.Lock()
		defer mu.Unlock()
		callOrder = append(callOrder, op)
	}

	api := testPublicControlAPI()
	origStart := api.start
	api.start = func(p *cupti.PCSamplingStartParams) error {
		record("native_start")
		return origStart(p)
	}
	origStop := api.stop
	api.stop = func(p *cupti.PCSamplingStopParams) error {
		record("native_stop")
		return origStop(p)
	}
	origDisable := api.disable
	api.disable = func(p *cupti.PCSamplingDisableParams) error {
		record("native_disable")
		return origDisable(p)
	}

	mockStarter := func(ctx cupti.Context, cfg Config) (*Session, error) {
		return startSession(ctx, cfg, api, testPublicInventory, testPublicNames, testPublicOnePC)
	}

	cfg := testPhaseConfig()
	cfg.Config.MaxReads = 3
	cfg.Limits.MaxBatches = 2

	syncCalled := false
	syncFn := func() error {
		record("sync")
		syncCalled = true
		return nil
	}

	workloadCalled := false
	workload := func(ctx context.Context) error {
		record("workload")
		workloadCalled = true
		return nil
	}

	report, err := capturePhase(context.Background(), 5, cfg, mockStarter, syncFn, workload)
	if err != nil {
		t.Fatalf("capturePhase failed: %v", err)
	}

	if !workloadCalled || !syncCalled {
		t.Fatalf("workloadCalled=%v, syncCalled=%v", workloadCalled, syncCalled)
	}

	// Verify order: start -> workload -> sync -> stop -> disable
	expectedOrder := []string{"native_start", "workload", "sync", "native_stop", "native_disable"}
	if len(callOrder) != len(expectedOrder) {
		t.Fatalf("unexpected call order: %v, want: %v", callOrder, expectedOrder)
	}
	for i, op := range expectedOrder {
		if callOrder[i] != op {
			t.Fatalf("call order at %d: got %s, want %s", i, callOrder[i], op)
		}
	}

	if !report.Info.Closed {
		t.Fatal("expected session to be closed")
	}
	if report.BatchesDrained == 0 {
		t.Fatal("expected at least one batch drained")
	}
	if report.TotalSamples == 0 {
		t.Fatal("expected non-zero total samples")
	}
	if report.Loss() {
		t.Fatalf("expected no loss, got dropped=%d remaining=%d", report.DroppedSamples, report.RemainingSamples)
	}
}

// TestPhaseCaptureWorkloadFailure verifies that workload errors are reported separately,
// synchronization is still attempted, and session is cleanly closed without leaking resources.
func TestPhaseCaptureWorkloadFailure(t *testing.T) {
	mockStarter := func(ctx cupti.Context, cfg Config) (*Session, error) {
		return startSession(ctx, cfg, testPublicControlAPI(), testPublicInventory, testPublicNames, testPublicEmptyQuery)
	}

	cfg := testPhaseConfig()
	expectedWorkloadErr := errors.New("workload failed")

	syncAttempted := false
	syncFn := func() error {
		syncAttempted = true
		return nil
	}

	workload := func(ctx context.Context) error {
		return expectedWorkloadErr
	}

	report, err := capturePhase(context.Background(), 5, cfg, mockStarter, syncFn, workload)
	if err == nil || !errors.Is(err, expectedWorkloadErr) {
		t.Fatalf("expected error wrapping %v, got %v", expectedWorkloadErr, err)
	}

	if !syncAttempted {
		t.Fatal("expected sync to be attempted even if workload fails")
	}
	if !errors.Is(report.WorkloadErr, expectedWorkloadErr) {
		t.Fatalf("expected report.WorkloadErr = %v, got %v", expectedWorkloadErr, report.WorkloadErr)
	}
	if !report.Info.Closed {
		t.Fatal("expected session to be cleanly closed on workload failure")
	}
}

// TestPhaseCaptureCancellation verifies context cancellation is reported separately in report.Canceled.
func TestPhaseCaptureCancellation(t *testing.T) {
	mockStarter := func(ctx cupti.Context, cfg Config) (*Session, error) {
		return startSession(ctx, cfg, testPublicControlAPI(), testPublicInventory, testPublicNames, testPublicEmptyQuery)
	}

	cfg := testPhaseConfig()

	ctx, cancel := context.WithCancel(context.Background())

	workload := func(ctx context.Context) error {
		cancel()
		<-ctx.Done()
		return ctx.Err()
	}

	report, err := capturePhase(ctx, 5, cfg, mockStarter, func() error { return nil }, workload)
	if err == nil || !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context.Canceled, got %v", err)
	}

	if !report.Canceled {
		t.Fatal("expected report.Canceled = true")
	}
	if !report.Info.Closed {
		t.Fatal("expected session closed")
	}
}

// TestPhaseCaptureConsumerFailure verifies consumer errors are reported separately in report.ConsumerErr.
func TestPhaseCaptureConsumerFailure(t *testing.T) {
	mockStarter := func(ctx cupti.Context, cfg Config) (*Session, error) {
		return startSession(ctx, cfg, testPublicControlAPI(), testPublicInventory, testPublicNames, testPublicOnePC)
	}

	consumerFailure := errors.New("consumer sink write error")
	cfg := testPhaseConfig()
	cfg.Consumer = func(b Batch) error {
		return consumerFailure
	}

	report, err := capturePhase(context.Background(), 5, cfg, mockStarter, func() error { return nil }, func(context.Context) error { return nil })
	_ = err
	if !errors.Is(report.ConsumerErr, consumerFailure) {
		t.Fatalf("expected report.ConsumerErr = %v, got %v", consumerFailure, report.ConsumerErr)
	}
	if !report.Info.Closed {
		t.Fatal("expected session closed")
	}
}

// TestPhaseCaptureLossReporting tests that dropped samples and hardware overflow
// are reported separately in the PhaseReport.
func TestPhaseCaptureLossReporting(t *testing.T) {
	mockStarter := func(ctx cupti.Context, cfg Config) (*Session, error) {
		queryWithLoss := func(p *cupti.PCSamplingGetDataParams) error {
			d := (*cupti.PCSamplingData)(nativePointer(p.PcSamplingData))
			d.TotalNumPcs = 1
			d.TotalSamples = 100
			d.DroppedSamples = 25
			d.HardwareBufferFull = 1
			pcs := unsafe.Slice((*cupti.PCSamplingPCData)(nativePointer(d.PPcData)), int(d.CollectNumPcs))
			pcs[0].FunctionName = 99
			pcs[0].StallReasonCount = 1
			reasons := unsafe.Slice((*cupti.PCSamplingStallReason)(nativePointer(pcs[0].StallReason)), 1)
			reasons[0] = cupti.PCSamplingStallReason{PcSamplingStallReasonIndex: 7, Samples: 10}
			return nil
		}
		return startSession(ctx, cfg, testPublicControlAPI(), testPublicInventory, testPublicNames, queryWithLoss)
	}

	cfg := testPhaseConfig()
	report, err := capturePhase(context.Background(), 5, cfg, mockStarter, func() error { return nil }, func(context.Context) error { return nil })
	_ = err

	if report.DroppedSamples < 25 {
		t.Fatalf("expected DroppedSamples >= 25, got %d", report.DroppedSamples)
	}
	if !report.HardwareFull {
		t.Fatal("expected HardwareFull = true")
	}
	if !report.Loss() {
		t.Fatal("expected report.Loss() = true")
	}
}

// TestPhaseCaptureRemainingSamples verifies that when MaxBatches limits drainage,
// remaining PCs / samples are reported separately.
func TestPhaseCaptureRemainingSamples(t *testing.T) {
	mockStarter := func(ctx cupti.Context, cfg Config) (*Session, error) {
		queryRemaining := func(p *cupti.PCSamplingGetDataParams) error {
			d := (*cupti.PCSamplingData)(nativePointer(p.PcSamplingData))
			d.TotalNumPcs = 1
			d.TotalSamples = 50
			d.RemainingNumPcs = 120
			pcs := unsafe.Slice((*cupti.PCSamplingPCData)(nativePointer(d.PPcData)), int(d.CollectNumPcs))
			pcs[0].FunctionName = 99
			pcs[0].StallReasonCount = 1
			reasons := unsafe.Slice((*cupti.PCSamplingStallReason)(nativePointer(pcs[0].StallReason)), 1)
			reasons[0] = cupti.PCSamplingStallReason{PcSamplingStallReasonIndex: 7, Samples: 5}
			return nil
		}
		return startSession(ctx, cfg, testPublicControlAPI(), testPublicInventory, testPublicNames, queryRemaining)
	}

	cfg := testPhaseConfig()
	cfg.Limits.MaxBatches = 1
	cfg.Config.MaxReads = 2

	report, err := capturePhase(context.Background(), 5, cfg, mockStarter, func() error { return nil }, func(context.Context) error { return nil })
	if err != nil {
		t.Fatalf("capturePhase error: %v", err)
	}

	if report.RemainingPCs != 120 {
		t.Fatalf("expected RemainingPCs = 120, got %d", report.RemainingPCs)
	}
	if !report.Loss() {
		t.Fatal("expected Loss() = true due to remaining PCs")
	}
}

// TestPhaseCaptureConcurrentDrainConsumer verifies the full concurrent ring buffer drain path.
func TestPhaseCaptureConcurrentDrainConsumer(t *testing.T) {
	var queryCalls int32
	mockStarter := func(ctx cupti.Context, cfg Config) (*Session, error) {
		cfg.BufferCount = 3
		cfg.MaxReads = 10
		query := func(p *cupti.PCSamplingGetDataParams) error {
			n := atomic.AddInt32(&queryCalls, 1)
			d := (*cupti.PCSamplingData)(nativePointer(p.PcSamplingData))
			d.TotalNumPcs = 1
			d.TotalSamples = uint64(n * 10)
			pcs := unsafe.Slice((*cupti.PCSamplingPCData)(nativePointer(d.PPcData)), int(d.CollectNumPcs))
			pcs[0].FunctionName = 0x1000
			pcs[0].StallReasonCount = 1
			reasons := unsafe.Slice((*cupti.PCSamplingStallReason)(nativePointer(pcs[0].StallReason)), 1)
			reasons[0] = cupti.PCSamplingStallReason{PcSamplingStallReasonIndex: 7, Samples: 1}
			return nil
		}
		return startSession(ctx, cfg, testPublicControlAPI(), testPublicInventory, testPublicNames, query)
	}

	var consumedBatches []Batch
	var cMu sync.Mutex

	cfg := testPhaseConfig()
	cfg.Config.BufferCount = 3
	cfg.Config.MaxOutputBytes = 608
	cfg.Config.MaxNames = 10
	cfg.Config.MaxReads = 10
	cfg.Limits.MaxBatches = 8
	cfg.Consumer = func(b Batch) error {
		cMu.Lock()
		consumedBatches = append(consumedBatches, b)
		cMu.Unlock()
		return nil
	}

	workload := func(ctx context.Context) error {
		time.Sleep(10 * time.Millisecond)
		return nil
	}

	report, err := capturePhase(context.Background(), 5, cfg, mockStarter, func() error { return nil }, workload)
	if err != nil {
		t.Fatalf("capturePhase: %v", err)
	}

	if !report.Info.Closed {
		t.Fatal("expected session closed")
	}
	if report.BatchesDrained == 0 {
		t.Fatal("expected batches drained")
	}
	cMu.Lock()
	count := len(consumedBatches)
	cMu.Unlock()
	if count == 0 {
		t.Fatal("expected consumer to receive batches")
	}
}

// TestPhaseCaptureNoPrematureTeardown verifies resources are not destroyed while
// collection or drain consumer is running.
func TestPhaseCaptureNoPrematureTeardown(t *testing.T) {
	sessionDestroyed := false
	consumerActive := false
	var mu sync.Mutex

	mockStarter := func(ctx cupti.Context, cfg Config) (*Session, error) {
		cfg.BufferCount = 3
		cfg.MaxReads = 10
		api := testPublicControlAPI()
		origDisable := api.disable
		api.disable = func(p *cupti.PCSamplingDisableParams) error {
			mu.Lock()
			if consumerActive {
				mu.Unlock()
				t.Error("native disable called while consumer was still active")
				return errors.New("premature teardown")
			}
			sessionDestroyed = true
			mu.Unlock()
			return origDisable(p)
		}
		return startSession(ctx, cfg, api, testPublicInventory, testPublicNames, testPublicOnePC)
	}

	cfg := testPhaseConfig()
	cfg.Config.BufferCount = 3
	cfg.Config.MaxOutputBytes = 608
	cfg.Config.MaxNames = 10
	cfg.Config.MaxReads = 10
	cfg.Limits.MaxBatches = 5
	cfg.Consumer = func(b Batch) error {
		mu.Lock()
		consumerActive = true
		mu.Unlock()
		time.Sleep(5 * time.Millisecond)
		mu.Lock()
		consumerActive = false
		mu.Unlock()
		return nil
	}

	report, err := capturePhase(context.Background(), 5, cfg, mockStarter, func() error { return nil }, func(context.Context) error {
		return nil
	})
	if err != nil {
		t.Fatalf("capturePhase error: %v", err)
	}
	mu.Lock()
	destroyed := sessionDestroyed
	mu.Unlock()
	if !destroyed {
		t.Fatal("expected session to be cleanly destroyed at end of phase")
	}
	if !report.Info.Closed {
		t.Fatal("expected report.Info.Closed = true")
	}
}

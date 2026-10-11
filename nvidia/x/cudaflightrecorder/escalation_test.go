package cudaflightrecorder_test

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"sync"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudaerrors"
	"github.com/tmc/lib/nvidia/x/cudaflightrecorder"
)

func TestBudgetsEnforced(t *testing.T) {
	budgets := cudaflightrecorder.QualificationBudgets()
	if budgets.MaxRingRecords != 65536 {
		t.Fatalf("expected 65536 ring records, got %d", budgets.MaxRingRecords)
	}
	if budgets.MaxRetainedBytes != 64<<20 {
		t.Fatalf("expected 64MB retained bytes, got %d", budgets.MaxRetainedBytes)
	}
	if budgets.MaxCaptureDuration != 5*time.Second {
		t.Fatalf("expected 5s max capture duration, got %v", budgets.MaxCaptureDuration)
	}
	if budgets.MaxConcurrentCaptures != 1 {
		t.Fatalf("expected 1 max concurrent capture, got %d", budgets.MaxConcurrentCaptures)
	}
	if budgets.MaxCPUPercent != 10.0 {
		t.Fatalf("expected 10.0 max CPU percent, got %f", budgets.MaxCPUPercent)
	}
}

func TestStateMachineLifecycle(t *testing.T) {
	rec, err := cudaflightrecorder.New(cudaflightrecorder.Config{
		Window:   5 * time.Second,
		Capacity: 100,
	})
	if err != nil {
		t.Fatalf("New recorder: %v", err)
	}
	if err := rec.Start(); err != nil {
		t.Fatalf("Start recorder: %v", err)
	}
	defer rec.Stop()

	fakeClock := cudaflightrecorder.NewFakeClock(time.Date(2026, 10, 10, 12, 0, 0, 0, time.UTC))
	budgets := cudaflightrecorder.QualificationBudgets()
	budgets.HysteresisCount = 1 // Single trigger for lifecycle test
	budgets.CooldownDuration = 2 * time.Second

	coord, err := cudaflightrecorder.NewEscalationCoordinator(rec, budgets,
		cudaflightrecorder.WithResourceSampler(&cudaflightrecorder.FakeResourceSampler{}),
		cudaflightrecorder.WithClock(fakeClock),
	)
	if err != nil {
		t.Fatalf("New coordinator: %v", err)
	}

	// 1. Initial state is Idle
	if state := coord.State(); state != cudaflightrecorder.StateIdle {
		t.Fatalf("expected initial state Idle, got %s", state)
	}

	// 2. Trigger while Idle is refused
	_, err = coord.Trigger(context.Background(), cudaflightrecorder.TriggerRequest{
		Reason: "test_idle",
	})
	if !errors.Is(err, cudaerrors.ErrUnsupported) {
		t.Fatalf("expected refusal wrapping ErrUnsupported while idle, got: %v", err)
	}

	// 3. Arm
	if err := coord.Arm(); err != nil {
		t.Fatalf("Arm: %v", err)
	}
	if state := coord.State(); state != cudaflightrecorder.StateArmed {
		t.Fatalf("expected state Armed, got %s", state)
	}

	// 4. Trigger -> captures and enters Cooldown
	receipt, err := coord.Trigger(context.Background(), cudaflightrecorder.TriggerRequest{
		Reason:      "latency_spike",
		MetricName:  "gpu_latency_ms",
		MetricValue: 120.0,
		Threshold:   100.0,
	})
	if err != nil {
		t.Fatalf("Trigger failed: %v", err)
	}
	if receipt.Outcome != "captured" {
		t.Fatalf("expected outcome captured, got %s", receipt.Outcome)
	}
	if state := coord.State(); state != cudaflightrecorder.StateCooldown {
		t.Fatalf("expected state Cooldown after capture, got %s", state)
	}

	// 5. Cooldown refusal
	_, err = coord.Trigger(context.Background(), cudaflightrecorder.TriggerRequest{
		Reason: "test_cooldown",
	})
	if !errors.Is(err, cudaerrors.ErrUnsupported) {
		t.Fatalf("expected cooldown refusal, got %v", err)
	}

	// 6. Advance clock past cooldown -> auto-rearm on next trigger or explicit Arm
	fakeClock.Advance(3 * time.Second)
	receipt2, err := coord.Trigger(context.Background(), cudaflightrecorder.TriggerRequest{
		Reason:      "second_incident",
		MetricName:  "gpu_latency_ms",
		MetricValue: 130.0,
		Threshold:   100.0,
	})
	if err != nil {
		t.Fatalf("Trigger after cooldown: %v", err)
	}
	if receipt2.Outcome != "captured" {
		t.Fatalf("expected outcome captured, got %s", receipt2.Outcome)
	}

	// 7. Stop coordinator
	if err := coord.Stop(); err != nil {
		t.Fatalf("Stop: %v", err)
	}
	if state := coord.State(); state != cudaflightrecorder.StateStopped {
		t.Fatalf("expected state Stopped, got %s", state)
	}
}

func TestTriggerHysteresisAndStorm(t *testing.T) {
	rec, err := cudaflightrecorder.New(cudaflightrecorder.Config{
		Window:   5 * time.Second,
		Capacity: 100,
	})
	if err != nil {
		t.Fatalf("New recorder: %v", err)
	}
	if err := rec.Start(); err != nil {
		t.Fatalf("Start recorder: %v", err)
	}
	defer rec.Stop()

	fakeClock := cudaflightrecorder.NewFakeClock(time.Date(2026, 10, 10, 12, 0, 0, 0, time.UTC))
	budgets := cudaflightrecorder.QualificationBudgets()
	budgets.HysteresisCount = 3 // Requires 3 triggers
	budgets.HysteresisWindow = 5 * time.Second
	budgets.CooldownDuration = 2 * time.Second

	coord, err := cudaflightrecorder.NewEscalationCoordinator(rec, budgets,
		cudaflightrecorder.WithResourceSampler(&cudaflightrecorder.FakeResourceSampler{}),
		cudaflightrecorder.WithClock(fakeClock),
	)
	if err != nil {
		t.Fatalf("New coordinator: %v", err)
	}
	if err := coord.Arm(); err != nil {
		t.Fatalf("Arm: %v", err)
	}

	req := cudaflightrecorder.TriggerRequest{
		Reason:      "power_spike",
		MetricName:  "power_milliwatts",
		MetricValue: 380000,
		Threshold:   350000,
	}

	// Trigger 1: Refused due to hysteresis
	r1, err1 := coord.Trigger(context.Background(), req)
	if err1 == nil || r1.RefusalReason != cudaflightrecorder.RefusalHysteresisNotMet {
		t.Fatalf("expected hysteresis refusal on trigger 1, got receipt=%+v, err=%v", r1, err1)
	}

	// Advance 1s
	fakeClock.Advance(1 * time.Second)

	// Trigger 2: Still below hysteresis count of 3
	r2, err2 := coord.Trigger(context.Background(), req)
	if err2 == nil || r2.RefusalReason != cudaflightrecorder.RefusalHysteresisNotMet {
		t.Fatalf("expected hysteresis refusal on trigger 2, got receipt=%+v, err=%v", r2, err2)
	}

	// Advance 1s
	fakeClock.Advance(1 * time.Second)

	// Trigger 3: Reaches threshold of 3 within window -> escalates to capture
	r3, err3 := coord.Trigger(context.Background(), req)
	if err3 != nil || r3.Outcome != "captured" {
		t.Fatalf("expected trigger 3 to escalate and capture, got receipt=%+v, err=%v", r3, err3)
	}

	// Trigger 4 immediately following (trigger storm during cooldown)
	r4, err4 := coord.Trigger(context.Background(), req)
	if err4 == nil || r4.RefusalReason != cudaflightrecorder.RefusalCooldownActive {
		t.Fatalf("expected cooldown refusal during storm, got receipt=%+v, err=%v", r4, err4)
	}

	// Verify history contains all receipts
	history := coord.History()
	if len(history) != 4 {
		t.Fatalf("expected 4 receipts in history, got %d", len(history))
	}
}

func TestTenantBudgetIsolation(t *testing.T) {
	rec, err := cudaflightrecorder.New(cudaflightrecorder.Config{
		Window:   5 * time.Second,
		Capacity: 100,
	})
	if err != nil {
		t.Fatalf("New recorder: %v", err)
	}
	if err := rec.Start(); err != nil {
		t.Fatalf("Start recorder: %v", err)
	}
	defer rec.Stop()

	fakeClock := cudaflightrecorder.NewFakeClock(time.Date(2026, 10, 10, 12, 0, 0, 0, time.UTC))
	budgets := cudaflightrecorder.QualificationBudgets()
	budgets.HysteresisCount = 1
	budgets.MaxTenantCapturesPerHour = 2 // Max 2 per hour per tenant
	budgets.CooldownDuration = 100 * time.Millisecond

	coord, err := cudaflightrecorder.NewEscalationCoordinator(rec, budgets,
		cudaflightrecorder.WithResourceSampler(&cudaflightrecorder.FakeResourceSampler{}),
		cudaflightrecorder.WithClock(fakeClock),
	)
	if err != nil {
		t.Fatalf("New coordinator: %v", err)
	}
	if err := coord.Arm(); err != nil {
		t.Fatalf("Arm: %v", err)
	}

	// Tenant A - Capture 1
	_, err = coord.Trigger(context.Background(), cudaflightrecorder.TriggerRequest{
		TenantID: "tenant-a",
		Reason:   "incident-1",
	})
	if err != nil {
		t.Fatalf("Tenant A capture 1: %v", err)
	}

	fakeClock.Advance(200 * time.Millisecond)

	// Tenant A - Capture 2
	_, err = coord.Trigger(context.Background(), cudaflightrecorder.TriggerRequest{
		TenantID: "tenant-a",
		Reason:   "incident-2",
	})
	if err != nil {
		t.Fatalf("Tenant A capture 2: %v", err)
	}

	fakeClock.Advance(200 * time.Millisecond)

	// Tenant A - Capture 3: Exceeds hourly rate budget (2 per hour)
	rA3, errA3 := coord.Trigger(context.Background(), cudaflightrecorder.TriggerRequest{
		TenantID: "tenant-a",
		Reason:   "incident-3",
	})
	if errA3 == nil || rA3.RefusalReason != cudaflightrecorder.RefusalTenantRate {
		t.Fatalf("expected Tenant A to be rate-refused, got receipt=%+v, err=%v", rA3, errA3)
	}

	// Tenant B - Capture 1: Independent tenant budget must succeed
	rB1, errB1 := coord.Trigger(context.Background(), cudaflightrecorder.TriggerRequest{
		TenantID: "tenant-b",
		Reason:   "incident-b-1",
	})
	if errB1 != nil || rB1.Outcome != "captured" {
		t.Fatalf("Tenant B should be admitted independently, got receipt=%+v, err=%v", rB1, errB1)
	}
}

func TestPreTriggerEvidencePreserved(t *testing.T) {
	rec, err := cudaflightrecorder.New(cudaflightrecorder.Config{
		Window:   10 * time.Second,
		Capacity: 1000,
	})
	if err != nil {
		t.Fatalf("New recorder: %v", err)
	}
	if err := rec.Start(); err != nil {
		t.Fatalf("Start recorder: %v", err)
	}
	defer rec.Stop()

	// Seed continuous records prior to incident
	baseTime := time.Date(2026, 10, 10, 12, 0, 0, 0, time.UTC).UnixNano()
	for i := 0; i < 20; i++ {
		rec.Push(cudaflightrecorder.Record{
			TimestampNs: baseTime + int64(i)*10_000_000, // 10ms intervals
			KernelName:  "matrix_multiply_kernel",
			DurationNs:  500_000,
			Labels: map[string]string{
				"layer": "attention",
			},
		})
	}
	// Push a driver policy observation
	rec.PushPolicyRecord(baseTime+200_000_000, "thermal_slowdown", map[string]string{"temp": "85"})

	fakeClock := cudaflightrecorder.NewFakeClock(time.Unix(0, baseTime+250_000_000))
	budgets := cudaflightrecorder.QualificationBudgets()
	budgets.HysteresisCount = 1

	bufFactory := cudaflightrecorder.NewBufferWriterFactory()
	coord, err := cudaflightrecorder.NewEscalationCoordinator(rec, budgets,
		cudaflightrecorder.WithResourceSampler(&cudaflightrecorder.FakeResourceSampler{}),
		cudaflightrecorder.WithClock(fakeClock),
		cudaflightrecorder.WithCaptureWriterFactory(bufFactory),
	)
	if err != nil {
		t.Fatalf("New coordinator: %v", err)
	}
	if err := coord.Arm(); err != nil {
		t.Fatalf("Arm: %v", err)
	}

	receipt, err := coord.Trigger(context.Background(), cudaflightrecorder.TriggerRequest{
		Reason:      "thermal_violation",
		MetricName:  "temperature_celsius",
		MetricValue: 86.0,
		Threshold:   82.0,
	})
	if err != nil {
		t.Fatalf("Trigger failed: %v", err)
	}

	// Verify pre-trigger report
	pre := receipt.Trigger.PreTrigger
	if !pre.Available {
		t.Errorf("expected pre-trigger to be available")
	}
	if pre.RecordCount < 20 {
		t.Errorf("expected >= 20 pre-trigger records, got %d", pre.RecordCount)
	}
	if pre.Truncated {
		t.Errorf("pre-trigger should not be truncated with sufficient capacity")
	}

	// Verify missing pre-trigger families are reported
	missing := pre.MissingPreTriggerFamilies
	var hasCUPTIActivity bool
	for _, m := range missing {
		if m == "cupti_activity" {
			hasCUPTIActivity = true
			break
		}
	}
	if !hasCUPTIActivity {
		t.Errorf("expected cupti_activity to be reported in missing pre-trigger families, got %v", missing)
	}

	// Verify captured tar bundle contains incident.json
	buf, ok := bufFactory.Buffers[receipt.IncidentID]
	if !ok {
		t.Fatalf("bundle for incident %s not found in factory", receipt.IncidentID)
	}
	bundle, err := cudaflightrecorder.ReadBundle(buf)
	if err != nil {
		t.Fatalf("ReadBundle failed: %v", err)
	}
	incidentFile, exists := bundle.Files["incident.json"]
	if !exists {
		t.Fatalf("incident.json missing from bundle")
	}
	var unmarshaled cudaflightrecorder.TriggerEvidence
	if err := json.Unmarshal(incidentFile, &unmarshaled); err != nil {
		t.Fatalf("unmarshal incident.json: %v", err)
	}
	if unmarshaled.Reason != "thermal_violation" {
		t.Errorf("incident.json Reason = %q, want thermal_violation", unmarshaled.Reason)
	}
}

func TestCUPTIOwnershipRefusalNoTheft(t *testing.T) {
	rec, err := cudaflightrecorder.New(cudaflightrecorder.Config{
		Window:   5 * time.Second,
		Capacity: 100,
	})
	if err != nil {
		t.Fatalf("New recorder: %v", err)
	}
	if err := rec.Start(); err != nil {
		t.Fatalf("Start recorder: %v", err)
	}
	defer rec.Stop()

	fakeNegotiator := &cudaflightrecorder.FakeCUPTINegotiator{}
	fakeNegotiator.SetConflict(true, nil) // External profiler holds session

	budgets := cudaflightrecorder.QualificationBudgets()
	budgets.HysteresisCount = 1

	coord, err := cudaflightrecorder.NewEscalationCoordinator(rec, budgets,
		cudaflightrecorder.WithResourceSampler(&cudaflightrecorder.FakeResourceSampler{}),
		cudaflightrecorder.WithCUPTINegotiator(fakeNegotiator),
	)
	if err != nil {
		t.Fatalf("New coordinator: %v", err)
	}
	if err := coord.Arm(); err != nil {
		t.Fatalf("Arm: %v", err)
	}

	receipt, err := coord.Trigger(context.Background(), cudaflightrecorder.TriggerRequest{
		Reason:        "anomaly",
		DeviceOrdinal: 0,
	})
	if err == nil {
		t.Fatalf("expected refusal error when CUPTI ownership conflicts, got nil")
	}
	if !errors.Is(err, cudaerrors.ErrUnsupported) {
		t.Fatalf("expected error wrapping ErrUnsupported, got %v", err)
	}
	if receipt.RefusalReason != cudaflightrecorder.RefusalCUPTIConflict {
		t.Fatalf("expected RefusalCUPTIConflict, got %q", receipt.RefusalReason)
	}
	if receipt.Outcome != "refused" {
		t.Fatalf("expected outcome refused, got %q", receipt.Outcome)
	}
}

type failingWriter struct {
	failAfter int
	written   int
}

func (f *failingWriter) Write(p []byte) (int, error) {
	if f.written >= f.failAfter {
		return 0, errors.New("simulated disk full")
	}
	f.written += len(p)
	return len(p), nil
}

type failingWriterFactory struct {
	failAfter int
}

func (f *failingWriterFactory) CreateWriter(id, tenant string) (io.Writer, func() error, error) {
	return &failingWriter{failAfter: f.failAfter}, func() error { return nil }, nil
}

func TestWriterFailurePropagated(t *testing.T) {
	rec, err := cudaflightrecorder.New(cudaflightrecorder.Config{
		Window:   5 * time.Second,
		Capacity: 100,
	})
	if err != nil {
		t.Fatalf("New recorder: %v", err)
	}
	if err := rec.Start(); err != nil {
		t.Fatalf("Start recorder: %v", err)
	}
	defer rec.Stop()

	budgets := cudaflightrecorder.QualificationBudgets()
	budgets.HysteresisCount = 1

	coord, err := cudaflightrecorder.NewEscalationCoordinator(rec, budgets,
		cudaflightrecorder.WithResourceSampler(&cudaflightrecorder.FakeResourceSampler{}),
		cudaflightrecorder.WithCaptureWriterFactory(&failingWriterFactory{failAfter: 10}),
	)
	if err != nil {
		t.Fatalf("New coordinator: %v", err)
	}
	if err := coord.Arm(); err != nil {
		t.Fatalf("Arm: %v", err)
	}

	receipt, err := coord.Trigger(context.Background(), cudaflightrecorder.TriggerRequest{
		Reason: "write_failure_test",
	})
	if err == nil {
		t.Fatalf("expected writer failure error, got nil")
	}
	if receipt.Outcome != "failed" {
		t.Fatalf("expected receipt outcome failed, got %q", receipt.Outcome)
	}
	if receipt.Error == "" {
		t.Fatalf("expected receipt Error to record write error")
	}

	// Coordinator must have transitioned to cooldown cleanly
	if state := coord.State(); state != cudaflightrecorder.StateCooldown {
		t.Fatalf("expected coordinator to be in Cooldown after writer failure, got %s", state)
	}
}

func TestCancellationSafety(t *testing.T) {
	rec, err := cudaflightrecorder.New(cudaflightrecorder.Config{
		Window:   5 * time.Second,
		Capacity: 100,
	})
	if err != nil {
		t.Fatalf("New recorder: %v", err)
	}
	if err := rec.Start(); err != nil {
		t.Fatalf("Start recorder: %v", err)
	}
	defer rec.Stop()

	budgets := cudaflightrecorder.QualificationBudgets()
	budgets.HysteresisCount = 1

	coord, err := cudaflightrecorder.NewEscalationCoordinator(rec, budgets, cudaflightrecorder.WithResourceSampler(&cudaflightrecorder.FakeResourceSampler{}))
	if err != nil {
		t.Fatalf("New coordinator: %v", err)
	}
	if err := coord.Arm(); err != nil {
		t.Fatalf("Arm: %v", err)
	}

	// Pre-canceled context
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	receipt, err := coord.Trigger(ctx, cudaflightrecorder.TriggerRequest{
		Reason: "canceled_test",
	})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context.Canceled, got: %v", err)
	}
	if receipt.RefusalReason != "context_canceled" {
		t.Fatalf("expected refusal reason context_canceled, got %q", receipt.RefusalReason)
	}
}

func TestIdempotentShutdown(t *testing.T) {
	rec, err := cudaflightrecorder.New(cudaflightrecorder.Config{
		Window:   5 * time.Second,
		Capacity: 100,
	})
	if err != nil {
		t.Fatalf("New recorder: %v", err)
	}
	if err := rec.Start(); err != nil {
		t.Fatalf("Start recorder: %v", err)
	}
	defer rec.Stop()

	coord, err := cudaflightrecorder.NewEscalationCoordinator(rec, cudaflightrecorder.QualificationBudgets(), cudaflightrecorder.WithResourceSampler(&cudaflightrecorder.FakeResourceSampler{}))
	if err != nil {
		t.Fatalf("New coordinator: %v", err)
	}
	if err := coord.Arm(); err != nil {
		t.Fatalf("Arm: %v", err)
	}

	// Repeated Stop calls must be harmless and idempotent
	for i := 0; i < 5; i++ {
		if err := coord.Stop(); err != nil {
			t.Fatalf("Stop call %d failed: %v", i, err)
		}
	}
	for i := 0; i < 3; i++ {
		if err := coord.Close(); err != nil {
			t.Fatalf("Close call %d failed: %v", i, err)
		}
	}

	// Operations after shutdown must refuse cleanly
	r, err := coord.Trigger(context.Background(), cudaflightrecorder.TriggerRequest{
		Reason: "after_stop",
	})
	if !errors.Is(err, cudaerrors.ErrUnsupported) {
		t.Fatalf("expected ErrUnsupported after stop, got %v", err)
	}
	if r.RefusalReason != cudaflightrecorder.RefusalStopped {
		t.Fatalf("expected RefusalStopped, got %q", r.RefusalReason)
	}
}

func TestEscalationStressAndRace(t *testing.T) {
	rec, err := cudaflightrecorder.New(cudaflightrecorder.Config{
		Window:   2 * time.Second,
		Capacity: 500,
	})
	if err != nil {
		t.Fatalf("New recorder: %v", err)
	}
	if err := rec.Start(); err != nil {
		t.Fatalf("Start recorder: %v", err)
	}
	defer rec.Stop()

	budgets := cudaflightrecorder.DefaultBudgets()
	budgets.HysteresisCount = 1
	budgets.CooldownDuration = 10 * time.Millisecond
	budgets.MaxConcurrentCaptures = 1
	budgets.MaxTenantConcurrency = 1

	coord, err := cudaflightrecorder.NewEscalationCoordinator(rec, budgets, cudaflightrecorder.WithResourceSampler(&cudaflightrecorder.FakeResourceSampler{}))
	if err != nil {
		t.Fatalf("New coordinator: %v", err)
	}
	if err := coord.Arm(); err != nil {
		t.Fatalf("Arm: %v", err)
	}

	var wg sync.WaitGroup
	// Concurrently push telemetry and trigger escalations across multiple tenants
	for worker := 0; worker < 8; worker++ {
		wg.Add(1)
		tenant := "tenant-" + string(rune('A'+worker%4))
		go func(w int, tID string) {
			defer wg.Done()
			for i := 0; i < 20; i++ {
				rec.Push(cudaflightrecorder.Record{
					TimestampNs: time.Now().UnixNano(),
					KernelName:  "stress_kernel",
					DurationNs:  100_000,
				})
				coord.Trigger(context.Background(), cudaflightrecorder.TriggerRequest{
					TenantID:    tID,
					Reason:      "stress_trigger",
					MetricName:  "gpu_util",
					MetricValue: float64(90 + i),
					Threshold:   85,
				})
			}
		}(worker, tenant)
	}

	wg.Wait()

	history := coord.History()
	if len(history) == 0 {
		t.Fatalf("expected non-empty history after concurrent stress test")
	}
}

func TestRawBudgetReceiptGeneration(t *testing.T) {
	receipt := cudaflightrecorder.IncidentReceipt{
		IncidentID: "inc-000042",
		Timestamp:  time.Date(2026, 10, 10, 12, 0, 0, 0, time.UTC),
		TenantID:   "prod-workload-01",
		State:      cudaflightrecorder.StateCooldown,
		Outcome:    "captured",
		Budgets:    cudaflightrecorder.QualificationBudgets(),
		Trigger: cudaflightrecorder.TriggerEvidence{
			TimestampMono: 1000000000,
			TimestampUnix: 1760097600000000000,
			TenantID:      "prod-workload-01",
			Reason:        "latency_spike",
			MetricName:    "sm_activity_pct",
			MetricValue:   99.5,
			Threshold:     95.0,
			PreTrigger: cudaflightrecorder.PreTriggerReport{
				Available:      true,
				WindowDuration: 5 * time.Second,
				RecordCount:    150,
			},
		},
		Perturbation: cudaflightrecorder.PerturbationReport{
			CaptureDuration:  12 * time.Millisecond,
			BundleBytes:      1048576,
			RecordsCaptured:  150,
			CPUPercentActual: 2.5,
			RSSBytesActual:   128 << 20,
		},
	}

	data, err := json.MarshalIndent(receipt, "", "  ")
	if err != nil {
		t.Fatalf("marshal receipt: %v", err)
	}
	if len(data) == 0 {
		t.Fatalf("receipt serialized to empty bytes")
	}

	var parsed cudaflightrecorder.IncidentReceipt
	if err := json.Unmarshal(data, &parsed); err != nil {
		t.Fatalf("unmarshal receipt: %v", err)
	}
	if parsed.IncidentID != "inc-000042" || parsed.Outcome != "captured" {
		t.Fatalf("mismatched parsed receipt: %+v", parsed)
	}
}

package cudatrial

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"
)

func TestInterleaveABBAOrder(t *testing.T) {
	var callOrder []string

	arms := []Arm{
		{
			Name: "armA",
			Run: func(ctx context.Context) (Receipt, error) {
				callOrder = append(callOrder, "armA")
				return Receipt{
					Complete: true,
					Digest:   "hashA",
				}, nil
			},
		},
		{
			Name: "armB",
			Run: func(ctx context.Context) (Receipt, error) {
				callOrder = append(callOrder, "armB")
				return Receipt{
					Complete: true,
					Digest:   "hashB",
				}, nil
			},
		},
	}

	receipts, err := Interleave(context.Background(), arms, 3)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// True ABBA ordering:
	// Round 1: armA, armB
	// Round 2: armB, armA
	// Round 3: armA, armB
	wantCalls := []string{
		"armA", "armB",
		"armB", "armA",
		"armA", "armB",
	}
	if len(callOrder) != len(wantCalls) {
		t.Fatalf("call count = %d, want %d", len(callOrder), len(wantCalls))
	}
	for i, call := range callOrder {
		if call != wantCalls[i] {
			t.Errorf("call %d = %q, want %q", i, call, wantCalls[i])
		}
	}

	if len(receipts) != 6 {
		t.Fatalf("receipt count = %d, want 6", len(receipts))
	}
	for i, r := range receipts {
		wantRound := (i / 2) + 1
		if r.Round != wantRound {
			t.Errorf("receipt %d round = %d, want %d", i, r.Round, wantRound)
		}
		if !r.Complete {
			t.Errorf("receipt %d complete = false, want true", i)
		}
	}
}

func TestInterleaveValidation(t *testing.T) {
	tests := []struct {
		name    string
		arms    []Arm
		rounds  int
		wantErr string
	}{
		{
			name:    "zero rounds",
			arms:    []Arm{{Name: "a", Run: func(ctx context.Context) (Receipt, error) { return Receipt{}, nil }}},
			rounds:  0,
			wantErr: "rounds must be positive",
		},
		{
			name:    "negative rounds",
			arms:    []Arm{{Name: "a", Run: func(ctx context.Context) (Receipt, error) { return Receipt{}, nil }}},
			rounds:  -2,
			wantErr: "rounds must be positive",
		},
		{
			name:    "empty arms",
			arms:    nil,
			rounds:  3,
			wantErr: "arms must not be empty",
		},
		{
			name:    "empty arm name",
			arms:    []Arm{{Name: "", Run: func(ctx context.Context) (Receipt, error) { return Receipt{}, nil }}},
			rounds:  1,
			wantErr: "empty Name",
		},
		{
			name:    "nil run func",
			arms:    []Arm{{Name: "a", Run: nil}},
			rounds:  1,
			wantErr: "nil Run function",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := Interleave(context.Background(), tt.arms, tt.rounds)
			if err == nil {
				t.Fatalf("expected error containing %q, got nil", tt.wantErr)
			}
			if !strings.Contains(err.Error(), tt.wantErr) {
				t.Errorf("error = %q, want substring %q", err.Error(), tt.wantErr)
			}
		})
	}
}

func TestInterleaveContextCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())

	count := 0
	arms := []Arm{
		{
			Name: "cancel_after_first",
			Run: func(ctx context.Context) (Receipt, error) {
				count++
				cancel()
				return Receipt{Complete: true}, nil
			},
		},
		{
			Name: "unreached",
			Run: func(ctx context.Context) (Receipt, error) {
				t.Error("unreached arm should not have run")
				return Receipt{Complete: true}, nil
			},
		},
	}

	_, err := Interleave(ctx, arms, 3)
	if err == nil {
		t.Fatal("expected error on canceled context, got nil")
	}
	if !errors.Is(err, context.Canceled) {
		t.Errorf("expected context.Canceled error, got: %v", err)
	}
	if count != 1 {
		t.Errorf("count = %d, want 1", count)
	}
}

func TestInterleaveErrorHandling(t *testing.T) {
	errBoom := errors.New("boom")

	arms := []Arm{
		{
			Name: "arm1",
			Run: func(ctx context.Context) (Receipt, error) {
				return Receipt{Complete: true, Digest: "d1"}, nil
			},
		},
		{
			Name: "failing_arm",
			Run: func(ctx context.Context) (Receipt, error) {
				return Receipt{}, errBoom
			},
		},
	}

	// Default: halt on first error
	receipts, err := Interleave(context.Background(), arms, 2)
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !errors.Is(err, errBoom) {
		t.Errorf("expected %v, got %v", errBoom, err)
	}
	if len(receipts) != 2 {
		t.Errorf("receipts returned = %d, want 2 (arm1 and failing_arm)", len(receipts))
	}

	// ContinueOnError: true
	receiptsCont, err := InterleaveWithOptions(context.Background(), arms, 2, Options{ContinueOnError: true})
	if err != nil {
		t.Fatalf("unexpected error with ContinueOnError: %v", err)
	}
	if len(receiptsCont) != 4 {
		t.Fatalf("receipts count = %d, want 4", len(receiptsCont))
	}
	if receiptsCont[1].Complete {
		t.Errorf("failed receipt should not be complete")
	}
	if !errors.Is(receiptsCont[1].Err, errBoom) {
		t.Errorf("receipt err = %v, want %v", receiptsCont[1].Err, errBoom)
	}
}

func TestSameArmExact(t *testing.T) {
	tests := []struct {
		name      string
		receipts  []Receipt
		wantExact bool
	}{
		{
			name: "consistent single arm",
			receipts: []Receipt{
				{Name: "a", Round: 1, Complete: true, Digest: "d1"},
				{Name: "a", Round: 2, Complete: true, Digest: "d1"},
			},
			wantExact: true,
		},
		{
			name: "divergent single arm",
			receipts: []Receipt{
				{Name: "a", Round: 1, Complete: true, Digest: "d1"},
				{Name: "a", Round: 2, Complete: true, Digest: "d2_different"},
			},
			wantExact: false,
		},
		{
			name: "different arms different digests are ok",
			receipts: []Receipt{
				{Name: "a", Round: 1, Complete: true, Digest: "da"},
				{Name: "b", Round: 1, Complete: true, Digest: "db"},
				{Name: "a", Round: 2, Complete: true, Digest: "da"},
				{Name: "b", Round: 2, Complete: true, Digest: "db"},
			},
			wantExact: true,
		},
		{
			name: "empty receipts fails",
			receipts: nil,
			wantExact: false,
		},
		{
			name: "arm with all failed runs fails",
			receipts: []Receipt{
				{Name: "a", Round: 1, Complete: false, Digest: "corrupted"},
				{Name: "a", Round: 2, Complete: false, Digest: "corrupted"},
			},
			wantExact: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, msg := SameArmExact(tt.receipts)
			if got != tt.wantExact {
				t.Errorf("SameArmExact() = %v (msg: %s), want %v", got, msg, tt.wantExact)
			}
		})
	}
}

func TestDurationPreferenceAndSummary(t *testing.T) {
	t0 := time.Now()
	receipts := []Receipt{
		{
			Name:           "b_arm",
			Round:          1,
			Complete:       true,
			Digest:         "h2",
			Started:        t0,
			Finished:       t0.Add(5 * time.Millisecond),
			DeviceDuration: 50 * time.Microsecond, // DeviceDuration preferred over wall
		},
		{
			Name:     "a_arm",
			Round:    1,
			Complete: true,
			Digest:   "h1",
			Started:  t0,
			Finished: t0.Add(2 * time.Millisecond),
		},
		{
			Name:     "a_arm",
			Round:    2,
			Complete: true,
			Digest:   "h1",
			Started:  t0,
			Finished: t0.Add(3 * time.Millisecond),
		},
	}

	if receipts[0].Duration() != 50*time.Microsecond {
		t.Errorf("b_arm Duration = %v, want DeviceDuration 50µs", receipts[0].Duration())
	}
	if receipts[1].Duration() != 2*time.Millisecond {
		t.Errorf("a_arm Duration = %v, want wall Duration 2ms", receipts[1].Duration())
	}

	digests := DistinctDigests(receipts)
	if len(digests) != 2 || digests[0] != "h1" || digests[1] != "h2" {
		t.Errorf("DistinctDigests = %v, want [h1 h2]", digests)
	}

	summary := FormatSummary(receipts)
	if !strings.Contains(summary, "MEDIAN_DUR") || !strings.Contains(summary, "CLOCK") || !strings.Contains(summary, "a_arm") || !strings.Contains(summary, "b_arm") {
		t.Errorf("summary missing headers or arms:\n%s", summary)
	}

	emptySummary := FormatSummary(nil)
	if emptySummary != "no receipts\n" {
		t.Errorf("empty summary = %q, want 'no receipts\\n'", emptySummary)
	}
}

func TestReceiptExplicitTiming(t *testing.T) {
	r := Receipt{Name: "test_receipt"}

	// Initially unmeasured
	if r.HasTiming(TimingGPUBusy) {
		t.Error("expected HasTiming(TimingGPUBusy) to be false")
	}
	if _, err := r.DurationFor(TimingGPUBusy); err == nil {
		t.Error("expected error for unmeasured timing, got nil")
	}

	// Reject empty timing kind
	if err := r.RecordTiming("", 10*time.Millisecond); err == nil {
		t.Error("expected error for empty timing kind, got nil")
	}

	// Reject negative duration
	if err := r.RecordTiming(TimingGPUBusy, -5*time.Millisecond); err == nil {
		t.Error("expected error for negative duration, got nil")
	}

	// Measured zero must be distinguishable from unmeasured
	if err := r.RecordTiming(TimingGPUBusy, 0); err != nil {
		t.Fatalf("unexpected error recording zero duration: %v", err)
	}
	if !r.HasTiming(TimingGPUBusy) {
		t.Error("expected HasTiming(TimingGPUBusy) to be true for measured zero")
	}
	d, err := r.DurationFor(TimingGPUBusy)
	if err != nil {
		t.Fatalf("DurationFor(TimingGPUBusy) failed for measured zero: %v", err)
	}
	if d != 0 {
		t.Errorf("duration = %v, want 0", d)
	}

	// Record other timings
	if err := r.RecordTiming(TimingSummedKernel, 15*time.Millisecond); err != nil {
		t.Fatalf("RecordTiming failed: %v", err)
	}
	if err := r.RecordTiming(TimingWallElapsed, 20*time.Millisecond); err != nil {
		t.Fatalf("RecordTiming failed: %v", err)
	}

	// Verify primary TimingKind is set to first recorded (TimingGPUBusy)
	if r.TimingKind != TimingGPUBusy {
		t.Errorf("TimingKind = %q, want %q", r.TimingKind, TimingGPUBusy)
	}
	if r.Duration() != 0 {
		t.Errorf("r.Duration() = %v, want 0 (from primary TimingGPUBusy)", r.Duration())
	}

	// Switch primary TimingKind
	r.TimingKind = TimingSummedKernel
	if r.Duration() != 15*time.Millisecond {
		t.Errorf("r.Duration() = %v, want 15ms", r.Duration())
	}
}

func TestNonAdditiveLatencyDecomposition(t *testing.T) {
	r := Receipt{
		Name: "test_decomp",
	}
	_ = r.RecordTiming(TimingWallElapsed, 100*time.Microsecond)
	_ = r.RecordTiming(TimingSummedKernel, 60*time.Microsecond)
	_ = r.RecordTiming(TimingGPUBusy, 50*time.Microsecond)
	r.QueueNs = 15000 // 15µs

	decomp := r.LatencyDecomposition()
	if decomp.WallNs != 100000 {
		t.Errorf("WallNs = %d, want 100000", decomp.WallNs)
	}
	if decomp.DeviceNs != 60000 {
		t.Errorf("DeviceNs = %d, want 60000", decomp.DeviceNs)
	}
	if decomp.GpuIntervalUnionNs != 50000 {
		t.Errorf("GpuIntervalUnionNs = %d, want 50000", decomp.GpuIntervalUnionNs)
	}
	if decomp.QueueNs != 15000 {
		t.Errorf("QueueNs = %d, want 15000", decomp.QueueNs)
	}

	// Naive sum must be explicitly rejected
	_, err := decomp.TotalDuration()
	if !errors.Is(err, ErrNaiveSumDoubleCounts) {
		t.Fatalf("TotalDuration err = %v, want %v", err, ErrNaiveSumDoubleCounts)
	}
}

func TestHardwareLockVerification(t *testing.T) {
	arms := []Arm{
		{
			Name: "a",
			Run: func(ctx context.Context) (Receipt, error) {
				return Receipt{Complete: true, Digest: "abc"}, nil
			},
		},
	}

	// Without lock, should fail with ErrHardwareLockRequired
	_, err := InterleaveWithOptions(context.Background(), arms, 1, Options{
		RequireHardwareLock: true,
		HardwareLockChecker: func() bool { return false },
	})
	if !errors.Is(err, ErrHardwareLockRequired) {
		t.Fatalf("expected %v, got %v", ErrHardwareLockRequired, err)
	}

	// With lock held, should succeed
	receipts, err := InterleaveWithOptions(context.Background(), arms, 1, Options{
		RequireHardwareLock: true,
		HardwareLockChecker: func() bool { return true },
	})
	if err != nil {
		t.Fatalf("unexpected error with lock held: %v", err)
	}
	if len(receipts) != 1 {
		t.Errorf("receipts len = %d, want 1", len(receipts))
	}
}

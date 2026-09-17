package cudacmp

import (
	"bytes"
	"context"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudaroofline"
	"github.com/tmc/lib/nvidia/x/cudatrial"
)

func TestNewTrial(t *testing.T) {
	tests := []struct {
		name    string
		cfg     TrialConfig
		wantErr bool
	}{
		{
			name: "valid config",
			cfg: TrialConfig{
				WorkloadTokensPath: "test.json",
				ContextLength:      4096,
				PrefillTokens:      2048,
				DecodeTokens:       16,
				InterleavedRuns:    5,
				WarmupRuns:         1,
				COraclePath:        "/bin/echo",
				Residency:          cudaroofline.DeviceResident,
			},
			wantErr: false,
		},
		{
			name: "negative interleaved runs",
			cfg: TrialConfig{
				InterleavedRuns: -1,
			},
			wantErr: true,
		},
		{
			name: "negative warmup runs",
			cfg: TrialConfig{
				WarmupRuns: -1,
			},
			wantErr: true,
		},
		{
			name: "negative prefill tokens",
			cfg: TrialConfig{
				PrefillTokens: -1,
			},
			wantErr: true,
		},
		{
			name: "negative decode tokens",
			cfg: TrialConfig{
				DecodeTokens: -1,
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			trial, err := NewTrial(tt.cfg)
			if (err != nil) != tt.wantErr {
				t.Fatalf("NewTrial() error = %v, wantErr %v", err, tt.wantErr)
			}
			if !tt.wantErr && trial == nil {
				t.Fatal("expected non-nil trial on success")
			}
			if !tt.wantErr && trial.Config().ContextLength != tt.cfg.ContextLength {
				t.Errorf("Config().ContextLength = %d, want %d", trial.Config().ContextLength, tt.cfg.ContextLength)
			}
		})
	}
}

func TestRender(t *testing.T) {
	reports := []DifferentialReport{
		{
			Phase:              PhasePrefill,
			Family:             FamilyExperts,
			Availability:       AvailableBoth,
			Available:          true,
			CObserved:          true,
			GoObserved:         true,
			CDurationNs:        12_500_000,
			GoDurationNs:       10_000_000,
			DeltaNs:            -2_500_000,
			Speedup:            1.25,
			NullDriftNs:        50_000,
			ExceedsDrift:       true,
			StatisticallyValid: true,
		},
		{
			Phase:              PhaseDecode,
			Family:             FamilyDenseQ8,
			Availability:       AvailableBoth,
			Available:          true,
			CObserved:          true,
			GoObserved:         true,
			CDurationNs:        8_000_000,
			GoDurationNs:       8_020_000,
			DeltaNs:            20_000,
			Speedup:            0.9975,
			NullDriftNs:        50_000,
			ExceedsDrift:       false,
			StatisticallyValid: false,
		},
		{
			Phase:        PhaseDecode,
			Family:       FamilyResidual,
			Availability: AvailableCOnly,
			Available:    false,
			CObserved:    true,
			GoObserved:   false,
			CDurationNs:  1_000_000,
			NullDriftNs:  50_000,
		},
	}

	var buf bytes.Buffer
	if err := Render(&buf, reports); err != nil {
		t.Fatalf("Render failed: %v", err)
	}

	out := buf.String()
	if !strings.Contains(out, "PHASE") || !strings.Contains(out, "FAMILY") {
		t.Errorf("missing headers in output: %s", out)
	}
	if !strings.Contains(out, "experts") || !strings.Contains(out, "dense_q8") {
		t.Errorf("missing family names in output: %s", out)
	}
	if !strings.Contains(out, "YES") || !strings.Contains(out, "NOISE") {
		t.Errorf("missing validity flags in output: %s", out)
	}
	if !strings.Contains(out, "UNAVAILABLE") {
		t.Errorf("expected UNAVAILABLE in output, got: %s", out)
	}
}

func TestHelperMethods(t *testing.T) {
	km := KernelMetric{
		Timing: LatencyDecomposition{DeviceNs: 5_000_000},
	}
	if got := km.Duration(); got != 5*time.Millisecond {
		t.Errorf("km.Duration() = %v, want 5ms", got)
	}

	pc := PhaseCensus{
		SummedKernelNs: 10_000_000,
		GPUBusyNs:      8_000_000,
	}
	if got := pc.SummedKernelDuration(); got != 10*time.Millisecond {
		t.Errorf("pc.SummedKernelDuration() = %v, want 10ms", got)
	}
	if got := pc.GPUBusyDuration(); got != 8*time.Millisecond {
		t.Errorf("pc.GPUBusyDuration() = %v, want 8ms", got)
	}
	if got := pc.Duration(); got != 10*time.Millisecond {
		t.Errorf("pc.Duration() = %v, want 10ms", got)
	}

	dr := DifferentialReport{
		CDurationNs:  15_000_000,
		GoDurationNs: 10_000_000,
		DeltaNs:      -5_000_000,
	}
	if got := dr.CDuration(); got != 15*time.Millisecond {
		t.Errorf("CDuration() = %v, want 15ms", got)
	}
	if got := dr.GoDuration(); got != 10*time.Millisecond {
		t.Errorf("GoDuration() = %v, want 10ms", got)
	}
	if got := dr.Delta(); got != -5*time.Millisecond {
		t.Errorf("Delta() = %v, want -5ms", got)
	}
}

func TestTrialRun(t *testing.T) {
	baseCalls := 0
	candCalls := 0

	baseArm := func(ctx context.Context, phase Phase) (cudatrial.Receipt, error) {
		baseCalls++
		r := cudatrial.Receipt{
			Name:     "baseline",
			Complete: true,
			Digest:   "hash_match",
		}
		_ = r.RecordTiming(cudatrial.TimingGPUBusy, 100*time.Millisecond)
		return r, nil
	}

	candArm := func(ctx context.Context, phase Phase) (cudatrial.Receipt, error) {
		candCalls++
		r := cudatrial.Receipt{
			Name:     "candidate",
			Complete: true,
			Digest:   "hash_match",
		}
		_ = r.RecordTiming(cudatrial.TimingGPUBusy, 80*time.Millisecond)
		return r, nil
	}

	cfg := TrialConfig{
		InterleavedRuns: 2,
		WarmupRuns:      1,
		PrefillTokens:   128,
		DecodeTokens:    0, // test only prefill
		TimingKind:      cudatrial.TimingGPUBusy,
		BaselineArm:     baseArm,
		CandidateArm:    candArm,
	}

	trial, err := NewTrial(cfg)
	if err != nil {
		t.Fatalf("NewTrial failed: %v", err)
	}

	res, err := trial.RunTrial(context.Background())
	if err != nil {
		t.Fatalf("RunTrial failed: %v", err)
	}

	// Warmup: 1 run each = 2 warmup receipts
	if len(res.WarmupReceipts) != 2 {
		t.Errorf("warmup receipts = %d, want 2", len(res.WarmupReceipts))
	}

	// Null control: 2 arms * 2 rounds = 4 null receipts
	if len(res.NullControlReceipts) != 4 {
		t.Errorf("null control receipts = %d, want 4", len(res.NullControlReceipts))
	}

	// Trial: 2 arms * 2 rounds = 4 receipts (2 baseline, 2 candidate)
	if len(res.BaselineReceipts) != 2 {
		t.Errorf("baseline receipts = %d, want 2", len(res.BaselineReceipts))
	}
	if len(res.CandidateReceipts) != 2 {
		t.Errorf("candidate receipts = %d, want 2", len(res.CandidateReceipts))
	}

	// Single report for prefill
	if len(res.Reports) != 1 {
		t.Fatalf("reports count = %d, want 1", len(res.Reports))
	}

	rep := res.Reports[0]
	if rep.Phase != PhasePrefill {
		t.Errorf("report phase = %v, want %v", rep.Phase, PhasePrefill)
	}
	if rep.CDurationNs != int64(100*time.Millisecond) {
		t.Errorf("CDurationNs = %v, want 100ms", rep.CDurationNs)
	}
	if rep.GoDurationNs != int64(80*time.Millisecond) {
		t.Errorf("GoDurationNs = %v, want 80ms", rep.GoDurationNs)
	}
	if rep.DeltaNs != int64(-20*time.Millisecond) {
		t.Errorf("DeltaNs = %v, want -20ms", rep.DeltaNs)
	}
	if rep.Speedup != 1.25 {
		t.Errorf("Speedup = %v, want 1.25", rep.Speedup)
	}
	if !rep.ExceedsDrift {
		t.Errorf("ExceedsDrift = false, want true (null drift was 0, delta was 20ms)")
	}
}

func TestTrialRejections(t *testing.T) {
	baseArm := func(ctx context.Context, phase Phase) (cudatrial.Receipt, error) {
		r := cudatrial.Receipt{
			Name:     "baseline",
			Complete: true,
			Digest:   "hash_base",
		}
		_ = r.RecordTiming(cudatrial.TimingGPUBusy, 100*time.Millisecond)
		return r, nil
	}

	t.Run("reject mismatched digest", func(t *testing.T) {
		candMismatched := func(ctx context.Context, phase Phase) (cudatrial.Receipt, error) {
			r := cudatrial.Receipt{
				Name:     "candidate",
				Complete: true,
				Digest:   "hash_different",
			}
			_ = r.RecordTiming(cudatrial.TimingGPUBusy, 80*time.Millisecond)
			return r, nil
		}

		trial, _ := NewTrial(TrialConfig{
			InterleavedRuns: 1,
			PrefillTokens:   10,
			BaselineArm:     baseArm,
			CandidateArm:    candMismatched,
			TimingKind:      cudatrial.TimingGPUBusy,
		})
		_, err := trial.Run(context.Background())
		if err == nil {
			t.Fatal("expected error on mismatched digest, got nil")
		}
		if !strings.Contains(err.Error(), "mismatch") {
			t.Errorf("error = %q, want substring 'mismatch'", err.Error())
		}
	})

	t.Run("reject incomplete candidate arm", func(t *testing.T) {
		candIncomplete := func(ctx context.Context, phase Phase) (cudatrial.Receipt, error) {
			return cudatrial.Receipt{
				Name:     "candidate",
				Complete: false,
			}, nil
		}

		trial, _ := NewTrial(TrialConfig{
			InterleavedRuns: 1,
			PrefillTokens:   10,
			BaselineArm:     baseArm,
			CandidateArm:    candIncomplete,
			TimingKind:      cudatrial.TimingGPUBusy,
		})
		_, err := trial.Run(context.Background())
		if err == nil {
			t.Fatal("expected error on incomplete candidate, got nil")
		}
	})

	t.Run("context cancellation halts trial", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		candArm := func(c context.Context, phase Phase) (cudatrial.Receipt, error) {
			cancel() // cancel context during execution
			r := cudatrial.Receipt{Name: "candidate", Complete: true, Digest: "hash_base"}
			_ = r.RecordTiming(cudatrial.TimingGPUBusy, 80*time.Millisecond)
			return r, nil
		}

		trial, _ := NewTrial(TrialConfig{
			InterleavedRuns: 5,
			PrefillTokens:   10,
			BaselineArm:     baseArm,
			CandidateArm:    candArm,
			TimingKind:      cudatrial.TimingGPUBusy,
		})
		_, err := trial.Run(ctx)
		if err == nil {
			t.Fatal("expected error on context cancellation, got nil")
		}
	})
}

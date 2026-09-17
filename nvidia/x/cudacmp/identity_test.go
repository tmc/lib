package cudacmp_test

import (
	"errors"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudacmp"
	"github.com/tmc/lib/nvidia/x/cudatrial"
)

func TestCompareReceipts(t *testing.T) {
	cRec := cudatrial.Receipt{
		Name:     "oracle_c",
		Round:    1,
		Complete: true,
	}
	_ = cRec.RecordTiming(cudatrial.TimingGPUBusy, 100*time.Millisecond)
	_ = cRec.RecordTiming(cudatrial.TimingWallElapsed, 120*time.Millisecond)

	goRec := cudatrial.Receipt{
		Name:     "candidate_go",
		Round:    1,
		Complete: true,
	}
	_ = goRec.RecordTiming(cudatrial.TimingGPUBusy, 80*time.Millisecond)
	_ = goRec.RecordTiming(cudatrial.TimingWallElapsed, 90*time.Millisecond)

	t.Run("successful comparison on GPUBusy", func(t *testing.T) {
		cmp, err := cudacmp.CompareReceipts(cRec, goRec, cudatrial.TimingGPUBusy, 5*time.Millisecond)
		if err != nil {
			t.Fatalf("CompareReceipts failed: %v", err)
		}
		if cmp.CDuration != 100*time.Millisecond {
			t.Errorf("CDuration = %v, want 100ms", cmp.CDuration)
		}
		if cmp.GoDuration != 80*time.Millisecond {
			t.Errorf("GoDuration = %v, want 80ms", cmp.GoDuration)
		}
		if cmp.Delta != -20*time.Millisecond {
			t.Errorf("Delta = %v, want -20ms", cmp.Delta)
		}
		if cmp.Speedup != 1.25 {
			t.Errorf("Speedup = %f, want 1.25", cmp.Speedup)
		}
		if !cmp.ExceedsDrift {
			t.Error("expected ExceedsDrift = true for |-20ms| > 5ms")
		}
	})

	t.Run("within noise floor", func(t *testing.T) {
		cmp, err := cudacmp.CompareReceipts(cRec, goRec, cudatrial.TimingGPUBusy, 25*time.Millisecond)
		if err != nil {
			t.Fatalf("CompareReceipts failed: %v", err)
		}
		if cmp.ExceedsDrift {
			t.Error("expected ExceedsDrift = false for |-20ms| <= 25ms")
		}
	})

	t.Run("reject unspecified timing kind", func(t *testing.T) {
		_, err := cudacmp.CompareReceipts(cRec, goRec, "", 5*time.Millisecond)
		if err == nil {
			t.Fatal("expected error for empty timing kind, got nil")
		}
		if !errors.Is(err, cudacmp.ErrIneligibleComparison) {
			t.Errorf("expected ErrIneligibleComparison, got %v", err)
		}
	})

	t.Run("reject unmeasured timing kind", func(t *testing.T) {
		_, err := cudacmp.CompareReceipts(cRec, goRec, cudatrial.TimingAPICall, 5*time.Millisecond)
		if err == nil {
			t.Fatal("expected error for unmeasured timing kind, got nil")
		}
		if !errors.Is(err, cudacmp.ErrIneligibleComparison) {
			t.Errorf("expected ErrIneligibleComparison, got %v", err)
		}
	})

	t.Run("reject mixed timing kinds when one arm lacks it", func(t *testing.T) {
		incompleteGo := cudatrial.Receipt{
			Name:     "candidate_go",
			Round:    1,
			Complete: true,
		}
		// incompleteGo only has WallElapsed, not GPUBusy
		_ = incompleteGo.RecordTiming(cudatrial.TimingWallElapsed, 90*time.Millisecond)

		_, err := cudacmp.CompareReceipts(cRec, incompleteGo, cudatrial.TimingGPUBusy, 5*time.Millisecond)
		if err == nil {
			t.Fatal("expected error when candidate lacks requested timing kind, got nil")
		}
		if !errors.Is(err, cudacmp.ErrIneligibleComparison) {
			t.Errorf("expected ErrIneligibleComparison, got %v", err)
		}
	})

	t.Run("reject incomplete baseline receipt", func(t *testing.T) {
		badC := cRec
		badC.Complete = false
		_, err := cudacmp.CompareReceipts(badC, goRec, cudatrial.TimingGPUBusy, 5*time.Millisecond)
		if err == nil {
			t.Fatal("expected error for incomplete baseline, got nil")
		}
	})

	t.Run("reject incomplete candidate receipt", func(t *testing.T) {
		badGo := goRec
		badGo.Complete = false
		_, err := cudacmp.CompareReceipts(cRec, badGo, cudatrial.TimingGPUBusy, 5*time.Millisecond)
		if err == nil {
			t.Fatal("expected error for incomplete candidate, got nil")
		}
	})

	t.Run("reject negative null drift", func(t *testing.T) {
		_, err := cudacmp.CompareReceipts(cRec, goRec, cudatrial.TimingGPUBusy, -1*time.Millisecond)
		if err == nil {
			t.Fatal("expected error for negative drift, got nil")
		}
	})
}

func TestCheckCompatibility(t *testing.T) {
	base := cudacmp.ComparisonIdentity{
		Version:      "v1",
		Phase:        "prefill",
		WorkloadName: "llama3-8b",
		WorkloadHash: "sha256:model123",
		TokensHash:   "sha256:tokens123",
		Revision:     "git:commitA",
		Device:       "NVIDIA H100",
		ComputeCap:   "sm_90a",
		Complete:     true,
	}

	cand := base
	cand.Revision = "git:commitB" // revision differences are permitted

	if err := cudacmp.CheckCompatibility(base, cand); err != nil {
		t.Fatalf("expected compatibility, got: %v", err)
	}

	cand.Phase = "decode"
	if err := cudacmp.CheckCompatibility(base, cand); err == nil {
		t.Fatal("expected error for phase mismatch, got nil")
	} else if !errors.Is(err, cudacmp.ErrIneligibleComparison) {
		t.Errorf("expected ErrIneligibleComparison, got %v", err)
	}
}

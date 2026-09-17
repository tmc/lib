package cudacmp_test

import (
	"errors"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudacmp"
)

func TestTimingDecomposition(t *testing.T) {
	decomp := cudacmp.TimingDecomposition{
		HostElapsedNs:      50_000_000,
		SummedKernelNs:     30_000_000,
		GPUIntervalUnionNs: 25_000_000,
		APICallNs:          2_000_000,
	}

	// 1. Verify separate non-additive fields are preserved
	if decomp.HostElapsedNs != 50_000_000 {
		t.Errorf("HostElapsedNs = %d, want 50ms", decomp.HostElapsedNs)
	}
	if decomp.SummedKernelNs != 30_000_000 {
		t.Errorf("SummedKernelNs = %d, want 30ms", decomp.SummedKernelNs)
	}
	if decomp.GPUIntervalUnionNs != 25_000_000 {
		t.Errorf("GPUIntervalUnionNs = %d, want 25ms", decomp.GPUIntervalUnionNs)
	}
	if decomp.APICallNs != 2_000_000 {
		t.Errorf("APICallNs = %d, want 2ms", decomp.APICallNs)
	}

	// 2. Reject naive sums that cause double-counting
	_, err := decomp.TotalDuration()
	if !errors.Is(err, cudacmp.ErrNaiveSumDoubleCounts) {
		t.Fatalf("expected ErrNaiveSumDoubleCounts, got: %v", err)
	}

	// 3. Unavailable measurements return Unmeasured rather than zero or wall-clock time.Since
	partialDecomp := cudacmp.TimingDecomposition{
		HostElapsedNs:      cudacmp.Unmeasured,
		SummedKernelNs:     cudacmp.Unmeasured,
		GPUIntervalUnionNs: 10_000_000,
		APICallNs:          cudacmp.Unmeasured,
	}
	if partialDecomp.HostElapsedNs != cudacmp.Unmeasured {
		t.Errorf("HostElapsedNs = %d, want Unmeasured (-1)", partialDecomp.HostElapsedNs)
	}
	if partialDecomp.SummedKernelNs != cudacmp.Unmeasured {
		t.Errorf("SummedKernelNs = %d, want Unmeasured (-1)", partialDecomp.SummedKernelNs)
	}

	// PhaseCensus timing decomposition
	pc := cudacmp.PhaseCensus{
		Phase:          cudacmp.PhasePrefill,
		SummedKernelNs: 12_000_000,
		GPUBusyNs:      10_000_000,
		HostElapsedNs:  cudacmp.Unmeasured,
		APICallNs:      cudacmp.Unmeasured,
	}
	td := pc.TimingDecomposition()
	if td.GPUIntervalUnionNs != 10_000_000 {
		t.Errorf("GPUIntervalUnionNs = %d, want 10ms", td.GPUIntervalUnionNs)
	}
	if td.HostElapsedNs != cudacmp.Unmeasured {
		t.Errorf("HostElapsedNs = %d, want Unmeasured", td.HostElapsedNs)
	}
}

func TestPhaseAlignment(t *testing.T) {
	// 1. Mismatched phase names must be rejected
	cPrefill := cudacmp.PhaseCensus{
		Phase: cudacmp.PhasePrefill,
		Families: map[string]cudacmp.KernelMetric{
			cudacmp.FamilyAttention: {
				Name:   "flash_attn",
				Family: cudacmp.FamilyAttention,
				Timing: cudacmp.LatencyDecomposition{DeviceNs: 5_000_000},
			},
		},
	}
	goDecode := cudacmp.PhaseCensus{
		Phase: cudacmp.PhaseDecode,
		Families: map[string]cudacmp.KernelMetric{
			cudacmp.FamilyAttention: {
				Name:   "flash_attn",
				Family: cudacmp.FamilyAttention,
				Timing: cudacmp.LatencyDecomposition{DeviceNs: 4_500_000},
			},
		},
	}

	_, err := cudacmp.ComparePhases(cPrefill, goDecode, 50_000, nil)
	if err == nil {
		t.Fatal("expected error comparing incompatible phases (prefill vs decode), got nil")
	}

	// 2. Missing arms labeled as unavailable/incomparable, with duration Unmeasured rather than zero
	goPrefillOneSided := cudacmp.PhaseCensus{
		Phase: cudacmp.PhasePrefill,
		Families: map[string]cudacmp.KernelMetric{
			cudacmp.FamilyNormPrep: {
				Name:   "rmsnorm",
				Family: cudacmp.FamilyNormPrep,
				Timing: cudacmp.LatencyDecomposition{DeviceNs: 1_000_000},
			},
		},
	}

	reports, err := cudacmp.ComparePhases(cPrefill, goPrefillOneSided, 50_000, nil)
	if err != nil {
		t.Fatalf("ComparePhases failed: %v", err)
	}

	var attnReport, normReport *cudacmp.DifferentialReport
	for i := range reports {
		if reports[i].Family == cudacmp.FamilyAttention {
			attnReport = &reports[i]
		}
		if reports[i].Family == cudacmp.FamilyNormPrep {
			normReport = &reports[i]
		}
	}

	if attnReport == nil || normReport == nil {
		t.Fatal("missing attention or norm reports")
	}

	// Attention was C-only: Go arm must be Unmeasured, not zero
	if attnReport.Availability != cudacmp.AvailableCOnly {
		t.Errorf("attn Availability = %v, want AvailableCOnly", attnReport.Availability)
	}
	if attnReport.Available {
		t.Errorf("attn Available = true, want false")
	}
	if attnReport.GoDurationNs != cudacmp.Unmeasured {
		t.Errorf("attn GoDurationNs = %d, want Unmeasured (-1), not zero", attnReport.GoDurationNs)
	}
	if attnReport.GoDuration() != time.Duration(cudacmp.Unmeasured) {
		t.Errorf("attn GoDuration() = %v, want Unmeasured", attnReport.GoDuration())
	}
	if attnReport.Delta() != time.Duration(cudacmp.Unmeasured) {
		t.Errorf("attn Delta() = %v, want Unmeasured", attnReport.Delta())
	}

	// Norm was Go-only: C arm must be Unmeasured, not zero
	if normReport.Availability != cudacmp.AvailableGoOnly {
		t.Errorf("norm Availability = %v, want AvailableGoOnly", normReport.Availability)
	}
	if normReport.Available {
		t.Errorf("norm Available = true, want false")
	}
	if normReport.CDurationNs != cudacmp.Unmeasured {
		t.Errorf("norm CDurationNs = %d, want Unmeasured (-1), not zero", normReport.CDurationNs)
	}
	if normReport.CDuration() != time.Duration(cudacmp.Unmeasured) {
		t.Errorf("norm CDuration() = %v, want Unmeasured", normReport.CDuration())
	}
}

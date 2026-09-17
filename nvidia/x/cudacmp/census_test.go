package cudacmp

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudacensus"
	"github.com/tmc/lib/nvidia/x/cudaroofline"
)

func findFixturePath(t *testing.T) string {
	// Check relative to current package directory
	path := filepath.Join("..", "cudacensus", "testdata", "capture")
	if _, err := os.Stat(path); err == nil {
		return path
	}
	t.Skip("fixture ../cudacensus/testdata/capture not found")
	return ""
}

func TestExtractPhaseCensusFromFixture(t *testing.T) {
	fixture := findFixturePath(t)
	c, err := cudacensus.ReadDir(fixture)
	if err != nil {
		t.Fatal(err)
	}

	prefillSpans := ExtractPhaseSpans(c, PhasePrefill)
	if len(prefillSpans) == 0 {
		t.Fatal("expected prefill spans in fixture")
	}

	decodeSpans := ExtractPhaseSpans(c, PhaseDecode)
	if len(decodeSpans) == 0 {
		t.Fatal("expected decode spans in fixture")
	}

	decodeCensus, err := ExtractPhaseCensus(c, PhaseDecode, nil, nil)
	if err != nil {
		t.Fatalf("ExtractPhaseCensus decode: %v", err)
	}

	if decodeCensus.Phase != PhaseDecode {
		t.Errorf("Phase = %v, want %v", decodeCensus.Phase, PhaseDecode)
	}
	if decodeCensus.SummedKernelNs <= 0 {
		t.Errorf("SummedKernelNs = %d, want > 0", decodeCensus.SummedKernelNs)
	}
	if decodeCensus.GPUBusyNs <= 0 {
		t.Errorf("GPUBusyNs = %d, want > 0", decodeCensus.GPUBusyNs)
	}
	if len(decodeCensus.Entries) == 0 {
		t.Error("expected non-empty census entries")
	}

	// Verify family aggregation
	if len(decodeCensus.Families) == 0 {
		t.Error("expected non-empty families map")
	}
	for fam, km := range decodeCensus.Families {
		if km.Family != fam {
			t.Errorf("km.Family = %q, want %q", km.Family, fam)
		}
		if km.Launches <= 0 {
			t.Errorf("family %s has launches <= 0", fam)
		}
	}
}

func TestComparePhases(t *testing.T) {
	cPhase := PhaseCensus{
		Phase:          PhasePrefill,
		SummedKernelNs: 20_000_000,
		GPUBusyNs:      18_000_000,
		Families: map[string]KernelMetric{
			FamilyExperts: {
				Family:   FamilyExperts,
				Launches: 10,
				Timing:   LatencyDecomposition{DeviceNs: 12_000_000},
			},
			FamilyAttention: {
				Family:   FamilyAttention,
				Launches: 5,
				Timing:   LatencyDecomposition{DeviceNs: 8_000_000},
			},
		},
	}

	goPhase := PhaseCensus{
		Phase:          PhasePrefill,
		SummedKernelNs: 16_000_000,
		GPUBusyNs:      14_000_000,
		Families: map[string]KernelMetric{
			FamilyExperts: {
				Family:   FamilyExperts,
				Launches: 10,
				Timing:   LatencyDecomposition{DeviceNs: 10_000_000},
			},
			FamilyAttention: {
				Family:   FamilyAttention,
				Launches: 5,
				Timing:   LatencyDecomposition{DeviceNs: 6_000_000},
			},
		},
	}

	nullDriftNs := int64(500_000) // 0.5 ms noise floor
	reports, err := ComparePhases(cPhase, goPhase, nullDriftNs, nil)
	if err != nil {
		t.Fatalf("ComparePhases failed: %v", err)
	}

	if len(reports) != 2 {
		t.Fatalf("reports = %d, want 2", len(reports))
	}

	// First is FamilyExperts
	r0 := reports[0]
	if r0.Family != FamilyExperts {
		t.Errorf("report[0].Family = %q, want %q", r0.Family, FamilyExperts)
	}
	if !r0.Available || r0.Availability != AvailableBoth {
		t.Errorf("report[0].Availability = %v, Available = %v; want AvailableBoth/true", r0.Availability, r0.Available)
	}
	if !r0.CObserved || !r0.GoObserved {
		t.Errorf("report[0] CObserved=%v, GoObserved=%v; want both true", r0.CObserved, r0.GoObserved)
	}
	if r0.CDurationNs != 12_000_000 || r0.GoDurationNs != 10_000_000 {
		t.Errorf("unexpected durations: C=%d, Go=%d", r0.CDurationNs, r0.GoDurationNs)
	}
	if r0.DeltaNs != -2_000_000 {
		t.Errorf("DeltaNs = %d, want -2000000", r0.DeltaNs)
	}
	// Speedup: 12ms / 10ms = 1.2x
	if diff := r0.Speedup - 1.2; diff < -1e-6 || diff > 1e-6 {
		t.Errorf("Speedup = %f, want 1.2", r0.Speedup)
	}
	if !r0.ExceedsDrift || !r0.StatisticallyValid {
		t.Errorf("ExceedsDrift = false, want true (|Delta| = 2ms > 0.5ms drift)")
	}

	// Check drift floor rejecting noise
	highDriftNs := int64(5_000_000) // 5 ms noise floor
	noiseReports, err := ComparePhases(cPhase, goPhase, highDriftNs, nil)
	if err != nil {
		t.Fatalf("ComparePhases high drift failed: %v", err)
	}
	if noiseReports[0].ExceedsDrift || noiseReports[0].StatisticallyValid {
		t.Errorf("expected ExceedsDrift = false with high drift floor")
	}
}

func TestComparePhasesWithRoofline(t *testing.T) {
	model, err := cudaroofline.NewModel("sm_121", cudaroofline.DeviceResident)
	if err != nil {
		t.Fatalf("NewModel: %v", err)
	}

	cPhase := PhaseCensus{
		Phase:          PhaseDecode,
		SummedKernelNs: 1_000_000,
		GPUBusyNs:      1_000_000,
		Families: map[string]KernelMetric{
			FamilyDenseQ8: {
				Family:     FamilyDenseQ8,
				Launches:   1,
				Timing:     LatencyDecomposition{DeviceNs: 1_000_000},
				BytesMoved: 200_000, // ~200 KB in 1 ms
				FLOPs:      400_000,
				Residency:  cudaroofline.DeviceResident,
			},
		},
	}

	goPhase := PhaseCensus{
		Phase:          PhaseDecode,
		SummedKernelNs: 800_000,
		GPUBusyNs:      800_000,
		Families: map[string]KernelMetric{
			FamilyDenseQ8: {
				Family:     FamilyDenseQ8,
				Launches:   1,
				Timing:     LatencyDecomposition{DeviceNs: 800_000},
				BytesMoved: 200_000,
				FLOPs:      400_000,
				Residency:  cudaroofline.DeviceResident,
			},
		},
	}

	reports, err := ComparePhases(cPhase, goPhase, 10_000, model)
	if err != nil {
		t.Fatalf("ComparePhases failed: %v", err)
	}
	if len(reports) != 1 {
		t.Fatalf("reports = %d, want 1", len(reports))
	}
	r := reports[0]
	if r.RooflineC == nil || r.RooflineGo == nil {
		t.Fatal("expected non-nil roofline points for both arms")
	}
	if r.RooflineGo.AttainedGBps <= 0 {
		t.Errorf("AttainedGBps = %f, want > 0", r.RooflineGo.AttainedGBps)
	}
}

func TestComparePhases_IncompatiblePhases(t *testing.T) {
	prefill := PhaseCensus{Phase: PhasePrefill}
	decode := PhaseCensus{Phase: PhaseDecode}
	empty := PhaseCensus{Phase: ""}

	if _, err := ComparePhases(prefill, decode, 1000, nil); err == nil {
		t.Error("expected error comparing prefill to decode, got nil")
	}
	if _, err := ComparePhases(empty, decode, 1000, nil); err == nil {
		t.Error("expected error comparing empty phase to decode, got nil")
	}
	if _, err := ComparePhases(prefill, empty, 1000, nil); err == nil {
		t.Error("expected error comparing prefill to empty phase, got nil")
	}
}

func TestComparePhases_InvalidDrift(t *testing.T) {
	p1 := PhaseCensus{Phase: PhasePrefill}
	p2 := PhaseCensus{Phase: PhasePrefill}

	if _, err := ComparePhases(p1, p2, -1, nil); err == nil {
		t.Error("expected error for negative null drift threshold, got nil")
	}
}

func TestComparePhases_OneSidedFamilies(t *testing.T) {
	cPhase := PhaseCensus{
		Phase:          PhasePrefill,
		SummedKernelNs: 12_000_000,
		Families: map[string]KernelMetric{
			FamilyExperts: {
				Family:   FamilyExperts,
				Launches: 5,
				Timing:   LatencyDecomposition{DeviceNs: 12_000_000},
			},
		},
	}

	goPhase := PhaseCensus{
		Phase:          PhasePrefill,
		SummedKernelNs: 8_000_000,
		Families: map[string]KernelMetric{
			FamilyAttention: {
				Family:   FamilyAttention,
				Launches: 3,
				Timing:   LatencyDecomposition{DeviceNs: 8_000_000},
			},
		},
	}

	reports, err := ComparePhases(cPhase, goPhase, 1000, nil)
	if err != nil {
		t.Fatalf("ComparePhases failed: %v", err)
	}

	if len(reports) != 2 {
		t.Fatalf("expected 2 reports, got %d", len(reports))
	}

	for _, r := range reports {
		if r.Available {
			t.Errorf("family %s: expected Available = false for one-sided family", r.Family)
		}
		if r.DeltaNs != 0 {
			t.Errorf("family %s: expected DeltaNs = 0, got %d", r.Family, r.DeltaNs)
		}
		if r.Speedup != 0 {
			t.Errorf("family %s: expected Speedup = 0, got %f", r.Family, r.Speedup)
		}
		if r.ExceedsDrift || r.StatisticallyValid {
			t.Errorf("family %s: expected ExceedsDrift = false for one-sided family", r.Family)
		}

		switch r.Family {
		case FamilyExperts:
			if r.Availability != AvailableCOnly {
				t.Errorf("experts: Availability = %v, want %v", r.Availability, AvailableCOnly)
			}
			if !r.CObserved || r.GoObserved {
				t.Errorf("experts: CObserved=%v, GoObserved=%v; want CObserved=true, GoObserved=false", r.CObserved, r.GoObserved)
			}
			if r.CDurationNs != 12_000_000 || r.GoDurationNs != 0 {
				t.Errorf("experts: CDurationNs=%d, GoDurationNs=%d; want 12ms and 0", r.CDurationNs, r.GoDurationNs)
			}
		case FamilyAttention:
			if r.Availability != AvailableGoOnly {
				t.Errorf("attention: Availability = %v, want %v", r.Availability, AvailableGoOnly)
			}
			if r.CObserved || !r.GoObserved {
				t.Errorf("attention: CObserved=%v, GoObserved=%v; want CObserved=false, GoObserved=true", r.CObserved, r.GoObserved)
			}
			if r.CDurationNs != 0 || r.GoDurationNs != 8_000_000 {
				t.Errorf("attention: CDurationNs=%d, GoDurationNs=%d; want 0 and 8ms", r.CDurationNs, r.GoDurationNs)
			}
		}
	}
}

func TestComparePhases_MeasuredZeroVsMissing(t *testing.T) {
	cPhase := PhaseCensus{
		Phase:          PhasePrefill,
		SummedKernelNs: 0,
		Families: map[string]KernelMetric{
			FamilyExperts: {
				Family:   FamilyExperts,
				Launches: 1,
				Timing:   LatencyDecomposition{DeviceNs: 0}, // explicitly measured zero
			},
		},
	}

	goPhase := PhaseCensus{
		Phase:          PhasePrefill,
		SummedKernelNs: 5_000_000,
		Families: map[string]KernelMetric{
			FamilyExperts: {
				Family:   FamilyExperts,
				Launches: 1,
				Timing:   LatencyDecomposition{DeviceNs: 5_000_000},
			},
		},
	}

	reports, err := ComparePhases(cPhase, goPhase, 1000, nil)
	if err != nil {
		t.Fatalf("ComparePhases failed: %v", err)
	}
	if len(reports) != 1 {
		t.Fatalf("expected 1 report, got %d", len(reports))
	}
	r := reports[0]
	// Measured zero means both were observed and comparison is available!
	if !r.Available || r.Availability != AvailableBoth {
		t.Errorf("expected Available = true for measured zero, got %v / %v", r.Available, r.Availability)
	}
	if !r.CObserved || !r.GoObserved {
		t.Errorf("expected both observed, got C=%v, Go=%v", r.CObserved, r.GoObserved)
	}
	if r.CDurationNs != 0 || r.GoDurationNs != 5_000_000 {
		t.Errorf("durations: C=%d, Go=%d", r.CDurationNs, r.GoDurationNs)
	}
	if r.DeltaNs != 5_000_000 {
		t.Errorf("DeltaNs = %d, want 5_000_000", r.DeltaNs)
	}
}

func TestIntervalUnion_TwoLaunches(t *testing.T) {
	// Acceptance test: Two launches [0, 10) and [5, 15) yield summed kernel duration 20 and busy union 15.
	launches := []cudacensus.Launch{
		{Name: "k1", StartNs: 0, EndNs: 10},
		{Name: "k2", StartNs: 5, EndNs: 15},
	}

	census, err := BuildPhaseCensus(PhasePrefill, launches, nil, nil)
	if err != nil {
		t.Fatalf("BuildPhaseCensus failed: %v", err)
	}

	if census.SummedKernelNs != 20 {
		t.Errorf("SummedKernelNs = %d, want 20", census.SummedKernelNs)
	}
	if census.GPUBusyNs != 15 {
		t.Errorf("GPUBusyNs = %d, want 15", census.GPUBusyNs)
	}
}

func TestIntervalUnion_NonOverlapping(t *testing.T) {
	intervals := []Interval{
		{StartNs: 0, EndNs: 10},
		{StartNs: 15, EndNs: 25},
	}
	union, err := UnionDuration(intervals)
	if err != nil {
		t.Fatalf("UnionDuration failed: %v", err)
	}
	if union != 20 {
		t.Errorf("UnionDuration = %d, want 20", union)
	}
}

func TestIntervalUnion_InvalidIntervals(t *testing.T) {
	tests := []struct {
		name      string
		intervals []Interval
	}{
		{
			name:      "end before start",
			intervals: []Interval{{StartNs: 10, EndNs: 5}},
		},
		{
			name:      "negative start",
			intervals: []Interval{{StartNs: -5, EndNs: 10}},
		},
		{
			name:      "negative end",
			intervals: []Interval{{StartNs: -10, EndNs: -5}},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := UnionDuration(tc.intervals)
			if err == nil {
				t.Fatalf("expected error for %s, got nil", tc.name)
			}
		})
	}
}

func TestBuildPhaseCensus_InvalidIntervals(t *testing.T) {
	badLaunches := []cudacensus.Launch{
		{Name: "k1", StartNs: 10, EndNs: 5},
	}
	if _, err := BuildPhaseCensus(PhasePrefill, badLaunches, nil, nil); err == nil {
		t.Error("expected error for bad interval in BuildPhaseCensus, got nil")
	}

	negLaunches := []cudacensus.Launch{
		{Name: "k1", StartNs: -1, EndNs: 5},
	}
	if _, err := BuildPhaseCensus(PhasePrefill, negLaunches, nil, nil); err == nil {
		t.Error("expected error for negative timestamp in BuildPhaseCensus, got nil")
	}
}


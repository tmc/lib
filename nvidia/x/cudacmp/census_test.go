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
	if decodeCensus.TotalNs <= 0 {
		t.Errorf("TotalNs = %d, want > 0", decodeCensus.TotalNs)
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
		Phase:   PhasePrefill,
		TotalNs: 20_000_000,
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
		Phase:   PhasePrefill,
		TotalNs: 16_000_000,
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
	reports := ComparePhases(cPhase, goPhase, nullDriftNs, nil)

	if len(reports) != 2 {
		t.Fatalf("reports = %d, want 2", len(reports))
	}

	// First is FamilyExperts
	r0 := reports[0]
	if r0.Family != FamilyExperts {
		t.Errorf("report[0].Family = %q, want %q", r0.Family, FamilyExperts)
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
	if !r0.StatisticallyValid {
		t.Errorf("StatisticallyValid = false, want true (|Delta| = 2ms > 0.5ms drift)")
	}

	// Check drift floor rejecting noise
	highDriftNs := int64(5_000_000) // 5 ms noise floor
	noiseReports := ComparePhases(cPhase, goPhase, highDriftNs, nil)
	if noiseReports[0].StatisticallyValid {
		t.Errorf("expected StatisticallyValid = false with high drift floor")
	}
}

func TestComparePhasesWithRoofline(t *testing.T) {
	model, err := cudaroofline.NewModel("sm_121", cudaroofline.DeviceResident)
	if err != nil {
		t.Fatalf("NewModel: %v", err)
	}

	cPhase := PhaseCensus{
		Phase:   PhaseDecode,
		TotalNs: 1_000_000,
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
		Phase:   PhaseDecode,
		TotalNs: 800_000,
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

	reports := ComparePhases(cPhase, goPhase, 10_000, model)
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

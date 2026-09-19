package cudaroofline

import (
	"strings"
	"testing"
)

// TestSinglePeakTableWithProvenance: a second peak table is the defect this
// checks for, and it does not announce itself as one -- it shows up later as two
// efficiencies for the same kernel. PeakTFLOPSFor must read knownCeilings
// rather than hold numbers of its own, and every entry must state where its
// numbers came from.
func TestSinglePeakTableWithProvenance(t *testing.T) {
	seen := map[string]float64{}
	for key, c := range knownCeilings {
		if err := ValidateCeiling(c); err != nil {
			t.Errorf("%s/%s: %v", key.arch, key.residency, err)
		}
		if c.Architecture != key.arch || c.Residency != key.residency {
			t.Errorf("%s/%s: entry describes %s/%s", key.arch, key.residency, c.Architecture, c.Residency)
		}
		// The compute peak belongs to the part, not to the memory path, so
		// two residencies of one architecture that disagree about it are two
		// tables wearing one name.
		if prev, ok := seen[key.arch]; ok && prev != c.PeakTFLOPS {
			t.Errorf("%s: two compute peaks, %v and %v", key.arch, prev, c.PeakTFLOPS)
		}
		seen[key.arch] = c.PeakTFLOPS

		peak, prov, err := PeakTFLOPSFor(key.arch, FP32)
		if err != nil {
			t.Errorf("PeakTFLOPSFor(%q, fp32): %v", key.arch, err)
			continue
		}
		if peak != c.PeakTFLOPS {
			t.Errorf("%s: PeakTFLOPSFor says %v, the table says %v", key.arch, peak, c.PeakTFLOPS)
		}
		if prov != c.ComputeProvenance {
			t.Errorf("%s: provenance %q, table says %q", key.arch, prov, c.ComputeProvenance)
		}
	}
}

// TestNoPeakInterpolationForUntestedSMVersion: an architecture between two known
// ones must fail, not land between them. Interpolation is the shape this defect
// takes, because the numbers look plausible.
func TestNoPeakInterpolationForUntestedSMVersion(t *testing.T) {
	for _, arch := range []string{"sm_87", "sm_88", "sm_100", "sm_100a", "sm_103", "sm_120"} {
		if _, err := LookupCeiling(arch, DeviceResident); err == nil {
			t.Errorf("LookupCeiling(%q) succeeded; no entry was ever measured or cited for it", arch)
		}
		if peak, _, err := PeakTFLOPSFor(arch, FP32); err == nil {
			t.Errorf("PeakTFLOPSFor(%q, fp32) returned %v; want an absence", arch, peak)
		}
	}
}

// TestSM100AndSM121PeaksDistinct: 00-common.md separates SM100 from SM121, and
// the failure mode is not two wrong numbers but one number answering for both.
// The table holds no SM100 entry, so the required behavior is that asking for
// SM100 fails rather than quietly returning GB10's figures.
func TestSM100AndSM121PeaksDistinct(t *testing.T) {
	gb10, err := LookupCeiling("sm_121", DeviceResident)
	if err != nil {
		t.Fatalf("sm_121: %v", err)
	}
	for _, arch := range []string{"sm_100", "sm_100a"} {
		c, err := LookupCeiling(arch, DeviceResident)
		if err == nil {
			if c.PeakTFLOPS == gb10.PeakTFLOPS && c.PeakMemoryGBps == gb10.PeakMemoryGBps {
				t.Errorf("%s returned GB10's ceiling; the two parts are distinct", arch)
			}
			continue
		}
		if !strings.Contains(err.Error(), arch) {
			t.Errorf("%s: error does not name the architecture: %v", arch, err)
		}
	}
}

// TestTensorPrecisionPeakReportedAbsent: an FP16 tensor peak derived from the
// FP32 pipe peak by a datasheet ratio would be a fabrication that every
// downstream efficiency inherits silently. Absence is the answer.
func TestTensorPrecisionPeakReportedAbsent(t *testing.T) {
	for _, p := range []Precision{FP64, FP16, TF32Tensor, FP16Tensor, BF16Tensor, FP8Tensor, INT8Tensor} {
		peak, _, err := PeakTFLOPSFor("sm_121", p)
		if err == nil {
			t.Errorf("PeakTFLOPSFor(sm_121, %s) returned %v; no such number was measured or cited", p, peak)
			continue
		}
		if peak != 0 {
			t.Errorf("%s: absent peak returned %v alongside its error", p, peak)
		}
		if !strings.Contains(err.Error(), string(p)) {
			t.Errorf("%s: error does not name the precision: %v", p, err)
		}
	}
}

func TestUnknownPrecisionRejected(t *testing.T) {
	if _, _, err := PeakTFLOPSFor("sm_121", Precision("fp4_tensor")); err == nil {
		t.Error("an unknown precision was accepted")
	} else if !strings.Contains(err.Error(), "unknown precision") {
		t.Errorf("error does not say the precision is unknown: %v", err)
	}
}

// TestUtilizationNotReportedAsEfficiency: the package must not offer a path from
// a busy-pipe fraction to an efficiency. Activity counters say a pipe was busy,
// not what it was busy with, and this package holds no precision attribution --
// so it must not accept a utilization figure at all.
func TestUtilizationNotReportedAsEfficiency(t *testing.T) {
	m := goldenModel(t)
	// Efficiency arises only from measured FLOPs, bytes and a duration. The
	// guard is structural: Evaluate takes counts, never a ratio.
	pt := m.Evaluate("k", 2_000_000, 162_400, 4000)
	if pt.Efficiency <= 0 {
		t.Fatalf("efficiency %v from real counts", pt.Efficiency)
	}
	if pt.LimitingAxis == NoLimit || pt.CeilingProvenance == "" {
		t.Error("an efficiency was reported without naming the limit it divides by")
	}
}

package cudaroofline

import "testing"

// TestKnownCeilingsAreMarked fails if any table entry is unusable or does not
// say where its numbers came from. A ninth architecture added without a
// provenance marker fails here.
func TestKnownCeilingsAreMarked(t *testing.T) {
	if len(knownCeilings) == 0 {
		t.Fatal("knownCeilings is empty")
	}
	for key, c := range knownCeilings {
		if err := ValidateCeiling(c); err != nil {
			t.Errorf("%v: %v", key, err)
		}
		if c.Architecture != key.arch {
			t.Errorf("%v: Architecture = %q, does not match its key", key, c.Architecture)
		}
		if c.Residency != key.residency {
			t.Errorf("%v: Residency = %q, does not match its key", key, c.Residency)
		}
	}
}

// TestOnlyGB10BandwidthIsMeasured guards the claim the package documentation
// makes. Marking an unmeasured figure Measured is the defect this package
// exists to prevent, so widening this set requires measurements to point at.
func TestOnlyGB10BandwidthIsMeasured(t *testing.T) {
	measured := map[string]bool{"sm_121": true, "sm_121a": true}
	for key, c := range knownCeilings {
		if c.MemoryProvenance == Measured && !measured[key.arch] {
			t.Errorf("%v: MemoryProvenance = %q, but no measurement for this part is recorded", key, Measured)
		}
		if c.MemoryProvenance != Measured && measured[key.arch] {
			t.Errorf("%v: MemoryProvenance = %q, want %q", key, c.MemoryProvenance, Measured)
		}
		if c.ComputeProvenance == Measured {
			t.Errorf("%v: ComputeProvenance = %q, but no compute peak has been measured here", key, Measured)
		}
	}
}

// TestUnconstrainedEfficiencyIsTheCutoff pins the named constant to the
// boundary Evaluate actually uses. The value is arbitrary and documented as
// such; what must not drift is the constant and the comparison agreeing, since
// the comment explaining the cutoff lives on the constant.
func TestUnconstrainedEfficiencyIsTheCutoff(t *testing.T) {
	// A round ceiling so efficiency is exactly attained GB/s over peak GB/s:
	// at intensity 1 FLOP/B the memory slope binds (the knee is at 40).
	model := NewModelWithCeiling(Ceiling{
		Architecture: "test", Residency: DeviceResident,
		PeakTFLOPS: 40.0, PeakMemoryGBps: 1000.0,
		ComputeProvenance: Unverified, MemoryProvenance: Unverified,
	})
	const durationNs = 1000000

	tests := []struct {
		name      string
		targetEff float64
		wantClass string
	}{
		{name: "just below the cutoff", targetEff: unconstrainedEfficiency - 0.01, wantClass: ClassUnconstrained},
		{name: "just above the cutoff", targetEff: unconstrainedEfficiency + 0.01, wantClass: ClassMemoryBound},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// eff = (bytes/durationNs) / PeakMemoryGBps, so bytes follows from
			// the target. flops = bytes fixes the intensity at 1 FLOP/B.
			bytes := uint64(tt.targetEff * model.Ceiling.PeakMemoryGBps * durationNs)
			pt := model.Evaluate(tt.name, bytes, bytes, durationNs)
			if diff := pt.Efficiency - tt.targetEff; diff < -1e-6 || diff > 1e-6 {
				t.Fatalf("setup: Efficiency = %v, want %v", pt.Efficiency, tt.targetEff)
			}
			if pt.Classification != tt.wantClass {
				t.Errorf("Classification = %q at efficiency %v, want %q (cutoff %v)",
					pt.Classification, pt.Efficiency, tt.wantClass, unconstrainedEfficiency)
			}
		})
	}
}

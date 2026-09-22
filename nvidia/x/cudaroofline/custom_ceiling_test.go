package cudaroofline

import (
	"bytes"
	"math"
	"testing"
)

func TestFloorUsesSuppliedComputeCeiling(t *testing.T) {
	for _, arch := range []string{"sm_121", "custom"} {
		t.Run(arch, func(t *testing.T) {
			m := NewModelWithCeiling(Ceiling{
				Architecture: arch, Residency: DeviceResident,
				PeakTFLOPS: 2, PeakMemoryGBps: 100,
				ComputeProvenance: Measured, MemoryProvenance: Unverified,
			})
			op := Operation{Name: "fixture", FLOPs: 20000, Bytes: 100, Precision: FP32}
			j := m.Judge(op, 9)
			if j.ComputeFloorNs != 10 || j.FloorNs != 10 || j.Axis != ComputeLimited || j.CeilingProvenance != Measured || j.Verdict != VerdictRejected {
				t.Fatalf("custom ceiling ignored: %+v", j)
			}
			if got := m.Judge(op, 10).Verdict; got != VerdictAccepted {
				t.Fatalf("at floor: %s", got)
			}
			var out bytes.Buffer
			if err := NewRecord(m, []Judgment{j}).Encode(&out); err != nil {
				t.Fatal(err)
			}
			r, err := DecodeRecord(&out)
			if err != nil || r.Device.PeakTFLOPS != 2 || r.Triples[0].ComputeFloorNs != 10 {
				t.Fatalf("record: %+v, %v", r, err)
			}
			for _, p := range []Precision{FP16, FP64, FP16Tensor, FP8Tensor} {
				op.Precision = p
				if j := m.Floor(op); j.ComputeFloorNs != FloorUnavailable || j.Basis != FloorFromMemoryOnly {
					t.Fatalf("invented %s peak: %+v", p, j)
				}
			}
			op.Precision = FP32
			m.Ceiling.PeakTFLOPS = 0
			if j := m.Floor(op); j.ComputeFloorNs != FloorUnavailable {
				t.Fatalf("restored unavailable peak from table: %+v", j)
			}
		})
	}
}

func TestValidateCeilingRejectsNonfinitePeaks(t *testing.T) {
	for _, peak := range []float64{math.NaN(), math.Inf(1), math.Inf(-1)} {
		for _, compute := range []bool{false, true} {
			c, err := LookupCeiling("sm_121", DeviceResident)
			if err != nil {
				t.Fatal(err)
			}
			if compute {
				c.PeakTFLOPS = peak
			} else {
				c.PeakMemoryGBps = peak
			}
			if err := ValidateCeiling(c); err == nil {
				t.Fatalf("accepted nonfinite peak: %+v", c)
			}
		}
	}
}

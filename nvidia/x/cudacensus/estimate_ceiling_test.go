package cudacensus

import (
	"math"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudaroofline"
)

func TestEstimateUsesModelComputePeak(t *testing.T) {
	for _, arch := range []string{"sm_121", "custom"} {
		t.Run(arch, func(t *testing.T) {
			m := cudaroofline.NewModelWithCeiling(cudaroofline.Ceiling{
				Architecture: arch, Residency: cudaroofline.DeviceResident,
				PeakTFLOPS: 2, PeakMemoryGBps: 100,
				ComputeProvenance: cudaroofline.Measured, MemoryProvenance: cudaroofline.Unverified,
			})
			in := Inputs{Readings: []Reading{{Counter: FP32Active, Value: 0.25, Available: true}}, Attribution: fp32Attribution(), Interval: time.Millisecond}
			for _, withMemory := range []bool{false, true} {
				if withMemory {
					in.Readings = append(in.Readings, Reading{Counter: DRAMBandwidthUtil, Value: 0.5, Available: true})
				}
				est := NewEstimate(m, in)
				if !est.Available || est.Rejected || est.AchievedTFLOPS != 0.5 || est.Derivation.PeakTFLOPS != 2 || est.Derivation.PeakProvenance != cudaroofline.Measured {
					t.Fatalf("withMemory=%v: %+v", withMemory, est)
				}
			}
			in.Attribution.Precision = cudaroofline.FP16Tensor
			if est := NewEstimate(m, in); est.Available || est.Derivation.UnavailableReason == "" {
				t.Fatalf("invented tensor peak: %+v", est)
			}
		})
	}
}

func TestEstimateRefusesInvalidModelComputePeak(t *testing.T) {
	in := Inputs{Readings: []Reading{{Counter: FP32Active, Value: 0.25, Available: true}}, Attribution: fp32Attribution(), Interval: time.Millisecond}
	for _, peak := range []float64{0, -1, math.NaN(), math.Inf(1), math.Inf(-1)} {
		m := testModel(t)
		m.Ceiling.PeakTFLOPS = peak
		if est := NewEstimate(m, in); est.Available || est.Confidence != ConfidenceUnavailable || est.Derivation.UnavailableReason == "" {
			t.Fatalf("peak %v: %+v", peak, est)
		}
	}
	m := testModel(t)
	m.Ceiling.ComputeProvenance = ""
	if est := NewEstimate(m, in); est.Available || est.Derivation.UnavailableReason == "" {
		t.Fatalf("unstated provenance: %+v", est)
	}
}

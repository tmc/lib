package cudacensus

import (
	"fmt"
	"math"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudaroofline"
)

func testPrecisionModel(t *testing.T) *cudaroofline.Model {
	t.Helper()
	c := cudaroofline.Ceiling{
		Architecture:      "sm_121a",
		Residency:         cudaroofline.DeviceResident,
		DeviceName:        "NVIDIA GB10 (constructed test profile)",
		MaxSMClockMHz:     2400,
		PeakTFLOPS:        40.0,
		PeakMemoryGBps:    239.6,
		ComputeProvenance: cudaroofline.Unverified,
		MemoryProvenance:  cudaroofline.Measured,
		PrecisionPeaks: map[cudaroofline.Precision]cudaroofline.PrecisionPeak{
			cudaroofline.FP64: {
				PeakTFLOPS: 1.25,
				Provenance: cudaroofline.Unverified,
				Citation:   "constructed test profile: dense FP64 pipe assertion",
			},
			cudaroofline.FP16: {
				PeakTFLOPS: 80.0,
				Provenance: cudaroofline.Unverified,
				Citation:   "constructed test profile: dense FP16 pipe assertion",
			},
			cudaroofline.TF32Tensor: {
				PeakTFLOPS: 80.0,
				Provenance: cudaroofline.Unverified,
				Citation:   "constructed test profile: dense TF32 tensor assertion",
			},
			cudaroofline.FP16Tensor: {
				PeakTFLOPS: 160.0,
				Provenance: cudaroofline.Unverified,
				Citation:   "constructed test profile: dense FP16 tensor assertion",
			},
			cudaroofline.BF16Tensor: {
				PeakTFLOPS: 160.0,
				Provenance: cudaroofline.Unverified,
				Citation:   "constructed test profile: dense BF16 tensor assertion",
			},
			cudaroofline.FP8Tensor: {
				PeakTFLOPS: 320.0,
				Provenance: cudaroofline.Unverified,
				Citation:   "constructed test profile: dense FP8 tensor assertion",
			},
		},
	}
	if err := cudaroofline.ValidateCeiling(c); err != nil {
		t.Fatalf("ValidateCeiling: %v", err)
	}
	return cudaroofline.NewModelWithCeiling(c)
}

func TestNewEstimateFloatingPrecisionCeilings(t *testing.T) {
	m := testPrecisionModel(t)

	for _, tc := range []struct {
		precision cudaroofline.Precision
		counter   Counter
		wantPeak  float64
	}{
		{cudaroofline.FP64, FP64Active, 1.25},
		{cudaroofline.FP32, FP32Active, 40.0},
		{cudaroofline.FP16, FP16Active, 80.0},
		{cudaroofline.TF32Tensor, HMMATensorActive, 80.0},
		{cudaroofline.FP16Tensor, HMMATensorActive, 160.0},
		{cudaroofline.BF16Tensor, HMMATensorActive, 160.0},
		{cudaroofline.FP8Tensor, HMMATensorActive, 320.0},
	} {
		t.Run(string(tc.precision), func(t *testing.T) {
			// High achieved throughput (75% of peak) with low DRAM utilization (0.01)
			// so the launch is on the compute plateau of tc.precision. For every
			// precision with wantPeak > 40.0, AchievedTFLOPS exceeds the 40.0 FP32
			// ceiling and would be falsely rejected if CeilingTFLOPS used FP32.
			est := NewEstimate(m, Inputs{
				Readings: []Reading{
					{Counter: tc.counter, Value: 0.75, Available: true},
					{Counter: DRAMBandwidthUtil, Value: 0.01, Available: true},
				},
				Attribution: Attribution{
					Precision: tc.precision,
					Method:    FromInstructionMix,
					Share:     1.0,
					Known:     true,
				},
				Interval: 10 * time.Millisecond,
			})
			if !est.Available || est.Rejected {
				t.Fatalf("%s estimate unavailable or rejected: available=%v rejected=%v reason=%q reject=%q",
					tc.precision, est.Available, est.Rejected, est.Derivation.UnavailableReason, est.RejectReason)
			}
			if est.Confidence != ConfidenceStaticMix {
				t.Errorf("Confidence = %q, want %q", est.Confidence, ConfidenceStaticMix)
			}
			wantAchieved := 0.75 * tc.wantPeak
			if math.Abs(est.AchievedTFLOPS-wantAchieved) > 1e-9 {
				t.Errorf("AchievedTFLOPS = %v, want %v", est.AchievedTFLOPS, wantAchieved)
			}
			wantCeiling, err := m.CeilingAtFor(tc.precision, est.ArithmeticIntensity)
			if err != nil {
				t.Fatalf("CeilingAtFor: %v", err)
			}
			if math.Abs(est.CeilingTFLOPS-wantCeiling) > 1e-9 {
				t.Errorf("CeilingTFLOPS = %v, want %v", est.CeilingTFLOPS, wantCeiling)
			}
			if est.Derivation.PeakTFLOPS != tc.wantPeak {
				t.Errorf("Derivation.PeakTFLOPS = %v, want %v", est.Derivation.PeakTFLOPS, tc.wantPeak)
			}
			if est.Derivation.DeviceName != m.Ceiling.DeviceName || est.Derivation.MaxSMClockMHz != m.Ceiling.MaxSMClockMHz {
				t.Errorf("Derivation identity = (%q, %d), want (%q, %d)",
					est.Derivation.DeviceName, est.Derivation.MaxSMClockMHz,
					m.Ceiling.DeviceName, m.Ceiling.MaxSMClockMHz)
			}
			if tc.precision != cudaroofline.FP32 && est.Derivation.PeakCitation == "" {
				t.Errorf("Derivation.PeakCitation is empty for %s", tc.precision)
			}

			// Without DRAMBandwidthUtil, CeilingTFLOPS must equal tc.wantPeak (not FP32 40.0).
			noBW := NewEstimate(m, Inputs{
				Readings: []Reading{
					{Counter: tc.counter, Value: 0.75, Available: true},
				},
				Attribution: Attribution{
					Precision: tc.precision,
					Method:    FromInstructionMix,
					Share:     1.0,
					Known:     true,
				},
				Interval: 10 * time.Millisecond,
			})
			if !noBW.Available || noBW.Rejected || noBW.CeilingTFLOPS != tc.wantPeak {
				t.Errorf("%s without DRAMBandwidthUtil: available=%v rejected=%v ceiling=%v, want available with ceiling %v",
					tc.precision, noBW.Available, noBW.Rejected, noBW.CeilingTFLOPS, tc.wantPeak)
			}
		})
	}
}

func TestNewEstimateStaticMixOnlyForAmbiguousTensorFormats(t *testing.T) {
	m := testPrecisionModel(t)

	// Ambiguous HMMA tensor formats (TF32Tensor, BF16Tensor, FP8Tensor) must be
	// refused when attributed via FromCounterRatio, even when the model has a
	// validated peak for them.
	for _, p := range []cudaroofline.Precision{
		cudaroofline.TF32Tensor,
		cudaroofline.BF16Tensor,
		cudaroofline.FP8Tensor,
	} {
		est := NewEstimate(m, Inputs{
			Readings: []Reading{
				{Counter: HMMATensorActive, Value: 0.5, Available: true},
				{Counter: DRAMBandwidthUtil, Value: 0.2, Available: true},
			},
			Attribution: Attribution{
				Precision: p,
				Method:    FromCounterRatio,
				Share:     0.9,
				Known:     true,
			},
			Interval: 10 * time.Millisecond,
		})
		if est.Available || est.Confidence != ConfidenceUnavailable {
			t.Fatalf("%s with FromCounterRatio produced available estimate: %+v", p, est)
		}
		if !strings.Contains(est.Derivation.UnavailableReason, string(FromInstructionMix)) {
			t.Errorf("%s UnavailableReason = %q, want it to require %s", p, est.Derivation.UnavailableReason, FromInstructionMix)
		}
	}

	// Calibrated FP16Tensor counter-ratio path remains supported and carries ConfidenceCounterRatio.
	fp16Ratio := NewEstimate(m, Inputs{
		Readings: []Reading{
			{Counter: HMMATensorActive, Value: 0.5, Available: true},
			{Counter: DRAMBandwidthUtil, Value: 0.2, Available: true},
		},
		Attribution: Attribution{
			Precision: cudaroofline.FP16Tensor,
			Method:    FromCounterRatio,
			Share:     0.9,
			Known:     true,
		},
		Interval: 10 * time.Millisecond,
	})
	if !fp16Ratio.Available || fp16Ratio.Confidence != ConfidenceCounterRatio || fp16Ratio.AchievedTFLOPS != 80.0 {
		t.Fatalf("FP16Tensor with FromCounterRatio = %+v, want Available=true Confidence=%q AchievedTFLOPS=80.0",
			fp16Ratio, ConfidenceCounterRatio)
	}

	// AnyTensorActive alone must never substitute for HMMATensorActive.
	anyOnly := NewEstimate(m, Inputs{
		Readings: []Reading{
			{Counter: AnyTensorActive, Value: 0.5, Available: true},
			{Counter: DRAMBandwidthUtil, Value: 0.2, Available: true},
		},
		Attribution: Attribution{
			Precision: cudaroofline.FP16Tensor,
			Method:    FromInstructionMix,
			Share:     1.0,
			Known:     true,
		},
		Interval: 10 * time.Millisecond,
	})
	if anyOnly.Available || !strings.Contains(anyOnly.Derivation.UnavailableReason, "no activity counter available") {
		t.Fatalf("AnyTensorActive alone produced %+v, want unavailable missing activity counter", anyOnly)
	}
}

func TestNewEstimateINT8TensorUnitBoundaryAndIdlePreservation(t *testing.T) {
	m := testPrecisionModel(t)

	// Non-idle INT8Tensor work must remain unavailable in TFLOPS fields.
	activeINT8 := NewEstimate(m, Inputs{
		Readings: []Reading{
			{Counter: SMActive, Value: 0.8, Available: true},
			{Counter: IMMATensorActive, Value: 0.8, Available: true},
			{Counter: DRAMBandwidthUtil, Value: 0.2, Available: true},
		},
		Attribution: Attribution{
			Precision: cudaroofline.INT8Tensor,
			Method:    FromInstructionMix,
			Share:     1.0,
			Known:     true,
		},
		Interval: 10 * time.Millisecond,
	})
	if activeINT8.Available || activeINT8.Confidence != ConfidenceUnavailable {
		t.Fatalf("non-idle INT8Tensor produced available estimate: %+v", activeINT8)
	}
	if activeINT8.AchievedTFLOPS != 0 || activeINT8.CeilingTFLOPS != 0 {
		t.Errorf("non-idle INT8Tensor populated TFLOPS fields: achieved=%v ceiling=%v", activeINT8.AchievedTFLOPS, activeINT8.CeilingTFLOPS)
	}
	if !strings.Contains(activeINT8.Derivation.UnavailableReason, "integer TOPS") ||
		!strings.Contains(activeINT8.Derivation.UnavailableReason, "FP32 non-tensor peak only") {
		t.Errorf("INT8Tensor UnavailableReason = %q, want integer TOPS and FP32-only explanation", activeINT8.Derivation.UnavailableReason)
	}

	// Broad-zero idle window still returns ConfidenceIdle (zero work is not positive integer throughput).
	idleINT8 := NewEstimate(m, Inputs{
		Readings: []Reading{
			{Counter: SMActive, Value: 0, Available: true},
			{Counter: IMMATensorActive, Value: 0, Available: true},
			{Counter: DRAMBandwidthUtil, Value: 0, Available: true},
		},
		Attribution: Attribution{
			Precision: cudaroofline.INT8Tensor,
			Method:    FromInstructionMix,
			Share:     1.0,
			Known:     true,
		},
		Interval: 10 * time.Millisecond,
	})
	if !idleINT8.Available || idleINT8.Confidence != ConfidenceIdle || idleINT8.AchievedTFLOPS != 0 {
		t.Fatalf("broad-zero idle INT8Tensor = %+v, want Available=true Confidence=%q AchievedTFLOPS=0",
			idleINT8, ConfidenceIdle)
	}
}

func TestNewEstimateDerivationSnapshotIsolation(t *testing.T) {
	m := testPrecisionModel(t)
	in := Inputs{
		Readings: []Reading{
			{Counter: HMMATensorActive, Value: 0.5, Available: true},
			{Counter: DRAMBandwidthUtil, Value: 0.1, Available: true},
		},
		Attribution: Attribution{
			Precision: cudaroofline.FP16Tensor,
			Method:    FromInstructionMix,
			Share:     1.0,
			Known:     true,
		},
		Interval: 10 * time.Millisecond,
	}
	first := NewEstimate(m, in)
	if !first.Available || first.AchievedTFLOPS != 80.0 {
		t.Fatalf("first estimate = %+v, want AchievedTFLOPS=80.0", first)
	}

	// Mutating m.Ceiling after first is returned must not alter first.Derivation,
	// while a subsequent NewEstimate call reflects the updated Model.Ceiling.
	m.Ceiling.DeviceName = "NVIDIA GB10 (updated profile)"
	m.Ceiling.PrecisionPeaks[cudaroofline.FP16Tensor] = cudaroofline.PrecisionPeak{
		PeakTFLOPS: 200.0,
		Provenance: cudaroofline.Measured,
		Citation:   "constructed test profile: updated FP16 tensor peak",
	}

	if first.Derivation.DeviceName != "NVIDIA GB10 (constructed test profile)" || first.Derivation.PeakTFLOPS != 160.0 {
		t.Errorf("first.Derivation mutated after return: %+v", first.Derivation)
	}

	second := NewEstimate(m, in)
	if !second.Available || second.AchievedTFLOPS != 100.0 || second.Derivation.PeakTFLOPS != 200.0 || second.Derivation.DeviceName != "NVIDIA GB10 (updated profile)" {
		t.Fatalf("second estimate after m.Ceiling update = %+v, want AchievedTFLOPS=100.0 PeakTFLOPS=200.0", second)
	}
}

func ExampleNewEstimate_precisionCeiling() {
	m := cudaroofline.NewModelWithCeiling(cudaroofline.Ceiling{
		Architecture:      "sm_121a",
		Residency:         cudaroofline.DeviceResident,
		DeviceName:        "NVIDIA GB10 (constructed example profile)",
		MaxSMClockMHz:     2400,
		PeakTFLOPS:        40.0,
		PeakMemoryGBps:    239.6,
		ComputeProvenance: cudaroofline.Unverified,
		MemoryProvenance:  cudaroofline.Measured,
		PrecisionPeaks: map[cudaroofline.Precision]cudaroofline.PrecisionPeak{
			cudaroofline.FP16Tensor: {
				PeakTFLOPS: 160.0,
				Provenance: cudaroofline.Unverified,
				Citation:   "constructed example profile: dense FP16 tensor assertion",
			},
		},
	})

	est := NewEstimate(m, Inputs{
		Readings: []Reading{
			{Counter: HMMATensorActive, Value: 0.5, Available: true},
			{Counter: DRAMBandwidthUtil, Value: 0.05, Available: true},
		},
		Attribution: Attribution{
			Precision: cudaroofline.FP16Tensor,
			Method:    FromInstructionMix,
			Share:     1.0,
			Known:     true,
		},
		Interval: 10 * time.Millisecond,
	})
	fmt.Printf("available=%v confidence=%s achieved=%.1f ceiling=%.1f peak=%.1f\n",
		est.Available, est.Confidence, est.AchievedTFLOPS, est.CeilingTFLOPS, est.Derivation.PeakTFLOPS)
	// Output:
	// available=true confidence=static_mix achieved=80.0 ceiling=160.0 peak=160.0
}

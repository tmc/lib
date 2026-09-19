package cudacensus

import (
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudaroofline"
)

// testModel is a model over the one architecture whose memory ceiling is
// measured, so that a test asserting on provenance is asserting on a real
// distinction rather than on a fixture's invention.
func testModel(t *testing.T) *cudaroofline.Model {
	t.Helper()
	m, err := cudaroofline.NewModel("sm_121", cudaroofline.DeviceResident)
	if err != nil {
		t.Fatalf("NewModel: %v", err)
	}
	return m
}

func fp32Attribution() Attribution {
	return Attribution{
		Precision: cudaroofline.FP32,
		Method:    FromInstructionMix,
		Share:     1,
		Known:     true,
	}
}

// TestEstimateCarriesDerivationRecord: the derivation is not optional decoration.
// Every field a reader would need to disagree with the figure is present, and the
// absent counter is still in the record as absent.
func TestEstimateCarriesDerivationRecord(t *testing.T) {
	m := testModel(t)
	in := Inputs{
		Readings: []Reading{
			{Counter: FP32Active, Value: 0.25, Available: true},
			{Counter: DRAMBandwidthUtil, Value: 0.5, Available: true},
			{Counter: HMMATensorActive, Available: false, Reason: "GPM metric not supported"},
		},
		Attribution: fp32Attribution(),
		Interval:    10 * time.Millisecond,
	}
	est := NewEstimate(m, in)
	if !est.Available {
		t.Fatalf("estimate unavailable: %s", est.Derivation.UnavailableReason)
	}

	d := est.Derivation
	if d.Architecture != m.Ceiling.Architecture {
		t.Errorf("Architecture = %q, want %q", d.Architecture, m.Ceiling.Architecture)
	}
	if d.Residency != m.Ceiling.Residency {
		t.Errorf("Residency = %q, want %q", d.Residency, m.Ceiling.Residency)
	}
	if d.PeakTFLOPS != m.Ceiling.PeakTFLOPS {
		t.Errorf("PeakTFLOPS = %v, want %v", d.PeakTFLOPS, m.Ceiling.PeakTFLOPS)
	}
	if d.PeakProvenance == "" {
		t.Error("the derivation records no provenance for the peak it scaled by")
	}
	if d.Attribution.Method != FromInstructionMix {
		t.Errorf("Attribution.Method = %q, want %q", d.Attribution.Method, FromInstructionMix)
	}
	if d.Interval != in.Interval {
		t.Errorf("Interval = %v, want %v", d.Interval, in.Interval)
	}
	if len(d.Readings) != len(in.Readings) {
		t.Errorf("Readings has %d entries, want %d: the absent counter belongs in the record", len(d.Readings), len(in.Readings))
	}
	if d.Note == "" {
		t.Error("the derivation states no qualification for an intensity built from two different bandwidth figures")
	}

	// 25% of a 40 TFLOPS peak.
	if want := 0.25 * m.Ceiling.PeakTFLOPS; est.AchievedTFLOPS != want {
		t.Errorf("AchievedTFLOPS = %v, want %v", est.AchievedTFLOPS, want)
	}
	if est.Confidence != ConfidenceStaticMix {
		t.Errorf("Confidence = %q, want %q", est.Confidence, ConfidenceStaticMix)
	}
}

// TestIdleDeviceEstimatesZeroWithConfidence: idle is a measurement. It must be
// distinguishable from an unavailable counter, which is the case below it.
func TestIdleDeviceEstimatesZeroWithConfidence(t *testing.T) {
	m := testModel(t)
	idle := NewEstimate(m, Inputs{
		Readings: []Reading{
			{Counter: FP32Active, Value: 0, Available: true},
			{Counter: DRAMBandwidthUtil, Value: 0, Available: true},
		},
		Attribution: fp32Attribution(),
		Interval:    time.Millisecond,
	})
	if !idle.Available {
		t.Fatalf("an idle device produced no estimate: %s", idle.Derivation.UnavailableReason)
	}
	if idle.AchievedTFLOPS != 0 {
		t.Errorf("AchievedTFLOPS = %v, want 0", idle.AchievedTFLOPS)
	}
	if idle.Confidence != ConfidenceIdle {
		t.Errorf("Confidence = %q, want %q", idle.Confidence, ConfidenceIdle)
	}

	silent := NewEstimate(m, Inputs{
		Readings: []Reading{
			{Counter: FP32Active, Available: false, Reason: "GPM metric not supported"},
			{Counter: DRAMBandwidthUtil, Available: false, Reason: "GPM metric not supported"},
		},
		Attribution: fp32Attribution(),
		Interval:    time.Millisecond,
	})
	if silent.Available {
		t.Fatal("a device that answered nothing produced an available estimate")
	}
	if silent.Confidence != ConfidenceUnavailable {
		t.Errorf("Confidence = %q, want %q", silent.Confidence, ConfidenceUnavailable)
	}
	if silent.AchievedTFLOPS != 0 || idle.Confidence == silent.Confidence {
		t.Error("idle and unanswered are not distinguishable in the result")
	}
}

// TestEstimateAboveRooflineRejectedWithInputs: an impossible figure is a bug in
// the estimator, reported with everything needed to find it.
//
// The case that can produce one is a mis-scaled activity reading. GPM reports
// some metrics as percentages and some as fractions, and a caller who passes 85.0
// where 0.85 was meant gets a figure eighty-five times the device's peak. With
// activity in [0, 1] the estimate cannot exceed the plateau at all, which is why
// this guard is about the input scale and not about the arithmetic.
func TestEstimateAboveRooflineRejectedWithInputs(t *testing.T) {
	m := testModel(t)
	est := NewEstimate(m, Inputs{
		Readings: []Reading{
			{Counter: FP32Active, Value: 85.0, Available: true}, // a percentage where a fraction was meant
			{Counter: DRAMBandwidthUtil, Value: 0.5, Available: true},
		},
		Attribution: fp32Attribution(),
		Interval:    time.Millisecond,
	})
	if !est.Rejected {
		t.Fatalf("a percentage-scaled activity reading was accepted: %.4f TFLOPS against a %.4f ceiling", est.AchievedTFLOPS, est.CeilingTFLOPS)
	}
	if est.Available {
		t.Error("a rejected estimate is still marked available")
	}
	for _, want := range []string{"TFLOPS", "ceiling", "intensity", string(cudaroofline.FP32), string(FromInstructionMix)} {
		if !strings.Contains(est.RejectReason, want) {
			t.Errorf("RejectReason = %q, want it to name %q", est.RejectReason, want)
		}
	}
}

// TestFractionalActivityCannotExceedThePlateau records the property the guard
// above rests on: with activity in [0, 1] the estimate is bounded by the peak by
// construction, so a rejection means the inputs were wrong, not the arithmetic.
func TestFractionalActivityCannotExceedThePlateau(t *testing.T) {
	m := testModel(t)
	for _, activity := range []float64{0.01, 0.5, 0.99, 1.0} {
		for _, util := range []float64{0.01, 0.5, 1.0} {
			est := NewEstimate(m, Inputs{
				Readings: []Reading{
					{Counter: FP32Active, Value: activity, Available: true},
					{Counter: DRAMBandwidthUtil, Value: util, Available: true},
				},
				Attribution: fp32Attribution(),
				Interval:    time.Millisecond,
			})
			if est.Rejected {
				t.Errorf("activity %v at util %v was rejected: %s", activity, util, est.RejectReason)
			}
			if est.AchievedTFLOPS > m.Ceiling.PeakTFLOPS {
				t.Errorf("activity %v produced %v TFLOPS above the %v peak", activity, est.AchievedTFLOPS, m.Ceiling.PeakTFLOPS)
			}
		}
	}
}

// TestTensorPrecisionEstimateUnavailableNotZero: the peak table holds FP32
// non-tensor peaks, so a tensor attribution has no multiplier. The honest result
// is no estimate, and the reason says the peak is what is missing.
func TestTensorPrecisionEstimateUnavailableNotZero(t *testing.T) {
	m := testModel(t)
	readings := []Reading{
		{Counter: HMMATensorActive, Value: 0.8, Available: true},
		{Counter: DRAMBandwidthUtil, Value: 0.2, Available: true},
	}
	est := NewEstimate(m, Inputs{
		Readings: readings,
		Attribution: Attribution{
			Precision: cudaroofline.FP16Tensor,
			Method:    FromInstructionMix,
			Share:     1,
			Known:     true,
		},
		Interval: time.Millisecond,
	})
	if est.Available {
		t.Fatalf("estimated %v TFLOPS for a precision with no peak", est.AchievedTFLOPS)
	}
	if est.Confidence != ConfidenceUnavailable {
		t.Errorf("Confidence = %q, want %q", est.Confidence, ConfidenceUnavailable)
	}
	if !strings.Contains(est.Derivation.UnavailableReason, "peak") {
		t.Errorf("UnavailableReason = %q, want it to name the missing peak", est.Derivation.UnavailableReason)
	}
	if len(est.Derivation.Readings) != len(readings) {
		t.Error("an unavailable estimate discarded the counters it did read")
	}
}

// TestUnattributedPrecisionEstimatesNothing: without a precision the activity is
// not a throughput, and a caller must not receive a number anyway.
func TestUnattributedPrecisionEstimatesNothing(t *testing.T) {
	m := testModel(t)
	est := NewEstimate(m, Inputs{
		Readings: []Reading{
			{Counter: AnyTensorActive, Value: 0.95, Available: true},
		},
		Attribution: Attribution{Method: Unattributed, Reason: "only any_tensor_active is available"},
		Interval:    time.Millisecond,
	})
	if est.Available {
		t.Fatalf("estimated %v TFLOPS with no attributed precision", est.AchievedTFLOPS)
	}
	if !strings.Contains(est.Derivation.UnavailableReason, "unattributed") {
		t.Errorf("UnavailableReason = %q, want it to say the precision was unattributed", est.Derivation.UnavailableReason)
	}
}

// TestEstimateRefusesUnknownInterval: a rate needs a denominator, and host wall
// time is not one.
func TestEstimateRefusesUnknownInterval(t *testing.T) {
	m := testModel(t)
	est := NewEstimate(m, Inputs{
		Readings:    []Reading{{Counter: FP32Active, Value: 0.5, Available: true}},
		Attribution: fp32Attribution(),
	})
	if est.Available {
		t.Fatal("estimated a rate over an interval of unknown length")
	}
	if !strings.Contains(est.Derivation.UnavailableReason, "interval") {
		t.Errorf("UnavailableReason = %q, want it to name the missing interval", est.Derivation.UnavailableReason)
	}
}

// TestConfidenceReflectsAttributionMethod: the two methods are not equally good
// and the estimate says which it used.
func TestConfidenceReflectsAttributionMethod(t *testing.T) {
	m := testModel(t)
	readings := []Reading{
		{Counter: FP32Active, Value: 0.1, Available: true},
		{Counter: DRAMBandwidthUtil, Value: 0.5, Available: true},
	}
	for _, tc := range []struct {
		method Method
		want   Confidence
	}{
		{FromInstructionMix, ConfidenceStaticMix},
		{FromCounterRatio, ConfidenceCounterRatio},
	} {
		attr := fp32Attribution()
		attr.Method = tc.method
		est := NewEstimate(m, Inputs{Readings: readings, Attribution: attr, Interval: time.Millisecond})
		if est.Confidence != tc.want {
			t.Errorf("%s: Confidence = %q, want %q", tc.method, est.Confidence, tc.want)
		}
	}
}

// TestUnmeasuredWhenCounterUnavailable: 00-common.md's absence policy reaches the
// estimate, not only the reading. A missing activity counter yields an estimate
// that is unavailable and names the counter, and every field a caller might read
// instead stays zero rather than being filled in from the other counters that were
// available.
func TestUnmeasuredWhenCounterUnavailable(t *testing.T) {
	m := testModel(t)
	est := NewEstimate(m, Inputs{
		Readings: []Reading{
			{Counter: FP32Active, Available: false, Reason: "GPM metric not supported"},
			{Counter: DRAMBandwidthUtil, Value: 0.5, Available: true},
		},
		Attribution: fp32Attribution(),
		Interval:    time.Millisecond,
	})
	if est.Available {
		t.Fatalf("estimated %v TFLOPS from an unavailable activity counter", est.AchievedTFLOPS)
	}
	if est.Confidence != ConfidenceUnavailable {
		t.Errorf("Confidence = %q, want %q", est.Confidence, ConfidenceUnavailable)
	}
	if !strings.Contains(est.Derivation.UnavailableReason, string(cudaroofline.FP32)) {
		t.Errorf("UnavailableReason = %q, want it to name the precision whose counter was absent", est.Derivation.UnavailableReason)
	}
	if est.AchievedTFLOPS != 0 || est.ArithmeticIntensity != 0 || est.CeilingTFLOPS != 0 {
		t.Errorf("an unavailable estimate carries numbers: %+v", est)
	}
	// The bandwidth counter that was available must not be promoted into an
	// intensity on its own: intensity is a ratio and it has no numerator here.
	if est.ArithmeticIntensity != 0 {
		t.Error("an intensity was derived without a FLOPs figure")
	}
}

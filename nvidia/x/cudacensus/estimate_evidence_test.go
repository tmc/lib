package cudacensus

import (
	"reflect"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/nvml"
)

func TestEstimateRetainsReadingsAcrossSamplerReuse(t *testing.T) {
	for _, tt := range []struct {
		name      string
		value     float64
		interval  time.Duration
		available bool
		rejected  bool
	}{
		{"accepted", 0.25, time.Millisecond, true, false},
		{"idle", 0, time.Millisecond, true, false},
		{"rejected", 25, time.Millisecond, false, true},
		{"unavailable", 0.25, 0, false, false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			want := []Counter{FP32Active, SMActive}
			dst := []Reading{
				{Counter: FP32Active, Value: tt.value, Available: true},
				{Counter: SMActive, Reason: "not supported"},
			}
			if tt.name == "idle" {
				dst[1] = Reading{Counter: SMActive, Available: true}
			}
			before := append([]Reading(nil), dst...)
			est := NewEstimate(testModel(t), Inputs{Readings: dst, Attribution: fp32Attribution(), Interval: tt.interval})
			if est.Available != tt.available || est.Rejected != tt.rejected {
				t.Fatalf("wrong result path: %+v", est)
			}
			// The actual GPM decoder writes the next tick into the same buffer.
			fillReadings(dst, want, recordedBlock(t, want, []nvml.Result{nvml.ErrNotSupported, nvml.Success}, []float64{0, 50}))
			if !reflect.DeepEqual(est.Derivation.Readings, before) {
				t.Fatalf("next tick changed retained evidence: %+v", est.Derivation.Readings)
			}
			// The caller may edit its owned estimate without altering the sampler.
			est.Derivation.Readings[0].Reason = "annotated"
			if dst[0].Reason == "annotated" {
				t.Fatal("estimate shares sampler storage")
			}
		})
	}
}

func TestEstimatesHaveIndependentEvidence(t *testing.T) {
	in := Inputs{Readings: []Reading{{Counter: FP32Active, Value: 0.5, Available: true}}, Attribution: fp32Attribution(), Interval: time.Millisecond}
	m := testModel(t)
	first, second := NewEstimate(m, in), NewEstimate(m, in)
	first.Derivation.Readings[0].Value = 1
	if second.Derivation.Readings[0].Value != 0.5 || in.Readings[0].Value != 0.5 {
		t.Fatal("estimates or input share retained storage")
	}
}

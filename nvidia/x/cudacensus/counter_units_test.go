package cudacensus

import (
	"math"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/nvml"
)

func TestGPMPercentageToEstimate(t *testing.T) {
	want := []Counter{FP32Active}
	dst := make([]Reading, 1)
	fillReadings(dst, want, recordedBlock(t, want, []nvml.Result{nvml.Success}, []float64{25}))
	m := testModel(t)
	est := NewEstimate(m, Inputs{Readings: dst, Attribution: fp32Attribution(), Interval: time.Millisecond})
	if !est.Available || est.AchievedTFLOPS != m.Ceiling.PeakTFLOPS/4 {
		t.Fatalf("percentage not normalized before estimation: %+v", est)
	}
}

func TestGPMPercentageBounds(t *testing.T) {
	for _, c := range AllCounters() {
		t.Run(string(c), func(t *testing.T) {
			want := []Counter{c}
			dst := []Reading{{Counter: c, Value: 0.5, Available: true}}
			for _, value := range []float64{0, 100, -1, 101, math.NaN(), math.Inf(1), math.Inf(-1)} {
				fillReadings(dst, want, recordedBlock(t, want, []nvml.Result{nvml.Success}, []float64{value}))
				if value == 0 || value == 100 {
					if !dst[0].Available || dst[0].Value != value/100 {
						t.Fatalf("boundary %v: %+v", value, dst[0])
					}
				} else if dst[0].Available || dst[0].Value != 0 || dst[0].Reason == "" {
					t.Fatalf("invalid %v retained: %+v", value, dst[0])
				}
			}
		})
	}
}

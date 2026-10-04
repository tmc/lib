package cudafuse

import (
	"errors"
	"testing"
	"time"
)

type mhcTimeMeasurement struct {
	Status      string
	Nanoseconds *int64
}

func mhcGPUEventTime(d time.Duration) (mhcTimeMeasurement, error) {
	if d <= 0 {
		return mhcTimeMeasurement{Status: "Unmeasured"}, errors.New("mhc: unavailable or invalid GPU event duration")
	}
	n := int64(d)
	return mhcTimeMeasurement{Status: "Measured", Nanoseconds: &n}, nil
}
func TestMHCTimingChannels(t *testing.T) {
	for _, d := range []time.Duration{0, -1} {
		m, err := mhcGPUEventTime(d)
		if err == nil || m.Status != "Unmeasured" || m.Nanoseconds != nil {
			t.Fatal("invalid event duration accepted", m, err)
		}
	}
	m, err := mhcGPUEventTime(3 * time.Microsecond)
	if err != nil || m.Status != "Measured" || m.Nanoseconds == nil || *m.Nanoseconds != 3000 {
		t.Fatal(m, err)
	}
	// No conversion from wall time is provided. The missing CUPTI channel is
	// explicitly Unmeasured and has no numeric value to add to event time.
	cupti := mhcTimeMeasurement{Status: "Unmeasured"}
	if cupti.Nanoseconds != nil {
		t.Fatal("missing activity duration fabricated")
	}
}

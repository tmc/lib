package cudacensus

import (
	"testing"
	"time"
)

func TestPartialCountersDoNotEstablishIdle(t *testing.T) {
	m := testModel(t)
	for _, c := range []Counter{DRAMBandwidthUtil, FP32Active, SMOccupancy, HMMATensorActive} {
		t.Run(string(c), func(t *testing.T) {
			in := Inputs{Readings: []Reading{{Counter: c, Available: true}}, Interval: time.Millisecond}
			est := NewEstimate(m, in)
			if est.Available || est.Confidence == ConfidenceIdle {
				t.Fatalf("one zero counter established idle: %+v", est)
			}
		})
	}
	in := Inputs{Readings: []Reading{
		{Counter: FP32Active, Reason: "unsupported"},
		{Counter: DRAMBandwidthUtil, Available: true},
	}, Attribution: fp32Attribution(), Interval: time.Millisecond}
	if est := NewEstimate(m, in); est.Available || est.Confidence == ConfidenceIdle {
		t.Fatalf("missing math counter plus zero bandwidth became idle: %+v", est)
	}
	// An attributed pipe can estimate zero, but this has attribution confidence,
	// not independent evidence that the whole device was idle.
	in.Readings = []Reading{{Counter: FP32Active, Available: true}}
	if est := NewEstimate(m, in); !est.Available || est.AchievedTFLOPS != 0 || est.Confidence != ConfidenceStaticMix {
		t.Fatalf("attributed zero: %+v", est)
	}
}

func TestBroadActivityEstablishesIdleUnlessContradicted(t *testing.T) {
	for _, c := range []Counter{SMActive, GraphicsActive} {
		in := Inputs{Readings: []Reading{
			{Counter: c, Available: true},
			{Counter: FP32Active, Reason: "unsupported"},
		}, Interval: time.Millisecond}
		if est := NewEstimate(testModel(t), in); !est.Available || est.Confidence != ConfidenceIdle || est.AchievedTFLOPS != 0 {
			t.Fatalf("broad zero %s: %+v", c, est)
		}
		for _, reverse := range []bool{false, true} {
			in.Readings[1] = Reading{Counter: FP32Active, Value: 0.25, Available: true}
			if reverse {
				in.Readings[0], in.Readings[1] = in.Readings[1], in.Readings[0]
			}
			if est := NewEstimate(testModel(t), in); est.Confidence == ConfidenceIdle {
				t.Fatalf("contradictory counters became idle: %+v", est)
			}
		}
	}
}

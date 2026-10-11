package cudaflightrecorder

import (
	"math"
	"testing"
	"time"
)

func detectorSample(t int64, value float64) DetectorObservation {
	return DetectorObservation{Metric: "latency", Unit: "ms", Epoch: "process-1", Clock: "mono-1", Timestamp: t, Value: value, Available: true}
}

func TestDetectorBaseline(t *testing.T) {
	for _, mode := range []string{"rolling", "ewma"} {
		t.Run(mode, func(t *testing.T) {
			d, err := NewDetector(DetectorConfig{Mode: mode, Window: 3, Warmup: 3, Alpha: .5, Delta: 5, MaxGap: time.Second})
			if err != nil {
				t.Fatal(err)
			}
			for i := int64(0); i < 3; i++ {
				if got := d.Observe(detectorSample(i, 10)); got.Ready {
					t.Fatalf("early decision: %+v", got)
				}
			}
			normal := d.Observe(detectorSample(3, 15))
			if !normal.Ready || normal.Anomaly || normal.Baseline != 10 || normal.Threshold != 15 {
				t.Fatalf("boundary: %+v", normal)
			}
			got := d.Observe(detectorSample(4, 30))
			wantSamples := 3
			if mode == "ewma" {
				wantSamples = 4
			}
			if !got.Ready || !got.Anomaly || got.Samples != wantSamples || got.Baseline <= 10 || got.Baseline >= 15 {
				t.Fatalf("anomaly: %+v", got)
			}
			for i := int64(5); i < 5000; i++ {
				d.Observe(detectorSample(i, 10))
			}
			if len(d.values) > 3 || d.count > 4096 {
				t.Fatal("unbounded state")
			}
			if mode == "rolling" && d.count != 3 {
				t.Fatal("wrong rolling population")
			}
		})
	}
}

func TestDetectorReset(t *testing.T) {
	tests := []struct {
		name, reason string
		change       func(*DetectorObservation)
	}{
		{"missing", "missing", func(o *DetectorObservation) { o.Available = false }},
		{"counter", "counter_reset", func(o *DetectorObservation) { o.CounterReset = true }},
		{"nan", "invalid_value", func(o *DetectorObservation) { o.Value = math.NaN() }},
		{"infinite", "invalid_value", func(o *DetectorObservation) { o.Value = math.Inf(1) }},
		{"negative time", "invalid_value", func(o *DetectorObservation) { o.Timestamp = -1 }},
		{"epoch", "identity_changed", func(o *DetectorObservation) { o.Epoch = "process-2" }},
		{"unit", "identity_changed", func(o *DetectorObservation) { o.Unit = "seconds" }},
		{"metric", "identity_changed", func(o *DetectorObservation) { o.Metric = "queue" }},
		{"phase", "identity_changed", func(o *DetectorObservation) { o.Phase = "load" }},
		{"clock", "identity_changed", func(o *DetectorObservation) { o.Clock = "mono-2" }},
		{"reversal", "clock_discontinuity", func(o *DetectorObservation) { o.Timestamp = 0 }},
		{"duplicate", "clock_discontinuity", func(o *DetectorObservation) { o.Timestamp = 1 }},
		{"gap", "expired", func(o *DetectorObservation) { o.Timestamp = 100 }},
		{"identity", "invalid_identity", func(o *DetectorObservation) { o.Epoch = "" }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			d, _ := NewDetector(DetectorConfig{Mode: "rolling", Window: 2, Warmup: 2, Delta: 5, MaxGap: 10})
			d.Observe(detectorSample(0, 10))
			d.Observe(detectorSample(1, 10))
			o := detectorSample(2, 100)
			tt.change(&o)
			got := d.Observe(o)
			if got.Ready || got.Anomaly || got.Reason != tt.reason {
				t.Fatalf("reset: %+v", got)
			}
			if d.count > 1 {
				t.Fatal("retained incompatible baseline")
			}
		})
	}
}

func TestDetectorInvalidConfig(t *testing.T) {
	valid := DetectorConfig{Mode: "ewma", Warmup: 2, Alpha: .5, Delta: 5, MaxGap: time.Second}
	tests := []DetectorConfig{{}, {Mode: "unknown", Warmup: 2, Delta: 5, MaxGap: time.Second}}
	for _, modify := range []func(*DetectorConfig){
		func(c *DetectorConfig) { c.Alpha = math.NaN() }, func(c *DetectorConfig) { c.Alpha = 0 }, func(c *DetectorConfig) { c.Alpha = 2 },
		func(c *DetectorConfig) { c.Delta = math.Inf(1) }, func(c *DetectorConfig) { c.Delta = -1 }, func(c *DetectorConfig) { c.Warmup = 4097 },
		func(c *DetectorConfig) { c.MaxGap = 0 }, func(c *DetectorConfig) { c.Mode = "rolling"; c.Window = 1 }, func(c *DetectorConfig) { c.Mode = "rolling"; c.Window = 4097 },
	} {
		c := valid
		modify(&c)
		tests = append(tests, c)
	}
	for _, c := range tests {
		if _, err := NewDetector(c); err == nil {
			t.Fatalf("accepted %+v", c)
		}
	}
	var zero Detector
	if got := zero.Observe(detectorSample(0, 10)); got.Reason != "unconfigured" {
		t.Fatalf("zero: %+v", got)
	}
}

func TestDetectorOverflow(t *testing.T) {
	d, _ := NewDetector(DetectorConfig{Mode: "ewma", Warmup: 1, Alpha: 1, Delta: math.MaxFloat64, MaxGap: time.Second})
	d.Observe(detectorSample(0, math.MaxFloat64))
	if got := d.Observe(detectorSample(1, math.MaxFloat64)); got.Reason != "numeric_overflow" || got.Ready {
		t.Fatalf("overflow: %+v", got)
	}
	if d.count != 0 {
		t.Fatal("overflow baseline retained")
	}
}

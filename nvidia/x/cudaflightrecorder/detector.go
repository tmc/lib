package cudaflightrecorder

import (
	"fmt"
	"math"
	"sync"
	"time"
)

// DetectorConfig freezes a high-side anomaly profile. Mode is "rolling" or
// "ewma". Rolling retains Window observations; EWMA uses Alpha in (0, 1].
// Warmup observations precede decisions. Delta is an absolute increase in the
// observation's unit. MaxGap bounds the interval between observations.
// Neither mode adjusts for seasonality or establishes a causal diagnosis.
type DetectorConfig struct {
	Mode   string
	Window int
	Alpha  float64
	Warmup int
	Delta  float64
	MaxGap time.Duration
}

// DetectorObservation is one comparable gauge/window value. Epoch identifies
// the process/model/instrument generation; Phase and Clock identify compatible
// workload and monotonic clock domains. Missing values are not healthy samples.
// CounterReset invalidates state; cumulative counters must be converted into
// comparable windows before observation.
type DetectorObservation struct {
	Metric, Unit, Epoch, Phase, Clock string
	Timestamp                         int64
	Value                             float64
	Available, CounterReset           bool
}

// DetectorDecision separates anomaly evidence from capture admission. Baseline
// and Threshold describe state before this observation. Samples is the rolling
// population or the EWMA valid observation count, saturated at 4096. Ready
// only means the detector completed warmup.
type DetectorDecision struct {
	Ready, Anomaly      bool
	Reason              string
	Samples             int
	Baseline, Threshold float64
}

// Detector is a bounded, concurrency-safe baseline. Its zero value is unusable;
// use NewDetector. Observations must arrive in timestamp order.
type Detector struct {
	mu     sync.Mutex
	config DetectorConfig
	key    [5]string
	last   int64
	count  int
	mean   float64
	values []float64
	next   int
}

// NewDetector validates and freezes a detector profile. Rolling state is limited
// to 4096 observations. Warmup is limited to 4096 in either mode.
func NewDetector(c DetectorConfig) (*Detector, error) {
	if c.Warmup < 1 || c.Warmup > 4096 || !finiteDetector(c.Delta) || c.Delta <= 0 || c.MaxGap <= 0 {
		return nil, fmt.Errorf("invalid detector warmup, delta or gap")
	}
	d := &Detector{config: c}
	switch c.Mode {
	case "rolling":
		if c.Window < c.Warmup || c.Window > 4096 {
			return nil, fmt.Errorf("invalid rolling window")
		}
		d.values = make([]float64, c.Window)
	case "ewma":
		if !finiteDetector(c.Alpha) || c.Alpha <= 0 || c.Alpha > 1 {
			return nil, fmt.Errorf("invalid ewma alpha")
		}
	default:
		return nil, fmt.Errorf("invalid detector mode")
	}
	return d, nil
}

func finiteDetector(v float64) bool { return !math.IsNaN(v) && !math.IsInf(v, 0) }

func (d *Detector) clear() {
	d.key = [5]string{}
	d.last, d.count, d.next, d.mean = 0, 0, 0, 0
	clear(d.values)
}

// Observe evaluates against the preceding baseline, then incorporates the value.
// Missing, invalid or reset values erase the baseline. Identity/phase changes,
// clock reversal and expired gaps start a new warmup with the current value.
// High anomalies are included in future baselines; callers must account for
// this adaptation when selecting a profile and comparing static thresholds.
func (d *Detector) Observe(o DetectorObservation) DetectorDecision {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.config.Mode == "" {
		return DetectorDecision{Reason: "unconfigured"}
	}
	key := [5]string{o.Metric, o.Unit, o.Epoch, o.Phase, o.Clock}
	reason := ""
	for i, s := range key {
		if len(s) > 256 || (i != 3 && s == "") {
			reason = "invalid_identity"
		}
	}
	if o.Timestamp < 0 || !finiteDetector(o.Value) {
		reason = "invalid_value"
	}
	if !o.Available {
		reason = "missing"
	}
	if o.CounterReset {
		reason = "counter_reset"
	}
	if reason != "" {
		d.clear()
		return DetectorDecision{Reason: reason}
	}
	if d.count > 0 {
		switch {
		case key != d.key:
			reason = "identity_changed"
		case o.Timestamp <= d.last:
			reason = "clock_discontinuity"
		case o.Timestamp-d.last > int64(d.config.MaxGap):
			reason = "expired"
		}
		if reason != "" {
			d.clear()
		}
	}
	result := DetectorDecision{Reason: "warmup", Samples: d.count, Baseline: d.mean, Threshold: d.mean + d.config.Delta}
	if !finiteDetector(result.Threshold) {
		d.clear()
		return DetectorDecision{Reason: "numeric_overflow"}
	}
	if d.count >= d.config.Warmup {
		result.Ready = true
		result.Reason = "normal"
		result.Anomaly = o.Value > result.Threshold
		if result.Anomaly {
			result.Reason = "high_anomaly"
		}
	}
	if reason != "" {
		result.Reason = reason
	}
	if d.count == 0 {
		d.mean = o.Value
	} else if d.config.Mode == "ewma" {
		// Weighted operands avoid overflowing an intermediate difference.
		d.mean = (1-d.config.Alpha)*d.mean + d.config.Alpha*o.Value
	}
	if d.config.Mode == "rolling" {
		d.values[d.next] = o.Value
		d.next = (d.next + 1) % len(d.values)
		n := min(d.count+1, len(d.values))
		// Divide before summation to avoid overflowing a finite population.
		d.mean = 0
		for _, v := range d.values[:n] {
			d.mean += v / float64(n)
		}
	}
	if !finiteDetector(d.mean) {
		d.clear()
		return DetectorDecision{Reason: "numeric_overflow"}
	}
	limit := 4096
	if d.config.Mode == "rolling" {
		limit = len(d.values)
	}
	d.count = min(d.count+1, limit)
	d.key, d.last = key, o.Timestamp
	return result
}

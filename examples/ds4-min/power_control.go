package main

import (
	"fmt"
	"math"
	"time"
)

// powerControl is request-local scheduling state, not model State. Reset and
// restore do not rewind its measured work averages.
type powerControl struct {
	percent int
	prefill bool
	layers  []float64
	decode  float64
	now     func() time.Time
	sleep   func(time.Duration)
}

func defaultPower(percent int) int {
	if percent == 0 {
		return 100
	}
	return percent
}

func validatePower(percent int, family string) error {
	if percent < 1 || percent > 100 {
		return fmt.Errorf("power must be between 1 and 100")
	}
	if percent != 100 && (family == "v41" || family == "glm" || family == "glm52" || family == "glm53") {
		return fmt.Errorf("power throttling is not supported for %s", family)
	}
	if percent != 100 {
		return powerSleepSupported()
	}
	return nil
}

func admitGroupedPower(p *powerControl) error {
	if p != nil {
		return fmt.Errorf("power throttling is not qualified for grouped speculative execution")
	}
	return nil
}

func newPowerControl(percent, layers int) *powerControl {
	if percent == 0 || percent == 100 {
		return nil
	}
	return &powerControl{percent: percent, layers: make([]float64, layers), now: time.Now, sleep: powerSleep}
}

func (p *powerControl) newRequest() *powerControl {
	if p == nil {
		return nil
	}
	q := newPowerControl(p.percent, len(p.layers))
	q.now, q.sleep = p.now, p.sleep
	return q
}

func (p *powerControl) start() time.Time {
	if p == nil {
		return time.Time{}
	}
	return p.now()
}

// finishLayer waits for real device completion before sampling or sleeping.
// Ordinary decode has one token boundary instead of per-layer boundaries.
func (p *powerControl) finishLayer(layer int, start time.Time, wait func() error) error {
	if p == nil || !p.prefill || layer < 0 || layer >= len(p.layers) {
		return nil
	}
	if err := wait(); err != nil {
		return err
	}
	p.layers[layer] = powerAverage(p.layers[layer], p.now().Sub(start).Seconds())
	p.pause(p.layers[layer])
	return nil
}

func (p *powerControl) finishToken(start time.Time) {
	if p == nil || p.prefill {
		return
	}
	p.decode = powerAverage(p.decode, p.now().Sub(start).Seconds())
	p.pause(p.decode)
}

func powerAverage(avg, sample float64) float64 {
	if sample <= 0 || math.IsNaN(sample) || math.IsInf(sample, 0) {
		return avg
	}
	if avg <= 0 || math.IsNaN(avg) || math.IsInf(avg, 0) {
		return sample
	}
	return avg*0.875 + sample*0.125
}

func (p *powerControl) pause(work float64) {
	seconds := work * float64(100-p.percent) / float64(p.percent)
	if seconds <= 0 || math.IsNaN(seconds) || math.IsInf(seconds, 0) {
		return
	}
	// Saturate rather than wrap an impossible wall-clock duration.
	nanos := seconds * 1e9
	if nanos >= float64(math.MaxInt64) {
		p.sleep(time.Duration(math.MaxInt64))
		return
	}
	p.sleep(time.Duration(nanos))
}

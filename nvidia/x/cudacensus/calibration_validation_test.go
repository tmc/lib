package cudacensus

import (
	"math"
	"strings"
	"testing"
	"time"
)

func TestCalibrationRefusesInvalidThroughput(t *testing.T) {
	for _, value := range []float64{-1, math.NaN(), math.Inf(1), math.Inf(-1)} {
		c := calibratedRun(t, 0.5)
		c.Estimate.AchievedTFLOPS = value
		if _, err := c.RelativeError(); err == nil {
			t.Fatalf("accepted throughput %v", value)
		}
		if _, err := c.Receipt("fixture", "fixture"); err == nil {
			t.Fatalf("receipt accepted throughput %v", value)
		}
		if !strings.Contains(c.Report(), "no comparison") {
			t.Fatalf("report hides invalidity: %s", c.Report())
		}
	}
	c := calibratedRun(t, 0.5)
	c.AnalyticFLOPs, c.Duration = 1, time.Second
	c.Estimate.AchievedTFLOPS = math.MaxFloat64
	if _, err := c.RelativeError(); err == nil {
		t.Fatal("accepted overflowing relative error")
	}
	// Zero remains a finite, maximally low estimate: its signed error is -1.
	c.Estimate.AchievedTFLOPS = 0
	if got, err := c.RelativeError(); err != nil || got != -1 {
		t.Fatalf("zero estimate error=%v, %v", got, err)
	}
}

func TestThresholdRefusesNonfiniteCalibration(t *testing.T) {
	for _, value := range []float64{math.NaN(), math.Inf(1), math.Inf(-1)} {
		r := goodReceipt()
		r.Error = value
		if err := r.Validate(); err == nil {
			t.Fatalf("accepted error %v", value)
		}
		if _, err := NewThreshold(0.8, r); err == nil {
			t.Fatal("used invalid receipt")
		}
		if _, err := NewThreshold(value, goodReceipt()); err == nil {
			t.Fatalf("accepted threshold %v", value)
		}
	}
}

func TestCalibrationReceiptCountsObservationsNotMetrics(t *testing.T) {
	c := calibratedRun(t, 0.5)
	for _, count := range []int{1, 2, len(AllCounters())} {
		c.Estimate.Derivation.Readings = make([]Reading, count)
		for i, counter := range AllCounters()[:count] {
			c.Estimate.Derivation.Readings[i] = Reading{Counter: counter, Value: 0.5, Available: true}
		}
		r, err := c.Receipt("fixture", "fixture")
		if err != nil || r.Samples != 1 {
			t.Fatalf("%d metrics: samples=%d, %v", count, r.Samples, err)
		}
	}
	c.Estimate.Derivation.Readings = nil
	if _, err := c.Receipt("fixture", "fixture"); err == nil {
		t.Fatal("receipt accepted no evidence")
	}
}

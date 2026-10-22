package main

import (
	"math"
	"strings"
	"testing"
)

func TestDSparkThresholdZero(t *testing.T) {
	for _, threshold := range []float64{0, 1, -.1, 1.1, math.NaN(), math.Inf(1)} {
		c := dsparkSupportConfig{path: "support.gguf", sha256: strings.Repeat("a", 64), confidence: threshold}
		err := c.validate(config{backend: "metal"})
		valid := threshold >= 0 && threshold <= 1
		if (err == nil) != valid {
			t.Errorf("confidence %g validation: %v", threshold, err)
		}
	}
	ids, probabilities, err := dsparkHeadReference([]float32{0, 1, 0, 1}, []float32{-100, -100}, []float32{0, 0}, []float32{0, 0}, []float32{1, 0}, 2, 2, 1, 1, 0, 0)
	if err != nil || len(ids) != 2 || len(probabilities) != 2 {
		t.Errorf("zero must disable confidence rejection: IDs=%v probabilities=%v error=%v", ids, probabilities, err)
	}
}

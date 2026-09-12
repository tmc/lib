package nccltest

import (
	"fmt"
	"math"
	"testing"
)

func ExampleTiming() {
	options := Options{Timing: TimingMaximum}
	fmt.Println(options.Timing)
	// Output: max
}

func TestRankTime(t *testing.T) {
	for _, tt := range []struct {
		mode Timing
		want float64
	}{
		{"", 4}, {TimingMean, 4}, {TimingRankZero, 3}, {TimingMinimum, 1}, {TimingMaximum, 8},
	} {
		got, err := rankTime([]float64{3, 1, 8}, tt.mode)
		if err != nil || got != tt.want {
			t.Fatalf("%q: got %g, %v; want %g", tt.mode, got, err, tt.want)
		}
	}
	for _, values := range [][]float64{nil, {0}, {-1}, {math.NaN()}, {math.Inf(1)}, {1, math.NaN()}} {
		if _, err := rankTime(values, TimingRankZero); err == nil {
			t.Fatalf("accepted invalid timings %v", values)
		}
	}
	if _, err := rankTime([]float64{1}, "bogus"); err == nil {
		t.Fatal("accepted unknown mode")
	}
}

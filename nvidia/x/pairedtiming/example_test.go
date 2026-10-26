package pairedtiming_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/pairedtiming"
)

func Example() {
	// Microseconds. The control graphs ran one and two copies of a tiny op.
	h, _ := pairedtiming.Intercept([]pairedtiming.ControlBlock{
		{One1: 5.2, Two1: 6.1, Two2: 6.1, One2: 5.2},
		{One1: 5.1, Two1: 6.1, Two2: 6.0, One2: 5.2},
		{One1: 5.2, Two1: 6.2, Two2: 6.1, One2: 5.1},
	})
	r, _ := pairedtiming.Paired([]pairedtiming.Block{
		{A1: 24.3, B1: 21.1, B2: 21.2, A2: 24.4},
		{A1: 24.1, B1: 21.0, B2: 21.3, A2: 24.2},
		{A1: 24.6, B1: 21.4, B2: 21.1, A2: 24.3},
	}, h, pairedtiming.Geometric)
	fmt.Printf("intercept %.2f us, speedup %.3f, baseline %.2f us\n", h, r.Ratio, r.Baseline)
	// Output:
	// intercept 4.25 us, speedup 1.188, baseline 20.05 us
}

func ExampleScore() {
	s, _ := pairedtiming.Score([]pairedtiming.Shape{
		{Load: "decode", Weight: 46, Baseline: 20, Candidate: 17},
		{Load: "decode", Weight: 16, Baseline: 31, Candidate: 30},
		{Load: "verify", Weight: 46, Baseline: 23, Candidate: 22},
		{Load: "verify", Weight: 16, Baseline: 36, Candidate: 37},
	})
	for _, l := range s.Loads {
		fmt.Printf("%s %.3f\n", l.Load, l.Speedup)
	}
	fmt.Printf("score %.3f\n", s.Speedup)
	// Output:
	// decode 1.122
	// verify 1.019
	// score 1.069
}

func ExampleBootstrap() {
	series := []pairedtiming.Series{{
		Name: "decode-1", Load: "decode", Weight: 1, Intercept: 4,
		Blocks: []pairedtiming.Block{
			{A1: 24, B1: 21, B2: 21, A2: 24},
			{A1: 25, B1: 21, B2: 22, A2: 24},
			{A1: 24, B1: 22, B2: 21, A2: 25},
			{A1: 24, B1: 21, B2: 21, A2: 23},
		},
	}}
	iv, _ := pairedtiming.Bootstrap(series, pairedtiming.Geometric, 1000, 1)
	fmt.Printf("95%% interval [%.3f, %.3f]\n", iv.Lo, iv.Hi)
	// Output:
	// 95% interval [1.147, 1.176]
}

func ExampleSignTest() {
	// The candidate won 14 of 15 blocks.
	fmt.Printf("p = %.5f\n", pairedtiming.SignTest(14, 1))
	// Output:
	// p = 0.00098
}

// A bound model: what would the suite score if every candidate ran at a
// fraction of its memory floor, the bytes it must move divided by a
// bandwidth ceiling measured on the same host, but no faster than a fixed
// per-call latency? Score the projected times as if they were measured.
func ExampleScore_bound() {
	const (
		ceiling    = 2740e9 / 1e6 // measured copy bandwidth, bytes per microsecond
		efficiency = 0.8          // fraction of the floor reached
		latency    = 2.5          // microseconds no kernel goes below
	)
	type op struct {
		load       string
		weight     float64
		baselineUS float64 // measured
		bytes      float64 // bytes the operation must move
	}
	ops := []op{
		{"decode", 1, 7.9, 8.39e6},
		{"decode", 1, 21.6, 37.7e6},
		{"verify", 1, 30.1, 67.1e6},
	}
	var shapes []pairedtiming.Shape
	for _, o := range ops {
		floor := o.bytes / ceiling
		shapes = append(shapes, pairedtiming.Shape{
			Load: o.load, Weight: o.weight, Baseline: o.baselineUS,
			Candidate: max(floor/efficiency, latency),
		})
	}
	s, _ := pairedtiming.Score(shapes)
	fmt.Printf("projected score %.2f\n", s.Speedup)
	// Output:
	// projected score 1.17
}

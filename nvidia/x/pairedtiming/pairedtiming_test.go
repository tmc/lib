package pairedtiming

import (
	"encoding/json"
	"math"
	"os"
	"testing"
)

func TestIntercept(t *testing.T) {
	tests := []struct {
		name    string
		blocks  []ControlBlock
		want    float64
		wantErr bool
	}{
		{"exact", []ControlBlock{{5, 6, 6, 5}}, 4, false},
		{"median of three", []ControlBlock{{5, 6, 6, 5}, {5, 7, 7, 5}, {6, 6, 6, 6}}, 4, false},
		// A linear drift of 0.5 per sample adds 1.5*0.5.
		{"linear drift biases", []ControlBlock{{5, 6.5, 7, 6.5}}, 4.75, false},
		{"unresolved", []ControlBlock{{5, 11, 11, 5}}, 0, true},
		{"zero", []ControlBlock{{5, 10, 10, 5}}, 0, true},
		{"nan", []ControlBlock{{math.NaN(), 6, 6, 5}}, 0, true},
		{"empty", nil, 0, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := Intercept(tt.blocks)
			if (err != nil) != tt.wantErr {
				t.Fatalf("Intercept() error = %v, wantErr %v", err, tt.wantErr)
			}
			if !tt.wantErr && math.Abs(got-tt.want) > 1e-12 {
				t.Errorf("Intercept() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestBlockRatio(t *testing.T) {
	tests := []struct {
		name    string
		b       Block
		h       float64
		e       Estimator
		want    float64
		wantErr bool
	}{
		{"geometric", Block{A1: 4, B1: 2, B2: 2, A2: 4}, 0, Geometric, 2, false},
		{"geometric intercept", Block{A1: 5, B1: 3, B2: 3, A2: 5}, 1, Geometric, 2, false},
		{"geometric drift", Block{A1: 4, B1: 2, B2: 4, A2: 8}, 0, Geometric, 2, false},
		{"sum", Block{A1: 4, B1: 1, B2: 3, A2: 4}, 0, Sum, 2, false},
		{"slower", Block{A1: 2, B1: 4, B2: 4, A2: 2}, 0, Geometric, 0.5, false},
		{"sample at intercept", Block{A1: 5, B1: 1, B2: 3, A2: 5}, 1, Geometric, 0, true},
		{"inf", Block{A1: math.Inf(1), B1: 1, B2: 1, A2: 1}, 0, Geometric, 0, true},
		{"bad estimator", Block{1, 1, 1, 1}, 0, Estimator(9), 0, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := tt.b.Ratio(tt.h, tt.e)
			if (err != nil) != tt.wantErr {
				t.Fatalf("Ratio() error = %v, wantErr %v", err, tt.wantErr)
			}
			if !tt.wantErr && math.Abs(got-tt.want) > 1e-12 {
				t.Errorf("Ratio() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestPaired(t *testing.T) {
	blocks := []Block{
		{A1: 11, B1: 6, B2: 6, A2: 11},
		{A1: 13, B1: 6, B2: 6, A2: 13},
		{A1: 9, B1: 6, B2: 6, A2: 9},
	}
	r, err := Paired(blocks, 1, Geometric)
	if err != nil {
		t.Fatal(err)
	}
	if r.Ratio != 2 || r.MinRatio != 1.6 || r.Baseline != 10 || r.Candidate != 5 {
		t.Errorf("Paired() = %+v, want ratio 2, min 1.6, baseline 10, candidate 5", r)
	}
	if _, err := Paired(nil, 0, Geometric); err == nil {
		t.Error("Paired(nil) succeeded")
	}
}

func TestScore(t *testing.T) {
	tests := []struct {
		name    string
		shapes  []Shape
		want    float64
		wantErr bool
	}{
		{"one shape", []Shape{{"a", 1, 4, 2}}, 2, false},
		// Weighted sums within a load: (1*10 + 1*1) / (1*5 + 1*1) = 11/6,
		// not the geometric mean of 2 and 1.
		{"weights within load", []Shape{{"a", 1, 10, 5}, {"a", 1, 1, 1}}, 11.0 / 6, false},
		{"geomean across loads", []Shape{{"a", 1, 4, 1}, {"b", 1, 1, 1}}, 2, false},
		{"weight scales", []Shape{{"a", 3, 10, 5}, {"a", 1, 2, 2}}, 32.0 / 17, false},
		{"zero weight", []Shape{{"a", 0, 1, 1}}, 0, true},
		{"negative time", []Shape{{"a", 1, -1, 1}}, 0, true},
		{"nan time", []Shape{{"a", 1, math.NaN(), 1}}, 0, true},
		{"empty", nil, 0, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := Score(tt.shapes)
			if (err != nil) != tt.wantErr {
				t.Fatalf("Score() error = %v, wantErr %v", err, tt.wantErr)
			}
			if !tt.wantErr && math.Abs(got.Speedup-tt.want) > 1e-12 {
				t.Errorf("Score() = %v, want %v", got.Speedup, tt.want)
			}
		})
	}
}

func TestSignTest(t *testing.T) {
	tests := []struct {
		wins, losses int
		want         float64
	}{
		{0, 0, 1},
		{1, 0, 1},
		{5, 0, 0.0625},
		{0, 5, 0.0625},
		{8, 2, 0.109375},
		{10, 10, 1},
		// 2 * sum_{i<=100} C(2000,i)/2^2000 underflows in linear space.
		{1900, 100, 0},
	}
	for _, tt := range tests {
		got := SignTest(tt.wins, tt.losses)
		if math.Abs(got-tt.want) > 1e-12 {
			t.Errorf("SignTest(%d, %d) = %v, want %v", tt.wins, tt.losses, got, tt.want)
		}
	}
}

func TestBootstrap(t *testing.T) {
	series := []Series{{Name: "s", Load: "a", Weight: 1, Blocks: []Block{
		{A1: 10, B1: 5, B2: 5, A2: 10},
		{A1: 12, B1: 5, B2: 5, A2: 12},
		{A1: 8, B1: 5, B2: 5, A2: 8},
		{A1: 11, B1: 5, B2: 5, A2: 11},
	}}}
	iv, err := Bootstrap(series, Geometric, 2000, 1)
	if err != nil {
		t.Fatal(err)
	}
	if !(iv.Lo >= 1.6 && iv.Lo <= iv.Hi && iv.Hi <= 2.4) {
		t.Errorf("Bootstrap() = %+v, want within block range [1.6, 2.4]", iv)
	}
	again, err := Bootstrap(series, Geometric, 2000, 1)
	if err != nil || again != iv {
		t.Errorf("Bootstrap() not reproducible: %+v then %+v (%v)", iv, again, err)
	}
	// Identical blocks have no spread.
	flat := []Series{{Load: "a", Weight: 1, Blocks: []Block{{4, 2, 2, 4}, {4, 2, 2, 4}}}}
	if iv, err := Bootstrap(flat, Geometric, 100, 7); err != nil || iv != (Interval{2, 2}) {
		t.Errorf("Bootstrap(flat) = %+v, %v; want {2 2}", iv, err)
	}
	if _, err := Bootstrap(series, Geometric, 0, 1); err == nil {
		t.Error("Bootstrap(draws=0) succeeded")
	}
}

// judge holds one scored H200 run: per-shape baseline and candidate samples
// in nanoseconds, the server's replay intercept, and the server's own
// paired ratios and score. The server, not this package, produced the
// expected values.
type judge struct {
	Speedup float64
	Shapes  []struct {
		Load           string
		Weight         float64
		InterceptNS    float64   `json:"intercept_ns"`
		OneNS          float64   `json:"one_ns"`
		TwoNS          float64   `json:"two_ns"`
		BaselineNS     float64   `json:"baseline_ns"`
		PairedRatio    float64   `json:"paired_ratio"`
		BaselineRunsNS []float64 `json:"baseline_runs_ns"`
		RunsNS         []float64 `json:"runs_ns"`
	}
}

func TestJudgeReconstruction(t *testing.T) {
	data, err := os.ReadFile("testdata/judge.json")
	if err != nil {
		t.Fatal(err)
	}
	var j judge
	if err := json.Unmarshal(data, &j); err != nil {
		t.Fatal(err)
	}
	if len(j.Shapes) == 0 {
		t.Fatal("no shapes in fixture")
	}
	var series []Series
	for i, s := range j.Shapes {
		if got := 2*s.OneNS - s.TwoNS; math.Abs(got-s.InterceptNS) > 1e-6 {
			t.Errorf("shape %d: 2*one-two = %v, judge intercept %v", i, got, s.InterceptNS)
		}
		a, b := s.BaselineRunsNS, s.RunsNS
		if len(a) != len(b) || len(a)%2 != 0 || len(a) == 0 {
			t.Fatalf("shape %d: %d baseline and %d candidate samples", i, len(a), len(b))
		}
		var blocks []Block
		for r := 0; r < len(a); r += 2 {
			blocks = append(blocks, Block{A1: a[r], B1: b[r], B2: b[r+1], A2: a[r+1]})
		}
		res, err := Paired(blocks, s.InterceptNS, Geometric)
		if err != nil {
			t.Fatalf("shape %d: %v", i, err)
		}
		if d := math.Abs(res.Ratio - s.PairedRatio); d > 1e-7 {
			t.Errorf("shape %d: ratio %v, judge %v (diff %g)", i, res.Ratio, s.PairedRatio, d)
		}
		series = append(series, Series{Load: s.Load, Weight: s.Weight, Intercept: s.InterceptNS, Blocks: blocks})
	}
	sc, _, err := ScoreSeries(series, Geometric)
	if err != nil {
		t.Fatal(err)
	}
	if d := math.Abs(sc.Speedup - j.Speedup); d > 1e-6 {
		t.Errorf("score %v, judge %v (diff %g)", sc.Speedup, j.Speedup, d)
	}
}

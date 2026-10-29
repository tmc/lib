package qualify

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"testing"
)

var gb10 = Ceiling{Device: "GB10", BandwidthGBs: 250, FLOPS: 100e12}

func records(name string, ns ...float64) []KernelRecord {
	r := make([]KernelRecord, len(ns))
	for i, d := range ns {
		r[i] = KernelRecord{Name: name, DurationNs: d, Bytes: 1 << 20, FLOPs: 1 << 20}
	}
	return r
}

func TestRoofline(t *testing.T) {
	tests := []struct {
		rec      KernelRecord
		bound    string
		idealNs  float64
		fraction float64
	}{
		// 250 MB at 250 GB/s is 1 ms; 1 GFLOP at 100 TFLOP/s is 10 us.
		{KernelRecord{"copy", 2e6, 250e6, 1e9}, "memory", 1e6, 0.5},
		// 1 TFLOP is 10 ms; 1 MB is 4 us.
		{KernelRecord{"gemm", 12.5e6, 1e6, 1e12}, "compute", 1e7, 0.8},
		{KernelRecord{"pure", 1e3, 0, 1e8}, "compute", 1e3, 1},
	}
	for _, tt := range tests {
		p, err := Roofline([]KernelRecord{tt.rec}, gb10)
		if err != nil {
			t.Fatal(err)
		}
		if p[0].Bound != tt.bound || math.Abs(p[0].IdealNs-tt.idealNs) > 1e-6*tt.idealNs || math.Abs(p[0].Fraction-tt.fraction) > 1e-9 {
			t.Errorf("Roofline(%s) = %+v, want %s bound, %v ns, fraction %v", tt.rec.Name, p[0], tt.bound, tt.idealNs, tt.fraction)
		}
	}
	for _, bad := range []KernelRecord{{"zero", 0, 1, 1}, {"empty", 1, 0, 0}, {"neg", 1, -1, 1}} {
		if _, err := Roofline([]KernelRecord{bad}, gb10); err == nil {
			t.Errorf("Roofline accepted %+v", bad)
		}
	}
	if _, err := Roofline(nil, Ceiling{BandwidthGBs: 0, FLOPS: 1}); err == nil {
		t.Error("Roofline accepted a zero bandwidth ceiling")
	}
}

func TestCompareRuns(t *testing.T) {
	base := Run{Ceiling: gb10, Records: records("k", 100, 101, 99, 100, 102, 98, 100, 101, 99)}
	tests := []struct {
		name      string
		cur       []float64
		regressed bool
		improved  bool
	}{
		{"same distribution", []float64{101, 99, 100, 100, 102, 98, 99, 101, 100}, false, false},
		{"clear slowdown", []float64{110, 111, 109, 112, 110, 108, 110, 111, 109}, true, false},
		// The mean rises tenfold; the median and the signs do not move.
		{"mean moves, median does not", []float64{100, 101, 99, 100, 99, 101, 100, 100, 10000}, false, false},
		{"slower but only 3 samples", []float64{120, 120, 120}, false, false},
		{"slower with 5 samples", []float64{120, 121, 119, 120, 122}, true, false},
		{"all slower, below threshold", []float64{103, 103, 104, 103, 102, 103, 104, 103, 103}, false, false},
		// The median rises 7% but 4 of 9 samples are faster.
		{"median up, signs split", []float64{90, 90, 90, 90, 107, 120, 120, 120, 120}, false, false},
		{"clear speedup", []float64{80, 81, 79, 80, 82, 78, 80, 81, 79}, false, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rs, err := CompareRuns(base, Run{Ceiling: gb10, Records: records("k", tt.cur...)}, RegressionConfig{})
			if err != nil {
				t.Fatal(err)
			}
			if len(rs) != 1 {
				t.Fatalf("%d results, want 1", len(rs))
			}
			r := rs[0]
			if r.Regressed != tt.regressed || r.Improved != tt.improved {
				t.Errorf("CompareRuns = %+v, want regressed %v improved %v", r, tt.regressed, tt.improved)
			}
		})
	}
}

func TestCompareRunsMeanMovesMedianDoesNot(t *testing.T) {
	base := records("k", 100, 101, 99, 100, 102, 98, 100, 101, 99)
	cur := records("k", 100, 101, 99, 100, 99, 101, 100, 100, 10000)
	mean := func(r []KernelRecord) float64 {
		s := 0.0
		for _, x := range r {
			s += x.DurationNs
		}
		return s / float64(len(r))
	}
	if mean(cur) < 10*mean(base) {
		t.Fatalf("fixture: mean %v vs %v does not move", mean(cur), mean(base))
	}
	rs, err := CompareRuns(Run{gb10, base}, Run{gb10, cur}, RegressionConfig{})
	if err != nil {
		t.Fatal(err)
	}
	if r := rs[0]; r.Regressed || r.Ratio != 1 {
		t.Errorf("CompareRuns = %+v, want ratio 1 and no regression", r)
	}
}

func TestCompareRunsCeiling(t *testing.T) {
	base := Run{Ceiling: gb10, Records: records("k", 100, 100, 100)}
	tests := []struct {
		name string
		c    Ceiling
		ok   bool
	}{
		{"same", gb10, true},
		{"bandwidth within 1%", Ceiling{"GB10", 251, 100e12}, true},
		{"bandwidth 2% lower", Ceiling{"GB10", 245, 100e12}, false},
		{"flops 5% higher", Ceiling{"GB10", 250, 105e12}, false},
		{"other device", Ceiling{"H100", 250, 100e12}, false},
	}
	for _, tt := range tests {
		_, err := CompareRuns(base, Run{Ceiling: tt.c, Records: records("k", 100)}, RegressionConfig{})
		if tt.ok && err != nil {
			t.Errorf("%s: %v", tt.name, err)
		}
		if !tt.ok && !errors.Is(err, ErrCeilingMismatch) {
			t.Errorf("%s: err = %v, want ErrCeilingMismatch", tt.name, err)
		}
	}
}

func TestCompareRunsMissingKernels(t *testing.T) {
	base := Run{Ceiling: gb10, Records: append(records("a", 1, 2, 3), records("b", 1)...)}
	cur := Run{Ceiling: gb10, Records: append(records("a", 1, 2, 3), records("c", 1)...)}
	rs, err := CompareRuns(base, cur, RegressionConfig{})
	if err != nil {
		t.Fatal(err)
	}
	got := fmt.Sprintln(rs[0].Name, rs[0].Note == "", rs[1].Name, rs[1].Note, rs[2].Name, rs[2].Note)
	if want := "a true b no current samples c no baseline samples\n"; got != want {
		t.Errorf("CompareRuns = %s, want %s", got, want)
	}
}

func TestRunJSONRoundTrip(t *testing.T) {
	in := Run{Ceiling: gb10, Records: records("k", 1, 2)}
	b, err := json.Marshal(in)
	if err != nil {
		t.Fatal(err)
	}
	var out Run
	if err := json.Unmarshal(b, &out); err != nil {
		t.Fatal(err)
	}
	if fmt.Sprint(out) != fmt.Sprint(in) {
		t.Errorf("round trip = %+v, want %+v", out, in)
	}
}

func TestBinomialTail(t *testing.T) {
	tests := []struct {
		k, n int
		want float64
	}{
		{0, 0, 1}, {0, 5, 1}, {5, 5, 1.0 / 32}, {3, 3, 1.0 / 8}, {9, 9, 1.0 / 512}, {5, 9, 0.5},
	}
	for _, tt := range tests {
		if got := binomialTail(tt.k, tt.n); math.Abs(got-tt.want) > 1e-12 {
			t.Errorf("binomialTail(%d, %d) = %v, want %v", tt.k, tt.n, got, tt.want)
		}
	}
}

func ExampleCompareRuns() {
	c := Ceiling{Device: "GB10", BandwidthGBs: 250, FLOPS: 100e12}
	run := func(ns ...float64) Run {
		r := Run{Ceiling: c}
		for _, d := range ns {
			r.Records = append(r.Records, KernelRecord{Name: "rmsnorm", DurationNs: d, Bytes: 2e6})
		}
		return r
	}
	base := run(10000, 10100, 9900, 10000, 10050)
	cur := run(11000, 11100, 10900, 11000, 11050)
	rs, err := CompareRuns(base, cur, RegressionConfig{})
	if err != nil {
		panic(err)
	}
	p, _ := Roofline(cur.Records[:1], c)
	fmt.Printf("%s: ratio %.2f p=%.3f regressed=%v roofline %.2f %s-bound\n",
		rs[0].Name, rs[0].Ratio, rs[0].PValue, rs[0].Regressed, p[0].Fraction, p[0].Bound)
	// Output: rmsnorm: ratio 1.10 p=0.031 regressed=true roofline 0.73 memory-bound
}

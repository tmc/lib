package okcprofile

import (
	"math"
	"strconv"
	"strings"
	"testing"
)

// TestElasticitiesMatchRecon checks the fixture against the values
// recon/analysis/kernel_elasticity.py (an independent Python
// implementation) prints for it, unrounded.
func TestElasticitiesMatchRecon(t *testing.T) {
	f := readFixture(t)
	tab, err := Elasticities(f)
	if err != nil {
		t.Fatal(err)
	}
	if tab.Loads != 3 || math.Abs(tab.Score-1.1874437818273236) > 1e-12 || math.Abs(tab.Coverage-0.9922378528993165) > 1e-12 {
		t.Errorf("loads %d score %.16g coverage %.16g", tab.Loads, tab.Score, tab.Coverage)
	}
	if math.Abs(tab.Score-f.Summary.Speedup) > 1e-12 {
		t.Errorf("score %.16g, judge %.16g", tab.Score, f.Summary.Speedup)
	}
	want := []struct {
		kernel string
		calls  int
		e      float64
	}{
		{"void pm::gemm_kernel<64, 8, true, true>(CUtensorMap_st, u", 2, 0.14217111890901998},
		{"void cutlass::device_kernel<flash::enable_sm90_or_later<f", 8, 0.11876137045272704},
		{"void pm::gemm_kernel<128, 8, true, true>(CUtensorMap_st, ", 2, 0.11777258219440019},
		{"void pm::gemm_kernel<256, 5, true, true>(CUtensorMap_st, ", 4, 0.08982647740172474},
		{"void pm::gemm_kernel<64, 8, false, true>(CUtensorMap_st, ", 2, 0.07434143899427602},
		{"nvjet_sm90_tst_192x192_64x4_2x1_v_bz_coopB_TNN", 12, 0.0712118972681116},
		{"void pm::gemm_kernel<128, 8, false, true>(CUtensorMap_st,", 2, 0.058152825471188406},
		{"void sglang::moe8_spark_norm_residual_kernel<8192l, 512, ", 16, 0.0490493707157755},
	}
	for i, w := range want {
		g := tab.Kernels[i]
		if g.Kernel != w.kernel || g.Calls != w.calls || math.Abs(g.E-w.e) > 1e-12 {
			t.Errorf("row %d = %q calls %d E %.16g; want %q %d %.16g", i, g.Kernel, g.Calls, g.E, w.kernel, w.calls, w.e)
		}
	}
	if !tab.Kernels[7].Merged {
		t.Error("norm_residual rows collide by name and must be marked Merged")
	}
}

// TestElasticityIsDerivative checks E against a finite difference: removing
// a fraction x of a kernel's time from the judged medians moves log(score) by
// about x·E.
func TestElasticityIsDerivative(t *testing.T) {
	us := func(v float64) string { return strconv.FormatFloat(v, 'f', -1, 64) }
	report := func(a, b float64) string {
		return "A " + us(a) + " µs 50.0% 1\nB " + us(b) + " µs 50.0% 1\ntotal 1 µs\n"
	}
	f := &File{
		Benchmarks: []Benchmark{
			{Case: Case{Load: "p", Weight: 3}, BaselineNS: 40000, MedianNS: 30000},
			{Case: Case{Load: "p", Weight: 1}, BaselineNS: 90000, MedianNS: 60000},
			{Case: Case{Load: "q", Weight: 2}, BaselineNS: 10000, MedianNS: 9000},
		},
		Profiles: []Profile{
			{Case: Case{Load: "p", Weight: 3}, Report: report(20, 5)},
			{Case: Case{Load: "p", Weight: 1}, Report: report(10, 40)},
			{Case: Case{Load: "q", Weight: 2}, Report: report(1, 7)},
		},
	}
	tab, err := Elasticities(f)
	if err != nil {
		t.Fatal(err)
	}
	kernelUS := map[string][]float64{"A": {20, 10, 1}, "B": {5, 40, 7}}
	const x = 1e-6
	for _, k := range tab.Kernels {
		g := *f
		g.Benchmarks = append([]Benchmark(nil), f.Benchmarks...)
		for i := range g.Benchmarks {
			g.Benchmarks[i].MedianNS -= x * kernelUS[k.Kernel][i] * 1e3
		}
		moved, err := Elasticities(&g)
		if err != nil {
			t.Fatal(err)
		}
		slope := (math.Log(moved.Score) - math.Log(tab.Score)) / x
		if math.Abs(slope-k.E) > 1e-5 {
			t.Errorf("%s: E %.8f, finite difference %.8f", k.Kernel, k.E, slope)
		}
	}
}

func TestElasticitiesErrors(t *testing.T) {
	ok := "k 1 µs 100.0% 1\ntotal 1 µs\n"
	b := []Benchmark{{Case: Case{Load: "p", Weight: 1}, BaselineNS: 2, MedianNS: 1}}
	tests := []struct {
		name string
		f    File
		want string
	}{
		{"no benchmarks", File{}, "no benchmarks"},
		{"zero median", File{Benchmarks: []Benchmark{{Case: Case{Load: "p", Weight: 1}, BaselineNS: 2}}}, "median 0 ns"},
		{"unknown load", File{Benchmarks: b, Profiles: []Profile{{Case: Case{Load: "q", Weight: 1}, Report: ok}}}, `load "q" has no benchmark`},
		{"bad report", File{Benchmarks: b, Profiles: []Profile{{Case: Case{Load: "p", Weight: 1}, Report: "x\n"}}}, "profile 0: line 1"},
	}
	for _, tt := range tests {
		_, err := Elasticities(&tt.f)
		if err == nil || !strings.Contains(err.Error(), tt.want) {
			t.Errorf("%s: error %v, want %q", tt.name, err, tt.want)
		}
	}
}

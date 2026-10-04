package cudahotswap

import (
	"math"
	"strings"
	"testing"
)

func TestCompareRefusesInvalidPolicyAndReference(t *testing.T) {
	for _, tol := range []Tolerance{{Abs: -1}, {Rel: -1}, {Abs: math.NaN()}, {Rel: math.NaN()}, {Abs: math.Inf(1)}, {Rel: math.Inf(1)}} {
		c, err := Compare([]float32{1}, []float32{1}, tol)
		if err == nil || c.OK() {
			t.Fatalf("invalid tolerance %v accepted: %v %+v", tol, err, c)
		}
	}
	for _, bad := range []float32{float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1))} {
		c, err := Compare([]float32{1}, []float32{bad}, Tolerance{})
		if err == nil || c.OK() {
			t.Fatalf("invalid reference %v accepted: %v %+v", bad, err, c)
		}
	}
	if (Comparison{}).OK() {
		t.Fatal("empty comparison reports success")
	}
}

func TestSummarizeClearsStaleRates(t *testing.T) {
	r := TimingReport{LaunchesPerSample: 1, SampleNanos: []int64{10, 20}}
	r.Summarize()
	if !r.Resolved {
		t.Fatal("healthy clock rejected")
	}
	r.SampleNanos = []int64{0}
	r.Summarize()
	if r.Resolved || r.MedianNanosPerLaunch != 0 || r.MinNanosPerLaunch != 0 || r.MaxNanosPerLaunch != 0 {
		t.Fatalf("stale rate: %+v", r)
	}
	if got := median([]int64{math.MaxInt64, math.MaxInt64}); got != float64(math.MaxInt64) {
		t.Fatalf("median overflow: %v", got)
	}
}

func TestTextAcceptsMissingAndShortDigests(t *testing.T) {
	for _, digest := range []string{"", "short", "0123456789abcdef0123456789abcdef"} {
		r := Report{Compile: &CompileReport{SourceSHA256: digest, PTXSHA256: digest}}
		text := r.Text()
		if !strings.Contains(text, "compile   source sha256 "+shortDigest(digest)) {
			t.Fatalf("digest omitted: %s", text)
		}
	}
}

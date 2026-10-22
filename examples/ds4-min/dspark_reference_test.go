package main

import (
	"math"
	"reflect"
	"testing"
)

func TestDSparkMarkovChainReference(t *testing.T) {
	// W1[0]=[1,0] selects token1; W1[1]=[0,1] selects token2.
	// Reading the original previous token for both rows is a source mutant
	// that would produce [1,1], which this value oracle rejects.
	logits := make([]float32, 6)
	w1 := []float32{1, 0, 0, 1, 1, 1}
	w2 := []float32{0, 0, 2, 0, 0, 3}
	got, p, err := dsparkHeadReference(logits, []float32{1, 1}, w1, w2, []float32{0, 0, 0}, 2, 3, 2, 1, 0, 0.5)
	if err != nil || !reflect.DeepEqual(got, []int{1, 2}) || !reflect.DeepEqual(p, []float32{0.5, 0.5}) {
		t.Fatalf("chain: %v %v %v", got, p, err)
	}
	if !reflect.DeepEqual(logits, make([]float32, 6)) {
		t.Fatal("reference changed caller logits")
	}
	_, _, err = dsparkHeadReference(logits, []float32{1, 1}, w1, w2, []float32{float32(math.NaN()), 0, 0}, 2, 3, 2, 1, 0, 0.5)
	if err == nil {
		t.Fatal("nonfinite confidence admitted")
	}
}

func TestDSparkConfidencePrefixReference(t *testing.T) {
	got, p, err := dsparkHeadReference([]float32{0, 1, 0, 1, 0, 1}, []float32{2, -2, 2}, []float32{0, 0}, []float32{0, 0}, []float32{1, 0}, 3, 2, 1, 1, 0, 0.7)
	if err != nil || !reflect.DeepEqual(got, []int{1}) || len(p) != 2 {
		t.Fatalf("prefix: %v %v %v", got, p, err)
	}
	for i, z := range []float64{2, -2} {
		want := 1 / (1 + math.Exp(-z))
		if math.Abs(float64(p[i])-want) > 1e-7 {
			t.Fatalf("sigmoid[%d] %g want %g", i, p[i], want)
		}
	}
}

func TestDSparkHCMeanReference(t *testing.T) {
	got, err := dsparkHCMean([]float32{1, 2, 3, 4, 5, 6, 7, 8}, 2, 4)
	if err != nil || !reflect.DeepEqual(got, []float32{4, 5}) {
		t.Fatalf("mean: %v %v", got, err)
	}
	// An embedding-major interpretation gives [2.5,6.5], so values test layout.
	if _, err := dsparkHCMean([]float32{1}, 2, 4); err == nil {
		t.Fatal("short state admitted")
	}
}

func TestDSparkMainProjectionReference(t *testing.T) {
	got, err := dsparkMainReference([]float32{1, 2, 3, 4}, []float32{1, 0, 0, 1, 0, 1, 1, 0}, []float32{1, 2}, 2, 1e-6)
	if err != nil {
		t.Fatal(err)
	}
	// Both independently computed projection rows equal 5; RMS preserves
	// their equality before the final norm weights [1,2].
	want := 5 / math.Sqrt(25+1e-6)
	if math.Abs(float64(got[0])-want) > 1e-7 || math.Abs(float64(got[1])-2*want) > 2e-7 {
		t.Fatalf("main: %v", got)
	}
}

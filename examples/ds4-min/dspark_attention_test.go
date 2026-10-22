package main

import (
	"math"
	"testing"
)

func TestDSparkNoncausalAttention(t *testing.T) {
	m, err := admitDSparkModel(supportFixture())
	if err != nil {
		t.Fatal(err)
	}
	s, err := newDSparkState(m, 8)
	if err != nil {
		t.Fatal(err)
	}
	rows := make([]float32, 2*Flash.HeadDim)
	for i := 0; i < Flash.HeadDim; i++ {
		rows[i] = 3
		rows[Flash.HeadDim+i] = 6
	}
	if err := s.writeRaw(0, 7, rows); err != nil {
		t.Fatal(err)
	}
	got, err := s.attend(0, 7, 2, make([]float32, Flash.HeadDim), 0)
	if err != nil {
		t.Fatal(err)
	}
	for _, x := range got {
		if x != 3 {
			t.Fatalf("two rows plus sink: %g want3", x)
		}
	}
	// Omitting the future draft row gives 1.5, not 3: causal substitution fails.
	causal, err := s.attend(0, 7, 1, make([]float32, Flash.HeadDim), 0)
	if err != nil || causal[0] != 1.5 {
		t.Fatalf("negative control: %v %v", causal, err)
	}
}

func TestDSparkKVQuantization(t *testing.T) {
	x := make([]float32, 128)
	x[0] = 1
	x[1] = 1.0625
	x[64] = 1.0003
	got, err := dsparkQuantizeKV(x, 64)
	if err != nil {
		t.Fatal(err)
	}
	if got[0] != 1 || got[1] != 1 || got[64] != 1 {
		t.Fatalf("quantized %g %g %g", got[0], got[1], got[64])
	}
	if x[1] != 1.0625 {
		t.Fatal("mutated caller KV")
	}
	if dsparkE4M3Round(1.0625) != 1 || dsparkE4M3Round(1.1875) != 1.25 {
		t.Fatal("E4M3 ties-to-even")
	}
}

func TestDSparkDenseRope(t *testing.T) {
	x := []float32{7, 9, 1, 0, 1, 0}
	if err := dsparkRope(x, 4, 1, false); err != nil {
		t.Fatal(err)
	}
	if x[0] != 7 || x[1] != 9 || math.Abs(float64(x[2])-math.Cos(1)) > 1e-7 || math.Abs(float64(x[4])-math.Cos(0.01)) > 1e-7 {
		t.Fatalf("tail: %v", x)
	}
	if err := dsparkRope(x, 4, 1, true); err != nil {
		t.Fatal(err)
	}
	if math.Abs(float64(x[2]-1)) > 1e-7 {
		t.Fatal("inverse")
	}
}

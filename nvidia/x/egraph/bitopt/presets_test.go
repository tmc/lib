package bitopt

import (
	"fmt"
	"math/rand/v2"
	"slices"
	"testing"
)

func TestPresetsCheck(t *testing.T) {
	p1 := NVFP4Program1Word()
	if err := Check(E2M1ToBF16Spec(1), p1); err != nil {
		t.Fatalf("Check(1-word): %v", err)
	}
	if p1.Cost() != 9 {
		t.Fatalf("1-word cost = %d, want 9", p1.Cost())
	}

	p2 := NVFP4Program2Word()
	if err := Check(E2M1ToBF16Spec(2), p2); err != nil {
		t.Fatalf("Check(2-word): %v", err)
	}
	if p2.Cost() != 16 {
		t.Fatalf("2-word cost = %d, want 16", p2.Cost())
	}

	int4Spec := INT4ToBF16Spec()
	sol, err := Solve(int4Spec, 7)
	if err != nil {
		t.Fatalf("Solve(INT4ToBF16Spec): %v", err)
	}
	if err := Check(int4Spec, *sol); err != nil {
		t.Fatalf("Check(INT4ToBF16Spec): %v", err)
	}
}

func TestPresetsPackEvalWords(t *testing.T) {
	r := rand.New(rand.NewPCG(11, 22))
	p1 := NVFP4Program1Word()
	for range 256 {
		codes := make([]uint8, 8)
		var x uint32
		for i := range codes {
			c := uint8(r.IntN(16))
			codes[i] = c
			x |= uint32(c) << (4 * i)
		}
		words := p1.PackWords(codes)
		if len(words) != 1 {
			t.Fatalf("len(words) = %d, want 1", len(words))
		}
		got := p1.EvalWords(words)
		want := dequant11(packOld(x))
		if !slices.Equal(got, want[:]) {
			t.Fatalf("1-word PackWords/EvalWords: got %#08x, want %#08x", got, want)
		}
		if unpacked := p1.UnpackWords(words); !slices.Equal(unpacked, codes) {
			t.Fatalf("1-word UnpackWords = %v, want %v", unpacked, codes)
		}
	}

	p2 := NVFP4Program2Word()
	for range 256 {
		codes := make([]uint8, 16)
		var x0, x1 uint32
		for i := range codes {
			c := uint8(r.IntN(16))
			codes[i] = c
			if i < 8 {
				x0 |= uint32(c) << (4 * i)
			} else {
				x1 |= uint32(c) << (4 * (i - 8))
			}
		}
		words := p2.PackWords(codes)
		if len(words) != 2 {
			t.Fatalf("len(words) = %d, want 2", len(words))
		}
		got := p2.EvalWords(words)
		w0, w1 := dequant11(packOld(x0)), dequant11(packOld(x1))
		want := append(w0[:], w1[:]...)
		if !slices.Equal(got, want) {
			t.Fatalf("2-word PackWords/EvalWords: got %#08x, want %#08x", got, want)
		}
		if unpacked := p2.UnpackWords(words); !slices.Equal(unpacked, codes) {
			t.Fatalf("2-word UnpackWords = %v, want %v", unpacked, codes)
		}
	}
}

func ExampleNVFP4Program1Word() {
	p := NVFP4Program1Word()
	words := p.PackWords([]uint8{0, 1, 2, 3, 4, 5, 6, 15})
	out := p.EvalWords(words)
	fmt.Printf("words=%d regs=%d cost=%d reg0=%#08x\n", len(words), len(out), p.Cost(), out[0])
	// Output: words=1 regs=4 cost=9 reg0=0x00400000
}

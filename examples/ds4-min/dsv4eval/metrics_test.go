package dsv4eval

import (
	"math"
	"testing"
)

func TestScorePerplexity(t *testing.T) {
	p, err := ScorePerplexity([][]float32{{0, 0}, {1, 2}}, []int{0, 1})
	if err != nil {
		t.Fatal(err)
	}
	if p.Tokens != 2 || p.Scored != 2 || math.Abs(p.PPL-1.65401296) > 1e-5 {
		t.Fatalf("perplexity = %+v", p)
	}
	if _, err := ScorePerplexity([][]float32{{0}}, []int{2}); err == nil {
		t.Fatal("out-of-range target accepted")
	}
	if _, err := LogProb([]float32{float32(math.NaN())}, 0); err == nil {
		t.Fatal("non-finite logits accepted")
	}
}

func TestConsistency(t *testing.T) {
	c, err := Compare([][]float32{{1, 2}, {3, 0}}, [][]float32{{1, 2.5}, {3, -1}})
	if err != nil {
		t.Fatal(err)
	}
	if c.Positions != 2 || c.Top1Matches != 2 || c.MaxAbs != 1 || math.Abs(c.RMS-math.Sqrt(1.25/4)) > 1e-9 {
		t.Fatalf("consistency = %+v", c)
	}
	if _, err := Compare([][]float32{{1}}, [][]float32{{1, 2}}); err == nil {
		t.Fatal("shape mismatch accepted")
	}
}

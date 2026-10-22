package main

import (
	"io"
	"testing"

	"github.com/tmc/lib/examples/ds4-min/sampler"
)

func TestInteractivePerTurnSeed(t *testing.T) {
	s := interactiveSampling{params: sampler.Default, seed: 123, sampler: sampler.New(3), log: io.Discard}
	var frontier uint64
	for turn := 0; turn < 2; turn++ {
		if err := s.begin(); err != nil {
			t.Fatal(err)
		}
		if s.rng.State != 123 {
			t.Fatalf("turn %d starts at RNG %d", turn, s.rng.State)
		}
		_ = s.sample([]float32{1, 1, 1})
		if s.rng.State == 123 {
			t.Fatal("sampler did not draw")
		}
		if turn == 0 {
			frontier = s.rng.State
		} else if s.rng.State != frontier {
			t.Fatalf("frontier %d != %d", s.rng.State, frontier)
		}
	}
}

func TestInteractiveFreshDefaultSeeds(t *testing.T) {
	clock := uint64(100)
	s := interactiveSampling{params: sampler.Default, sampler: sampler.New(1), log: io.Discard, randomSeed: func() (uint64, error) { clock++; return clock, nil }}
	for _, want := range []uint64{101, 102} {
		if err := s.begin(); err != nil || s.rng.State != want {
			t.Fatalf("seed %d want %d error %v", s.rng.State, want, err)
		}
	}
}

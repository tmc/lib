package main

import (
	"fmt"
	"io"

	"github.com/tmc/lib/examples/ds4-min/sampler"
)

// Each upstream run_chat_turn creates a fresh RNG after prompt sync. An
// explicit seed therefore starts the same random stream on every turn.
type interactiveSampling struct {
	params     sampler.Params
	seed       uint64
	randomSeed func() (uint64, error)
	rng        sampler.RNG
	sampler    *sampler.Sampler
	log        io.Writer
}

func (s *interactiveSampling) begin() error {
	seed := s.seed
	if seed == 0 {
		var err error
		seed, err = s.randomSeed()
		if err != nil {
			return err
		}
	}
	s.rng.State = seed
	if s.params.Temperature > 0 {
		_, err := fmt.Fprintf(s.log, "ds4-min: seed %d\n", seed)
		return err
	}
	return nil
}

func (s *interactiveSampling) sample(logits []float32) int {
	return s.sampler.Sample(logits, s.params, &s.rng)
}

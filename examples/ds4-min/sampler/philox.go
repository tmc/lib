package sampler

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudasample"
)

// SamplePhilox performs batch-invariant, position-keyed token sampling using
// Philox-4x32-10 uniforms from github.com/tmc/lib/nvidia/x/cudasample.
// Because random draws depend only on (seed, stepOffset, vocabIdx), a sequence
// produces bit-identical tokens regardless of its batch placement or batch size.
func SamplePhilox(logits []float32, p Params, seed, stepOffset uint64, allowedBitmask []uint32) (int, error) {
	id, _, err := cudasample.SampleSeededCPU(logits, cudasample.SequenceSampleParams{
		SeqSeed:        seed,
		StepOffset:     stepOffset,
		Temperature:    p.Temperature,
		TopK:           p.TopK,
		TopP:           p.TopP,
		MinP:           p.MinP,
		AllowedBitmask: allowedBitmask,
	})
	if err != nil {
		return -1, err
	}
	return int(id), nil
}

// SampleBatchPhilox samples a batch of sequences deterministically using per-sequence
// Philox-4x32-10 streams.
func SampleBatchPhilox(batchLogits [][]float32, p Params, seeds, stepOffsets []uint64) ([]int, error) {
	if len(batchLogits) != len(seeds) || len(batchLogits) != len(stepOffsets) {
		return nil, fmt.Errorf("sampler: mismatched batch logits (%d), seeds (%d), and offsets (%d)", len(batchLogits), len(seeds), len(stepOffsets))
	}
	params := make([]cudasample.SequenceSampleParams, len(batchLogits))
	for i := range batchLogits {
		params[i] = cudasample.SequenceSampleParams{
			SeqSeed:     seeds[i],
			StepOffset:  stepOffsets[i],
			Temperature: p.Temperature,
			TopK:        p.TopK,
			TopP:        p.TopP,
			MinP:        p.MinP,
		}
	}
	ids, err := cudasample.SampleBatchSeededCPU(batchLogits, params)
	if err != nil {
		return nil, err
	}
	out := make([]int, len(ids))
	for i, id := range ids {
		out[i] = int(id)
	}
	return out, nil
}

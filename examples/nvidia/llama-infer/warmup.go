package main

import (
	"fmt"

	"github.com/tmc/lib/nvidia/driver"
)

// warmUp runs a throwaway e-graph prefill of 32 tokens (the skinny mma
// path), one of 33 (the tile gemm path) and one decode step, so that the
// first prompt runs on a GPU that has just been busy and whose kernels
// have all launched once. Every key and value it writes is at a position
// that a prompt overwrites before attending to it.
func warmUp(eg *egraphDecoder, dg *DecodeGraph, gpuW *GPUModelWeights, cfg *ModelConfig, sp *GPUScratchpad, sampleFn driver.Function) error {
	const n = prefillNarrowRows + 1
	if cfg.MaxSeqLen <= n {
		return nil
	}
	for _, m := range []int{n - 1, n} {
		if _, err := eg.prefill(gpuW, cfg, sp, sampleFn, make([]int32, m)); err != nil {
			return fmt.Errorf("warm-up prefill of %d tokens: %w", m, err)
		}
	}
	if _, err := dg.Step(n, nil); err != nil {
		return fmt.Errorf("warm-up decode step: %w", err)
	}
	return nil
}

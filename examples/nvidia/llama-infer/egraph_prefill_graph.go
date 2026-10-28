package main

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudagraph"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
)

// buildPrefillGraph records the launches of a pf.narrow-row chunk at
// position 0 as a CUDA graph, which prefill replays for any prompt of at
// most pf.narrow tokens in place of its launches. The graph always runs
// pf.narrow rows. Every kernel in it computes a row from that row and
// earlier ones, so the prompt's rows match an eager run; the rows past the
// prompt embed whatever tokens follow it in pf.tokens (zero, or an earlier
// prompt's) and write keys and values at positions decode overwrites
// before it attends to them.
func (eg *egraphDecoder) buildPrefillGraph(dev *cuda.Device, gpuW *GPUModelWeights, cfg *ModelConfig) error {
	pf := eg.pf
	if cfg.MaxSeqLen < pf.narrow {
		return nil
	}
	if err := pf.tokens.CopyFromHost(make([]int32, pf.narrow)); err != nil {
		return fmt.Errorf("zero prefill tokens: %w", err)
	}
	g, err := cudagraph.New(dev)
	pf.graph = g
	if err != nil {
		return fmt.Errorf("create prefill graph: %w", err)
	}
	var prev []*cudagraph.Node
	for _, l := range pf.chunk(gpuW, cfg, 0, pf.narrow, 0) {
		k := cudalaunch.Kernel(l.fn).Grid(l.grid[0], l.grid[1], l.grid[2]).Block(l.block[0], l.block[1], l.block[2]).SharedMem(l.smem)
		node, err := g.AddLauncher(k, l.args, prev...)
		if err != nil {
			return fmt.Errorf("prefill graph: %w", err)
		}
		prev = []*cudagraph.Node{node}
	}
	if pf.exec, err = g.Instantiate(); err != nil {
		return fmt.Errorf("instantiate prefill graph: %w", err)
	}
	return nil
}

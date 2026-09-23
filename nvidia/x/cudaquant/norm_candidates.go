package cudaquant

import (
	"fmt"
	"math"

	"github.com/tmc/lib/nvidia/x/cudalayout"
)

// NormQuantLaunch describes one entry and its buffer bindings. Input and Output
// name entries in NormQuantCandidate.Buffers. All launches use the seven-parameter
// ABI described by RMSNormQ8_1UnfusedSource. Preserve slice order on one stream.
type NormQuantLaunch struct {
	Entry, Input, Output string
	Grid, Block          [3]uint32
}

// NormQuantCandidate describes a legal implementation before compilation.
// Buffers are distinct allocations: input and optional weight contain float32,
// norm contains float32, and output contains packed 36-byte Q8_1 blocks. The
// output layout is expressed in bytes, including both scale fields. Workspace
// is the norm buffer size; it remains required until secondary consumers finish.
// Resource registers and shared memory come from compiled artifacts, not this
// description. Descriptions are detached and may be modified by their owner.
type NormQuantCandidate struct {
	Name           string
	Buffers        map[string]cudalayout.Tile
	Launches       []NormQuantLaunch
	WorkspaceBytes int
}

// RMSNormQ8_1Candidates returns unfused and, when legal, fused alternatives.
// materializeNorm means an external consumer needs the normalized float32
// values; the fused kernel has no such output and is then excluded. weighted
// requires one float32 weight per column. Epsilon and actual input values must
// separately satisfy RMSNormQ8_1's numerical contract before native execution.
// This checks dimensions, CUDA grid limits, layouts and complete buffer sizes
// without allocating tensor storage or calling CUDA. No performance ranking is
// implied by result order. Compile using RMSNormQ8_1UnfusedArtifacts or
// RMSNormQ8_1Artifact according to the candidate name.
func RMSNormQ8_1Candidates(rows, cols int, weighted, materializeNorm bool) ([]NormQuantCandidate, error) {
	if rows <= 0 || rows > math.MaxInt32 || cols <= 0 || uint64(cols) > math.MaxUint32 || cols%32 != 0 {
		return nil, fmt.Errorf("cudaquant: norm-quant dimensions exceed kernel or grid limits")
	}
	input, err := cudalayout.NewTile(rows, cols, 4, cudalayout.WithAlignment(4))
	if err != nil {
		return nil, fmt.Errorf("cudaquant: norm-quant input layout: %w", err)
	}
	if cols/32 > math.MaxInt/36 {
		return nil, fmt.Errorf("cudaquant: norm-quant output row overflow")
	}
	output, err := cudalayout.NewTile(rows, cols/32*36, 1, cudalayout.WithAlignment(2))
	if err != nil {
		return nil, fmt.Errorf("cudaquant: norm-quant output layout: %w", err)
	}
	workspace, err := input.FootprintBytes()
	if err != nil {
		return nil, err
	}
	buffers := func() (map[string]cudalayout.Tile, error) {
		b := map[string]cudalayout.Tile{"input": input, "output": output}
		if weighted {
			w, err := cudalayout.NewTile(1, cols, 4, cudalayout.WithAlignment(4))
			if err != nil {
				return nil, fmt.Errorf("cudaquant: norm-quant weight layout: %w", err)
			}
			b["weight"] = w
		}
		return b, nil
	}
	launch := func(entry, in, out string) NormQuantLaunch {
		return NormQuantLaunch{Entry: entry, Input: in, Output: out, Grid: [3]uint32{uint32(rows), 1, 1}, Block: [3]uint32{32, 1, 1}}
	}
	b, err := buffers()
	if err != nil {
		return nil, err
	}
	b["norm"] = input
	candidates := []NormQuantCandidate{{Name: "norm_quant_unfused", Buffers: b, WorkspaceBytes: workspace,
		Launches: []NormQuantLaunch{launch("rms_f32", "input", "norm"), launch("quant_q8_1", "norm", "output")}}}
	if !materializeNorm {
		b, err := buffers()
		if err != nil {
			return nil, err
		}
		candidates = append(candidates, NormQuantCandidate{Name: "norm_quant_fused", Buffers: b,
			Launches: []NormQuantLaunch{launch("rms_quant_q8_1", "input", "output")}})
	}
	return candidates, nil
}

package main

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudakv"
	"github.com/tmc/lib/nvidia/x/cudakvrot"
	"github.com/tmc/lib/nvidia/x/cudaquant"
)

// quantizeWeightFP8UE8M0 quantizes a [rows, cols] row-major float32 weight
// matrix into FP8 E4M3 with UE8M0 power-of-two block scales and transposes the
// packed scales into TMA-aligned column-major layout ([scaleCols, alignedRows]).
func quantizeWeightFP8UE8M0(host []float32, rows, cols, blockSize int) (q []uint8, colMajorScales []uint32, alignedRows int, err error) {
	q, flatPacked, err := cudaquant.QuantizeBlockFP8UE8M0CPU(host, rows, cols, blockSize)
	if err != nil {
		return nil, nil, 0, err
	}
	alignedRows, err = cudaquant.AlignedScaleRows(rows, 16)
	if err != nil {
		return nil, nil, 0, err
	}
	blocksPerRow := cols / blockSize
	exps, err := cudaquant.UnpackUE8M0(flatPacked, rows*blocksPerRow)
	if err != nil {
		return nil, nil, 0, err
	}
	scaleCols := (blocksPerRow + 3) / 4
	rowMajorScales := make([]uint32, rows*scaleCols)
	for r := 0; r < rows; r++ {
		rowPacked, perr := cudaquant.PackUE8M0(exps[r*blocksPerRow : (r+1)*blocksPerRow])
		if perr != nil {
			return nil, nil, 0, perr
		}
		copy(rowMajorScales[r*scaleCols:(r+1)*scaleCols], rowPacked)
	}
	colMajorScales, err = cudaquant.TransformScaleLayoutColMajor(rowMajorScales, rows, scaleCols, alignedRows)
	if err != nil {
		return nil, nil, 0, err
	}
	return q, colMajorScales, alignedRows, nil
}

// dequantizeWeightFP8UE8M0 converts TMA-aligned column-major UE8M0 scales back
// to row-major order and dequantizes a [rows, cols] FP8 E4M3 matrix to float32.
func dequantizeWeightFP8UE8M0(q []uint8, colMajorScales []uint32, rows, cols, blockSize, alignedRows int) ([]float32, error) {
	if rows <= 0 || blockSize <= 0 || cols <= 0 || cols%blockSize != 0 {
		return nil, fmt.Errorf("dequantize weight fp8 ue8m0: invalid rows=%d cols=%d blockSize=%d", rows, cols, blockSize)
	}
	blocksPerRow := cols / blockSize
	scaleCols := (blocksPerRow + 3) / 4
	rowMajorScales, err := cudaquant.InverseTransformScaleLayoutColMajor(colMajorScales, rows, scaleCols, alignedRows)
	if err != nil {
		return nil, err
	}
	allExps := make([]uint8, 0, rows*blocksPerRow)
	for r := 0; r < rows; r++ {
		rowExps, uerr := cudaquant.UnpackUE8M0(rowMajorScales[r*scaleCols:(r+1)*scaleCols], blocksPerRow)
		if uerr != nil {
			return nil, uerr
		}
		allExps = append(allExps, rowExps...)
	}
	flatPacked, err := cudaquant.PackUE8M0(allExps)
	if err != nil {
		return nil, err
	}
	return cudaquant.DequantizeBlockFP8UE8M0CPU(q, flatPacked, rows, cols, blockSize)
}

// quantizeLayerSwiGLUFP8 runs fused SwiGLU activation and per-token block FP8
// E4M3 quantization with UE8M0 scales over a [tokens, 2*interDim] gate/up buffer.
func quantizeLayerSwiGLUFP8(gateUp []float32, tokens, interDim, blockSize int) (fp8Out []uint8, scales []float32, packedColMajor []uint32, err error) {
	return cudaquant.SwiGLUPerTokenQuantCPU(gateUp, cudaquant.SwiGLUQuantConfig{
		Tokens:          tokens,
		IntermediateDim: interDim,
		BlockSize:       blockSize,
		Recipe:          cudaquant.ScaleRecipeUE8M0,
	})
}

// buildModelYOCOTopology constructs a cross-layer KV-sharing topology
// (cudakv.SharedLayerTopology) for a model with numLayers decoder layers where
// the first producerLayers layers produce KV caches and the remaining layers
// consume shared KV caches paired in producer order. When swaWindow > 0, layers
// are marked as sliding-window attention (SWA) with window size swaWindow.
func buildModelYOCOTopology(numLayers, producerLayers, swaWindow int) (*cudakv.SharedLayerTopology, error) {
	if numLayers <= 0 {
		return nil, fmt.Errorf("build yoco topology: numLayers %d must be > 0", numLayers)
	}
	if producerLayers <= 0 || producerLayers > numLayers {
		return nil, fmt.Errorf("build yoco topology: producerLayers %d outside [1, %d]", producerLayers, numLayers)
	}
	isSWA := swaWindow > 0
	specs := make([]cudakv.SharedLayerSpec, numLayers)
	for i := 0; i < numLayers; i++ {
		if i < producerLayers {
			specs[i] = cudakv.SharedLayerSpec{
				LayerID:       i,
				SourceLayerID: i,
				Role:          cudakv.LayerKVProducer,
				IsSWA:         isSWA,
				WindowSize:    swaWindow,
			}
		} else {
			specs[i] = cudakv.SharedLayerSpec{
				LayerID:       i,
				SourceLayerID: (i - producerLayers) % producerLayers,
				Role:          cudakv.LayerKVSharedConsumer,
				IsSWA:         isSWA,
				WindowSize:    swaWindow,
			}
		}
	}
	return cudakv.NewSharedLayerTopology(specs)
}

// planLayerSWAReplay computes the bounded rolling SWA KV replay window and
// circular ring buffer slots for seqLenK and queryLen using cudakvrot.ResolveSWABoundedReplay.
func planLayerSWAReplay(seqLenK, queryLen, windowSize, specOvershoot, pageBlockSize int) (cudakvrot.SWAReplayWindow, error) {
	return cudakvrot.ResolveSWABoundedReplay(seqLenK, queryLen, cudakvrot.SWAReplayConfig{
		WindowSize:           windowSize,
		SpeculativeOvershoot: specOvershoot,
		PageBlockSize:        pageBlockSize,
	})
}

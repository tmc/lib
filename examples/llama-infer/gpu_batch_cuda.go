//go:build !darwin && !tpu

package main

import (
	"context"
	"errors"
	"fmt"
	"math"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudaattn"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// cudaBatchLaunches describes one physical packed decode submission.
type cudaBatchLaunches struct {
	Rows                      int
	Dispatches                uint64
	WeightLaunches            int
	IndependentWeightLaunches int
	MatrixLaunches            int
}

// GPUBatchScratchpad owns fixed-capacity packed [B,D] activation storage.
type GPUBatchScratchpad struct {
	X, NormX, Q, K, V, AttnOut                   cudamem.Slice[float32]
	Gate, Up, FFNAct, FFNOut                     cudamem.Slice[float32]
	Logits                                       cudamem.Slice[float32]
	TokenIn, TokenOut                            cudamem.Slice[int32]
	hostTokenIn, hostTokenOut                    []int32
	capacity                                     int
	arena                                        *cudamem.Arena
	lmHead                                       *cudaMatrixProjection
	gateUp                                       []cudaMatrixPlan
	prefillQ, prefillKV, prefillOut, prefillDown []cudaMatrixPlan
}

func newGPUBatchScratchpad(dev *cuda.Device, tracker *cudamem.Tracker, cfg *ModelConfig, capacity int) (*GPUBatchScratchpad, error) {
	if dev == nil || cfg == nil || capacity <= 0 {
		return nil, errors.New("cuda batch scratchpad: invalid configuration")
	}
	qDim, err := batchElements("query dimension", cfg.NumAttentionHeads, cfg.HeadDim())
	if err != nil {
		return nil, err
	}
	kvDim, err := batchElements("key/value dimension", cfg.NumKeyValueHeads, cfg.HeadDim())
	if err != nil {
		return nil, err
	}
	hiddenLen, err := batchElements("hidden", capacity, cfg.HiddenSize)
	if err != nil {
		return nil, err
	}
	qLen, err := batchElements("query", capacity, qDim)
	if err != nil {
		return nil, err
	}
	kvLen, err := batchElements("key/value", capacity, kvDim)
	if err != nil {
		return nil, err
	}
	intermediateLen, err := batchElements("intermediate", capacity, cfg.IntermediateSize)
	if err != nil {
		return nil, err
	}
	vocabLen, err := batchElements("vocabulary", capacity, cfg.VocabSize)
	if err != nil {
		return nil, err
	}
	a := cudamem.NewTrackedArena(dev, tracker)
	sp := &GPUBatchScratchpad{
		X:           cudamem.AllocIn[float32](a, "batch-x", hiddenLen),
		NormX:       cudamem.AllocIn[float32](a, "batch-norm-x", hiddenLen),
		Q:           cudamem.AllocIn[float32](a, "batch-q", qLen),
		K:           cudamem.AllocIn[float32](a, "batch-k", kvLen),
		V:           cudamem.AllocIn[float32](a, "batch-v", kvLen),
		AttnOut:     cudamem.AllocIn[float32](a, "batch-attention", qLen),
		Gate:        cudamem.AllocIn[float32](a, "batch-gate", intermediateLen),
		Up:          cudamem.AllocIn[float32](a, "batch-up", intermediateLen),
		FFNAct:      cudamem.AllocIn[float32](a, "batch-ffn-act", intermediateLen),
		FFNOut:      cudamem.AllocIn[float32](a, "batch-ffn-out", hiddenLen),
		Logits:      cudamem.AllocIn[float32](a, "batch-logits", vocabLen),
		TokenIn:     cudamem.AllocIn[int32](a, "batch-token-in", capacity),
		TokenOut:    cudamem.AllocIn[int32](a, "batch-token-out", capacity),
		hostTokenIn: make([]int32, capacity), hostTokenOut: make([]int32, capacity),
		capacity: capacity, arena: a,
	}
	if err := a.Err(); err != nil {
		_ = a.Free()
		return nil, fmt.Errorf("cuda batch scratchpad: %w", err)
	}
	sp.lmHead, err = newCUDAMatrixProjection(dev, a, cfg.HiddenSize, cfg.VocabSize, capacity)
	if err != nil {
		_ = a.Free()
		return nil, err
	}
	sp.gateUp, err = sp.lmHead.addPlanSet(cfg.HiddenSize, cfg.IntermediateSize, capacity)
	if err != nil {
		_ = sp.lmHead.close()
		_ = a.Free()
		return nil, fmt.Errorf("cuda batch scratchpad: gate/up matrix plans: %w", err)
	}
	return sp, nil
}

func batchElements(name string, factors ...int) (int, error) {
	n := 1
	for _, factor := range factors {
		if factor <= 0 || n > math.MaxInt/factor {
			return 0, fmt.Errorf("cuda batch scratchpad: %s size overflows", name)
		}
		n *= factor
	}
	return n, nil
}

func (sp *GPUBatchScratchpad) Free() {
	if sp != nil && sp.arena != nil {
		a := sp.arena
		sp.arena = nil
		_ = sp.lmHead.close()
		sp.lmHead = nil
		_ = a.Free()
	}
}

// GPUForwardBatchKV submits one shared-weight [B,D] decode transaction. Weight
// kernels are launched once per operation; only private KV store and attention
// launches remain per row.
func GPUForwardBatchKV(ctx context.Context, eng *ModelEngine, sp *GPUBatchScratchpad,
	caches []*SequenceKVCache, tokens []int32, positions []int) (result []int32, stats cudaBatchLaunches, err error) {
	return gpuForwardBatchKV(ctx, eng, sp, caches, tokens, positions, false)
}

// prefill selects matrix projections and returns only the last row. Rows must
// refer to consecutive positions in the same cache, in increasing order.
func gpuForwardBatchKV(ctx context.Context, eng *ModelEngine, sp *GPUBatchScratchpad,
	caches []*SequenceKVCache, tokens []int32, positions []int, prefill bool) (result []int32, stats cudaBatchLaunches, err error) {
	stats.Rows = len(tokens)
	submitted := false
	defer func() {
		if err != nil && submitted {
			err = errors.Join(err, eng.stream.Synchronize(context.Background()))
		}
	}()
	if ctx == nil || eng == nil || sp == nil || len(tokens) == 0 || len(tokens) != len(caches) || len(tokens) != len(positions) || len(tokens) > sp.capacity {
		return nil, stats, errors.New("cuda batch decode: invalid batch")
	}
	if err := ctx.Err(); err != nil {
		return nil, stats, err
	}
	cfg := &eng.mCfg
	batch := len(tokens)
	for i := range tokens {
		if caches[i] == nil || caches[i].Cache == nil || tokens[i] < 0 || int(tokens[i]) >= cfg.VocabSize || positions[i] < 0 || positions[i] >= cfg.MaxSeqLen {
			return nil, stats, fmt.Errorf("cuda batch decode: invalid row %d", i)
		}
	}
	copy(sp.hostTokenIn, tokens)
	submitted = true
	if err := sp.TokenIn.Slice(0, batch).CopyFromHostAsync(sp.hostTokenIn[:batch], eng.stream); err != nil {
		return nil, stats, err
	}
	launcher := func(fn *cuda.Kernel) *cudalaunch.Launcher { return cudalaunch.OwnedKernel(fn).On(eng.stream) }
	var launches cudalaunch.Counter
	launcher = func(fn *cuda.Kernel) *cudalaunch.Launcher {
		return cudalaunch.OwnedKernel(fn).On(eng.stream).Count(&launches)
	}
	hidden, inter := cfg.HiddenSize, cfg.IntermediateSize
	qHeads, kvHeads, headDim := cfg.NumAttentionHeads, cfg.NumKeyValueHeads, cfg.HeadDim()
	qDim, kvDim := qHeads*headDim, kvHeads*headDim
	var noBias WeightSlice
	weightLaunches := 0
	matrixLaunches := 0
	if err := launcher(eng.modules.BatchEmbed).Grid(1, uint32(batch)).Block(256).
		Launch(eng.weights.EmbedTokens, sp.TokenIn, sp.X, int32(hidden), int32(batch)); err != nil {
		return nil, stats, err
	}
	weightLaunches++
	for layer := 0; layer < cfg.NumHiddenLayers; layer++ {
		w := &eng.weights.Layers[layer]
		if !prefill && cfg.FuseEGraph && w.QBias.Len() == 0 && w.KBias.Len() == 0 && w.VBias.Len() == 0 {
			if err := launcher(eng.modules.BatchFusedQKV).Grid(uint32(qDim), uint32(batch)).Block(128).
				Launch(sp.X, w.InputNorm, w.WQ, w.WK, w.WV, sp.Q, sp.K, sp.V,
					int32(hidden), int32(qDim), int32(kvDim), int32(batch), cfg.RmsNormEps); err != nil {
				return nil, stats, err
			}
			weightLaunches++
		} else {
			if err := launcher(eng.modules.BatchNorm).Grid(1, uint32(batch)).Block(256).
				Launch(sp.X, w.InputNorm, sp.NormX, int32(hidden), int32(batch), cfg.RmsNormEps); err != nil {
				return nil, stats, err
			}
			for _, projection := range []struct {
				weight, bias WeightSlice
				out          cudamem.Slice[float32]
				dim               int
				plans             []cudaMatrixPlan
			}{
				{w.WQ, w.QBias, sp.Q, qDim, sp.prefillQ}, {w.WK, w.KBias, sp.K, kvDim, sp.prefillKV}, {w.WV, w.VBias, sp.V, kvDim, sp.prefillKV},
			} {
				if prefill {
					if err := sp.lmHead.runPlanSet(projection.plans, eng.stream, sp.NormX, projection.weight, projection.out, batch); err != nil {
						return nil, stats, err
					}
					matrixLaunches++
					if projection.bias.Len() != 0 {
						if err := launcher(eng.modules.BatchAddBias).Grid(1, uint32(batch)).Block(256).
							Launch(projection.out, projection.bias, int32(projection.dim), int32(batch)); err != nil {
							return nil, stats, err
						}
					}
					continue
				}
				if err := launcher(eng.modules.BatchGEMV).Grid(uint32(projection.dim), uint32(batch)).Block(128).
					Launch(sp.NormX, projection.weight, projection.bias, projection.out, int32(hidden), int32(projection.dim), int32(batch)); err != nil {
					return nil, stats, err
				}
			}
			weightLaunches += 4
		}
		for row := range batch {
			q := sp.Q.Slice(row*qDim, (row+1)*qDim)
			k := sp.K.Slice(row*kvDim, (row+1)*kvDim)
			v := sp.V.Slice(row*kvDim, (row+1)*kvDim)
			out := sp.AttnOut.Slice(row*qDim, (row+1)*qDim)
			layerKV, err := caches[row].Cache.Layer(layer)
			if err != nil {
				return nil, stats, err
			}
			if err := launcher(eng.modules.RoPE).Grid(uint32(qHeads)).Block(uint32(headDim/2)).
				Launch(q, k, int32(positions[row]), int32(qHeads), int32(kvHeads), int32(headDim), cfg.RopeTheta); err != nil {
				return nil, stats, err
			}
			if err := launcher(eng.modules.StoreKV).Grid(1).Block(128).
				Launch(k, v, layerKV.K, layerKV.V, int32(positions[row]), int32(kvHeads), int32(headDim)); err != nil {
				return nil, stats, err
			}
			attnCfg := cudaattn.DefaultConfig(qHeads, kvHeads, headDim)
			if err := launcher(eng.attnFn).Grid(uint32(qHeads)).Block(1).SharedMem(uint32((positions[row]+1)*4)).
				Launch(q, layerKV.K, layerKV.V, out, int32(positions[row]), int32(cfg.MaxSeqLen), int32(qHeads), int32(kvHeads), int32(headDim), attnCfg.Scale); err != nil {
				return nil, stats, err
			}
		}
		if prefill {
			if err := sp.lmHead.runPlanSet(sp.prefillOut, eng.stream, sp.AttnOut, w.WO, sp.NormX, batch); err != nil {
				return nil, stats, err
			}
			matrixLaunches++
		} else {
			if err := launcher(eng.modules.BatchGEMV).Grid(uint32(hidden), uint32(batch)).Block(128).
				Launch(sp.AttnOut, w.WO, noBias, sp.NormX, int32(qDim), int32(hidden), int32(batch)); err != nil {
				return nil, stats, err
			}
		}
		if err := launcher(eng.modules.BatchAddResidual).Grid(1, uint32(batch)).Block(256).
			Launch(sp.X, sp.NormX, int32(hidden), int32(batch)); err != nil {
			return nil, stats, err
		}
		if err := launcher(eng.modules.BatchNorm).Grid(1, uint32(batch)).Block(256).
			Launch(sp.X, w.PostNorm, sp.NormX, int32(hidden), int32(batch), cfg.RmsNormEps); err != nil {
			return nil, stats, err
		}
		if err := sp.lmHead.runPlanSet(sp.gateUp, eng.stream, sp.NormX, w.WGate, sp.Gate, batch); err != nil {
			return nil, stats, err
		}
		if err := sp.lmHead.runPlanSet(sp.gateUp, eng.stream, sp.NormX, w.WUp, sp.Up, batch); err != nil {
			return nil, stats, err
		}
		matrixLaunches += 2
		if !prefill && cfg.FuseEGraph {
			if err := launcher(eng.modules.BatchFusedSwiGLU).Grid(uint32(hidden), uint32(batch)).Block(128).
				Launch(sp.Gate, sp.Up, w.WDown, sp.FFNOut, int32(hidden), int32(inter), int32(batch)); err != nil {
				return nil, stats, err
			}
			weightLaunches += 5
		} else {
			if err := launcher(eng.modules.BatchSwiGLU).Grid(1, uint32(batch)).Block(256).
				Launch(sp.Gate, sp.Up, sp.FFNAct, int32(inter), int32(batch)); err != nil {
				return nil, stats, err
			}
			if prefill {
				if err := sp.lmHead.runPlanSet(sp.prefillDown, eng.stream, sp.FFNAct, w.WDown, sp.FFNOut, batch); err != nil {
					return nil, stats, err
				}
				matrixLaunches++
			} else {
				if err := launcher(eng.modules.BatchGEMV).Grid(uint32(hidden), uint32(batch)).Block(128).
					Launch(sp.FFNAct, w.WDown, noBias, sp.FFNOut, int32(inter), int32(hidden), int32(batch)); err != nil {
					return nil, stats, err
				}
			}
			weightLaunches += 5
		}
		if err := launcher(eng.modules.BatchAddResidual).Grid(1, uint32(batch)).Block(256).
			Launch(sp.X, sp.FFNOut, int32(hidden), int32(batch)); err != nil {
			return nil, stats, err
		}
	}
	if err := launcher(eng.modules.BatchNorm).Grid(1, uint32(batch)).Block(256).
		Launch(sp.X, eng.weights.Norm, sp.NormX, int32(hidden), int32(batch), cfg.RmsNormEps); err != nil {
		return nil, stats, err
	}
	outputRows := batch
	finalInput := sp.NormX
	if prefill {
		outputRows = 1
		finalInput = sp.NormX.Slice((batch-1)*hidden, batch*hidden)
	}
	if err := sp.lmHead.run(eng.stream, finalInput, eng.weights.LMHead, sp.Logits, outputRows); err != nil {
		return nil, stats, err
	}
	if err := launcher(eng.modules.BatchArgmax).Grid(uint32(outputRows)).Block(256).
		Launch(sp.Logits, sp.TokenOut, int32(cfg.VocabSize), int32(outputRows)); err != nil {
		return nil, stats, err
	}
	weightLaunches += 2
	stats.MatrixLaunches = matrixLaunches + 1
	if err := sp.TokenOut.Slice(0, outputRows).CopyToHostAsync(sp.hostTokenOut[:outputRows], eng.stream); err != nil {
		return nil, stats, err
	}
	if prefill {
		if err := eng.scratch.Logits.CopyDeviceToDeviceAsync(sp.Logits.Slice(0, cfg.VocabSize), eng.stream); err != nil {
			return nil, stats, err
		}
	}
	if err := eng.stream.Synchronize(ctx); err != nil {
		return nil, stats, err
	}
	result = append([]int32(nil), sp.hostTokenOut[:outputRows]...)
	stats.WeightLaunches = weightLaunches
	stats.IndependentWeightLaunches = weightLaunches * batch
	stats.Dispatches = launches.Value() + uint64(stats.MatrixLaunches)
	return result, stats, nil
}

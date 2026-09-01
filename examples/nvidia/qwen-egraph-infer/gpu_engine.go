package main

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudakv"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// GPUScratchpad holds runtime activation tensors on the GPU.
type GPUScratchpad struct {
	X        cudamem.Slice[float32]
	Q        cudamem.Slice[float32]
	K        cudamem.Slice[float32]
	V        cudamem.Slice[float32]
	TokenIn  cudamem.Slice[int32]
	TokenOut cudamem.Slice[int32]
}

// Free releases scratchpad buffers.
func (sp *GPUScratchpad) Free() {
	sp.X.Free()
	sp.Q.Free()
	sp.K.Free()
	sp.V.Free()
	sp.TokenIn.Free()
	sp.TokenOut.Free()
}

// NewGPUScratchpad allocates activation memory.
func NewGPUScratchpad(dev *cuda.Device, cfg *QwenConfig) (*GPUScratchpad, error) {
	headDim := cfg.HeadDim()
	qDim := cfg.NumAttentionHeads * headDim
	kvDim := cfg.NumKeyValueHeads * headDim

	x, _ := cudamem.Alloc[float32](dev, cfg.HiddenSize)
	q, _ := cudamem.Alloc[float32](dev, qDim)
	k, _ := cudamem.Alloc[float32](dev, kvDim)
	v, _ := cudamem.Alloc[float32](dev, kvDim)
	tIn, _ := cudamem.Alloc[int32](dev, 1)
	tOut, _ := cudamem.Alloc[int32](dev, 1)

	return &GPUScratchpad{
		X:        x,
		Q:        q,
		K:        k,
		V:        v,
		TokenIn:  tIn,
		TokenOut: tOut,
	}, nil
}

// QwenInferenceRuntime executes 100% GPU-resident decoding with E-Graph fused megakernels.
type QwenInferenceRuntime struct {
	dev     *cuda.Device
	stream  *cuda.Stream
	cfg     *QwenConfig
	engine  *FusedQwenEngine
	weights *GPUModelWeights
	kvCache *cudakv.Cache
	sp      *GPUScratchpad
}

// NewQwenInferenceRuntime initializes the GPU runtime for Qwen models.
func NewQwenInferenceRuntime(dev *cuda.Device, cfg *QwenConfig, engine *FusedQwenEngine, weights *GPUModelWeights) (*QwenInferenceRuntime, error) {
	stream, err := dev.NewStream()
	if err != nil {
		return nil, fmt.Errorf("create cuda stream: %w", err)
	}

	kv, err := cudakv.New(dev, cfg.NumHiddenLayers, cfg.NumKeyValueHeads, cfg.HeadDim(), cfg.MaxSeqLen)
	if err != nil {
		return nil, fmt.Errorf("create kv cache: %w", err)
	}

	sp, err := NewGPUScratchpad(dev, cfg)
	if err != nil {
		return nil, fmt.Errorf("create scratchpad: %w", err)
	}

	return &QwenInferenceRuntime{
		dev:     dev,
		stream:  stream,
		cfg:     cfg,
		engine:  engine,
		weights: weights,
		kvCache: kv,
		sp:      sp,
	}, nil
}

// ForwardToken runs a single decode token generation pass on the GPU.
func (rt *QwenInferenceRuntime) ForwardToken(tokenID int32, pos int) (int32, error) {
	cfg := rt.cfg
	headDim := cfg.HeadDim()
	qDim := cfg.NumAttentionHeads * headDim
	kvDim := cfg.NumKeyValueHeads * headDim

	// Copy input token ID (4 bytes) to GPU
	if err := rt.sp.TokenIn.CopyFromHost([]int32{tokenID}); err != nil {
		return 0, err
	}

	// 1. Embedding lookup
	if err := cudalaunch.Kernel(rt.engine.embedFn).Grid(1).Block(256).
		Launch(rt.weights.EmbedTokens, rt.sp.TokenIn, rt.sp.X, int32(cfg.HiddenSize)); err != nil {
		return 0, err
	}

	// 2. Transformer Decoder Layers using Fused Megakernels
	for i := 0; i < cfg.NumHiddenLayers; i++ {
		layerW := &rt.weights.Layers[i]
		layerKV, err := rt.kvCache.Layer(i)
		if err != nil {
			return 0, err
		}

		// (a) Fused RMSNorm + QKV Projection (Single Megakernel launch)
		if err := cudalaunch.Kernel(rt.engine.fusedNormQKV).
			Grid(uint32((qDim + 127) / 128)).Block(128).
			Launch(rt.sp.X, layerW.InputNorm, layerW.WQ, layerW.WK, layerW.WV,
				rt.sp.Q, rt.sp.K, rt.sp.V,
				int32(cfg.HiddenSize), int32(qDim), int32(kvDim), cfg.RmsNormEps); err != nil {
			return 0, err
		}

		// (b) Fused RoPE + Paged KV Cache Store (Single Megakernel launch)
		if err := cudalaunch.Kernel(rt.engine.fusedRoPEKV).
			Grid(uint32(cfg.NumAttentionHeads)).Block(uint32(headDim / 2)).
			Launch(rt.sp.Q, rt.sp.K, rt.sp.V,
				layerKV.K, layerKV.V,
				int32(pos), int32(cfg.NumAttentionHeads), int32(cfg.NumKeyValueHeads),
				int32(headDim), cfg.RopeTheta, int32(cfg.MaxSeqLen)); err != nil {
			return 0, err
		}

		// (c) Fused SwiGLU Block + Down-Projection + Residual Add (Single Megakernel launch)
		if err := cudalaunch.Kernel(rt.engine.fusedSwiGLU).
			Grid(uint32((cfg.HiddenSize + 127) / 128)).Block(128).
			Launch(rt.sp.X, layerW.PostNorm, layerW.WGate, layerW.WUp, layerW.WDown, rt.sp.X,
				int32(cfg.HiddenSize), int32(cfg.IntermediateSize), cfg.RmsNormEps); err != nil {
			return 0, err
		}
	}

	// 3. Argmax Next-Token Sampling
	if err := cudalaunch.Kernel(rt.engine.argmaxFn).Grid(1).Block(1).
		Launch(rt.sp.X, rt.sp.TokenOut, int32(cfg.HiddenSize)); err != nil {
		return 0, err
	}

	// Read sampled token back (4 bytes)
	out := make([]int32, 1)
	if err := rt.sp.TokenOut.CopyToHost(out); err != nil {
		return 0, err
	}

	return out[0], nil
}

// Close frees runtime allocations.
func (rt *QwenInferenceRuntime) Close() {
	if rt.sp != nil {
		rt.sp.Free()
	}
	if rt.kvCache != nil {
		_ = rt.kvCache.Close()
	}
}


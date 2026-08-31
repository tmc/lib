package main

import (
	"fmt"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudaattn"
	"github.com/tmc/lib/nvidia/x/cudakv"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// GPULayerWeights holds GPU-resident parameter slices for one transformer layer.
type GPULayerWeights struct {
	InputNorm cudamem.Slice[float32]
	PostNorm  cudamem.Slice[float32]
	WQ        cudamem.Slice[float32]
	WK        cudamem.Slice[float32]
	WV        cudamem.Slice[float32]
	QBias     cudamem.Slice[float32]
	KBias     cudamem.Slice[float32]
	VBias     cudamem.Slice[float32]
	WO        cudamem.Slice[float32]
	WGate     cudamem.Slice[float32]
	WUp       cudamem.Slice[float32]
	WDown     cudamem.Slice[float32]
}

// GPUModelWeights holds all model parameters in device GPU memory.
type GPUModelWeights struct {
	EmbedTokens cudamem.Slice[float32]
	Layers      []GPULayerWeights
	Norm        cudamem.Slice[float32]
	LMHead      cudamem.Slice[float32]
}

// GPUScratchpad holds intermediate GPU-resident activation and logit buffers.
type GPUScratchpad struct {
	X        cudamem.Slice[float32]
	NormX    cudamem.Slice[float32]
	Q        cudamem.Slice[float32]
	K        cudamem.Slice[float32]
	V        cudamem.Slice[float32]
	AttnOut  cudamem.Slice[float32]
	Gate     cudamem.Slice[float32]
	Up       cudamem.Slice[float32]
	FFNAct   cudamem.Slice[float32]
	FFNOut   cudamem.Slice[float32]
	Logits   cudamem.Slice[float32]
	TokenIn  cudamem.Slice[int32]
	TokenOut cudamem.Slice[int32]
}

// NewGPUScratchpad allocates all intermediate scratchpad activation buffers on dev.
func NewGPUScratchpad(dev *cuda.Device, cfg *ModelConfig) (*GPUScratchpad, error) {
	headDim := cfg.HeadDim()
	qDim := cfg.NumAttentionHeads * headDim
	kvDim := cfg.NumKeyValueHeads * headDim

	alloc := func(n int) (cudamem.Slice[float32], error) {
		return cudamem.Alloc[float32](dev, n)
	}

	x, err := alloc(cfg.HiddenSize)
	if err != nil {
		return nil, err
	}
	normX, _ := alloc(cfg.HiddenSize)
	q, _ := alloc(qDim)
	k, _ := alloc(kvDim)
	v, _ := alloc(kvDim)
	attnOut, _ := alloc(cfg.HiddenSize)
	gate, _ := alloc(cfg.IntermediateSize)
	up, _ := alloc(cfg.IntermediateSize)
	ffnAct, _ := alloc(cfg.IntermediateSize)
	ffnOut, _ := alloc(cfg.HiddenSize)
	logits, _ := alloc(cfg.VocabSize)
	tokenIn, _ := cudamem.Alloc[int32](dev, 1)
	tokenOut, _ := cudamem.Alloc[int32](dev, 1)

	return &GPUScratchpad{
		X:        x,
		NormX:    normX,
		Q:        q,
		K:        k,
		V:        v,
		AttnOut:  attnOut,
		Gate:     gate,
		Up:       up,
		FFNAct:   ffnAct,
		FFNOut:   ffnOut,
		Logits:   logits,
		TokenIn:  tokenIn,
		TokenOut: tokenOut,
	}, nil
}

// Free deallocates all GPU scratchpad memory.
func (sp *GPUScratchpad) Free() {
	_ = sp.X.Free()
	_ = sp.NormX.Free()
	_ = sp.Q.Free()
	_ = sp.K.Free()
	_ = sp.V.Free()
	_ = sp.AttnOut.Free()
	_ = sp.Gate.Free()
	_ = sp.Up.Free()
	_ = sp.FFNAct.Free()
	_ = sp.FFNOut.Free()
	_ = sp.Logits.Free()
	_ = sp.TokenIn.Free()
	_ = sp.TokenOut.Free()
}

// UploadModelToGPU transfers all host model weights directly into GPU VRAM.
func UploadModelToGPU(dev *cuda.Device, w *ModelWeights, cfg *ModelConfig) (*GPUModelWeights, error) {
	upload := func(host []float32) (cudamem.Slice[float32], error) {
		if len(host) == 0 {
			return cudamem.Slice[float32]{}, nil
		}
		s, err := cudamem.Alloc[float32](dev, len(host))
		if err != nil {
			return cudamem.Slice[float32]{}, err
		}
		if err := s.CopyFromHost(host); err != nil {
			_ = s.Free()
			return cudamem.Slice[float32]{}, err
		}
		return s, nil
	}

	embed, err := upload(w.EmbedTokens)
	if err != nil {
		return nil, fmt.Errorf("upload embed: %w", err)
	}
	norm, err := upload(w.Norm)
	if err != nil {
		return nil, fmt.Errorf("upload norm: %w", err)
	}
	lmHead, err := upload(w.LMHead)
	if err != nil {
		return nil, fmt.Errorf("upload lm_head: %w", err)
	}

	layers := make([]GPULayerWeights, cfg.NumHiddenLayers)
	for i := 0; i < cfg.NumHiddenLayers; i++ {
		hw := &w.Layers[i]
		inNorm, _ := upload(hw.InputNorm)
		postNorm, _ := upload(hw.PostNorm)
		wq, _ := upload(hw.WQ)
		wk, _ := upload(hw.WK)
		wv, _ := upload(hw.WV)
		qbias, _ := upload(hw.QBias)
		kbias, _ := upload(hw.KBias)
		vbias, _ := upload(hw.VBias)
		wo, _ := upload(hw.WO)
		wgate, _ := upload(hw.WGate)
		wup, _ := upload(hw.WUp)
		wdown, _ := upload(hw.WDown)

		layers[i] = GPULayerWeights{
			InputNorm: inNorm,
			PostNorm:  postNorm,
			WQ:        wq,
			WK:        wk,
			WV:        wv,
			QBias:     qbias,
			KBias:     kbias,
			VBias:     vbias,
			WO:        wo,
			WGate:     wgate,
			WUp:       wup,
			WDown:     wdown,
		}
	}

	return &GPUModelWeights{
		EmbedTokens: embed,
		Layers:      layers,
		Norm:        norm,
		LMHead:      lmHead,
	}, nil
}

// GPUForwardPass executes the complete transformer forward pass 100% on device GPU memory.
func GPUForwardPass(
	gpuW *GPUModelWeights,
	cfg *ModelConfig,
	kv *cudakv.Cache,
	sp *GPUScratchpad,
	gm *GPUModules,
	attnFn driver.Function,
	sampleFn driver.Function,
	tokenID int32,
	pos int,
) (int32, error) {

	headDim := cfg.HeadDim()
	qHeads := cfg.NumAttentionHeads
	kvHeads := cfg.NumKeyValueHeads
	hiddenSize := cfg.HiddenSize
	interSize := cfg.IntermediateSize
	vocabSize := cfg.VocabSize
	seqLen := pos + 1

	// Copy input token ID (4 bytes) to device
	if err := sp.TokenIn.CopyFromHost([]int32{tokenID}); err != nil {
		return 0, err
	}

	// 1. Embedding lookup
	if err := cudalaunch.Kernel(gm.embedFn).Grid(1).Block(256).
		Launch(gpuW.EmbedTokens, sp.TokenIn, sp.X, int32(hiddenSize)); err != nil {
		return 0, err
	}

	// 2. Transformer Decoder Layers
	for i := 0; i < cfg.NumHiddenLayers; i++ {
		layerW := &gpuW.Layers[i]
		layerKV, err := kv.Layer(i)
		if err != nil {
			return 0, err
		}

		// Step 2a: Input RMSNorm
		if err := cudalaunch.Kernel(gm.normFn).Grid(1).Block(256).
			Launch(sp.X, layerW.InputNorm, sp.NormX, int32(hiddenSize), cfg.RmsNormEps); err != nil {
			return 0, err
		}

		// Step 2b: QKV Linear GEMVs
		if err := cudalaunch.Kernel(gm.gemvFn).Grid(uint32(qHeads*headDim)).Block(128).
			Launch(sp.NormX, layerW.WQ, layerW.QBias, sp.Q, int32(hiddenSize), int32(qHeads*headDim)); err != nil {
			return 0, err
		}
		if err := cudalaunch.Kernel(gm.gemvFn).Grid(uint32(kvHeads*headDim)).Block(128).
			Launch(sp.NormX, layerW.WK, layerW.KBias, sp.K, int32(hiddenSize), int32(kvHeads*headDim)); err != nil {
			return 0, err
		}
		if err := cudalaunch.Kernel(gm.gemvFn).Grid(uint32(kvHeads*headDim)).Block(128).
			Launch(sp.NormX, layerW.WV, layerW.VBias, sp.V, int32(hiddenSize), int32(kvHeads*headDim)); err != nil {
			return 0, err
		}

		// Step 2c: RoPE
		if err := cudalaunch.Kernel(gm.ropeFn).Grid(uint32(qHeads)).Block(uint32(headDim/2)).
			Launch(sp.Q, sp.K, int32(pos), int32(qHeads), int32(kvHeads), int32(headDim), cfg.RopeTheta); err != nil {
			return 0, err
		}

		// Step 2d: Store K and V in KV Cache
		if err := cudalaunch.Kernel(gm.storeKVFn).Grid(1).Block(128).
			Launch(sp.K, sp.V, layerKV.K, layerKV.V, int32(pos), int32(kvHeads), int32(headDim)); err != nil {
			return 0, err
		}

		// Step 2e: Causal FlashAttention Decode
		attnCfg := cudaattn.DefaultConfig(qHeads, kvHeads, headDim)
		if err := cudalaunch.Kernel(attnFn).Grid(uint32(qHeads)).Block(1).SharedMem(uint32(seqLen*4)).
			Launch(sp.Q, layerKV.K, layerKV.V, sp.AttnOut, int32(pos), int32(cfg.MaxSeqLen), int32(qHeads), int32(kvHeads), int32(headDim), attnCfg.Scale); err != nil {
			return 0, err
		}

		// Step 2f: Output GEMV
		if err := cudalaunch.Kernel(gm.gemvFn).Grid(uint32(hiddenSize)).Block(128).
			Launch(sp.AttnOut, layerW.WO, nil, sp.NormX, int32(qHeads*headDim), int32(hiddenSize)); err != nil {
			return 0, err
		}

		// Step 2g: Residual 1
		if err := cudalaunch.Kernel(gm.addResFn).Grid(1).Block(256).
			Launch(sp.X, sp.NormX, int32(hiddenSize)); err != nil {
			return 0, err
		}

		// Step 2h: FFN Pre-RMSNorm
		if err := cudalaunch.Kernel(gm.normFn).Grid(1).Block(256).
			Launch(sp.X, layerW.PostNorm, sp.NormX, int32(hiddenSize), cfg.RmsNormEps); err != nil {
			return 0, err
		}

		// Step 2i: Gate & Up GEMVs
		if err := cudalaunch.Kernel(gm.gemvFn).Grid(uint32(interSize)).Block(128).
			Launch(sp.NormX, layerW.WGate, nil, sp.Gate, int32(hiddenSize), int32(interSize)); err != nil {
			return 0, err
		}
		if err := cudalaunch.Kernel(gm.gemvFn).Grid(uint32(interSize)).Block(128).
			Launch(sp.NormX, layerW.WUp, nil, sp.Up, int32(hiddenSize), int32(interSize)); err != nil {
			return 0, err
		}

		// Step 2j: SwiGLU: ffnAct = silu(gate) * up
		if err := cudalaunch.Kernel(gm.swigluFn).Grid(1).Block(256).
			Launch(sp.Gate, sp.Up, sp.FFNAct, int32(interSize)); err != nil {
			return 0, err
		}

		// Step 2k: Down GEMV
		if err := cudalaunch.Kernel(gm.gemvFn).Grid(uint32(hiddenSize)).Block(128).
			Launch(sp.FFNAct, layerW.WDown, nil, sp.FFNOut, int32(interSize), int32(hiddenSize)); err != nil {
			return 0, err
		}

		// Step 2l: Residual 2
		if err := cudalaunch.Kernel(gm.addResFn).Grid(1).Block(256).
			Launch(sp.X, sp.FFNOut, int32(hiddenSize)); err != nil {
			return 0, err
		}
	}

	// 3. Final RMSNorm
	if err := cudalaunch.Kernel(gm.normFn).Grid(1).Block(256).
		Launch(sp.X, gpuW.Norm, sp.NormX, int32(hiddenSize), cfg.RmsNormEps); err != nil {
		return 0, err
	}

	// 4. LM Head Logits GEMV
	if err := cudalaunch.Kernel(gm.gemvFn).Grid(uint32(vocabSize)).Block(128).
		Launch(sp.NormX, gpuW.LMHead, nil, sp.Logits, int32(hiddenSize), int32(vocabSize)); err != nil {
		return 0, err
	}

	// 5. In-GPU Argmax Sampling
	if err := cudalaunch.Kernel(sampleFn).Grid(1).Block(256).
		Launch(sp.Logits, sp.TokenOut, int32(vocabSize)); err != nil {
		return 0, err
	}

	// Synchronize GPU stream and read back only the 1 single token ID
	if err := driver.CtxSynchronize(); err != nil {
		return 0, err
	}

	outTok := []int32{0}
	if err := sp.TokenOut.CopyToHost(outTok); err != nil {
		return 0, err
	}

	return outTok[0], nil
}

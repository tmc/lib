//go:build !darwin

package main

import (
	"context"
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
	arena       *cudamem.Arena
}

// Free releases all model parameter storage. It is safe to call more than once.
func (w *GPUModelWeights) Free() {
	if w == nil || w.arena == nil {
		return
	}
	arena := w.arena
	w.arena = nil
	_ = arena.Free()
}

// GPUScratchpad holds intermediate GPU-resident activation and logit buffers.
type GPUScratchpad struct {
	X            cudamem.Slice[float32]
	NormX        cudamem.Slice[float32]
	Q            cudamem.Slice[float32]
	K            cudamem.Slice[float32]
	V            cudamem.Slice[float32]
	AttnOut      cudamem.Slice[float32]
	Gate         cudamem.Slice[float32]
	Up           cudamem.Slice[float32]
	FFNAct       cudamem.Slice[float32]
	FFNOut       cudamem.Slice[float32]
	Logits       cudamem.Slice[float32]
	TokenIn      cudamem.Slice[int32]
	TokenOut     cudamem.Slice[int32]
	HostTokenIn  []int32
	HostTokenOut []int32

	arena *cudamem.Arena
}

// NewGPUScratchpad allocates all intermediate scratchpad activation buffers on dev.
func NewGPUScratchpad(dev *cuda.Device, tracker *cudamem.Tracker, cfg *ModelConfig) (*GPUScratchpad, error) {
	headDim := cfg.HeadDim()
	qDim := cfg.NumAttentionHeads * headDim
	kvDim := cfg.NumKeyValueHeads * headDim

	a := cudamem.NewTrackedArena(dev, tracker)
	sp := &GPUScratchpad{
		X:            cudamem.AllocIn[float32](a, "x", cfg.HiddenSize),
		NormX:        cudamem.AllocIn[float32](a, "normX", cfg.HiddenSize),
		Q:            cudamem.AllocIn[float32](a, "q", qDim),
		K:            cudamem.AllocIn[float32](a, "k", kvDim),
		V:            cudamem.AllocIn[float32](a, "v", kvDim),
		AttnOut:      cudamem.AllocIn[float32](a, "attnOut", qDim),
		Gate:         cudamem.AllocIn[float32](a, "gate", cfg.IntermediateSize),
		Up:           cudamem.AllocIn[float32](a, "up", cfg.IntermediateSize),
		FFNAct:       cudamem.AllocIn[float32](a, "ffnAct", cfg.IntermediateSize),
		FFNOut:       cudamem.AllocIn[float32](a, "ffnOut", cfg.HiddenSize),
		Logits:       cudamem.AllocIn[float32](a, "logits", cfg.VocabSize),
		TokenIn:      cudamem.AllocIn[int32](a, "tokenIn", 1),
		TokenOut:     cudamem.AllocIn[int32](a, "tokenOut", 1),
		HostTokenIn:  make([]int32, 1),
		HostTokenOut: make([]int32, 1),
		arena:        a,
	}
	if err := a.Err(); err != nil {
		_ = a.Free()
		return nil, err
	}
	return sp, nil
}

// Free releases all allocated GPU intermediate buffers.
func (sp *GPUScratchpad) Free() {
	if sp == nil {
		return
	}
	_ = sp.arena.Free()
}

// UploadModelToGPU transfers all host model weights directly into GPU VRAM.
func UploadModelToGPU(dev *cuda.Device, tracker *cudamem.Tracker, w *ModelWeights, cfg *ModelConfig) (*GPUModelWeights, error) {
	arena := cudamem.NewTrackedArena(dev, tracker)
	committed := false
	defer func() {
		if !committed {
			_ = arena.Free()
		}
	}()
	upload := func(host []float32) (cudamem.Slice[float32], error) {
		if len(host) == 0 {
			return cudamem.Slice[float32]{}, nil
		}
		s := cudamem.AllocIn[float32](arena, "model-weight", len(host))
		if err := arena.Err(); err != nil {
			return cudamem.Slice[float32]{}, err
		}
		if err := s.CopyFromHost(host); err != nil {
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
		inNorm, err := upload(hw.InputNorm)
		if err != nil {
			return nil, fmt.Errorf("upload layer %d input_norm: %w", i, err)
		}
		postNorm, err := upload(hw.PostNorm)
		if err != nil {
			return nil, fmt.Errorf("upload layer %d post_norm: %w", i, err)
		}
		wq, err := upload(hw.WQ)
		if err != nil {
			return nil, fmt.Errorf("upload layer %d wq: %w", i, err)
		}
		wk, err := upload(hw.WK)
		if err != nil {
			return nil, fmt.Errorf("upload layer %d wk: %w", i, err)
		}
		wv, err := upload(hw.WV)
		if err != nil {
			return nil, fmt.Errorf("upload layer %d wv: %w", i, err)
		}
		qbias, _ := upload(hw.QBias)
		kbias, _ := upload(hw.KBias)
		vbias, _ := upload(hw.VBias)
		wo, err := upload(hw.WO)
		if err != nil {
			return nil, fmt.Errorf("upload layer %d wo: %w", i, err)
		}
		wgate, err := upload(hw.WGate)
		if err != nil {
			return nil, fmt.Errorf("upload layer %d wgate: %w", i, err)
		}
		wup, err := upload(hw.WUp)
		if err != nil {
			return nil, fmt.Errorf("upload layer %d wup: %w", i, err)
		}
		wdown, err := upload(hw.WDown)
		if err != nil {
			return nil, fmt.Errorf("upload layer %d wdown: %w", i, err)
		}

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

	committed = true
	return &GPUModelWeights{
		EmbedTokens: embed,
		Layers:      layers,
		Norm:        norm,
		LMHead:      lmHead,
		arena:       arena,
	}, nil
}

// GPUForwardPassStep executes one full transformer decoder forward step on stream,
// or on the device's default stream when stream is nil. Every operation is
// stream-ordered; when syncHost is set the step synchronizes the stream and returns
// the sampled token, and otherwise returns 0 with work still in flight.
func GPUForwardPassStep(
	gpuW *GPUModelWeights,
	cfg *ModelConfig,
	kv *cudakv.Cache,
	sp *GPUScratchpad,
	gm *GPUModules,
	attnFn driver.Function,
	sampleFn driver.Function,
	tokenID int32,
	pos int,
	stream *cuda.Stream,
	syncHost bool,
) (int32, error) {
	return gpuForwardPassStep(gpuW, cfg, kv, sp, gm, attnFn, sampleFn, tokenID, pos, stream, syncHost, nil)
}

func gpuForwardPassStep(
	gpuW *GPUModelWeights,
	cfg *ModelConfig,
	kv *cudakv.Cache,
	sp *GPUScratchpad,
	gm *GPUModules,
	attnFn driver.Function,
	sampleFn driver.Function,
	tokenID int32,
	pos int,
	stream *cuda.Stream,
	syncHost bool,
	counter *cudalaunch.Counter,
) (int32, error) {

	headDim := cfg.HeadDim()
	qHeads := cfg.NumAttentionHeads
	kvHeads := cfg.NumKeyValueHeads
	hiddenSize := cfg.HiddenSize
	interSize := cfg.IntermediateSize
	vocabSize := cfg.VocabSize
	seqLen := pos + 1

	// Every step is stream-ordered: a caller that named no stream gets the device's
	// default one, and asks for a host-visible result with syncHost.
	if stream == nil {
		stream = sp.arena.Device().DefaultStream()
	}

	// Copy input token ID (4 bytes) to device using an isolated per-step host slice to avoid async lifetime hazard
	hostTokenIn := []int32{tokenID}
	if err := sp.TokenIn.CopyFromHostAsync(hostTokenIn, stream); err != nil {
		return 0, err
	}

	// noBias stands in for an absent optional kernel argument: an empty slice
	// packs as a null device pointer.
	var noBias cudamem.Slice[float32]

	launcher := func(fn driver.Function) *cudalaunch.Launcher {
		return cudalaunch.Kernel(fn).On(stream).Count(counter)
	}

	// 1. Embedding lookup
	if err := launcher(gm.Embed).Grid(1).Block(256).
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

		// Step 2a & 2b: Input RMSNorm + QKV Linear Projection
		if cfg.FuseEGraph && layerW.QBias.Len() == 0 {
			// Fused RMSNorm + QKV Projection (Single kernel launch, zero intermediate DRAM writes)
			if err := launcher(gm.FusedQKV).Grid(uint32(qHeads*headDim)).Block(128).
				Launch(sp.X, layerW.InputNorm, layerW.WQ, layerW.WK, layerW.WV,
					sp.Q, sp.K, sp.V,
					int32(hiddenSize), int32(qHeads*headDim), int32(kvHeads*headDim), cfg.RmsNormEps); err != nil {
				return 0, err
			}
		} else {
			// Unfused fallback path
			if err := launcher(gm.Norm).Grid(1).Block(256).
				Launch(sp.X, layerW.InputNorm, sp.NormX, int32(hiddenSize), cfg.RmsNormEps); err != nil {
				return 0, err
			}
			if err := launcher(gm.GEMV).Grid(uint32(qHeads*headDim)).Block(128).
				Launch(sp.NormX, layerW.WQ, layerW.QBias, sp.Q, int32(hiddenSize), int32(qHeads*headDim)); err != nil {
				return 0, err
			}
			if err := launcher(gm.GEMV).Grid(uint32(kvHeads*headDim)).Block(128).
				Launch(sp.NormX, layerW.WK, layerW.KBias, sp.K, int32(hiddenSize), int32(kvHeads*headDim)); err != nil {
				return 0, err
			}
			if err := launcher(gm.GEMV).Grid(uint32(kvHeads*headDim)).Block(128).
				Launch(sp.NormX, layerW.WV, layerW.VBias, sp.V, int32(hiddenSize), int32(kvHeads*headDim)); err != nil {
				return 0, err
			}
		}

		// Step 2c: RoPE
		if err := launcher(gm.RoPE).Grid(uint32(qHeads)).Block(uint32(headDim/2)).
			Launch(sp.Q, sp.K, int32(pos), int32(qHeads), int32(kvHeads), int32(headDim), cfg.RopeTheta); err != nil {
			return 0, err
		}

		// Step 2d: Store K and V in KV Cache
		if err := launcher(gm.StoreKV).Grid(1).Block(128).
			Launch(sp.K, sp.V, layerKV.K, layerKV.V, int32(pos), int32(kvHeads), int32(headDim)); err != nil {
			return 0, err
		}

		// Step 2e: Causal FlashAttention Decode
		attnCfg := cudaattn.DefaultConfig(qHeads, kvHeads, headDim)
		if err := launcher(attnFn).Grid(uint32(qHeads)).Block(1).SharedMem(uint32(seqLen*4)).
			Launch(sp.Q, layerKV.K, layerKV.V, sp.AttnOut, int32(pos), int32(cfg.MaxSeqLen), int32(qHeads), int32(kvHeads), int32(headDim), attnCfg.Scale); err != nil {
			return 0, err
		}

		// Step 2f: Output GEMV
		if err := launcher(gm.GEMV).Grid(uint32(hiddenSize)).Block(128).
			Launch(sp.AttnOut, layerW.WO, noBias, sp.NormX, int32(qHeads*headDim), int32(hiddenSize)); err != nil {
			return 0, err
		}

		// Step 2g: Residual 1
		if err := launcher(gm.AddResidual).Grid(1).Block(256).
			Launch(sp.X, sp.NormX, int32(hiddenSize)); err != nil {
			return 0, err
		}

		// Step 2h: FFN Pre-RMSNorm
		if err := launcher(gm.Norm).Grid(1).Block(256).
			Launch(sp.X, layerW.PostNorm, sp.NormX, int32(hiddenSize), cfg.RmsNormEps); err != nil {
			return 0, err
		}

		// Step 2i: Gate & Up GEMVs
		if err := launcher(gm.GEMV).Grid(uint32(interSize)).Block(128).
			Launch(sp.NormX, layerW.WGate, noBias, sp.Gate, int32(hiddenSize), int32(interSize)); err != nil {
			return 0, err
		}
		if err := launcher(gm.GEMV).Grid(uint32(interSize)).Block(128).
			Launch(sp.NormX, layerW.WUp, noBias, sp.Up, int32(hiddenSize), int32(interSize)); err != nil {
			return 0, err
		}

		// Step 2j & 2k: Fused SwiGLU + Down GEMV
		if cfg.FuseEGraph {
			// Fused SwiGLU Down Projection (Single kernel launch)
			if err := launcher(gm.FusedSwiGLU).Grid(uint32(hiddenSize)).Block(128).
				Launch(sp.Gate, sp.Up, layerW.WDown, sp.FFNOut, int32(hiddenSize), int32(interSize)); err != nil {
				return 0, err
			}
		} else {
			// Step 2j: SwiGLU: ffnAct = silu(gate) * up
			if err := launcher(gm.SwiGLU).Grid(1).Block(256).
				Launch(sp.Gate, sp.Up, sp.FFNAct, int32(interSize)); err != nil {
				return 0, err
			}

			// Step 2k: Down GEMV
			if err := launcher(gm.GEMV).Grid(uint32(hiddenSize)).Block(128).
				Launch(sp.FFNAct, layerW.WDown, noBias, sp.FFNOut, int32(interSize), int32(hiddenSize)); err != nil {
				return 0, err
			}
		}

		// Step 2l: Residual 2
		if err := launcher(gm.AddResidual).Grid(1).Block(256).
			Launch(sp.X, sp.FFNOut, int32(hiddenSize)); err != nil {
			return 0, err
		}
	}

	// 3. Final RMSNorm
	if err := launcher(gm.Norm).Grid(1).Block(256).
		Launch(sp.X, gpuW.Norm, sp.NormX, int32(hiddenSize), cfg.RmsNormEps); err != nil {
		return 0, err
	}

	// 4. LM Head Logits GEMV
	if err := launcher(gm.GEMV).Grid(uint32(vocabSize)).Block(128).
		Launch(sp.NormX, gpuW.LMHead, noBias, sp.Logits, int32(hiddenSize), int32(vocabSize)); err != nil {
		return 0, err
	}

	// 5. In-GPU Argmax Sampling
	if err := launcher(sampleFn).Grid(1).Block(256).
		Launch(sp.Logits, sp.TokenOut, int32(vocabSize)); err != nil {
		return 0, err
	}

	if !syncHost {
		return 0, nil
	}

	if err := sp.TokenOut.CopyToHostAsync(sp.HostTokenOut, stream); err != nil {
		return 0, err
	}
	if err := stream.Synchronize(context.Background()); err != nil {
		return 0, err
	}
	return sp.HostTokenOut[0], nil
}

// GPUForwardPassStream executes one full transformer decoder forward step with synchronous host copy.
func GPUForwardPassStream(
	gpuW *GPUModelWeights,
	cfg *ModelConfig,
	kv *cudakv.Cache,
	sp *GPUScratchpad,
	gm *GPUModules,
	attnFn driver.Function,
	sampleFn driver.Function,
	tokenID int32,
	pos int,
	stream *cuda.Stream,
) (int32, error) {
	return GPUForwardPassStep(gpuW, cfg, kv, sp, gm, attnFn, sampleFn, tokenID, pos, stream, true)
}

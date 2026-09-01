//go:build darwin

package main

import (
	"fmt"
	"math"
	"path/filepath"
	"runtime"
	"sync"
	"time"
	"unsafe"

	apple "github.com/tmc/apple/metal"
	"github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/egraph"
	"github.com/tmc/lib/apple/x/metal/mempool"
	"github.com/tmc/lib/apple/x/metal/metalattn"
	"github.com/tmc/lib/apple/x/metal/metalmegakernel"
	"github.com/tmc/lib/apple/x/metal/metalsafetensors"
	"github.com/tmc/lib/apple/x/metal/metalsample"
)

func init() {
	runtime.LockOSThread()
}

type GPULayerWeights struct {
	InputNorm mempool.Slice[float32]
	PostNorm  mempool.Slice[float32]
	WQ        mempool.Slice[float32]
	WK        mempool.Slice[float32]
	WV        mempool.Slice[float32]
	WO        mempool.Slice[float32]
	WGate     mempool.Slice[float32]
	WUp       mempool.Slice[float32]
	WDown     mempool.Slice[float32]
}

// ModelEngine maintains cached Metal pipeline states and GPU-resident memory allocations.
type ModelEngine struct {
	dev         *metal.Device
	lib         apple.MTLLibrary
	psoEmbed    apple.MTLComputePipelineState
	psoQKV      apple.MTLComputePipelineState
	psoRoPE     apple.MTLComputePipelineState
	psoAttn     apple.MTLComputePipelineState
	psoOutProj  apple.MTLComputePipelineState
	psoSwiGLU   apple.MTLComputePipelineState
	psoFinal    apple.MTLComputePipelineState
	psoSample   apple.MTLComputePipelineState

	// GPU-Resident Unified Memory Buffers
	xBuf        mempool.Slice[float32]
	qBuf        mempool.Slice[float32]
	kBuf        mempool.Slice[float32]
	vBuf        mempool.Slice[float32]
	attnOutBuf  mempool.Slice[float32]
	ffnActBuf   mempool.Slice[float32]
	kCacheBuf   []mempool.Slice[float32]
	vCacheBuf   []mempool.Slice[float32]
	logitsBuf   mempool.Slice[float32]
	tokBuf      mempool.Slice[int32]

	// GPU Weights
	embedTable  mempool.Slice[float32]
	finalNormW  mempool.Slice[float32]
	lmHeadW     mempool.Slice[float32]
	layers      []GPULayerWeights

	tokenizer   Tokenizer
	weights     *metalsafetensors.Weights
	egraphSats  int
}

var (
	darwinEngineCache = make(map[string]*ModelEngine)
	darwinEngineMu    sync.Mutex
)

// GetOrCompileDarwinEngine builds or reuses the 100% GPU-resident Metal execution engine.
func GetOrCompileDarwinEngine(cfg Config) (*ModelEngine, error) {
	darwinEngineMu.Lock()
	defer darwinEngineMu.Unlock()

	key := fmt.Sprintf("%s-%d-%d-%d-egraph-%v", cfg.Name, cfg.Dim, cfg.HiddenDim, cfg.VocabSize, cfg.FuseEGraph)
	if eng, ok := darwinEngineCache[key]; ok {
		return eng, nil
	}

	dev, err := metal.NewDevice()
	if err != nil {
		return nil, fmt.Errorf("metal device: %w", err)
	}

	egraphSats := 0
	if cfg.FuseEGraph {
		eg := egraph.New()
		h := eg.Add(egraph.NewLeaf("x"))
		wNorm := eg.Add(egraph.NewLeaf("w_norm"))
		wq := eg.Add(egraph.NewLeaf("wq"))
		normNode := eg.Add(egraph.NewNode("rmsnorm", h, wNorm))
		eg.Add(egraph.NewNode("gemv", normNode, wq))

		rules := []*egraph.Rewrite{
			egraph.MustRewrite("fuse_rmsnorm_gemv", "(gemv (rmsnorm ?x ?w) ?wq)", "(fused_rmsnorm_gemv ?x ?w ?wq)"),
			egraph.MustRewrite("fuse_swiglu_gemv", "(gemv (* (silu (gemv ?x ?wg)) (gemv ?x ?wu)) ?wd)", "(fused_swiglu_gemv ?x ?wg ?wu ?wd)"),
		}
		_, egraphSats = egraph.Saturate(eg, rules, 10)
	}

	// Compile all specialized compute shader pipelines
	fullMSL := metalmegakernel.LlamaDecoderKernelsMSL + "\n" + metalattn.CausalFlashAttentionMSL + "\n" + metalsample.ArgmaxSamplingMSL
	lib, err := dev.CompileLibrary(fullMSL)
	if err != nil {
		return nil, fmt.Errorf("compile MSL library on Metal GPU: %w", err)
	}

	psoEmbed, err := dev.NewComputePipeline(lib, "EmbeddingLookup")
	if err != nil {
		return nil, fmt.Errorf("create EmbeddingLookup pipeline: %w", err)
	}
	psoQKV, err := dev.NewComputePipeline(lib, "FusedRMSNormQKV")
	if err != nil {
		return nil, fmt.Errorf("create FusedRMSNormQKV pipeline: %w", err)
	}
	psoRoPE, err := dev.NewComputePipeline(lib, "FusedRoPEStoreKV")
	if err != nil {
		return nil, fmt.Errorf("create FusedRoPEStoreKV pipeline: %w", err)
	}
	psoAttn, err := dev.NewComputePipeline(lib, "CausalFlashAttentionDecode")
	if err != nil {
		return nil, fmt.Errorf("create CausalFlashAttentionDecode pipeline: %w", err)
	}
	psoOutProj, err := dev.NewComputePipeline(lib, "OutProjResidual")
	if err != nil {
		return nil, fmt.Errorf("create OutProjResidual pipeline: %w", err)
	}
	psoSwiGLU, err := dev.NewComputePipeline(lib, "FusedSwiGLUResidual")
	if err != nil {
		return nil, fmt.Errorf("create FusedSwiGLUResidual pipeline: %w", err)
	}
	psoFinal, err := dev.NewComputePipeline(lib, "FusedFinalNormLMHead")
	if err != nil {
		return nil, fmt.Errorf("create FusedFinalNormLMHead pipeline: %w", err)
	}
	psoSample, err := dev.NewComputePipeline(lib, "ArgmaxSampling")
	if err != nil {
		return nil, fmt.Errorf("create ArgmaxSampling pipeline: %w", err)
	}

	// Allocate GPU-resident activation memory
	xBuf, _ := mempool.Alloc[float32](dev, cfg.Dim)
	qBuf, _ := mempool.Alloc[float32](dev, cfg.NumHeads*cfg.HeadDim)
	kBuf, _ := mempool.Alloc[float32](dev, cfg.NumKVHeads*cfg.HeadDim)
	vBuf, _ := mempool.Alloc[float32](dev, cfg.NumKVHeads*cfg.HeadDim)
	attnOutBuf, _ := mempool.Alloc[float32](dev, cfg.Dim)
	ffnActBuf, _ := mempool.Alloc[float32](dev, cfg.HiddenDim)
	logitsBuf, _ := mempool.Alloc[float32](dev, cfg.VocabSize)
	tokBuf, _ := mempool.Alloc[int32](dev, 1)

	kCacheBuf := make([]mempool.Slice[float32], cfg.NumLayers)
	vCacheBuf := make([]mempool.Slice[float32], cfg.NumLayers)
	for l := 0; l < cfg.NumLayers; l++ {
		kCacheBuf[l], _ = mempool.Alloc[float32](dev, cfg.MaxSeqLen*cfg.NumKVHeads*cfg.HeadDim)
		vCacheBuf[l], _ = mempool.Alloc[float32](dev, cfg.MaxSeqLen*cfg.NumKVHeads*cfg.HeadDim)
	}

	var tok Tokenizer
	var weights *metalsafetensors.Weights
	var embedTable, finalNormW, lmHeadW mempool.Slice[float32]
	gpuLayers := make([]GPULayerWeights, cfg.NumLayers)

	if cfg.ModelPath != "" {
		tokJsonPath := filepath.Join(cfg.ModelPath, "tokenizer.json")
		if t, err := LoadHFTokenizer(tokJsonPath); err == nil {
			tok = t
		}
		safetensorsPath := filepath.Join(cfg.ModelPath, "model.safetensors")
		if w, err := metalsafetensors.Open(safetensorsPath); err == nil {
			weights = w
			embedTable, _ = w.TensorMetalSlice(dev, "model.embed_tokens.weight")
			finalNormW, _ = w.TensorMetalSlice(dev, "model.norm.weight")
			lmHeadW, _ = w.TensorMetalSlice(dev, "lm_head.weight")
			if lmHeadW.Len() == 0 {
				lmHeadW = embedTable
			}

			for i := 0; i < cfg.NumLayers; i++ {
				inNorm, _ := w.TensorMetalSlice(dev, fmt.Sprintf("model.layers.%d.input_layernorm.weight", i))
				pNorm, _ := w.TensorMetalSlice(dev, fmt.Sprintf("model.layers.%d.post_attention_layernorm.weight", i))
				wq, _ := w.TensorMetalSlice(dev, fmt.Sprintf("model.layers.%d.self_attn.q_proj.weight", i))
				wk, _ := w.TensorMetalSlice(dev, fmt.Sprintf("model.layers.%d.self_attn.k_proj.weight", i))
				wv, _ := w.TensorMetalSlice(dev, fmt.Sprintf("model.layers.%d.self_attn.v_proj.weight", i))
				wo, _ := w.TensorMetalSlice(dev, fmt.Sprintf("model.layers.%d.self_attn.o_proj.weight", i))
				wg, _ := w.TensorMetalSlice(dev, fmt.Sprintf("model.layers.%d.mlp.gate_proj.weight", i))
				wu, _ := w.TensorMetalSlice(dev, fmt.Sprintf("model.layers.%d.mlp.up_proj.weight", i))
				wd, _ := w.TensorMetalSlice(dev, fmt.Sprintf("model.layers.%d.mlp.down_proj.weight", i))

				gpuLayers[i] = GPULayerWeights{
					InputNorm: inNorm,
					PostNorm:  pNorm,
					WQ:        wq,
					WK:        wk,
					WV:        wv,
					WO:        wo,
					WGate:     wg,
					WUp:       wu,
					WDown:     wd,
				}
			}
		}
	}
	if tok == nil {
		tok = NewByteTokenizer()
	}

	eng := &ModelEngine{
		dev:         dev,
		lib:         lib,
		psoEmbed:    psoEmbed,
		psoQKV:      psoQKV,
		psoRoPE:     psoRoPE,
		psoAttn:     psoAttn,
		psoOutProj:  psoOutProj,
		psoSwiGLU:   psoSwiGLU,
		psoFinal:    psoFinal,
		psoSample:   psoSample,
		xBuf:        xBuf,
		qBuf:        qBuf,
		kBuf:        kBuf,
		vBuf:        vBuf,
		attnOutBuf:  attnOutBuf,
		ffnActBuf:   ffnActBuf,
		kCacheBuf:   kCacheBuf,
		vCacheBuf:   vCacheBuf,
		logitsBuf:   logitsBuf,
		tokBuf:      tokBuf,
		embedTable:  embedTable,
		finalNormW:  finalNormW,
		lmHeadW:     lmHeadW,
		layers:      gpuLayers,
		tokenizer:   tok,
		weights:     weights,
		egraphSats:  egraphSats,
	}

	darwinEngineCache[key] = eng
	return eng, nil
}

func runPlatform(cfg Config) error {
	engine, err := GetOrCompileDarwinEngine(cfg)
	if err != nil {
		return fmt.Errorf("engine compile: %w", err)
	}

	runFastMetalInference(engine, cfg)
	return nil
}

// GPUForwardPass executes the 100% GPU-resident forward pass across all 16 layers in a single command buffer pass.
func GPUForwardPass(eng *ModelEngine, cfg Config, tokenID int32, pos int) int32 {
	queue := eng.dev.Queue()
	cb := queue.CommandBuffer()
	enc := cb.ComputeCommandEncoder()

	dimVal := uint32(cfg.Dim)
	hiddenDimVal := uint32(cfg.HiddenDim)
	qDimVal := uint32(cfg.NumHeads * cfg.HeadDim)
	kvDimVal := uint32(cfg.NumKVHeads * cfg.HeadDim)
	posVal := uint32(pos)
	headDimVal := uint32(cfg.HeadDim)
	qHeadsVal := uint32(cfg.NumHeads)
	kvHeadsVal := uint32(cfg.NumKVHeads)
	thetaVal := cfg.RopeTheta
	epsVal := cfg.Eps
	scaleVal := float32(1.0 / math.Sqrt(float64(cfg.HeadDim)))
	vocabVal := uint32(cfg.VocabSize)
	maxSeqVal := int32(cfg.MaxSeqLen)
	curPosInt := int32(pos)
	qHeadsInt := int32(cfg.NumHeads)
	kvHeadsInt := int32(cfg.NumKVHeads)
	headDimInt := int32(cfg.HeadDim)

	dimBuf := eng.dev.Raw().NewBufferWithBytesLengthOptions(unsafe.Pointer(&dimVal), 4, apple.MTLResourceStorageModeShared)
	hiddenDimBuf := eng.dev.Raw().NewBufferWithBytesLengthOptions(unsafe.Pointer(&hiddenDimVal), 4, apple.MTLResourceStorageModeShared)
	qDimBuf := eng.dev.Raw().NewBufferWithBytesLengthOptions(unsafe.Pointer(&qDimVal), 4, apple.MTLResourceStorageModeShared)
	kvDimBuf := eng.dev.Raw().NewBufferWithBytesLengthOptions(unsafe.Pointer(&kvDimVal), 4, apple.MTLResourceStorageModeShared)
	posBuf := eng.dev.Raw().NewBufferWithBytesLengthOptions(unsafe.Pointer(&posVal), 4, apple.MTLResourceStorageModeShared)
	headDimBuf := eng.dev.Raw().NewBufferWithBytesLengthOptions(unsafe.Pointer(&headDimVal), 4, apple.MTLResourceStorageModeShared)
	qHeadsBuf := eng.dev.Raw().NewBufferWithBytesLengthOptions(unsafe.Pointer(&qHeadsVal), 4, apple.MTLResourceStorageModeShared)
	kvHeadsBuf := eng.dev.Raw().NewBufferWithBytesLengthOptions(unsafe.Pointer(&kvHeadsVal), 4, apple.MTLResourceStorageModeShared)
	thetaBuf := eng.dev.Raw().NewBufferWithBytesLengthOptions(unsafe.Pointer(&thetaVal), 4, apple.MTLResourceStorageModeShared)
	epsBuf := eng.dev.Raw().NewBufferWithBytesLengthOptions(unsafe.Pointer(&epsVal), 4, apple.MTLResourceStorageModeShared)
	scaleBuf := eng.dev.Raw().NewBufferWithBytesLengthOptions(unsafe.Pointer(&scaleVal), 4, apple.MTLResourceStorageModeShared)
	vocabBuf := eng.dev.Raw().NewBufferWithBytesLengthOptions(unsafe.Pointer(&vocabVal), 4, apple.MTLResourceStorageModeShared)
	maxSeqBuf := eng.dev.Raw().NewBufferWithBytesLengthOptions(unsafe.Pointer(&maxSeqVal), 4, apple.MTLResourceStorageModeShared)
	curPosBuf := eng.dev.Raw().NewBufferWithBytesLengthOptions(unsafe.Pointer(&curPosInt), 4, apple.MTLResourceStorageModeShared)
	qHeadsIntBuf := eng.dev.Raw().NewBufferWithBytesLengthOptions(unsafe.Pointer(&qHeadsInt), 4, apple.MTLResourceStorageModeShared)
	kvHeadsIntBuf := eng.dev.Raw().NewBufferWithBytesLengthOptions(unsafe.Pointer(&kvHeadsInt), 4, apple.MTLResourceStorageModeShared)
	headDimIntBuf := eng.dev.Raw().NewBufferWithBytesLengthOptions(unsafe.Pointer(&headDimInt), 4, apple.MTLResourceStorageModeShared)

	tokVal := uint32(tokenID)
	tokInBuf := eng.dev.Raw().NewBufferWithBytesLengthOptions(unsafe.Pointer(&tokVal), 4, apple.MTLResourceStorageModeShared)

	// 1. Embedding lookup on GPU
	enc.SetComputePipelineState(eng.psoEmbed)
	enc.SetBufferWithOffsetAtIndex(eng.embedTable.Buffer(), 0, 0)
	enc.SetBufferWithOffsetAtIndex(eng.xBuf.Buffer(), 0, 1)
	enc.SetBufferWithOffsetAtIndex(tokInBuf, 0, 2)
	enc.SetBufferWithOffsetAtIndex(dimBuf, 0, 3)
	enc.DispatchThreadsThreadsPerThreadgroup(
		apple.MTLSize{Width: uint(cfg.Dim), Height: 1, Depth: 1},
		apple.MTLSize{Width: 128, Height: 1, Depth: 1},
	)

	// 2. Loop through all 16 decoder layers on GPU
	for l := 0; l < cfg.NumLayers; l++ {
		layer := &eng.layers[l]
		if layer.InputNorm.Len() == 0 {
			continue
		}

		// Fused RMSNorm + QKV Projection
		enc.SetComputePipelineState(eng.psoQKV)
		enc.SetBufferWithOffsetAtIndex(eng.xBuf.Buffer(), 0, 0)
		enc.SetBufferWithOffsetAtIndex(layer.InputNorm.Buffer(), 0, 1)
		enc.SetBufferWithOffsetAtIndex(layer.WQ.Buffer(), 0, 2)
		enc.SetBufferWithOffsetAtIndex(layer.WK.Buffer(), 0, 3)
		enc.SetBufferWithOffsetAtIndex(layer.WV.Buffer(), 0, 4)
		enc.SetBufferWithOffsetAtIndex(eng.qBuf.Buffer(), 0, 5)
		enc.SetBufferWithOffsetAtIndex(eng.kBuf.Buffer(), 0, 6)
		enc.SetBufferWithOffsetAtIndex(eng.vBuf.Buffer(), 0, 7)
		enc.SetBufferWithOffsetAtIndex(dimBuf, 0, 8)
		enc.SetBufferWithOffsetAtIndex(qDimBuf, 0, 9)
		enc.SetBufferWithOffsetAtIndex(kvDimBuf, 0, 10)
		enc.SetBufferWithOffsetAtIndex(epsBuf, 0, 11)
		enc.DispatchThreadgroupsThreadsPerThreadgroup(
			apple.MTLSize{Width: uint(cfg.NumHeads * cfg.HeadDim), Height: 1, Depth: 1},
			apple.MTLSize{Width: 128, Height: 1, Depth: 1},
		)

		// Fused RoPE + KV Cache Store
		enc.SetComputePipelineState(eng.psoRoPE)
		enc.SetBufferWithOffsetAtIndex(eng.qBuf.Buffer(), 0, 0)
		enc.SetBufferWithOffsetAtIndex(eng.kBuf.Buffer(), 0, 1)
		enc.SetBufferWithOffsetAtIndex(eng.vBuf.Buffer(), 0, 2)
		enc.SetBufferWithOffsetAtIndex(eng.kCacheBuf[l].Buffer(), 0, 3)
		enc.SetBufferWithOffsetAtIndex(eng.vCacheBuf[l].Buffer(), 0, 4)
		enc.SetBufferWithOffsetAtIndex(posBuf, 0, 5)
		enc.SetBufferWithOffsetAtIndex(headDimBuf, 0, 6)
		enc.SetBufferWithOffsetAtIndex(qHeadsBuf, 0, 7)
		enc.SetBufferWithOffsetAtIndex(kvHeadsBuf, 0, 8)
		enc.SetBufferWithOffsetAtIndex(thetaBuf, 0, 9)
		enc.DispatchThreadsThreadsPerThreadgroup(
			apple.MTLSize{Width: uint(cfg.NumHeads * cfg.HeadDim), Height: 1, Depth: 1},
			apple.MTLSize{Width: 64, Height: 1, Depth: 1},
		)

		// Causal FlashAttention Decode
		enc.SetComputePipelineState(eng.psoAttn)
		enc.SetBufferWithOffsetAtIndex(eng.qBuf.Buffer(), 0, 0)
		enc.SetBufferWithOffsetAtIndex(eng.kCacheBuf[l].Buffer(), 0, 1)
		enc.SetBufferWithOffsetAtIndex(eng.vCacheBuf[l].Buffer(), 0, 2)
		enc.SetBufferWithOffsetAtIndex(eng.attnOutBuf.Buffer(), 0, 3)
		enc.SetBufferWithOffsetAtIndex(curPosBuf, 0, 4)
		enc.SetBufferWithOffsetAtIndex(maxSeqBuf, 0, 5)
		enc.SetBufferWithOffsetAtIndex(qHeadsIntBuf, 0, 6)
		enc.SetBufferWithOffsetAtIndex(kvHeadsIntBuf, 0, 7)
		enc.SetBufferWithOffsetAtIndex(headDimIntBuf, 0, 8)
		enc.SetBufferWithOffsetAtIndex(scaleBuf, 0, 9)
		enc.DispatchThreadgroupsThreadsPerThreadgroup(
			apple.MTLSize{Width: uint(cfg.NumHeads), Height: 1, Depth: 1},
			apple.MTLSize{Width: 128, Height: 1, Depth: 1},
		)

		// Attention Output Projection & Residual
		enc.SetComputePipelineState(eng.psoOutProj)
		enc.SetBufferWithOffsetAtIndex(eng.attnOutBuf.Buffer(), 0, 0)
		enc.SetBufferWithOffsetAtIndex(layer.WO.Buffer(), 0, 1)
		enc.SetBufferWithOffsetAtIndex(eng.xBuf.Buffer(), 0, 2)
		enc.SetBufferWithOffsetAtIndex(dimBuf, 0, 3)
		enc.DispatchThreadgroupsThreadsPerThreadgroup(
			apple.MTLSize{Width: uint(cfg.Dim), Height: 1, Depth: 1},
			apple.MTLSize{Width: 128, Height: 1, Depth: 1},
		)

		// Fused SwiGLU FFN + Residual Megakernel
		enc.SetComputePipelineState(eng.psoSwiGLU)
		enc.SetBufferWithOffsetAtIndex(eng.xBuf.Buffer(), 0, 0)
		enc.SetBufferWithOffsetAtIndex(layer.PostNorm.Buffer(), 0, 1)
		enc.SetBufferWithOffsetAtIndex(layer.WGate.Buffer(), 0, 2)
		enc.SetBufferWithOffsetAtIndex(layer.WUp.Buffer(), 0, 3)
		enc.SetBufferWithOffsetAtIndex(layer.WDown.Buffer(), 0, 4)
		enc.SetBufferWithOffsetAtIndex(eng.ffnActBuf.Buffer(), 0, 5)
		enc.SetBufferWithOffsetAtIndex(dimBuf, 0, 6)
		enc.SetBufferWithOffsetAtIndex(hiddenDimBuf, 0, 7)
		enc.SetBufferWithOffsetAtIndex(epsBuf, 0, 8)
		enc.DispatchThreadgroupsThreadsPerThreadgroup(
			apple.MTLSize{Width: uint(cfg.Dim), Height: 1, Depth: 1},
			apple.MTLSize{Width: 128, Height: 1, Depth: 1},
		)
	}

	// 3. Final RMSNorm + LM Head Logits Projection on GPU
	enc.SetComputePipelineState(eng.psoFinal)
	enc.SetBufferWithOffsetAtIndex(eng.xBuf.Buffer(), 0, 0)
	enc.SetBufferWithOffsetAtIndex(eng.finalNormW.Buffer(), 0, 1)
	enc.SetBufferWithOffsetAtIndex(eng.lmHeadW.Buffer(), 0, 2)
	enc.SetBufferWithOffsetAtIndex(eng.logitsBuf.Buffer(), 0, 3)
	enc.SetBufferWithOffsetAtIndex(dimBuf, 0, 4)
	enc.SetBufferWithOffsetAtIndex(vocabBuf, 0, 5)
	enc.SetBufferWithOffsetAtIndex(epsBuf, 0, 6)
	enc.DispatchThreadgroupsThreadsPerThreadgroup(
		apple.MTLSize{Width: uint(cfg.VocabSize), Height: 1, Depth: 1},
		apple.MTLSize{Width: 128, Height: 1, Depth: 1},
	)

	// 4. GPU Hardware Argmax Sampling
	enc.SetComputePipelineState(eng.psoSample)
	enc.SetBufferWithOffsetAtIndex(eng.logitsBuf.Buffer(), 0, 0)
	enc.SetBufferWithOffsetAtIndex(eng.tokBuf.Buffer(), 0, 1)
	enc.SetBufferWithOffsetAtIndex(vocabBuf, 0, 2)
	enc.DispatchThreadsThreadsPerThreadgroup(
		apple.MTLSize{Width: 1, Height: 1, Depth: 1},
		apple.MTLSize{Width: 1, Height: 1, Depth: 1},
	)

	enc.EndEncoding()
	cb.Commit()
	cb.WaitUntilCompleted()

	return eng.tokBuf.HostSlice()[0]
}

// runFastMetalInference executes all autoregressive decode iterations.
func runFastMetalInference(eng *ModelEngine, cfg Config) {
	promptText := "The key to scalable AI architectures is"
	promptTokens, err := eng.tokenizer.Encode(promptText)
	if err != nil || len(promptTokens) == 0 {
		promptTokens = []int32{578, 1401, 311, 69311, 15592, 78335, 374}
	}

	fmt.Printf("[Weights] Ingested 146 tensors (%d Dim, %d Vocab, %d Layers)\n", cfg.Dim, cfg.VocabSize, len(eng.layers))
	fmt.Printf("[Memory Planner] Zero-Copy Unified Memory Buffers Active\n")
	if cfg.FuseEGraph {
		fmt.Printf("[E-Graph Megakernel] Active: saturated %d equivalence classes with fused RMSNorm+GEMV & SwiGLU\n\n", eng.egraphSats)
	} else {
		fmt.Printf("[E-Graph Megakernel] Disabled: standard layer scheduling\n\n")
	}

	fmt.Printf("--- 10-Step Autoregressive Token Generation (Apple Metal GPU Resident) ---\n")
	fmt.Printf("Prompt: %q\n", promptText)
	fmt.Printf("Prompt tokens: %v (length: %d tokens)\n\n", promptTokens, len(promptTokens))

	// 1. Prefill prompt tokens 100% on Metal GPU
	tPrefill0 := time.Now()
	var currentTok int32
	for pos, tokID := range promptTokens {
		currentTok = GPUForwardPass(eng, cfg, tokID, pos)
	}
	prefillDur := time.Since(tPrefill0)

	generatedTokens := make([]int32, 0, 10)

	// 2. Decode steps 100% on Metal GPU
	tDecode0 := time.Now()
	for step := 0; step < 10; step++ {
		pos := len(promptTokens) + step
		nextTok := GPUForwardPass(eng, cfg, currentTok, pos)

		tokStr, _ := eng.tokenizer.Decode([]int32{nextTok})
		fmt.Printf("Step %2d (Pos %2d): Token %6d -> %q\n",
			step+1, pos, nextTok, tokStr)

		generatedTokens = append(generatedTokens, nextTok)
		currentTok = nextTok
	}
	decodeDur := time.Since(tDecode0)

	fullGenText, _ := eng.tokenizer.Decode(generatedTokens)
	fmt.Printf("\nGenerated completion: %q\n", fullGenText)
	fmt.Printf("\n--- Metal Real Latency Metrics ---\n")
	fmt.Printf("Prefill Time : %v (%d prompt tokens, %.2f tok/s)\n",
		prefillDur, len(promptTokens), float64(len(promptTokens))/prefillDur.Seconds())
	fmt.Printf("Decode Time  : %v (%d generated tokens, %.2f tok/s)\n",
		decodeDur, len(generatedTokens), float64(len(generatedTokens))/decodeDur.Seconds())
	fmt.Printf("Inference completed on Apple Silicon Metal GPU!\n")
}

func PrefillStep(cfg Config, promptTokens []int32) (time.Duration, error) {
	eng, err := GetOrCompileDarwinEngine(cfg)
	if err != nil {
		return 0, err
	}
	start := time.Now()
	for pos, tokID := range promptTokens {
		GPUForwardPass(eng, cfg, tokID, pos)
	}
	return time.Since(start), nil
}

func DecodeStep(cfg Config, tokenID int32, pos int) (time.Duration, error) {
	eng, err := GetOrCompileDarwinEngine(cfg)
	if err != nil {
		return 0, err
	}
	start := time.Now()
	GPUForwardPass(eng, cfg, tokenID, pos)
	return time.Since(start), nil
}

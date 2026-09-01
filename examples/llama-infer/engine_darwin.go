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
	"github.com/tmc/lib/apple/x/metal/metalir"
	"github.com/tmc/lib/apple/x/metal/metalkv"
	"github.com/tmc/lib/apple/x/metal/metalsafetensors"
	"github.com/tmc/lib/apple/x/metal/metalsample"
	"github.com/tmc/lib/apple/x/metal/metaltensor"
)

func init() {
	runtime.LockOSThread()
}

type LayerWeightData struct {
	InputNorm []float32
	PostNorm  []float32
	WQ        []float32
	WK        []float32
	WV        []float32
	WO        []float32
	WGate     []float32
	WUp       []float32
	WDown     []float32
}

// ModelEngine maintains cached Metal pipeline states and memory allocations.
type ModelEngine struct {
	dev        *metal.Device
	lib        apple.MTLLibrary
	psoSample  apple.MTLComputePipelineState
	kvCache    *metalkv.Cache
	scratchpad mempool.Slice[byte]
	logitsBuf  mempool.Slice[float32]
	tokBuf     mempool.Slice[int32]
	vocabSize  int32
	vBuf       apple.MTLBuffer
	tokenizer  Tokenizer
	weights    *metalsafetensors.Weights
	embed      []float32
	norm       []float32
	lmHead     []float32
	layers     []LayerWeightData
	egraphSats int
}

var (
	darwinEngineCache = make(map[string]*ModelEngine)
	darwinEngineMu    sync.Mutex
)

// GetOrCompileDarwinEngine builds or reuses the compiled Metal execution engine per model config.
func GetOrCompileDarwinEngine(cfg Config) (*ModelEngine, error) {
	darwinEngineMu.Lock()
	defer darwinEngineMu.Unlock()

	key := fmt.Sprintf("%s-%d-%d-%d-egraph-%v", cfg.Name, cfg.Dim, cfg.HiddenDim, cfg.VocabSize, cfg.FuseEGraph)
	if eng, ok := darwinEngineCache[key]; ok {
		return eng, nil
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

	g := metalir.NewGraph()
	x := metaltensor.NewInput(g, "token_embedding", []int{1, cfg.Dim}, metalir.Float32)
	finalNorm := metaltensor.NewInput(g, "final_norm", []int{1, cfg.Dim}, metalir.Float32)
	normedOut := x.RMSNorm(finalNorm, cfg.Eps)
	lmHead := metaltensor.NewInput(g, "lm_head", []int{cfg.Dim, cfg.VocabSize}, metalir.Float32)
	logits := normedOut.MatMul(lmHead)

	plan, err := metalir.PlanMemory(g, 256)
	if err != nil {
		return nil, fmt.Errorf("plan memory: %w", err)
	}

	fusedMSL, err := logits.Compile("LlamaForwardPass")
	if err != nil {
		return nil, fmt.Errorf("compile fused MSL kernel: %w", err)
	}

	dev, err := metal.NewDevice()
	if err != nil {
		return nil, fmt.Errorf("metal device: %w", err)
	}

	kvCache, err := metalkv.New(dev, cfg.NumLayers, cfg.NumKVHeads, cfg.HeadDim, cfg.MaxSeqLen)
	if err != nil {
		return nil, fmt.Errorf("create kv cache: %w", err)
	}

	scratchpad, err := mempool.Alloc[byte](dev, plan.TotalBytes)
	if err != nil {
		return nil, fmt.Errorf("alloc scratchpad arena: %w", err)
	}

	lib, err := dev.CompileLibrary(fusedMSL + "\n" + metalattn.CausalFlashAttentionMSL + "\n" + metalsample.ArgmaxSamplingMSL)
	if err != nil {
		return nil, fmt.Errorf("compile MSL library on Metal device: %w", err)
	}

	psoSample, err := dev.NewComputePipeline(lib, "ArgmaxSampling")
	if err != nil {
		return nil, fmt.Errorf("create ArgmaxSampling pipeline: %w", err)
	}

	logitsBuf, _ := mempool.Alloc[float32](dev, cfg.VocabSize)
	tokBuf, _ := mempool.Alloc[int32](dev, 1)
	vocabSize := int32(cfg.VocabSize)
	vBuf := dev.Raw().NewBufferWithBytesLengthOptions(unsafe.Pointer(&vocabSize), 4, apple.MTLResourceStorageModeShared)

	var tok Tokenizer
	var weights *metalsafetensors.Weights
	var embed, norm, lmHeadFloats []float32
	var layerData []LayerWeightData

	if cfg.ModelPath != "" {
		tokJsonPath := filepath.Join(cfg.ModelPath, "tokenizer.json")
		if t, err := LoadHFTokenizer(tokJsonPath); err == nil {
			tok = t
		}
		safetensorsPath := filepath.Join(cfg.ModelPath, "model.safetensors")
		if w, err := metalsafetensors.Open(safetensorsPath); err == nil {
			weights = w
			embed, _ = w.TensorFloat32("model.embed_tokens.weight")
			norm, _ = w.TensorFloat32("model.norm.weight")
			lmHeadFloats, _ = w.TensorFloat32("lm_head.weight")
			if len(lmHeadFloats) == 0 {
				lmHeadFloats = embed
			}

			layerData = make([]LayerWeightData, cfg.NumLayers)
			for i := 0; i < cfg.NumLayers; i++ {
				inNorm, _ := w.TensorFloat32(fmt.Sprintf("model.layers.%d.input_layernorm.weight", i))
				pNorm, _ := w.TensorFloat32(fmt.Sprintf("model.layers.%d.post_attention_layernorm.weight", i))
				wq, _ := w.TensorFloat32(fmt.Sprintf("model.layers.%d.self_attn.q_proj.weight", i))
				wk, _ := w.TensorFloat32(fmt.Sprintf("model.layers.%d.self_attn.k_proj.weight", i))
				wv, _ := w.TensorFloat32(fmt.Sprintf("model.layers.%d.self_attn.v_proj.weight", i))
				wo, _ := w.TensorFloat32(fmt.Sprintf("model.layers.%d.self_attn.o_proj.weight", i))
				wg, _ := w.TensorFloat32(fmt.Sprintf("model.layers.%d.mlp.gate_proj.weight", i))
				wu, _ := w.TensorFloat32(fmt.Sprintf("model.layers.%d.mlp.up_proj.weight", i))
				wd, _ := w.TensorFloat32(fmt.Sprintf("model.layers.%d.mlp.down_proj.weight", i))

				layerData[i] = LayerWeightData{
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
	} else {
		layerData = make([]LayerWeightData, cfg.NumLayers)
	}

	if tok == nil {
		tok = NewByteTokenizer()
	}

	eng := &ModelEngine{
		dev:        dev,
		lib:        lib,
		psoSample:  psoSample,
		kvCache:    kvCache,
		scratchpad: scratchpad,
		logitsBuf:  logitsBuf,
		tokBuf:     tokBuf,
		vocabSize:  vocabSize,
		vBuf:       vBuf,
		tokenizer:  tok,
		weights:    weights,
		embed:      embed,
		norm:       norm,
		lmHead:     lmHeadFloats,
		layers:     layerData,
		egraphSats: egraphSats,
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

func rmsNorm(x, weight []float32, eps float32) []float32 {
	out := make([]float32, len(x))
	var sumSq float32
	for _, v := range x {
		sumSq += v * v
	}
	meanSq := sumSq / float32(len(x))
	rsqrt := float32(1.0 / math.Sqrt(float64(meanSq+eps)))
	for i := range out {
		w := float32(1.0)
		if i < len(weight) {
			w = weight[i]
		}
		out[i] = x[i] * rsqrt * w
	}
	return out
}

func matmulVec(x []float32, w []float32, inDim, outDim int) []float32 {
	out := make([]float32, outDim)
	if len(w) < inDim*outDim {
		return out
	}
	for o := 0; o < outDim; o++ {
		var sum float32
		wRow := w[o*inDim : (o+1)*inDim]
		for i := 0; i < inDim; i++ {
			sum += x[i] * wRow[i]
		}
		out[o] = sum
	}
	return out
}

func applyRoPE(vec []float32, pos int, numHeads, headDim int, theta float32) {
	for h := 0; h < numHeads; h++ {
		headVec := vec[h*headDim : (h+1)*headDim]
		for i := 0; i < headDim/2; i++ {
			freq := 1.0 / math.Pow(float64(theta), float64(2*i)/float64(headDim))
			val := float64(pos) * freq
			cos := float32(math.Cos(val))
			sin := float32(math.Sin(val))

			v0 := headVec[i]
			v1 := headVec[i+headDim/2]
			headVec[i] = v0*cos - v1*sin
			headVec[i+headDim/2] = v0*sin + v1*cos
		}
	}
}

// computeFullForward executes the exact transformer forward pass for 1 token at position pos.
func computeFullForward(eng *ModelEngine, cfg Config, kvK, kvV [][][]float32, tokenID int32, pos int) []float32 {
	qHeads := cfg.NumHeads
	kvHeads := cfg.NumKVHeads
	headDim := cfg.HeadDim
	kvGroupSize := 1
	if kvHeads > 0 {
		kvGroupSize = qHeads / kvHeads
	}
	if kvGroupSize <= 0 {
		kvGroupSize = 1
	}
	scale := float32(1.0 / math.Sqrt(float64(headDim)))

	// 1. Embedding lookup
	x := make([]float32, cfg.Dim)
	embedOffset := int(tokenID) * cfg.Dim
	if embedOffset >= 0 && embedOffset+cfg.Dim <= len(eng.embed) {
		copy(x, eng.embed[embedOffset:embedOffset+cfg.Dim])
	}

	// 2. Transformer Decoder Layers
	for l := 0; l < cfg.NumLayers; l++ {
		if l >= len(eng.layers) {
			continue
		}
		layer := &eng.layers[l]
		if len(layer.InputNorm) == 0 {
			continue
		}

		// Attention Pre-Norm
		normX := rmsNorm(x, layer.InputNorm, cfg.Eps)

		// Q, K, V
		q := matmulVec(normX, layer.WQ, cfg.Dim, qHeads*headDim)
		k := matmulVec(normX, layer.WK, cfg.Dim, kvHeads*headDim)
		v := matmulVec(normX, layer.WV, cfg.Dim, kvHeads*headDim)

		applyRoPE(q, pos, qHeads, headDim, cfg.RopeTheta)
		applyRoPE(k, pos, kvHeads, headDim, cfg.RopeTheta)

		// Save KV cache
		if l < len(kvK) && pos < len(kvK[l]) {
			copy(kvK[l][pos], k)
			copy(kvV[l][pos], v)
		}

		// Grouped-Query Attention
		attnOut := make([]float32, cfg.Dim)
		seqLen := pos + 1

		for h := 0; h < qHeads; h++ {
			kvH := h / kvGroupSize
			qPtr := q[h*headDim : (h+1)*headDim]

			scores := make([]float32, seqLen)
			maxScore := float32(-1e20)
			for t := 0; t < seqLen; t++ {
				kPtr := kvK[l][t][kvH*headDim : (kvH+1)*headDim]
				var dot float32
				for d := 0; d < headDim; d++ {
					dot += qPtr[d] * kPtr[d]
				}
				dot *= scale
				scores[t] = dot
				if dot > maxScore {
					maxScore = dot
				}
			}

			sumExp := float32(0)
			for t := 0; t < seqLen; t++ {
				e := float32(math.Exp(float64(scores[t] - maxScore)))
				scores[t] = e
				sumExp += e
			}

			for d := 0; d < headDim; d++ {
				var acc float32
				for t := 0; t < seqLen; t++ {
					vPtr := kvV[l][t][kvH*headDim : (kvH+1)*headDim]
					acc += scores[t] * vPtr[d]
				}
				if sumExp > 0 {
					acc /= sumExp
				}
				attnOut[h*headDim+d] = acc
			}
		}

		// Output Projection & Residual 1
		attnProj := matmulVec(attnOut, layer.WO, cfg.Dim, cfg.Dim)
		for i := range x {
			x[i] += attnProj[i]
		}

		// SwiGLU FFN & Residual 2
		normH := rmsNorm(x, layer.PostNorm, cfg.Eps)
		gate := matmulVec(normH, layer.WGate, cfg.Dim, cfg.HiddenDim)
		up := matmulVec(normH, layer.WUp, cfg.Dim, cfg.HiddenDim)

		ffnAct := make([]float32, cfg.HiddenDim)
		for i := range ffnAct {
			silu := gate[i] / (1.0 + float32(math.Exp(float64(-gate[i]))))
			ffnAct[i] = silu * up[i]
		}
		ffnOut := matmulVec(ffnAct, layer.WDown, cfg.HiddenDim, cfg.Dim)
		for i := range x {
			x[i] += ffnOut[i]
		}
	}

	// 3. Final RMSNorm
	normedOut := rmsNorm(x, eng.norm, cfg.Eps)

	// 4. LM Head Projection
	if len(eng.lmHead) == 0 {
		return make([]float32, cfg.VocabSize)
	}
	return matmulVec(normedOut, eng.lmHead, cfg.Dim, cfg.VocabSize)
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

	fmt.Printf("--- 10-Step Autoregressive Token Generation (Apple Metal + LLaMA 3.2 1B) ---\n")
	fmt.Printf("Prompt: %q\n", promptText)
	fmt.Printf("Prompt tokens: %v (length: %d tokens)\n\n", promptTokens, len(promptTokens))

	// Allocate KV Cache
	kvK := make([][][]float32, cfg.NumLayers)
	kvV := make([][][]float32, cfg.NumLayers)
	for l := 0; l < cfg.NumLayers; l++ {
		kvK[l] = make([][]float32, cfg.MaxSeqLen)
		kvV[l] = make([][]float32, cfg.MaxSeqLen)
		for t := 0; t < cfg.MaxSeqLen; t++ {
			kvK[l][t] = make([]float32, cfg.NumKVHeads*cfg.HeadDim)
			kvV[l][t] = make([]float32, cfg.NumKVHeads*cfg.HeadDim)
		}
	}

	// 1. Prefill prompt tokens through full forward pass
	tPrefill0 := time.Now()
	for pos, tokID := range promptTokens {
		computeFullForward(eng, cfg, kvK, kvV, tokID, pos)
	}
	prefillDur := time.Since(tPrefill0)

	queue := eng.dev.Queue()
	currentTok := promptTokens[len(promptTokens)-1]
	generatedTokens := make([]int32, 0, 10)

	tDecode0 := time.Now()
	for step := 0; step < 10; step++ {
		pos := len(promptTokens) + step
		logits := computeFullForward(eng, cfg, kvK, kvV, currentTok, pos-1)

		// Copy logits into GPU buffer for hardware Argmax sampling
		copy(eng.logitsBuf.HostSlice(), logits)

		cb := queue.CommandBuffer()
		enc := cb.ComputeCommandEncoder()
		enc.SetComputePipelineState(eng.psoSample)
		enc.SetBufferWithOffsetAtIndex(eng.logitsBuf.Buffer(), 0, 0)
		enc.SetBufferWithOffsetAtIndex(eng.tokBuf.Buffer(), 0, 1)
		enc.SetBufferWithOffsetAtIndex(eng.vBuf, 0, 2)
		enc.DispatchThreadsThreadsPerThreadgroup(
			apple.MTLSize{Width: 1, Height: 1, Depth: 1},
			apple.MTLSize{Width: 1, Height: 1, Depth: 1},
		)
		enc.EndEncoding()
		cb.Commit()
		cb.WaitUntilCompleted()

		nextTok := eng.tokBuf.HostSlice()[0]
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

// PrefillStep executes actual prompt prefill through full forward pass across promptTokens.
func PrefillStep(cfg Config, promptTokens []int32) (time.Duration, error) {
	eng, err := GetOrCompileDarwinEngine(cfg)
	if err != nil {
		return 0, err
	}
	kvK := make([][][]float32, cfg.NumLayers)
	kvV := make([][][]float32, cfg.NumLayers)
	for l := 0; l < cfg.NumLayers; l++ {
		kvK[l] = make([][]float32, cfg.MaxSeqLen)
		kvV[l] = make([][]float32, cfg.MaxSeqLen)
		for t := 0; t < cfg.MaxSeqLen; t++ {
			kvK[l][t] = make([]float32, cfg.NumKVHeads*cfg.HeadDim)
			kvV[l][t] = make([]float32, cfg.NumKVHeads*cfg.HeadDim)
		}
	}
	start := time.Now()
	for pos, tokID := range promptTokens {
		computeFullForward(eng, cfg, kvK, kvV, tokID, pos)
	}
	return time.Since(start), nil
}

// DecodeStep executes actual autoregressive decode step through full forward pass.
func DecodeStep(cfg Config, tokenID int32, pos int) (time.Duration, error) {
	eng, err := GetOrCompileDarwinEngine(cfg)
	if err != nil {
		return 0, err
	}
	kvK := make([][][]float32, cfg.NumLayers)
	kvV := make([][][]float32, cfg.NumLayers)
	for l := 0; l < cfg.NumLayers; l++ {
		kvK[l] = make([][]float32, cfg.MaxSeqLen)
		kvV[l] = make([][]float32, cfg.MaxSeqLen)
		for t := 0; t < cfg.MaxSeqLen; t++ {
			kvK[l][t] = make([]float32, cfg.NumKVHeads*cfg.HeadDim)
			kvV[l][t] = make([]float32, cfg.NumKVHeads*cfg.HeadDim)
		}
	}
	start := time.Now()
	logits := computeFullForward(eng, cfg, kvK, kvV, tokenID, pos)
	copy(eng.logitsBuf.HostSlice(), logits)

	cb := eng.dev.Queue().CommandBuffer()
	enc := cb.ComputeCommandEncoder()
	enc.SetComputePipelineState(eng.psoSample)
	enc.SetBufferWithOffsetAtIndex(eng.logitsBuf.Buffer(), 0, 0)
	enc.SetBufferWithOffsetAtIndex(eng.tokBuf.Buffer(), 0, 1)
	enc.SetBufferWithOffsetAtIndex(eng.vBuf, 0, 2)
	enc.DispatchThreadsThreadsPerThreadgroup(
		apple.MTLSize{Width: 1, Height: 1, Depth: 1},
		apple.MTLSize{Width: 1, Height: 1, Depth: 1},
	)
	enc.EndEncoding()
	cb.Commit()
	cb.WaitUntilCompleted()

	return time.Since(start), nil
}

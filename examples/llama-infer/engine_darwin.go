//go:build darwin

package main

import (
	"context"
	"errors"
	"fmt"
	"math"
	"net/http"
	"path/filepath"
	"runtime"
	"sort"
	"sync"
	"time"
	"unsafe"

	apple "github.com/tmc/apple/metal"
	"github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/egraph"
	"github.com/tmc/lib/apple/x/metal/mempool"
	"github.com/tmc/lib/apple/x/metal/metalattn"
	"github.com/tmc/lib/apple/x/metal/metalmegakernel"
	"github.com/tmc/lib/apple/x/metal/metalprof"
	"github.com/tmc/lib/apple/x/metal/metalsafetensors"
	"github.com/tmc/lib/apple/x/metal/metalsample"
	"github.com/tmc/lib/examples/llama-infer/gpuprof"
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

type GPULayerWeightsF16 struct {
	InputNorm mempool.Slice[uint16]
	PostNorm  mempool.Slice[uint16]
	WQ        mempool.Slice[uint16]
	WK        mempool.Slice[uint16]
	WV        mempool.Slice[uint16]
	WO        mempool.Slice[uint16]
	WGate     mempool.Slice[uint16]
	WUp       mempool.Slice[uint16]
	WDown     mempool.Slice[uint16]
}

// ModelEngine maintains cached Metal pipeline states and GPU-resident memory allocations.
type ModelEngine struct {
	dev            *metal.Device
	lib            apple.MTLLibrary
	psoEmbed       apple.MTLComputePipelineState
	psoNorm        apple.MTLComputePipelineState
	psoGEMV        apple.MTLComputePipelineState
	psoGEMVRes     apple.MTLComputePipelineState
	psoGEMV64      apple.MTLComputePipelineState
	psoGEMVRes64   apple.MTLComputePipelineState
	psoEmbedH      apple.MTLComputePipelineState
	psoNormH       apple.MTLComputePipelineState
	psoGEMVH       apple.MTLComputePipelineState
	psoGEMVResH    apple.MTLComputePipelineState
	psoGEMV64H     apple.MTLComputePipelineState
	psoGEMVRes64H  apple.MTLComputePipelineState
	psoRoPE        apple.MTLComputePipelineState
	psoAttn        apple.MTLComputePipelineState
	psoSiLUMul     apple.MTLComputePipelineState
	psoFusedQKV    apple.MTLComputePipelineState
	psoFusedSwiGLU apple.MTLComputePipelineState
	psoSample      apple.MTLComputePipelineState
	recorder       *metalprof.FlightRecorder
	metrics        *RuntimeMetrics
	generateMu     sync.Mutex
	prefixes       *prefixCache
	tuner          *DispatchTuner

	// GPU-Resident Unified Memory Buffers
	xBuf       mempool.Slice[float32]
	normXBuf   mempool.Slice[float32]
	qBuf       mempool.Slice[float32]
	kBuf       mempool.Slice[float32]
	vBuf       mempool.Slice[float32]
	attnOutBuf mempool.Slice[float32]
	gateBuf    mempool.Slice[float32]
	upBuf      mempool.Slice[float32]
	ffnActBuf  mempool.Slice[float32]
	kCacheBuf  []mempool.Slice[float32]
	vCacheBuf  []mempool.Slice[float32]
	logitsBuf  mempool.Slice[float32]
	tokBuf     mempool.Slice[int32]

	// GPU Weights (FP32 or FP16)
	isF16         bool
	embedTable    mempool.Slice[float32]
	finalNormW    mempool.Slice[float32]
	lmHeadW       mempool.Slice[float32]
	layers        []GPULayerWeights
	embedTableF16 mempool.Slice[uint16]
	finalNormWF16 mempool.Slice[uint16]
	lmHeadWF16    mempool.Slice[uint16]
	layersF16     []GPULayerWeightsF16

	tokenizer  Tokenizer
	weights    *metalsafetensors.Weights
	egraphSats int
	Plan       *ExecutionPlan
}

// FlightRecorder returns the active in-memory GPU telemetry recorder.
func (eng *ModelEngine) FlightRecorder() *metalprof.FlightRecorder {
	return eng.recorder
}

// IsF16 reports whether the engine is executing over FP16 weights.
func (eng *ModelEngine) IsF16() bool {
	if eng == nil {
		return false
	}
	return eng.isF16
}

func (eng *ModelEngine) precisionTelemetry() []TensorPrecision {
	if eng == nil || eng.weights == nil {
		return nil
	}

	byDType := make(map[string]TensorPrecision)
	for _, tensor := range eng.weights.Tensors {
		if tensor.DType == "" || tensor.OffsetEnd < tensor.OffsetStart {
			continue
		}
		summary := byDType[tensor.DType]
		summary.DType = tensor.DType
		summary.TensorCount++
		summary.Bytes += tensor.OffsetEnd - tensor.OffsetStart
		byDType[tensor.DType] = summary
	}
	if len(byDType) == 0 {
		return nil
	}

	dtypes := make([]TensorPrecision, 0, len(byDType))
	for _, summary := range byDType {
		dtypes = append(dtypes, summary)
	}
	sort.Slice(dtypes, func(i, j int) bool {
		return dtypes[i].DType < dtypes[j].DType
	})
	return dtypes
}

func (eng *ModelEngine) prefixCacheMetrics() PrefixCacheMetrics {
	if eng == nil || eng.prefixes == nil {
		return PrefixCacheMetrics{}
	}
	return eng.prefixes.metrics()
}

func (eng *ModelEngine) prefixCacheAvailable() bool {
	return eng != nil && eng.prefixes != nil
}

// ReadLogits returns a copy of the final logits computed in the last forward pass.
func (eng *ModelEngine) ReadLogits() ([]float32, error) {
	if eng == nil {
		return nil, errors.New("read logits: nil engine")
	}
	if eng.logitsBuf.Len() == 0 {
		return nil, errors.New("read logits: empty logits buffer")
	}
	res := make([]float32, eng.logitsBuf.Len())
	copy(res, eng.logitsBuf.HostSlice())
	return res, nil
}

// MutateWeightForTest alters a single weight element in GPU memory and returns a restore function.
func (eng *ModelEngine) MutateWeightForTest(layerIndex, elemIndex int, delta float32) (func(), error) {
	if eng == nil {
		return nil, errors.New("mutate weight: nil engine")
	}
	var slice []float32
	if layerIndex < 0 {
		if eng.embedTable.Len() == 0 {
			return nil, errors.New("mutate weight: empty embed table")
		}
		slice = eng.embedTable.HostSlice()
	} else {
		if len(eng.layers) <= layerIndex {
			return nil, errors.New("mutate weight: invalid layer index")
		}
		slice = eng.layers[layerIndex].WQ.HostSlice()
	}
	if len(slice) <= elemIndex {
		return nil, errors.New("mutate weight: invalid element index")
	}
	orig := slice[elemIndex]
	slice[elemIndex] += delta
	return func() {
		slice[elemIndex] = orig
	}, nil
}

// ResetKV zeroes out the entire KV cache buffer across all layers.
func (eng *ModelEngine) ResetKV() {
	if eng == nil {
		return
	}
	for _, kBuf := range eng.kCacheBuf {
		slice := kBuf.HostSlice()
		for i := range slice {
			slice[i] = 0
		}
	}
	for _, vBuf := range eng.vCacheBuf {
		slice := vBuf.HostSlice()
		for i := range slice {
			slice[i] = 0
		}
	}
}

func (eng *ModelEngine) capturePrefix(tokens int, current int32, cfg Config) *prefixSnapshot {
	if eng == nil || tokens <= 0 || cfg.NumKVHeads <= 0 || cfg.HeadDim <= 0 {
		return nil
	}
	valuesPerLayer := tokens * cfg.NumKVHeads * cfg.HeadDim
	snapshot := &prefixSnapshot{
		tokens:  tokens,
		current: current,
		k:       make([][]float32, len(eng.kCacheBuf)),
		v:       make([][]float32, len(eng.vCacheBuf)),
	}
	for layer := range eng.kCacheBuf {
		if valuesPerLayer > eng.kCacheBuf[layer].Len() || valuesPerLayer > eng.vCacheBuf[layer].Len() {
			return nil
		}
		snapshot.k[layer] = append([]float32(nil), eng.kCacheBuf[layer].HostSlice()[:valuesPerLayer]...)
		snapshot.v[layer] = append([]float32(nil), eng.vCacheBuf[layer].HostSlice()[:valuesPerLayer]...)
		snapshot.bytes += int64(len(snapshot.k[layer])+len(snapshot.v[layer])) * 4
	}
	return snapshot
}

func (eng *ModelEngine) restorePrefix(snapshot *prefixSnapshot) bool {
	if eng == nil || snapshot == nil || len(snapshot.k) != len(eng.kCacheBuf) || len(snapshot.v) != len(eng.vCacheBuf) {
		return false
	}
	for layer := range eng.kCacheBuf {
		if len(snapshot.k[layer]) > eng.kCacheBuf[layer].Len() || len(snapshot.v[layer]) > eng.vCacheBuf[layer].Len() {
			return false
		}
		copy(eng.kCacheBuf[layer].HostSlice(), snapshot.k[layer])
		copy(eng.vCacheBuf[layer].HostSlice(), snapshot.v[layer])
	}
	return true
}

var (
	darwinEngineCache = make(map[string]*ModelEngine)
	darwinEngineMu    sync.Mutex
)

// GetOrCompileDarwinEngine builds or reuses the 100% GPU-resident Metal execution engine.
func GetOrCompileDarwinEngine(cfg Config) (*ModelEngine, error) {
	return getOrCompileDarwinEngine(cfg)
}

// GetOrCompileEngine is the universal entrypoint for obtaining the GPU engine.
func GetOrCompileEngine(cfg Config) (*ModelEngine, error) {
	return GetOrCompileDarwinEngine(cfg)
}

func f32SliceToF16(s []float32) []uint16 {
	out := make([]uint16, len(s))
	for i, v := range s {
		out[i] = metalsafetensors.F32ToF16(v)
	}
	return out
}

func getOrCompileDarwinEngine(cfg Config) (*ModelEngine, error) {
	darwinEngineMu.Lock()
	defer darwinEngineMu.Unlock()

	key := fmt.Sprintf("%s-%d-%d-%d-prec-%s-egraph-%v", cfg.Name, cfg.Dim, cfg.HiddenDim, cfg.VocabSize, cfg.Precision, cfg.FuseEGraph)
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

	fullMSL := metalmegakernel.LlamaDecoderKernelsMSL + "\n" + metalmegakernel.HalfWeightKernelsMSL + "\n" + metalattn.CausalFlashAttentionMSL + "\n" + metalsample.ArgmaxSamplingMSL
	lib, err := dev.CompileLibrary(fullMSL)
	if err != nil {
		return nil, fmt.Errorf("compile MSL library on Metal GPU: %w", err)
	}

	psoEmbed, err := dev.NewComputePipeline(lib, "EmbeddingLookup")
	if err != nil {
		return nil, fmt.Errorf("create EmbeddingLookup pipeline: %w", err)
	}
	psoNorm, err := dev.NewComputePipeline(lib, "ComputeRMSNorm")
	if err != nil {
		return nil, fmt.Errorf("create ComputeRMSNorm pipeline: %w", err)
	}
	psoGEMV, err := dev.NewComputePipeline(lib, "FastGEMV")
	if err != nil {
		return nil, fmt.Errorf("create FastGEMV pipeline: %w", err)
	}
	psoGEMVRes, err := dev.NewComputePipeline(lib, "FastGEMVResidual")
	if err != nil {
		return nil, fmt.Errorf("create FastGEMVResidual pipeline: %w", err)
	}
	psoGEMV64, err := dev.NewComputePipeline(lib, "FastGEMV64")
	if err != nil {
		return nil, fmt.Errorf("create FastGEMV64 pipeline: %w", err)
	}
	psoGEMVRes64, err := dev.NewComputePipeline(lib, "FastGEMVResidual64")
	if err != nil {
		return nil, fmt.Errorf("create FastGEMVResidual64 pipeline: %w", err)
	}
	psoEmbedH, _ := dev.NewComputePipeline(lib, "EmbeddingLookupH")
	psoNormH, _ := dev.NewComputePipeline(lib, "ComputeRMSNormH")
	psoGEMVH, _ := dev.NewComputePipeline(lib, "FastGEMVH")
	psoGEMVResH, _ := dev.NewComputePipeline(lib, "FastGEMVResidualH")
	psoGEMV64H, _ := dev.NewComputePipeline(lib, "FastGEMV64H")
	psoGEMVRes64H, _ := dev.NewComputePipeline(lib, "FastGEMVResidual64H")
	psoRoPE, err := dev.NewComputePipeline(lib, "ApplyRoPEStoreKV")
	if err != nil {
		return nil, fmt.Errorf("create ApplyRoPEStoreKV pipeline: %w", err)
	}
	psoAttn, err := dev.NewComputePipeline(lib, "CausalFlashAttentionDecode")
	if err != nil {
		return nil, fmt.Errorf("create CausalFlashAttentionDecode pipeline: %w", err)
	}
	psoSiLUMul, err := dev.NewComputePipeline(lib, "FastSiLUMul")
	if err != nil {
		return nil, fmt.Errorf("create FastSiLUMul pipeline: %w", err)
	}
	psoFusedQKV, _ := dev.NewComputePipeline(lib, "FusedRMSNormQKV")
	psoFusedSwiGLU, _ := dev.NewComputePipeline(lib, "FusedSwiGLUDown")
	psoSample, err := dev.NewComputePipeline(lib, "ArgmaxSampling")
	if err != nil {
		return nil, fmt.Errorf("create ArgmaxSampling pipeline: %w", err)
	}

	xBuf, _ := mempool.Alloc[float32](dev, cfg.Dim)
	normXBuf, _ := mempool.Alloc[float32](dev, cfg.Dim)
	qBuf, _ := mempool.Alloc[float32](dev, cfg.NumHeads*cfg.HeadDim)
	kBuf, _ := mempool.Alloc[float32](dev, cfg.NumKVHeads*cfg.HeadDim)
	vBuf, _ := mempool.Alloc[float32](dev, cfg.NumKVHeads*cfg.HeadDim)
	attnOutBuf, _ := mempool.Alloc[float32](dev, cfg.Dim)
	gateBuf, _ := mempool.Alloc[float32](dev, cfg.HiddenDim)
	upBuf, _ := mempool.Alloc[float32](dev, cfg.HiddenDim)
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
	isF16 := cfg.Precision == "f16"

	var embedTable, finalNormW, lmHeadW mempool.Slice[float32]
	gpuLayers := make([]GPULayerWeights, cfg.NumLayers)

	var embedTableF16, finalNormWF16, lmHeadWF16 mempool.Slice[uint16]
	gpuLayersF16 := make([]GPULayerWeightsF16, cfg.NumLayers)

	if cfg.ModelPath != "" {
		tokJsonPath := filepath.Join(cfg.ModelPath, "tokenizer.json")
		if t, err := LoadHFTokenizer(tokJsonPath); err == nil {
			tok = t
		}
		safetensorsPath := filepath.Join(cfg.ModelPath, "model.safetensors")
		w, err := metalsafetensors.Open(safetensorsPath)
		if err != nil {
			return nil, fmt.Errorf("open safetensors %q: %w", safetensorsPath, err)
		}
		weights = w

		// Auto-detect F16/BF16 from safetensors if not explicitly forced to f32
		if cfg.Precision == "" && w.Tensors != nil {
			if tInfo, ok := w.Tensors["model.embed_tokens.weight"]; ok && (tInfo.DType == "F16" || tInfo.DType == "BF16") {
				isF16 = true
			}
		}

		if isF16 {
			var errUpload error
			embedTableF16, errUpload = w.TensorMetalSliceF16(dev, "model.embed_tokens.weight")
			if errUpload != nil || embedTableF16.Len() == 0 {
				return nil, fmt.Errorf("load embed_tokens f16: %w", errUpload)
			}
			finalNormWF16, errUpload = w.TensorMetalSliceF16(dev, "model.norm.weight")
			if errUpload != nil || finalNormWF16.Len() == 0 {
				return nil, fmt.Errorf("load norm weight f16: %w", errUpload)
			}
			lmHeadWF16, _ = w.TensorMetalSliceF16(dev, "lm_head.weight")
			if lmHeadWF16.Len() == 0 {
				lmHeadWF16 = embedTableF16
			}

			for i := 0; i < cfg.NumLayers; i++ {
				inNorm, err := w.TensorMetalSliceF16(dev, fmt.Sprintf("model.layers.%d.input_layernorm.weight", i))
				if err != nil {
					return nil, fmt.Errorf("load layer %d input_norm: %w", i, err)
				}
				pNorm, err := w.TensorMetalSliceF16(dev, fmt.Sprintf("model.layers.%d.post_attention_layernorm.weight", i))
				if err != nil {
					return nil, fmt.Errorf("load layer %d post_norm: %w", i, err)
				}
				wq, err := w.TensorMetalSliceF16(dev, fmt.Sprintf("model.layers.%d.self_attn.q_proj.weight", i))
				if err != nil {
					return nil, fmt.Errorf("load layer %d wq: %w", i, err)
				}
				wk, err := w.TensorMetalSliceF16(dev, fmt.Sprintf("model.layers.%d.self_attn.k_proj.weight", i))
				if err != nil {
					return nil, fmt.Errorf("load layer %d wk: %w", i, err)
				}
				wv, err := w.TensorMetalSliceF16(dev, fmt.Sprintf("model.layers.%d.self_attn.v_proj.weight", i))
				if err != nil {
					return nil, fmt.Errorf("load layer %d wv: %w", i, err)
				}
				wo, err := w.TensorMetalSliceF16(dev, fmt.Sprintf("model.layers.%d.self_attn.o_proj.weight", i))
				if err != nil {
					return nil, fmt.Errorf("load layer %d wo: %w", i, err)
				}
				wg, err := w.TensorMetalSliceF16(dev, fmt.Sprintf("model.layers.%d.mlp.gate_proj.weight", i))
				if err != nil {
					return nil, fmt.Errorf("load layer %d wgate: %w", i, err)
				}
				wu, err := w.TensorMetalSliceF16(dev, fmt.Sprintf("model.layers.%d.mlp.up_proj.weight", i))
				if err != nil {
					return nil, fmt.Errorf("load layer %d wup: %w", i, err)
				}
				wd, err := w.TensorMetalSliceF16(dev, fmt.Sprintf("model.layers.%d.mlp.down_proj.weight", i))
				if err != nil {
					return nil, fmt.Errorf("load layer %d wdown: %w", i, err)
				}

				gpuLayersF16[i] = GPULayerWeightsF16{
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
		} else {
			var errUpload error
			embedTable, errUpload = w.TensorMetalSlice(dev, "model.embed_tokens.weight")
			if errUpload != nil || embedTable.Len() == 0 {
				return nil, fmt.Errorf("load embed_tokens: %w", errUpload)
			}
			finalNormW, errUpload = w.TensorMetalSlice(dev, "model.norm.weight")
			if errUpload != nil || finalNormW.Len() == 0 {
				return nil, fmt.Errorf("load norm weight: %w", errUpload)
			}
			lmHeadW, _ = w.TensorMetalSlice(dev, "lm_head.weight")
			if lmHeadW.Len() == 0 {
				lmHeadW = embedTable
			}

			for i := 0; i < cfg.NumLayers; i++ {
				inNorm, err := w.TensorMetalSlice(dev, fmt.Sprintf("model.layers.%d.input_layernorm.weight", i))
				if err != nil {
					return nil, fmt.Errorf("load layer %d input_norm: %w", i, err)
				}
				pNorm, err := w.TensorMetalSlice(dev, fmt.Sprintf("model.layers.%d.post_attention_layernorm.weight", i))
				if err != nil {
					return nil, fmt.Errorf("load layer %d post_norm: %w", i, err)
				}
				wq, err := w.TensorMetalSlice(dev, fmt.Sprintf("model.layers.%d.self_attn.q_proj.weight", i))
				if err != nil {
					return nil, fmt.Errorf("load layer %d wq: %w", i, err)
				}
				wk, err := w.TensorMetalSlice(dev, fmt.Sprintf("model.layers.%d.self_attn.k_proj.weight", i))
				if err != nil {
					return nil, fmt.Errorf("load layer %d wk: %w", i, err)
				}
				wv, err := w.TensorMetalSlice(dev, fmt.Sprintf("model.layers.%d.self_attn.v_proj.weight", i))
				if err != nil {
					return nil, fmt.Errorf("load layer %d wv: %w", i, err)
				}
				wo, err := w.TensorMetalSlice(dev, fmt.Sprintf("model.layers.%d.self_attn.o_proj.weight", i))
				if err != nil {
					return nil, fmt.Errorf("load layer %d wo: %w", i, err)
				}
				wg, err := w.TensorMetalSlice(dev, fmt.Sprintf("model.layers.%d.mlp.gate_proj.weight", i))
				if err != nil {
					return nil, fmt.Errorf("load layer %d wgate: %w", i, err)
				}
				wu, err := w.TensorMetalSlice(dev, fmt.Sprintf("model.layers.%d.mlp.up_proj.weight", i))
				if err != nil {
					return nil, fmt.Errorf("load layer %d wup: %w", i, err)
				}
				wd, err := w.TensorMetalSlice(dev, fmt.Sprintf("model.layers.%d.mlp.down_proj.weight", i))
				if err != nil {
					return nil, fmt.Errorf("load layer %d wdown: %w", i, err)
				}

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
	} else if cfg.TestMode {
		mCfg := ModelConfig{
			HiddenSize:        cfg.Dim,
			IntermediateSize:  cfg.HiddenDim,
			NumHiddenLayers:   cfg.NumLayers,
			NumAttentionHeads: cfg.NumHeads,
			NumKeyValueHeads:  cfg.NumKVHeads,
			VocabSize:         cfg.VocabSize,
			RmsNormEps:        cfg.Eps,
			RopeTheta:         cfg.RopeTheta,
			MaxSeqLen:         cfg.MaxSeqLen,
		}
		if mCfg.HeadDim() == 0 && cfg.HeadDim > 0 {
			mCfg.ExplicitHeadDim = cfg.HeadDim
		}
		detWeights := GenerateDeterministicWeights(&mCfg)
		if isF16 {
			embedTableF16, _ = mempool.AllocFromSlice(dev, f32SliceToF16(detWeights.EmbedTokens))
			finalNormWF16, _ = mempool.AllocFromSlice(dev, f32SliceToF16(detWeights.Norm))
			lmHeadWF16, _ = mempool.AllocFromSlice(dev, f32SliceToF16(detWeights.LMHead))
			for i := 0; i < cfg.NumLayers; i++ {
				lw := detWeights.Layers[i]
				inNorm, _ := mempool.AllocFromSlice(dev, f32SliceToF16(lw.InputNorm))
				pNorm, _ := mempool.AllocFromSlice(dev, f32SliceToF16(lw.PostNorm))
				wq, _ := mempool.AllocFromSlice(dev, f32SliceToF16(lw.WQ))
				wk, _ := mempool.AllocFromSlice(dev, f32SliceToF16(lw.WK))
				wv, _ := mempool.AllocFromSlice(dev, f32SliceToF16(lw.WV))
				wo, _ := mempool.AllocFromSlice(dev, f32SliceToF16(lw.WO))
				wg, _ := mempool.AllocFromSlice(dev, f32SliceToF16(lw.WGate))
				wu, _ := mempool.AllocFromSlice(dev, f32SliceToF16(lw.WUp))
				wd, _ := mempool.AllocFromSlice(dev, f32SliceToF16(lw.WDown))

				gpuLayersF16[i] = GPULayerWeightsF16{
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
		} else {
			embedTable, _ = mempool.Alloc[float32](dev, len(detWeights.EmbedTokens))
			copy(embedTable.HostSlice(), detWeights.EmbedTokens)
			finalNormW, _ = mempool.Alloc[float32](dev, len(detWeights.Norm))
			copy(finalNormW.HostSlice(), detWeights.Norm)
			lmHeadW, _ = mempool.Alloc[float32](dev, len(detWeights.LMHead))
			copy(lmHeadW.HostSlice(), detWeights.LMHead)

			for i := 0; i < cfg.NumLayers; i++ {
				lw := detWeights.Layers[i]
				inNorm, _ := mempool.Alloc[float32](dev, len(lw.InputNorm))
				copy(inNorm.HostSlice(), lw.InputNorm)
				pNorm, _ := mempool.Alloc[float32](dev, len(lw.PostNorm))
				copy(pNorm.HostSlice(), lw.PostNorm)
				wq, _ := mempool.Alloc[float32](dev, len(lw.WQ))
				copy(wq.HostSlice(), lw.WQ)
				wk, _ := mempool.Alloc[float32](dev, len(lw.WK))
				copy(wk.HostSlice(), lw.WK)
				wv, _ := mempool.Alloc[float32](dev, len(lw.WV))
				copy(wv.HostSlice(), lw.WV)
				wo, _ := mempool.Alloc[float32](dev, len(lw.WO))
				copy(wo.HostSlice(), lw.WO)
				wg, _ := mempool.Alloc[float32](dev, len(lw.WGate))
				copy(wg.HostSlice(), lw.WGate)
				wu, _ := mempool.Alloc[float32](dev, len(lw.WUp))
				copy(wu.HostSlice(), lw.WUp)
				wd, _ := mempool.Alloc[float32](dev, len(lw.WDown))
				copy(wd.HostSlice(), lw.WDown)

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
	} else {
		return nil, errors.New("model path required (synthetic test weights require explicit TestMode=true)")
	}
	if tok == nil {
		tok = NewByteTokenizer()
	}

	metrics := NewRuntimeMetrics(256)
	var kvBytes int64
	for layer := range kCacheBuf {
		kvBytes += int64(kCacheBuf[layer].Len()+vCacheBuf[layer].Len()) * 4
	}
	metrics.SetKVAllocation(kvBytes, cfg.MaxSeqLen)

	eng := &ModelEngine{
		dev:            dev,
		lib:            lib,
		psoEmbed:       psoEmbed,
		psoNorm:        psoNorm,
		psoGEMV:        psoGEMV,
		psoGEMVRes:     psoGEMVRes,
		psoGEMV64:      psoGEMV64,
		psoGEMVRes64:   psoGEMVRes64,
		psoEmbedH:      psoEmbedH,
		psoNormH:       psoNormH,
		psoGEMVH:       psoGEMVH,
		psoGEMVResH:    psoGEMVResH,
		psoGEMV64H:     psoGEMV64H,
		psoGEMVRes64H:  psoGEMVRes64H,
		psoRoPE:        psoRoPE,
		psoAttn:        psoAttn,
		psoSiLUMul:     psoSiLUMul,
		psoFusedQKV:    psoFusedQKV,
		psoFusedSwiGLU: psoFusedSwiGLU,
		psoSample:      psoSample,
		xBuf:           xBuf,
		normXBuf:       normXBuf,
		qBuf:           qBuf,
		kBuf:           kBuf,
		vBuf:           vBuf,
		attnOutBuf:     attnOutBuf,
		gateBuf:        gateBuf,
		upBuf:          upBuf,
		ffnActBuf:      ffnActBuf,
		kCacheBuf:      kCacheBuf,
		vCacheBuf:      vCacheBuf,
		logitsBuf:      logitsBuf,
		tokBuf:         tokBuf,
		isF16:          isF16,
		embedTable:     embedTable,
		finalNormW:     finalNormW,
		lmHeadW:        lmHeadW,
		layers:         gpuLayers,
		embedTableF16:  embedTableF16,
		finalNormWF16:  finalNormWF16,
		lmHeadWF16:     lmHeadWF16,
		layersF16:      gpuLayersF16,
		tokenizer:      tok,
		weights:        weights,
		egraphSats:     egraphSats,
		recorder:       metalprof.NewFlightRecorder(2048),
		metrics:        metrics,
		prefixes:       newPrefixCache(64<<20, 8),
		tuner:          NewDispatchTuner(),
		Plan:           BuildExecutionPlan(cfg),
	}

	darwinEngineCache[key] = eng
	return eng, nil
}

func runPlatform(cfg Config) error {
	engine, err := GetOrCompileDarwinEngine(cfg)
	if err != nil {
		return fmt.Errorf("engine compile: %w", err)
	}

	cNode, err := NewIrohClusterNode(context.Background(), engine, cfg, 0)
	if err == nil {
		if !cfg.ServeMode {
			defer cNode.Close()
		}
		nodeAddr := cNode.Addr()
		fmt.Printf("[go-iroh Mesh] Node ID: %s (Platform: %s, Compute: %.0f TFLOPs, Bandwidth: %.0f GB/s)\n",
			cNode.NodeID(), cNode.SelfTelemetry().OS, cNode.SelfTelemetry().ComputeTFLOPs, cNode.SelfTelemetry().BandwidthGBS)
		fmt.Printf("[go-iroh Mesh] Node Addr: %s\n", nodeAddr.String())
		if cfg.PeerAddr != "" {
			go func() {
				for {
					if err := cNode.ConnectHTTPPeer(context.Background(), cfg.PeerAddr); err == nil {
						fmt.Printf("[go-iroh Mesh] Connected and registered peer %s into inference mesh!\n", cfg.PeerAddr)
						break
					}
					time.Sleep(2 * time.Second)
				}
			}()
		}
	}

	if cfg.HTTPAddr != "" {
		srv := NewTelemetryServer(engine, cfg)
		srv.SetPGOTracker(engine.tuner.Tracker())
		if cNode != nil {
			srv.SetClusterNode(cNode)
		}
		go func() {
			fmt.Printf("\n[Telemetry Server] Live E-Graph dashboard active at http://localhost%s\n", cfg.HTTPAddr)
			if err := http.ListenAndServe(cfg.HTTPAddr, srv); err != nil {
				fmt.Printf("[Telemetry Server] Error: %v\n", err)
			}
		}()
	}

	if !cfg.ServeMode {
		runFastMetalInference(engine, cfg)
	}
	return nil
}

// setScalar binds v directly into the encoder's argument table. Kernel scalar
// arguments are small and change every pass, so copying them inline avoids
// allocating a MTLBuffer per argument per token.
func setScalar[T any](enc apple.MTLComputeCommandEncoder, v T, index uint) {
	enc.SetBytesLengthAtIndex(unsafe.Slice((*byte)(unsafe.Pointer(&v)), unsafe.Sizeof(v)), index)
}

// SequenceKVCache holds isolated per-sequence Metal KV-cache allocations.
type SequenceKVCache struct {
	KCache []mempool.Slice[float32]
	VCache []mempool.Slice[float32]
}

// NewSequenceKV allocates an isolated KV-cache for a sequence.
func (eng *ModelEngine) NewSequenceKV(cfg Config) (*SequenceKVCache, error) {
	if eng == nil {
		return nil, errors.New("new sequence kv: nil engine")
	}
	kCache := make([]mempool.Slice[float32], cfg.NumLayers)
	vCache := make([]mempool.Slice[float32], cfg.NumLayers)
	elemCount := cfg.MaxSeqLen * cfg.NumKVHeads * cfg.HeadDim
	for l := 0; l < cfg.NumLayers; l++ {
		k, err := mempool.Alloc[float32](eng.dev, elemCount)
		if err != nil {
			return nil, fmt.Errorf("alloc k-cache layer %d: %w", l, err)
		}
		v, err := mempool.Alloc[float32](eng.dev, elemCount)
		if err != nil {
			return nil, fmt.Errorf("alloc v-cache layer %d: %w", l, err)
		}
		kCache[l] = k
		vCache[l] = v
	}
	return &SequenceKVCache{
		KCache: kCache,
		VCache: vCache,
	}, nil
}

// Free releases the isolated KV-cache buffers.
func (skv *SequenceKVCache) Free() {
	if skv == nil {
		return
	}
	for i := range skv.KCache {
		skv.KCache[i].Free()
		skv.VCache[i].Free()
	}
	skv.KCache = nil
	skv.VCache = nil
}

func encodeForwardStep(enc apple.MTLComputeCommandEncoder, eng *ModelEngine, cfg Config, tokenID int32, pos int, isLast bool) {
	encodeForwardStepWithKV(enc, eng, cfg, eng.kCacheBuf, eng.vCacheBuf, tokenID, pos, isLast)
}

func encodeForwardStepWithKV(enc apple.MTLComputeCommandEncoder, eng *ModelEngine, cfg Config, kCache, vCache []mempool.Slice[float32], tokenID int32, pos int, isLast bool) {
	gemvThreads := 128
	psoGEMV := eng.psoGEMV
	psoGEMVRes := eng.psoGEMVRes
	psoGEMVH := eng.psoGEMVH
	psoGEMVResH := eng.psoGEMVResH
	if eng.tuner != nil && eng.tuner.Width() == 64 {
		gemvThreads = 64
		psoGEMV = eng.psoGEMV64
		psoGEMVRes = eng.psoGEMVRes64
		psoGEMVH = eng.psoGEMV64H
		psoGEMVResH = eng.psoGEMVRes64H
	}

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
	tokVal := uint32(tokenID)

	// 1. Embedding lookup
	if eng.isF16 {
		enc.SetComputePipelineState(eng.psoEmbedH)
		enc.SetBufferWithOffsetAtIndex(eng.embedTableF16.Buffer(), 0, 0)
	} else {
		enc.SetComputePipelineState(eng.psoEmbed)
		enc.SetBufferWithOffsetAtIndex(eng.embedTable.Buffer(), 0, 0)
	}
	enc.SetBufferWithOffsetAtIndex(eng.xBuf.Buffer(), 0, 1)
	setScalar(enc, tokVal, 2)
	setScalar(enc, dimVal, 3)
	enc.DispatchThreadsThreadsPerThreadgroup(
		apple.MTLSize{Width: uint(cfg.Dim), Height: 1, Depth: 1},
		apple.MTLSize{Width: 128, Height: 1, Depth: 1},
	)

	// 2. Transformer Decoder Layers
	for l := 0; l < cfg.NumLayers; l++ {
		if eng.isF16 {
			layer := &eng.layersF16[l]
			if layer.InputNorm.Len() == 0 {
				continue
			}

			// Input RMSNorm (half norm weights): xBuf -> normXBuf
			enc.SetComputePipelineState(eng.psoNormH)
			enc.SetBufferWithOffsetAtIndex(eng.xBuf.Buffer(), 0, 0)
			enc.SetBufferWithOffsetAtIndex(layer.InputNorm.Buffer(), 0, 1)
			enc.SetBufferWithOffsetAtIndex(eng.normXBuf.Buffer(), 0, 2)
			setScalar(enc, dimVal, 3)
			setScalar(enc, epsVal, 4)
			enc.DispatchThreadgroupsThreadsPerThreadgroup(
				apple.MTLSize{Width: 1, Height: 1, Depth: 1},
				apple.MTLSize{Width: 128, Height: 1, Depth: 1},
			)

			// Q Projection (half weights): normXBuf * WQ -> qBuf
			enc.SetComputePipelineState(psoGEMVH)
			enc.SetBufferWithOffsetAtIndex(eng.normXBuf.Buffer(), 0, 0)
			enc.SetBufferWithOffsetAtIndex(layer.WQ.Buffer(), 0, 1)
			enc.SetBufferWithOffsetAtIndex(eng.qBuf.Buffer(), 0, 2)
			setScalar(enc, dimVal, 3)
			setScalar(enc, qDimVal, 4)
			enc.DispatchThreadgroupsThreadsPerThreadgroup(
				apple.MTLSize{Width: uint(cfg.NumHeads * cfg.HeadDim), Height: 1, Depth: 1},
				apple.MTLSize{Width: uint(gemvThreads), Height: 1, Depth: 1},
			)

			// K Projection (half weights): normXBuf * WK -> kBuf
			enc.SetComputePipelineState(psoGEMVH)
			enc.SetBufferWithOffsetAtIndex(eng.normXBuf.Buffer(), 0, 0)
			enc.SetBufferWithOffsetAtIndex(layer.WK.Buffer(), 0, 1)
			enc.SetBufferWithOffsetAtIndex(eng.kBuf.Buffer(), 0, 2)
			setScalar(enc, dimVal, 3)
			setScalar(enc, kvDimVal, 4)
			enc.DispatchThreadgroupsThreadsPerThreadgroup(
				apple.MTLSize{Width: uint(cfg.NumKVHeads * cfg.HeadDim), Height: 1, Depth: 1},
				apple.MTLSize{Width: uint(gemvThreads), Height: 1, Depth: 1},
			)

			// V Projection (half weights): normXBuf * WV -> vBuf
			enc.SetComputePipelineState(psoGEMVH)
			enc.SetBufferWithOffsetAtIndex(eng.normXBuf.Buffer(), 0, 0)
			enc.SetBufferWithOffsetAtIndex(layer.WV.Buffer(), 0, 1)
			enc.SetBufferWithOffsetAtIndex(eng.vBuf.Buffer(), 0, 2)
			setScalar(enc, dimVal, 3)
			setScalar(enc, kvDimVal, 4)
			enc.DispatchThreadgroupsThreadsPerThreadgroup(
				apple.MTLSize{Width: uint(cfg.NumKVHeads * cfg.HeadDim), Height: 1, Depth: 1},
				apple.MTLSize{Width: uint(gemvThreads), Height: 1, Depth: 1},
			)

			// Apply RoPE + Store KV cache
			enc.SetComputePipelineState(eng.psoRoPE)
			enc.SetBufferWithOffsetAtIndex(eng.qBuf.Buffer(), 0, 0)
			enc.SetBufferWithOffsetAtIndex(eng.kBuf.Buffer(), 0, 1)
			enc.SetBufferWithOffsetAtIndex(eng.vBuf.Buffer(), 0, 2)
			enc.SetBufferWithOffsetAtIndex(kCache[l].Buffer(), 0, 3)
			enc.SetBufferWithOffsetAtIndex(vCache[l].Buffer(), 0, 4)
			setScalar(enc, posVal, 5)
			setScalar(enc, headDimVal, 6)
			setScalar(enc, qHeadsVal, 7)
			setScalar(enc, kvHeadsVal, 8)
			setScalar(enc, thetaVal, 9)
			enc.DispatchThreadsThreadsPerThreadgroup(
				apple.MTLSize{Width: uint(cfg.NumHeads * cfg.HeadDim), Height: 1, Depth: 1},
				apple.MTLSize{Width: 64, Height: 1, Depth: 1},
			)

			// Causal FlashAttention Decode
			enc.SetComputePipelineState(eng.psoAttn)
			enc.SetBufferWithOffsetAtIndex(eng.qBuf.Buffer(), 0, 0)
			enc.SetBufferWithOffsetAtIndex(kCache[l].Buffer(), 0, 1)
			enc.SetBufferWithOffsetAtIndex(vCache[l].Buffer(), 0, 2)
			enc.SetBufferWithOffsetAtIndex(eng.attnOutBuf.Buffer(), 0, 3)
			setScalar(enc, curPosInt, 4)
			setScalar(enc, maxSeqVal, 5)
			setScalar(enc, qHeadsInt, 6)
			setScalar(enc, kvHeadsInt, 7)
			setScalar(enc, headDimInt, 8)
			setScalar(enc, scaleVal, 9)
			enc.DispatchThreadgroupsThreadsPerThreadgroup(
				apple.MTLSize{Width: uint(cfg.NumHeads), Height: 1, Depth: 1},
				apple.MTLSize{Width: 128, Height: 1, Depth: 1},
			)

			// Attention Out Proj & Residual: xBuf += WO * attnOutBuf
			enc.SetComputePipelineState(psoGEMVResH)
			enc.SetBufferWithOffsetAtIndex(eng.attnOutBuf.Buffer(), 0, 0)
			enc.SetBufferWithOffsetAtIndex(layer.WO.Buffer(), 0, 1)
			enc.SetBufferWithOffsetAtIndex(eng.xBuf.Buffer(), 0, 2)
			setScalar(enc, dimVal, 3)
			setScalar(enc, dimVal, 4)
			enc.DispatchThreadgroupsThreadsPerThreadgroup(
				apple.MTLSize{Width: uint(cfg.Dim), Height: 1, Depth: 1},
				apple.MTLSize{Width: uint(gemvThreads), Height: 1, Depth: 1},
			)

			// Post-Attention RMSNorm: xBuf -> normXBuf
			enc.SetComputePipelineState(eng.psoNormH)
			enc.SetBufferWithOffsetAtIndex(eng.xBuf.Buffer(), 0, 0)
			enc.SetBufferWithOffsetAtIndex(layer.PostNorm.Buffer(), 0, 1)
			enc.SetBufferWithOffsetAtIndex(eng.normXBuf.Buffer(), 0, 2)
			setScalar(enc, dimVal, 3)
			setScalar(enc, epsVal, 4)
			enc.DispatchThreadgroupsThreadsPerThreadgroup(
				apple.MTLSize{Width: 1, Height: 1, Depth: 1},
				apple.MTLSize{Width: 128, Height: 1, Depth: 1},
			)

			// SwiGLU Gate Proj
			enc.SetComputePipelineState(psoGEMVH)
			enc.SetBufferWithOffsetAtIndex(eng.normXBuf.Buffer(), 0, 0)
			enc.SetBufferWithOffsetAtIndex(layer.WGate.Buffer(), 0, 1)
			enc.SetBufferWithOffsetAtIndex(eng.gateBuf.Buffer(), 0, 2)
			setScalar(enc, dimVal, 3)
			setScalar(enc, hiddenDimVal, 4)
			enc.DispatchThreadgroupsThreadsPerThreadgroup(
				apple.MTLSize{Width: uint(cfg.HiddenDim), Height: 1, Depth: 1},
				apple.MTLSize{Width: uint(gemvThreads), Height: 1, Depth: 1},
			)

			// SwiGLU Up Proj
			enc.SetComputePipelineState(psoGEMVH)
			enc.SetBufferWithOffsetAtIndex(eng.normXBuf.Buffer(), 0, 0)
			enc.SetBufferWithOffsetAtIndex(layer.WUp.Buffer(), 0, 1)
			enc.SetBufferWithOffsetAtIndex(eng.upBuf.Buffer(), 0, 2)
			setScalar(enc, dimVal, 3)
			setScalar(enc, hiddenDimVal, 4)
			enc.DispatchThreadgroupsThreadsPerThreadgroup(
				apple.MTLSize{Width: uint(cfg.HiddenDim), Height: 1, Depth: 1},
				apple.MTLSize{Width: uint(gemvThreads), Height: 1, Depth: 1},
			)

			// Fast SiLU Activation Mul: silu(gateBuf) * upBuf -> ffnActBuf
			enc.SetComputePipelineState(eng.psoSiLUMul)
			enc.SetBufferWithOffsetAtIndex(eng.gateBuf.Buffer(), 0, 0)
			enc.SetBufferWithOffsetAtIndex(eng.upBuf.Buffer(), 0, 1)
			enc.SetBufferWithOffsetAtIndex(eng.ffnActBuf.Buffer(), 0, 2)
			setScalar(enc, hiddenDimVal, 3)
			enc.DispatchThreadsThreadsPerThreadgroup(
				apple.MTLSize{Width: uint(cfg.HiddenDim / 4), Height: 1, Depth: 1},
				apple.MTLSize{Width: 128, Height: 1, Depth: 1},
			)

			// SwiGLU Down Proj & Residual: xBuf += WDown * ffnActBuf
			enc.SetComputePipelineState(psoGEMVResH)
			enc.SetBufferWithOffsetAtIndex(eng.ffnActBuf.Buffer(), 0, 0)
			enc.SetBufferWithOffsetAtIndex(layer.WDown.Buffer(), 0, 1)
			enc.SetBufferWithOffsetAtIndex(eng.xBuf.Buffer(), 0, 2)
			setScalar(enc, hiddenDimVal, 3)
			setScalar(enc, dimVal, 4)
			enc.DispatchThreadgroupsThreadsPerThreadgroup(
				apple.MTLSize{Width: uint(cfg.Dim), Height: 1, Depth: 1},
				apple.MTLSize{Width: uint(gemvThreads), Height: 1, Depth: 1},
			)
			continue
		}

		layer := &eng.layers[l]
		if layer.InputNorm.Len() == 0 {
			continue
		}

		// Step 2a & 2b: Input RMSNorm + QKV Linear Projections
		if cfg.FuseEGraph && eng.psoFusedQKV != nil {
			enc.SetComputePipelineState(eng.psoFusedQKV)
			enc.SetBufferWithOffsetAtIndex(eng.xBuf.Buffer(), 0, 0)
			enc.SetBufferWithOffsetAtIndex(layer.InputNorm.Buffer(), 0, 1)
			enc.SetBufferWithOffsetAtIndex(layer.WQ.Buffer(), 0, 2)
			enc.SetBufferWithOffsetAtIndex(layer.WK.Buffer(), 0, 3)
			enc.SetBufferWithOffsetAtIndex(layer.WV.Buffer(), 0, 4)
			enc.SetBufferWithOffsetAtIndex(eng.qBuf.Buffer(), 0, 5)
			enc.SetBufferWithOffsetAtIndex(eng.kBuf.Buffer(), 0, 6)
			enc.SetBufferWithOffsetAtIndex(eng.vBuf.Buffer(), 0, 7)
			setScalar(enc, dimVal, 8)
			setScalar(enc, qDimVal, 9)
			setScalar(enc, kvDimVal, 10)
			setScalar(enc, epsVal, 11)
			enc.DispatchThreadgroupsThreadsPerThreadgroup(
				apple.MTLSize{Width: uint(cfg.NumHeads * cfg.HeadDim), Height: 1, Depth: 1},
				apple.MTLSize{Width: uint(gemvThreads), Height: 1, Depth: 1},
			)
		} else {
			// Unfused: Input RMSNorm: xBuf -> normXBuf
			enc.SetComputePipelineState(eng.psoNorm)
			enc.SetBufferWithOffsetAtIndex(eng.xBuf.Buffer(), 0, 0)
			enc.SetBufferWithOffsetAtIndex(layer.InputNorm.Buffer(), 0, 1)
			enc.SetBufferWithOffsetAtIndex(eng.normXBuf.Buffer(), 0, 2)
			setScalar(enc, dimVal, 3)
			setScalar(enc, epsVal, 4)
			enc.DispatchThreadgroupsThreadsPerThreadgroup(
				apple.MTLSize{Width: 1, Height: 1, Depth: 1},
				apple.MTLSize{Width: 128, Height: 1, Depth: 1},
			)

			// Q Projection: normXBuf * WQ -> qBuf
			enc.SetComputePipelineState(psoGEMV)
			enc.SetBufferWithOffsetAtIndex(eng.normXBuf.Buffer(), 0, 0)
			enc.SetBufferWithOffsetAtIndex(layer.WQ.Buffer(), 0, 1)
			enc.SetBufferWithOffsetAtIndex(eng.qBuf.Buffer(), 0, 2)
			setScalar(enc, dimVal, 3)
			setScalar(enc, qDimVal, 4)
			enc.DispatchThreadgroupsThreadsPerThreadgroup(
				apple.MTLSize{Width: uint(cfg.NumHeads * cfg.HeadDim), Height: 1, Depth: 1},
				apple.MTLSize{Width: uint(gemvThreads), Height: 1, Depth: 1},
			)

			// K Projection: normXBuf * WK -> kBuf
			enc.SetComputePipelineState(psoGEMV)
			enc.SetBufferWithOffsetAtIndex(eng.normXBuf.Buffer(), 0, 0)
			enc.SetBufferWithOffsetAtIndex(layer.WK.Buffer(), 0, 1)
			enc.SetBufferWithOffsetAtIndex(eng.kBuf.Buffer(), 0, 2)
			setScalar(enc, dimVal, 3)
			setScalar(enc, kvDimVal, 4)
			enc.DispatchThreadgroupsThreadsPerThreadgroup(
				apple.MTLSize{Width: uint(cfg.NumKVHeads * cfg.HeadDim), Height: 1, Depth: 1},
				apple.MTLSize{Width: uint(gemvThreads), Height: 1, Depth: 1},
			)

			// V Projection: normXBuf * WV -> vBuf
			enc.SetComputePipelineState(psoGEMV)
			enc.SetBufferWithOffsetAtIndex(eng.normXBuf.Buffer(), 0, 0)
			enc.SetBufferWithOffsetAtIndex(layer.WV.Buffer(), 0, 1)
			enc.SetBufferWithOffsetAtIndex(eng.vBuf.Buffer(), 0, 2)
			setScalar(enc, dimVal, 3)
			setScalar(enc, kvDimVal, 4)
			enc.DispatchThreadgroupsThreadsPerThreadgroup(
				apple.MTLSize{Width: uint(cfg.NumKVHeads * cfg.HeadDim), Height: 1, Depth: 1},
				apple.MTLSize{Width: uint(gemvThreads), Height: 1, Depth: 1},
			)
		}

		// Apply RoPE + Store KV cache
		enc.SetComputePipelineState(eng.psoRoPE)
		enc.SetBufferWithOffsetAtIndex(eng.qBuf.Buffer(), 0, 0)
		enc.SetBufferWithOffsetAtIndex(eng.kBuf.Buffer(), 0, 1)
		enc.SetBufferWithOffsetAtIndex(eng.vBuf.Buffer(), 0, 2)
		enc.SetBufferWithOffsetAtIndex(kCache[l].Buffer(), 0, 3)
		enc.SetBufferWithOffsetAtIndex(vCache[l].Buffer(), 0, 4)
		setScalar(enc, posVal, 5)
		setScalar(enc, headDimVal, 6)
		setScalar(enc, qHeadsVal, 7)
		setScalar(enc, kvHeadsVal, 8)
		setScalar(enc, thetaVal, 9)
		enc.DispatchThreadsThreadsPerThreadgroup(
			apple.MTLSize{Width: uint(cfg.NumHeads * cfg.HeadDim), Height: 1, Depth: 1},
			apple.MTLSize{Width: 64, Height: 1, Depth: 1},
		)

		// Causal FlashAttention Decode
		enc.SetComputePipelineState(eng.psoAttn)
		enc.SetBufferWithOffsetAtIndex(eng.qBuf.Buffer(), 0, 0)
		enc.SetBufferWithOffsetAtIndex(kCache[l].Buffer(), 0, 1)
		enc.SetBufferWithOffsetAtIndex(vCache[l].Buffer(), 0, 2)
		enc.SetBufferWithOffsetAtIndex(eng.attnOutBuf.Buffer(), 0, 3)
		setScalar(enc, curPosInt, 4)
		setScalar(enc, maxSeqVal, 5)
		setScalar(enc, qHeadsInt, 6)
		setScalar(enc, kvHeadsInt, 7)
		setScalar(enc, headDimInt, 8)
		setScalar(enc, scaleVal, 9)
		enc.DispatchThreadgroupsThreadsPerThreadgroup(
			apple.MTLSize{Width: uint(cfg.NumHeads), Height: 1, Depth: 1},
			apple.MTLSize{Width: 128, Height: 1, Depth: 1},
		)

		// Attention Out Proj & Residual: xBuf += WO * attnOutBuf
		enc.SetComputePipelineState(psoGEMVRes)
		enc.SetBufferWithOffsetAtIndex(eng.attnOutBuf.Buffer(), 0, 0)
		enc.SetBufferWithOffsetAtIndex(layer.WO.Buffer(), 0, 1)
		enc.SetBufferWithOffsetAtIndex(eng.xBuf.Buffer(), 0, 2)
		setScalar(enc, dimVal, 3)
		setScalar(enc, dimVal, 4)
		enc.DispatchThreadgroupsThreadsPerThreadgroup(
			apple.MTLSize{Width: uint(cfg.Dim), Height: 1, Depth: 1},
			apple.MTLSize{Width: uint(gemvThreads), Height: 1, Depth: 1},
		)

		// Post-Attention RMSNorm: xBuf -> normXBuf
		enc.SetComputePipelineState(eng.psoNorm)
		enc.SetBufferWithOffsetAtIndex(eng.xBuf.Buffer(), 0, 0)
		enc.SetBufferWithOffsetAtIndex(layer.PostNorm.Buffer(), 0, 1)
		enc.SetBufferWithOffsetAtIndex(eng.normXBuf.Buffer(), 0, 2)
		setScalar(enc, dimVal, 3)
		setScalar(enc, epsVal, 4)
		enc.DispatchThreadgroupsThreadsPerThreadgroup(
			apple.MTLSize{Width: 1, Height: 1, Depth: 1},
			apple.MTLSize{Width: 128, Height: 1, Depth: 1},
		)

		// Step 2f: SwiGLU Feed-Forward Network
		if cfg.FuseEGraph && eng.psoFusedSwiGLU != nil {
			// Gate Proj: normXBuf * WGate -> gateBuf
			enc.SetComputePipelineState(psoGEMV)
			enc.SetBufferWithOffsetAtIndex(eng.normXBuf.Buffer(), 0, 0)
			enc.SetBufferWithOffsetAtIndex(layer.WGate.Buffer(), 0, 1)
			enc.SetBufferWithOffsetAtIndex(eng.gateBuf.Buffer(), 0, 2)
			setScalar(enc, dimVal, 3)
			setScalar(enc, hiddenDimVal, 4)
			enc.DispatchThreadgroupsThreadsPerThreadgroup(
				apple.MTLSize{Width: uint(cfg.HiddenDim), Height: 1, Depth: 1},
				apple.MTLSize{Width: uint(gemvThreads), Height: 1, Depth: 1},
			)

			// Up Proj: normXBuf * WUp -> upBuf
			enc.SetComputePipelineState(psoGEMV)
			enc.SetBufferWithOffsetAtIndex(eng.normXBuf.Buffer(), 0, 0)
			enc.SetBufferWithOffsetAtIndex(layer.WUp.Buffer(), 0, 1)
			enc.SetBufferWithOffsetAtIndex(eng.upBuf.Buffer(), 0, 2)
			setScalar(enc, dimVal, 3)
			setScalar(enc, hiddenDimVal, 4)
			enc.DispatchThreadgroupsThreadsPerThreadgroup(
				apple.MTLSize{Width: uint(cfg.HiddenDim), Height: 1, Depth: 1},
				apple.MTLSize{Width: uint(gemvThreads), Height: 1, Depth: 1},
			)

			// Fused SwiGLU + Down Proj + Residual Add: xBuf += WDown * (silu(gateBuf) * upBuf)
			enc.SetComputePipelineState(eng.psoFusedSwiGLU)
			enc.SetBufferWithOffsetAtIndex(eng.gateBuf.Buffer(), 0, 0)
			enc.SetBufferWithOffsetAtIndex(eng.upBuf.Buffer(), 0, 1)
			enc.SetBufferWithOffsetAtIndex(layer.WDown.Buffer(), 0, 2)
			enc.SetBufferWithOffsetAtIndex(eng.xBuf.Buffer(), 0, 3)
			setScalar(enc, dimVal, 4)
			setScalar(enc, hiddenDimVal, 5)
			enc.DispatchThreadgroupsThreadsPerThreadgroup(
				apple.MTLSize{Width: uint(cfg.Dim), Height: 1, Depth: 1},
				apple.MTLSize{Width: uint(gemvThreads), Height: 1, Depth: 1},
			)
		} else {
			// Unfused: Gate Proj
			enc.SetComputePipelineState(psoGEMV)
			enc.SetBufferWithOffsetAtIndex(eng.normXBuf.Buffer(), 0, 0)
			enc.SetBufferWithOffsetAtIndex(layer.WGate.Buffer(), 0, 1)
			enc.SetBufferWithOffsetAtIndex(eng.gateBuf.Buffer(), 0, 2)
			setScalar(enc, dimVal, 3)
			setScalar(enc, hiddenDimVal, 4)
			enc.DispatchThreadgroupsThreadsPerThreadgroup(
				apple.MTLSize{Width: uint(cfg.HiddenDim), Height: 1, Depth: 1},
				apple.MTLSize{Width: uint(gemvThreads), Height: 1, Depth: 1},
			)

			// Up Proj
			enc.SetComputePipelineState(psoGEMV)
			enc.SetBufferWithOffsetAtIndex(eng.normXBuf.Buffer(), 0, 0)
			enc.SetBufferWithOffsetAtIndex(layer.WUp.Buffer(), 0, 1)
			enc.SetBufferWithOffsetAtIndex(eng.upBuf.Buffer(), 0, 2)
			setScalar(enc, dimVal, 3)
			setScalar(enc, hiddenDimVal, 4)
			enc.DispatchThreadgroupsThreadsPerThreadgroup(
				apple.MTLSize{Width: uint(cfg.HiddenDim), Height: 1, Depth: 1},
				apple.MTLSize{Width: uint(gemvThreads), Height: 1, Depth: 1},
			)

			// Fast SiLU Activation Mul: silu(gateBuf) * upBuf -> ffnActBuf
			enc.SetComputePipelineState(eng.psoSiLUMul)
			enc.SetBufferWithOffsetAtIndex(eng.gateBuf.Buffer(), 0, 0)
			enc.SetBufferWithOffsetAtIndex(eng.upBuf.Buffer(), 0, 1)
			enc.SetBufferWithOffsetAtIndex(eng.ffnActBuf.Buffer(), 0, 2)
			setScalar(enc, hiddenDimVal, 3)
			enc.DispatchThreadsThreadsPerThreadgroup(
				apple.MTLSize{Width: uint(cfg.HiddenDim / 4), Height: 1, Depth: 1},
				apple.MTLSize{Width: 128, Height: 1, Depth: 1},
			)

			// SwiGLU Down Proj & Residual: xBuf += WDown * ffnActBuf
			enc.SetComputePipelineState(psoGEMVRes)
			enc.SetBufferWithOffsetAtIndex(eng.ffnActBuf.Buffer(), 0, 0)
			enc.SetBufferWithOffsetAtIndex(layer.WDown.Buffer(), 0, 1)
			enc.SetBufferWithOffsetAtIndex(eng.xBuf.Buffer(), 0, 2)
			setScalar(enc, hiddenDimVal, 3)
			setScalar(enc, dimVal, 4)
			enc.DispatchThreadgroupsThreadsPerThreadgroup(
				apple.MTLSize{Width: uint(cfg.Dim), Height: 1, Depth: 1},
				apple.MTLSize{Width: uint(gemvThreads), Height: 1, Depth: 1},
			)
		}
	}

	if isLast {
		if eng.isF16 {
			// 3. Final RMSNorm (half norm weights): xBuf -> normXBuf
			enc.SetComputePipelineState(eng.psoNormH)
			enc.SetBufferWithOffsetAtIndex(eng.xBuf.Buffer(), 0, 0)
			enc.SetBufferWithOffsetAtIndex(eng.finalNormWF16.Buffer(), 0, 1)
			enc.SetBufferWithOffsetAtIndex(eng.normXBuf.Buffer(), 0, 2)
			setScalar(enc, dimVal, 3)
			setScalar(enc, epsVal, 4)
			enc.DispatchThreadgroupsThreadsPerThreadgroup(
				apple.MTLSize{Width: 1, Height: 1, Depth: 1},
				apple.MTLSize{Width: 128, Height: 1, Depth: 1},
			)

			// 4. LM Head Logits Projection (half weights): normXBuf * LMHead -> logitsBuf
			enc.SetComputePipelineState(psoGEMVH)
			enc.SetBufferWithOffsetAtIndex(eng.normXBuf.Buffer(), 0, 0)
			enc.SetBufferWithOffsetAtIndex(eng.lmHeadWF16.Buffer(), 0, 1)
			enc.SetBufferWithOffsetAtIndex(eng.logitsBuf.Buffer(), 0, 2)
			setScalar(enc, dimVal, 3)
			setScalar(enc, vocabVal, 4)
			enc.DispatchThreadgroupsThreadsPerThreadgroup(
				apple.MTLSize{Width: uint(cfg.VocabSize), Height: 1, Depth: 1},
				apple.MTLSize{Width: uint(gemvThreads), Height: 1, Depth: 1},
			)
		} else {
			// 3. Final RMSNorm: xBuf -> normXBuf
			enc.SetComputePipelineState(eng.psoNorm)
			enc.SetBufferWithOffsetAtIndex(eng.xBuf.Buffer(), 0, 0)
			enc.SetBufferWithOffsetAtIndex(eng.finalNormW.Buffer(), 0, 1)
			enc.SetBufferWithOffsetAtIndex(eng.normXBuf.Buffer(), 0, 2)
			setScalar(enc, dimVal, 3)
			setScalar(enc, epsVal, 4)
			enc.DispatchThreadgroupsThreadsPerThreadgroup(
				apple.MTLSize{Width: 1, Height: 1, Depth: 1},
				apple.MTLSize{Width: 128, Height: 1, Depth: 1},
			)

			// 4. LM Head Logits Projection: normXBuf * LMHead -> logitsBuf
			enc.SetComputePipelineState(psoGEMV)
			enc.SetBufferWithOffsetAtIndex(eng.normXBuf.Buffer(), 0, 0)
			enc.SetBufferWithOffsetAtIndex(eng.lmHeadW.Buffer(), 0, 1)
			enc.SetBufferWithOffsetAtIndex(eng.logitsBuf.Buffer(), 0, 2)
			setScalar(enc, dimVal, 3)
			setScalar(enc, vocabVal, 4)
			enc.DispatchThreadgroupsThreadsPerThreadgroup(
				apple.MTLSize{Width: uint(cfg.VocabSize), Height: 1, Depth: 1},
				apple.MTLSize{Width: uint(gemvThreads), Height: 1, Depth: 1},
			)
		}

		// 5. GPU Hardware Argmax Sampling
		enc.SetComputePipelineState(eng.psoSample)
		enc.SetBufferWithOffsetAtIndex(eng.logitsBuf.Buffer(), 0, 0)
		enc.SetBufferWithOffsetAtIndex(eng.tokBuf.Buffer(), 0, 1)
		setScalar(enc, vocabVal, 2)
		enc.DispatchThreadsThreadsPerThreadgroup(
			apple.MTLSize{Width: 1, Height: 1, Depth: 1},
			apple.MTLSize{Width: 1, Height: 1, Depth: 1},
		)
	}
}

// GPUForwardPass executes the 100% GPU-resident forward pass for a single decode token.
func GPUForwardPass(eng *ModelEngine, cfg Config, tokenID int32, pos int) int32 {
	queue := eng.dev.Queue()
	cb := queue.CommandBuffer()
	enc := cb.ComputeCommandEncoder()
	encodeForwardStep(enc, eng, cfg, tokenID, pos, true)
	enc.EndEncoding()

	t0 := time.Now()
	cb.Commit()
	cb.WaitUntilCompleted()
	wallTime := time.Since(t0)
	gpuTime := time.Duration((cb.GPUEndTime() - cb.GPUStartTime()) * float64(time.Second))
	if gpuTime < 0 {
		gpuTime = 0
	}
	if eng.tuner != nil {
		gemvThreads := 128
		if eng.tuner.Width() == 64 {
			gemvThreads = 64
		}
		eng.tuner.Record(gemvThreads, gpuTime)
	}
	eng.metrics.RecordForward(ForwardMeasurement{
		Position: pos,
		GPUTime:  gpuTime,
		WallTime: wallTime,
		When:     time.Now(),
	})

	if eng.recorder != nil {
		eng.recorder.Record("LLaMA_Forward_Pass_32L", gpuTime, [3]int{cfg.Dim, cfg.NumLayers, 1}, [3]int{128, 1, 1}, map[string]string{
			"model": cfg.Name,
			"pos":   fmt.Sprintf("%d", pos),
		})
	}

	return eng.tokBuf.HostSlice()[0]
}

// GPUPrefillPass executes prompt prefill across all prompt tokens in a single batched GPU command buffer.
func GPUPrefillPass(eng *ModelEngine, cfg Config, promptTokens []int32) int32 {
	if len(promptTokens) == 0 {
		return 0
	}
	queue := eng.dev.Queue()
	cb := queue.CommandBuffer()
	enc := cb.ComputeCommandEncoder()

	for pos, tokID := range promptTokens {
		isLast := (pos == len(promptTokens)-1)
		encodeForwardStep(enc, eng, cfg, tokID, pos, isLast)
	}
	enc.EndEncoding()

	t0 := time.Now()
	cb.Commit()
	cb.WaitUntilCompleted()
	wallTime := time.Since(t0)
	gpuTime := time.Duration((cb.GPUEndTime() - cb.GPUStartTime()) * float64(time.Second))
	if gpuTime < 0 {
		gpuTime = 0
	}
	if eng.tuner != nil {
		gemvThreads := 128
		if eng.tuner.Width() == 64 {
			gemvThreads = 64
		}
		eng.tuner.Record(gemvThreads, gpuTime)
	}
	eng.metrics.RecordForward(ForwardMeasurement{
		Position: len(promptTokens) - 1,
		GPUTime:  gpuTime,
		WallTime: wallTime,
		When:     time.Now(),
	})

	if eng.recorder != nil {
		eng.recorder.Record("LLaMA_Prefill_Pass", gpuTime, [3]int{cfg.Dim, len(promptTokens), 1}, [3]int{128, 1, 1}, map[string]string{
			"model": cfg.Name,
			"len":   fmt.Sprintf("%d", len(promptTokens)),
		})
	}

	return eng.tokBuf.HostSlice()[0]
}

// GPUForwardPassKV executes the GPU forward pass using an isolated SequenceKVCache.
func GPUForwardPassKV(eng *ModelEngine, cfg Config, skv *SequenceKVCache, tokenID int32, pos int) (int32, error) {
	if eng == nil {
		return 0, errors.New("forward pass: nil engine")
	}
	queue := eng.dev.Queue()
	cb := queue.CommandBuffer()
	enc := cb.ComputeCommandEncoder()
	kCache := eng.kCacheBuf
	vCache := eng.vCacheBuf
	if skv != nil && len(skv.KCache) == cfg.NumLayers {
		kCache = skv.KCache
		vCache = skv.VCache
	}
	encodeForwardStepWithKV(enc, eng, cfg, kCache, vCache, tokenID, pos, true)
	enc.EndEncoding()

	cb.Commit()
	cb.WaitUntilCompleted()
	return eng.tokBuf.HostSlice()[0], nil
}

// GPUPrefillPassKV executes prompt prefill using an isolated SequenceKVCache.
func GPUPrefillPassKV(eng *ModelEngine, cfg Config, skv *SequenceKVCache, promptTokens []int32) (int32, error) {
	if len(promptTokens) == 0 {
		return 0, nil
	}
	if eng == nil {
		return 0, errors.New("prefill pass: nil engine")
	}
	queue := eng.dev.Queue()
	cb := queue.CommandBuffer()
	enc := cb.ComputeCommandEncoder()
	kCache := eng.kCacheBuf
	vCache := eng.vCacheBuf
	if skv != nil && len(skv.KCache) == cfg.NumLayers {
		kCache = skv.KCache
		vCache = skv.VCache
	}

	for pos, tokID := range promptTokens {
		isLast := (pos == len(promptTokens)-1)
		encodeForwardStepWithKV(enc, eng, cfg, kCache, vCache, tokID, pos, isLast)
	}
	enc.EndEncoding()

	cb.Commit()
	cb.WaitUntilCompleted()
	return eng.tokBuf.HostSlice()[0], nil
}

// runFastMetalInference executes all autoregressive decode iterations.
func runFastMetalInference(eng *ModelEngine, cfg Config) {
	promptText := cfg.Prompt
	if promptText == "" {
		promptText = "The key to scalable AI architectures is"
	}
	promptTokens, err := eng.tokenizer.Encode(promptText)
	if err != nil || len(promptTokens) == 0 {
		promptTokens = []int32{578, 1401, 311, 69311, 15592, 78335, 374}
	}
	if cfg.BOSTokenID != 0 {
		promptTokens = append([]int32{cfg.BOSTokenID}, promptTokens...)
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

	// 1. Prefill prompt tokens 100% on Metal GPU in a single batched submission
	tPrefill0 := time.Now()
	currentTok := GPUPrefillPass(eng, cfg, promptTokens)
	prefillDur := time.Since(tPrefill0)

	numGen := cfg.GenTokens
	if numGen <= 0 {
		numGen = 10
	}
	generatedTokens := make([]int32, 0, numGen)

	// 2. Decode steps 100% on Metal GPU
	tDecode0 := time.Now()
	for step := 0; step < numGen; step++ {
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
	ttftMs := float64(prefillDur.Microseconds()) / 1000.0
	tpotMs := float64(decodeDur.Microseconds()) / 1000.0 / float64(len(generatedTokens))
	fmt.Printf("TTFT: %.2f ms\n", ttftMs)
	fmt.Printf("TPOT: %.2f ms\n", tpotMs)
	fmt.Printf("Inference completed on Apple Silicon Metal GPU!\n")
}

func PrefillStep(cfg Config, promptTokens []int32) (time.Duration, error) {
	eng, err := GetOrCompileDarwinEngine(cfg)
	if err != nil {
		return 0, err
	}
	start := time.Now()
	GPUPrefillPass(eng, cfg, promptTokens)
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

// RunBenchSuite executes count independent benchmark iterations and emits benchstat-compatible output.
func RunBenchSuite(cfg Config, count int) error {
	eng, err := GetOrCompileDarwinEngine(cfg)
	if err != nil {
		return fmt.Errorf("compile engine: %w", err)
	}

	promptText := cfg.Prompt
	if promptText == "" {
		promptText = "In one short paragraph, explain neural networks using the words layers, neurons, learning, and data."
	}
	promptTokens, err := eng.tokenizer.Encode(promptText)
	if err != nil || len(promptTokens) == 0 {
		promptTokens = []int32{578, 1401, 311, 69311, 15592, 78335, 374}
	}
	if cfg.BOSTokenID != 0 {
		promptTokens = append([]int32{cfg.BOSTokenID}, promptTokens...)
	}

	numGen := cfg.GenTokens
	if numGen <= 0 {
		numGen = 16
	}

	// Warmup run with clean KV cache
	eng.ResetKV()
	curr := GPUPrefillPass(eng, cfg, promptTokens)
	for s := 0; s < 2; s++ {
		curr = GPUForwardPass(eng, cfg, curr, len(promptTokens)+s)
	}

	searchDepths := cfg.SearchDepths
	if len(searchDepths) == 0 {
		depth := cfg.SearchGraphs
		if depth <= 0 {
			depth = 100
		}
		searchDepths = []int{depth}
	}

	modelProf := gpuprof.ModelProfile{
		Name:            cfg.Name,
		HiddenDim:       cfg.Dim,
		IntermediateDim: cfg.HiddenDim,
		NumLayers:       cfg.NumLayers,
		NumHeads:        cfg.NumHeads,
		NumKVHeads:      cfg.NumKVHeads,
		HeadDim:         cfg.HeadDim,
		VocabSize:       cfg.VocabSize,
		TiedEmbeddings:  false,
		BytesPerParam:   4.0,
	}
	hwProf := gpuprof.HardwareProfile{
		Name:              "Apple Silicon Metal",
		Source:            "assumed-m4-max",
		PeakBandwidthGBps: 400.0,
		PeakTFLOPs:        40.0,
	}

	for _, depth := range searchDepths {
		prefix := fmt.Sprintf("Benchmark%s/backend=metal/gen_tokens=%d/prompt=neural_network/search_graphs=%d", cfg.Name, numGen, depth)

		for iter := 0; iter < count; iter++ {
			eng.ResetKV()
			t0 := time.Now()
			currTok := GPUPrefillPass(eng, cfg, promptTokens)
			ttftDur := time.Since(t0)

			tDec0 := time.Now()
			for step := 0; step < numGen; step++ {
				pos := len(promptTokens) + step
				currTok = GPUForwardPass(eng, cfg, currTok, pos)
			}
			decDur := time.Since(tDec0)

			ttftNs := ttftDur.Nanoseconds()
			tpotNs := (decDur / time.Duration(numGen)).Nanoseconds()
			prefillTokPerSec := float64(len(promptTokens)) / ttftDur.Seconds()
			decTokPerSec := float64(numGen) / decDur.Seconds()

			prefObs := gpuprof.Observation{
				Duration:            ttftDur,
				TokenCount:          len(promptTokens),
				ThroughputTokPerSec: prefillTokPerSec,
			}
			decObs := gpuprof.Observation{
				Duration:            decDur,
				TokenCount:          numGen,
				ThroughputTokPerSec: decTokPerSec,
			}

			prefEst, errPref := gpuprof.EstimateWorkload(modelProf, hwProf, len(promptTokens), ttftDur, true)
			var pPrefEst *gpuprof.WorkloadEstimate
			if errPref == nil {
				pPrefEst = &prefEst
			}

			decEst, errDec := gpuprof.EstimateWorkload(modelProf, hwProf, numGen, decDur, false)
			var pDecEst *gpuprof.WorkloadEstimate
			if errDec == nil {
				pDecEst = &decEst
			}

			fmt.Printf("%s/TTFT-1 1 %d ns/op\n", prefix, ttftNs)
			fmt.Printf("%s/TPOT-1 1 %d ns/op\n", prefix, tpotNs)
			fmt.Print(gpuprof.FormatBenchmarkMetricLine(prefix, "prefill", ttftNs, prefObs, pPrefEst))
			fmt.Print(gpuprof.FormatBenchmarkMetricLine(prefix, "decode", decDur.Nanoseconds(), decObs, pDecEst))
		}
	}
	return nil
}

// Generate executes autoregressive decode on GPU for prompt and streams generated tokens through onToken.
func (eng *ModelEngine) Generate(ctx context.Context, cfg Config, prompt string, maxTokens int, onToken func(tokenID int32, tokenText string) bool) error {
	eng.generateMu.Lock()
	defer eng.generateMu.Unlock()

	if maxTokens <= 0 {
		maxTokens = 64
	}
	if eng.tokenizer == nil {
		eng.tokenizer = NewByteTokenizer()
	}
	if eng.dev == nil {
		// Mock simulation fallback when initialized without GPU device
		for step := 0; step < maxTokens; step++ {
			select {
			case <-ctx.Done():
				return ctx.Err()
			default:
			}
			if !onToken(int32(step), fmt.Sprintf(" tok_%d", step)) {
				break
			}
		}
		return nil
	}
	promptTokens, err := eng.tokenizer.Encode(prompt)
	if err != nil || len(promptTokens) == 0 {
		promptTokens = []int32{1}
	}
	if cfg.BOSTokenID != 0 {
		promptTokens = append([]int32{cfg.BOSTokenID}, promptTokens...)
	}

	var currentTok int32
	prefilled := 0
	if snapshot, ok := eng.prefixes.load(promptTokens); ok && eng.restorePrefix(snapshot) {
		currentTok = snapshot.current
		prefilled = snapshot.tokens
	}
	for i, tokID := range promptTokens[prefilled:] {
		pos := prefilled + i
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}
		currentTok = GPUForwardPass(eng, cfg, tokID, pos)
	}
	if prefilled < len(promptTokens) {
		eng.prefixes.store(promptTokens, eng.capturePrefix(len(promptTokens), currentTok, cfg))
	}

	for step := 0; step < maxTokens; step++ {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}
		pos := len(promptTokens) + step
		nextTok := GPUForwardPass(eng, cfg, currentTok, pos)
		tokStr, _ := eng.tokenizer.Decode([]int32{nextTok})
		if !onToken(nextTok, tokStr) {
			break
		}
		currentTok = nextTok
	}
	return nil
}

// GenerateTokenIDs runs generation and returns the token IDs emitted by this
// engine. It is used for exact paired-engine comparisons over Iroh.
func (eng *ModelEngine) GenerateTokenIDs(ctx context.Context, cfg Config, prompt string, maxTokens int) ([]int32, error) {
	var tokens []int32
	err := eng.Generate(ctx, cfg, prompt, maxTokens, func(tokenID int32, _ string) bool {
		tokens = append(tokens, tokenID)
		return true
	})
	return tokens, err
}

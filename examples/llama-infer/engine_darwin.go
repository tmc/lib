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

	apple "github.com/tmc/apple/metal"
	"github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/egraph"
	"github.com/tmc/lib/apple/x/metal/mempool"
	"github.com/tmc/lib/apple/x/metal/metalattn"
	"github.com/tmc/lib/apple/x/metal/metalmegakernel"
	"github.com/tmc/lib/apple/x/metal/metalmem"
	"github.com/tmc/lib/apple/x/metal/metalprof"
	"github.com/tmc/lib/apple/x/metal/metalsafetensors"
	"github.com/tmc/lib/apple/x/metal/metalsample"
	"github.com/tmc/lib/apple/x/metal/metalx"
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

// decodeKernels names every compute pipeline the decode and prefill paths
// dispatch. metalx.Bind resolves the whole set from one compiled library, so a
// kernel that is renamed or dropped from the MSL fails at engine construction
// with the field that named it, not with a nil pipeline state at dispatch.
type decodeKernels struct {
	Embed      apple.MTLComputePipelineState `metal:"EmbeddingLookup"`
	Norm       apple.MTLComputePipelineState `metal:"ComputeRMSNorm"`
	GEMV       apple.MTLComputePipelineState `metal:"FastGEMV"`
	GEMVRes    apple.MTLComputePipelineState `metal:"FastGEMVResidual"`
	GEMV64     apple.MTLComputePipelineState `metal:"FastGEMV64"`
	GEMVRes64  apple.MTLComputePipelineState `metal:"FastGEMVResidual64"`
	EmbedH     apple.MTLComputePipelineState `metal:"EmbeddingLookupH"`
	NormH      apple.MTLComputePipelineState `metal:"ComputeRMSNormH"`
	GEMVH      apple.MTLComputePipelineState `metal:"FastGEMVH"`
	GEMVResH   apple.MTLComputePipelineState `metal:"FastGEMVResidualH"`
	GEMV64H    apple.MTLComputePipelineState `metal:"FastGEMV64H"`
	GEMVRes64H apple.MTLComputePipelineState `metal:"FastGEMVResidual64H"`

	RoPE        apple.MTLComputePipelineState `metal:"ApplyRoPEStoreKV"`
	Attn        apple.MTLComputePipelineState `metal:"CausalFlashAttentionDecode"`
	SiLUMul     apple.MTLComputePipelineState `metal:"FastSiLUMul"`
	FusedQKV    apple.MTLComputePipelineState `metal:"FusedRMSNormQKV"`
	FusedSwiGLU apple.MTLComputePipelineState `metal:"FusedSwiGLUDown"`
	Sample      apple.MTLComputePipelineState `metal:"ArgmaxSampling"`

	BatchedEmbed    apple.MTLComputePipelineState `metal:"BatchedEmbeddingLookup"`
	BatchedNorm     apple.MTLComputePipelineState `metal:"BatchedRMSNorm"`
	BatchedGEMM     apple.MTLComputePipelineState `metal:"BatchedGEMM"`
	BatchedGEMMRes  apple.MTLComputePipelineState `metal:"BatchedGEMMResidual"`
	BatchedRoPE     apple.MTLComputePipelineState `metal:"BatchedRoPE"`
	BatchedAttn     apple.MTLComputePipelineState `metal:"BatchedCausalFlashAttention"`
	BatchedSiLU     apple.MTLComputePipelineState `metal:"BatchedSiLUMul"`
	BatchedEmbedH   apple.MTLComputePipelineState `metal:"BatchedEmbeddingLookupH"`
	BatchedNormH    apple.MTLComputePipelineState `metal:"BatchedRMSNormH"`
	BatchedGEMMH    apple.MTLComputePipelineState `metal:"BatchedGEMMH"`
	BatchedGEMMResH apple.MTLComputePipelineState `metal:"BatchedGEMMResidualH"`
}

// ModelEngine maintains cached Metal pipeline states and GPU-resident memory allocations.
type ModelEngine struct {
	dev        *metal.Device
	lib        apple.MTLLibrary
	pso        decodeKernels
	mem        *metalmem.Arena
	recorder   *metalprof.FlightRecorder
	metrics    *RuntimeMetrics
	generateMu sync.Mutex
	prefixes   *prefixCache
	tuner      *DispatchTuner

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

	// Reusable Batched Prefill Shape Arena Buffers
	prefill          *metalmem.Arena
	prefillMaxSeqLen int
	prefillXMat      mempool.Slice[float32]
	prefillNormXMat  mempool.Slice[float32]
	prefillQMat      mempool.Slice[float32]
	prefillKMat      mempool.Slice[float32]
	prefillVMat      mempool.Slice[float32]
	prefillAttnOut   mempool.Slice[float32]
	prefillGateMat   mempool.Slice[float32]
	prefillUpMat     mempool.Slice[float32]
	prefillFFNAct    mempool.Slice[float32]
	prefillTokIn     mempool.Slice[int32]

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
	cacheKey   string
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

// restorePrefixPartial restores exactly 'tokens' count of KV entries from the snapshot,
// ensuring no stale downstream entries are left unmanaged.
func (eng *ModelEngine) restorePrefixPartial(snapshot *prefixSnapshot, tokens int, cfg Config) bool {
	if eng == nil || snapshot == nil || tokens <= 0 || len(snapshot.k) != len(eng.kCacheBuf) || len(snapshot.v) != len(eng.vCacheBuf) {
		return false
	}
	valuesPerLayer := tokens * cfg.NumKVHeads * cfg.HeadDim
	for layer := range eng.kCacheBuf {
		if valuesPerLayer > len(snapshot.k[layer]) || valuesPerLayer > len(snapshot.v[layer]) ||
			valuesPerLayer > eng.kCacheBuf[layer].Len() || valuesPerLayer > eng.vCacheBuf[layer].Len() {
			return false
		}
		copy(eng.kCacheBuf[layer].HostSlice()[:valuesPerLayer], snapshot.k[layer][:valuesPerLayer])
		copy(eng.vCacheBuf[layer].HostSlice()[:valuesPerLayer], snapshot.v[layer][:valuesPerLayer])
	}
	return true
}

// RestorePrefixPartial restores exactly 'tokens' count of KV entries into the isolated SequenceKVCache.
func (skv *SequenceKVCache) RestorePrefixPartial(snapshot *prefixSnapshot, tokens int, cfg Config) bool {
	if skv == nil || snapshot == nil || tokens <= 0 || len(snapshot.k) != len(skv.KCache) || len(snapshot.v) != len(skv.VCache) {
		return false
	}
	valuesPerLayer := tokens * cfg.NumKVHeads * cfg.HeadDim
	for layer := range skv.KCache {
		if valuesPerLayer > len(snapshot.k[layer]) || valuesPerLayer > len(snapshot.v[layer]) ||
			valuesPerLayer > skv.KCache[layer].Len() || valuesPerLayer > skv.VCache[layer].Len() {
			return false
		}
		copy(skv.KCache[layer].HostSlice()[:valuesPerLayer], snapshot.k[layer][:valuesPerLayer])
		copy(skv.VCache[layer].HostSlice()[:valuesPerLayer], snapshot.v[layer][:valuesPerLayer])
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

	key := fmt.Sprintf("%s|%s|%d|%d|%d|%d|%d|%s|%v", cfg.Name, cfg.ModelPath, cfg.Dim, cfg.HiddenDim, cfg.NumLayers, cfg.VocabSize, cfg.MaxSeqLen, cfg.Precision, cfg.FuseEGraph)
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

	var pso decodeKernels
	if err := metalx.Bind(dev, lib, &pso); err != nil {
		return nil, fmt.Errorf("bind decode kernels: %w", err)
	}

	// One arena for every buffer whose lifetime is the engine's: checked once,
	// released once, and nothing leaks when a later allocation fails.
	mem := metalmem.NewArena(dev)
	xBuf := metalmem.AllocIn[float32](mem, "x", cfg.Dim)
	normXBuf := metalmem.AllocIn[float32](mem, "normX", cfg.Dim)
	qBuf := metalmem.AllocIn[float32](mem, "q", cfg.NumHeads*cfg.HeadDim)
	kBuf := metalmem.AllocIn[float32](mem, "k", cfg.NumKVHeads*cfg.HeadDim)
	vBuf := metalmem.AllocIn[float32](mem, "v", cfg.NumKVHeads*cfg.HeadDim)
	attnOutBuf := metalmem.AllocIn[float32](mem, "attnOut", cfg.Dim)
	gateBuf := metalmem.AllocIn[float32](mem, "gate", cfg.HiddenDim)
	upBuf := metalmem.AllocIn[float32](mem, "up", cfg.HiddenDim)
	ffnActBuf := metalmem.AllocIn[float32](mem, "ffnAct", cfg.HiddenDim)
	logitsBuf := metalmem.AllocIn[float32](mem, "logits", cfg.VocabSize)
	tokBuf := metalmem.AllocIn[int32](mem, "token", 1)

	kvPerLayer := cfg.MaxSeqLen * cfg.NumKVHeads * cfg.HeadDim
	kCacheBuf := make([]mempool.Slice[float32], cfg.NumLayers)
	vCacheBuf := make([]mempool.Slice[float32], cfg.NumLayers)
	for l := range cfg.NumLayers {
		kCacheBuf[l] = metalmem.AllocIn[float32](mem, fmt.Sprintf("kCache[%d]", l), kvPerLayer)
		vCacheBuf[l] = metalmem.AllocIn[float32](mem, fmt.Sprintf("vCache[%d]", l), kvPerLayer)
	}
	if err := mem.Err(); err != nil {
		mem.Free()
		return nil, err
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
			// lm_head is absent in tied-embedding checkpoints; its error is
			// the "no such tensor" that the fallback below handles.
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
			// lm_head is absent in tied-embedding checkpoints; its error is
			// the "no such tensor" that the fallback below handles.
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
			embedTableF16 = metalmem.UploadIn(mem, "embed", f32SliceToF16(detWeights.EmbedTokens))
			finalNormWF16 = metalmem.UploadIn(mem, "norm", f32SliceToF16(detWeights.Norm))
			lmHeadWF16 = metalmem.UploadIn(mem, "lm_head", f32SliceToF16(detWeights.LMHead))
			for i := range cfg.NumLayers {
				lw := detWeights.Layers[i]
				name := func(part string) string { return fmt.Sprintf("layer[%d].%s", i, part) }
				gpuLayersF16[i] = GPULayerWeightsF16{
					InputNorm: metalmem.UploadIn(mem, name("input_norm"), f32SliceToF16(lw.InputNorm)),
					PostNorm:  metalmem.UploadIn(mem, name("post_norm"), f32SliceToF16(lw.PostNorm)),
					WQ:        metalmem.UploadIn(mem, name("wq"), f32SliceToF16(lw.WQ)),
					WK:        metalmem.UploadIn(mem, name("wk"), f32SliceToF16(lw.WK)),
					WV:        metalmem.UploadIn(mem, name("wv"), f32SliceToF16(lw.WV)),
					WO:        metalmem.UploadIn(mem, name("wo"), f32SliceToF16(lw.WO)),
					WGate:     metalmem.UploadIn(mem, name("wgate"), f32SliceToF16(lw.WGate)),
					WUp:       metalmem.UploadIn(mem, name("wup"), f32SliceToF16(lw.WUp)),
					WDown:     metalmem.UploadIn(mem, name("wdown"), f32SliceToF16(lw.WDown)),
				}
			}
		} else {
			embedTable = metalmem.UploadIn(mem, "embed", detWeights.EmbedTokens)
			finalNormW = metalmem.UploadIn(mem, "norm", detWeights.Norm)
			lmHeadW = metalmem.UploadIn(mem, "lm_head", detWeights.LMHead)
			for i := range cfg.NumLayers {
				lw := detWeights.Layers[i]
				name := func(part string) string { return fmt.Sprintf("layer[%d].%s", i, part) }
				gpuLayers[i] = GPULayerWeights{
					InputNorm: metalmem.UploadIn(mem, name("input_norm"), lw.InputNorm),
					PostNorm:  metalmem.UploadIn(mem, name("post_norm"), lw.PostNorm),
					WQ:        metalmem.UploadIn(mem, name("wq"), lw.WQ),
					WK:        metalmem.UploadIn(mem, name("wk"), lw.WK),
					WV:        metalmem.UploadIn(mem, name("wv"), lw.WV),
					WO:        metalmem.UploadIn(mem, name("wo"), lw.WO),
					WGate:     metalmem.UploadIn(mem, name("wgate"), lw.WGate),
					WUp:       metalmem.UploadIn(mem, name("wup"), lw.WUp),
					WDown:     metalmem.UploadIn(mem, name("wdown"), lw.WDown),
				}
			}
		}
		if err := mem.Err(); err != nil {
			mem.Free()
			return nil, fmt.Errorf("upload deterministic weights: %w", err)
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
		dev:           dev,
		lib:           lib,
		pso:           pso,
		mem:           mem,
		xBuf:          xBuf,
		normXBuf:      normXBuf,
		qBuf:          qBuf,
		kBuf:          kBuf,
		vBuf:          vBuf,
		attnOutBuf:    attnOutBuf,
		gateBuf:       gateBuf,
		upBuf:         upBuf,
		ffnActBuf:     ffnActBuf,
		kCacheBuf:     kCacheBuf,
		vCacheBuf:     vCacheBuf,
		logitsBuf:     logitsBuf,
		tokBuf:        tokBuf,
		isF16:         isF16,
		embedTable:    embedTable,
		finalNormW:    finalNormW,
		lmHeadW:       lmHeadW,
		layers:        gpuLayers,
		embedTableF16: embedTableF16,
		finalNormWF16: finalNormWF16,
		lmHeadWF16:    lmHeadWF16,
		layersF16:     gpuLayersF16,
		tokenizer:     tok,
		weights:       weights,
		egraphSats:    egraphSats,
		recorder:      metalprof.NewFlightRecorder(2048),
		metrics:       metrics,
		prefixes:      newPrefixCache(64<<20, 8),
		tuner:         NewDispatchTuner(),
		Plan:          BuildExecutionPlan(cfg),
		cacheKey:      key,
	}
	if err := ValidatePlan(eng.Plan, cfg); err != nil {
		return nil, fmt.Errorf("validate compiled plan: %w", err)
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

// CapturePrefix captures a prefix snapshot from the isolated SequenceKVCache.
func (skv *SequenceKVCache) CapturePrefix(tokens int, current int32, cfg Config) *prefixSnapshot {
	if skv == nil || tokens <= 0 || cfg.NumKVHeads <= 0 || cfg.HeadDim <= 0 {
		return nil
	}
	valuesPerLayer := tokens * cfg.NumKVHeads * cfg.HeadDim
	snapshot := &prefixSnapshot{
		tokens:  tokens,
		current: current,
		k:       make([][]float32, len(skv.KCache)),
		v:       make([][]float32, len(skv.VCache)),
	}
	for layer := range skv.KCache {
		if valuesPerLayer > skv.KCache[layer].Len() || valuesPerLayer > skv.VCache[layer].Len() {
			return nil
		}
		snapshot.k[layer] = append([]float32(nil), skv.KCache[layer].HostSlice()[:valuesPerLayer]...)
		snapshot.v[layer] = append([]float32(nil), skv.VCache[layer].HostSlice()[:valuesPerLayer]...)
		snapshot.bytes += int64(len(snapshot.k[layer])+len(snapshot.v[layer])) * 4
	}
	return snapshot
}

// ExecutePlan executes the authoritative, immutable ExecutionPlan on the GPU compute command encoder.
func ExecutePlan(
	enc apple.MTLComputeCommandEncoder,
	eng *ModelEngine,
	plan *ExecutionPlan,
	cfg Config,
	kCache, vCache []mempool.Slice[float32],
	tokenID int32,
	pos int,
	isLast bool,
) error {
	if eng == nil {
		return errors.New("execute plan: nil engine")
	}
	return ExecutePlanWithLogits(enc, eng, plan, cfg, kCache, vCache, eng.logitsBuf, tokenID, pos, isLast)
}

// ExecutePlanWithLogits executes the authoritative plan directing output logits to a target buffer.
func ExecutePlanWithLogits(
	enc apple.MTLComputeCommandEncoder,
	eng *ModelEngine,
	plan *ExecutionPlan,
	cfg Config,
	kCache, vCache []mempool.Slice[float32],
	logitsBuf mempool.Slice[float32],
	tokenID int32,
	pos int,
	isLast bool,
) error {
	if plan == nil || eng == nil {
		return errors.New("execute plan: nil plan or engine")
	}
	if err := plan.Validate(cfg); err != nil {
		return fmt.Errorf("execute plan validation: %w", err)
	}
	targetLogits := eng.logitsBuf
	if logitsBuf.Len() > 0 {
		targetLogits = logitsBuf
	}

	dimVal := uint32(cfg.Dim)
	hiddenDimVal := uint32(cfg.HiddenDim)
	qDimVal := uint32(cfg.NumHeads * cfg.HeadDim)
	kvDimVal := uint32(cfg.NumKVHeads * cfg.HeadDim)
	epsVal := cfg.Eps
	thetaVal := cfg.RopeTheta
	posVal := uint32(pos)
	headDimVal := uint32(cfg.HeadDim)
	qHeadsVal := uint32(cfg.NumHeads)
	kvHeadsVal := uint32(cfg.NumKVHeads)
	scaleVal := float32(1.0 / math.Sqrt(float64(cfg.HeadDim)))
	vocabVal := uint32(cfg.VocabSize)
	tokVal := uint32(tokenID)

	gemvThreads := 128
	psoGEMV := eng.pso.GEMV
	psoGEMVRes := eng.pso.GEMVRes
	psoGEMVH := eng.pso.GEMVH
	psoGEMVResH := eng.pso.GEMVResH
	if eng.tuner != nil && eng.tuner.Width() == 64 {
		gemvThreads = 64
		psoGEMV = eng.pso.GEMV64
		psoGEMVRes = eng.pso.GEMVRes64
		psoGEMVH = eng.pso.GEMV64H
		psoGEMVResH = eng.pso.GEMVRes64H
	}

	for _, op := range plan.Ops {
		l := op.LayerIndex
		switch op.Name {
		case "EmbeddingLookup":
			if eng.isF16 {
				enc.SetComputePipelineState(eng.pso.EmbedH)
				enc.SetBufferWithOffsetAtIndex(eng.embedTableF16.Buffer(), 0, 0)
			} else {
				enc.SetComputePipelineState(eng.pso.Embed)
				enc.SetBufferWithOffsetAtIndex(eng.embedTable.Buffer(), 0, 0)
			}
			enc.SetBufferWithOffsetAtIndex(eng.xBuf.Buffer(), 0, 1)
			metal.SetScalar(enc, tokVal, 2)
			metal.SetScalar(enc, dimVal, 3)
			enc.DispatchThreadsThreadsPerThreadgroup(
				metal.Threads(cfg.Dim),
				metal.Threads(128),
			)

		case "FusedRMSNormQKV":
			if eng.isF16 {
				layer := &eng.layersF16[l]
				enc.SetComputePipelineState(eng.pso.NormH)
				enc.SetBufferWithOffsetAtIndex(eng.xBuf.Buffer(), 0, 0)
				enc.SetBufferWithOffsetAtIndex(layer.InputNorm.Buffer(), 0, 1)
				enc.SetBufferWithOffsetAtIndex(eng.normXBuf.Buffer(), 0, 2)
				metal.SetScalar(enc, dimVal, 3)
				metal.SetScalar(enc, epsVal, 4)
				enc.DispatchThreadgroupsThreadsPerThreadgroup(
					metal.Threads(1),
					metal.Threads(128),
				)

				enc.SetComputePipelineState(psoGEMVH)
				enc.SetBufferWithOffsetAtIndex(eng.normXBuf.Buffer(), 0, 0)
				enc.SetBufferWithOffsetAtIndex(layer.WQ.Buffer(), 0, 1)
				enc.SetBufferWithOffsetAtIndex(eng.qBuf.Buffer(), 0, 2)
				metal.SetScalar(enc, dimVal, 3)
				metal.SetScalar(enc, qDimVal, 4)
				enc.DispatchThreadgroupsThreadsPerThreadgroup(
					metal.Threads(cfg.NumHeads*cfg.HeadDim),
					metal.Threads(gemvThreads),
				)

				enc.SetComputePipelineState(psoGEMVH)
				enc.SetBufferWithOffsetAtIndex(eng.normXBuf.Buffer(), 0, 0)
				enc.SetBufferWithOffsetAtIndex(layer.WK.Buffer(), 0, 1)
				enc.SetBufferWithOffsetAtIndex(eng.kBuf.Buffer(), 0, 2)
				metal.SetScalar(enc, dimVal, 3)
				metal.SetScalar(enc, kvDimVal, 4)
				enc.DispatchThreadgroupsThreadsPerThreadgroup(
					metal.Threads(cfg.NumKVHeads*cfg.HeadDim),
					metal.Threads(gemvThreads),
				)

				enc.SetComputePipelineState(psoGEMVH)
				enc.SetBufferWithOffsetAtIndex(eng.normXBuf.Buffer(), 0, 0)
				enc.SetBufferWithOffsetAtIndex(layer.WV.Buffer(), 0, 1)
				enc.SetBufferWithOffsetAtIndex(eng.vBuf.Buffer(), 0, 2)
				metal.SetScalar(enc, dimVal, 3)
				metal.SetScalar(enc, kvDimVal, 4)
				enc.DispatchThreadgroupsThreadsPerThreadgroup(
					metal.Threads(cfg.NumKVHeads*cfg.HeadDim),
					metal.Threads(gemvThreads),
				)
			} else {
				layer := &eng.layers[l]
				enc.SetComputePipelineState(eng.pso.FusedQKV)
				enc.SetBufferWithOffsetAtIndex(eng.xBuf.Buffer(), 0, 0)
				enc.SetBufferWithOffsetAtIndex(layer.InputNorm.Buffer(), 0, 1)
				enc.SetBufferWithOffsetAtIndex(layer.WQ.Buffer(), 0, 2)
				enc.SetBufferWithOffsetAtIndex(layer.WK.Buffer(), 0, 3)
				enc.SetBufferWithOffsetAtIndex(layer.WV.Buffer(), 0, 4)
				enc.SetBufferWithOffsetAtIndex(eng.qBuf.Buffer(), 0, 5)
				enc.SetBufferWithOffsetAtIndex(eng.kBuf.Buffer(), 0, 6)
				enc.SetBufferWithOffsetAtIndex(eng.vBuf.Buffer(), 0, 7)
				metal.SetScalar(enc, dimVal, 8)
				metal.SetScalar(enc, qDimVal, 9)
				metal.SetScalar(enc, kvDimVal, 10)
				metal.SetScalar(enc, epsVal, 11)
				maxRows := cfg.NumHeads * cfg.HeadDim
				if cfg.NumKVHeads*cfg.HeadDim > maxRows {
					maxRows = cfg.NumKVHeads * cfg.HeadDim
				}
				enc.DispatchThreadgroupsThreadsPerThreadgroup(
					metal.Threads(maxRows),
					metal.Threads(128),
				)
			}

		case "ComputeRMSNorm":
			if eng.isF16 {
				layer := &eng.layersF16[l]
				enc.SetComputePipelineState(eng.pso.NormH)
				enc.SetBufferWithOffsetAtIndex(eng.xBuf.Buffer(), 0, 0)
				enc.SetBufferWithOffsetAtIndex(layer.InputNorm.Buffer(), 0, 1)
				enc.SetBufferWithOffsetAtIndex(eng.normXBuf.Buffer(), 0, 2)
			} else {
				layer := &eng.layers[l]
				enc.SetComputePipelineState(eng.pso.Norm)
				enc.SetBufferWithOffsetAtIndex(eng.xBuf.Buffer(), 0, 0)
				enc.SetBufferWithOffsetAtIndex(layer.InputNorm.Buffer(), 0, 1)
				enc.SetBufferWithOffsetAtIndex(eng.normXBuf.Buffer(), 0, 2)
			}
			metal.SetScalar(enc, dimVal, 3)
			metal.SetScalar(enc, epsVal, 4)
			enc.DispatchThreadgroupsThreadsPerThreadgroup(
				metal.Threads(1),
				metal.Threads(128),
			)

		case "FastGEMV_Q":
			if eng.isF16 {
				layer := &eng.layersF16[l]
				enc.SetComputePipelineState(psoGEMVH)
				enc.SetBufferWithOffsetAtIndex(eng.normXBuf.Buffer(), 0, 0)
				enc.SetBufferWithOffsetAtIndex(layer.WQ.Buffer(), 0, 1)
			} else {
				layer := &eng.layers[l]
				enc.SetComputePipelineState(psoGEMV)
				enc.SetBufferWithOffsetAtIndex(eng.normXBuf.Buffer(), 0, 0)
				enc.SetBufferWithOffsetAtIndex(layer.WQ.Buffer(), 0, 1)
			}
			enc.SetBufferWithOffsetAtIndex(eng.qBuf.Buffer(), 0, 2)
			metal.SetScalar(enc, dimVal, 3)
			metal.SetScalar(enc, qDimVal, 4)
			enc.DispatchThreadgroupsThreadsPerThreadgroup(
				metal.Threads(cfg.NumHeads*cfg.HeadDim),
				metal.Threads(gemvThreads),
			)

		case "FastGEMV_K":
			if eng.isF16 {
				layer := &eng.layersF16[l]
				enc.SetComputePipelineState(psoGEMVH)
				enc.SetBufferWithOffsetAtIndex(eng.normXBuf.Buffer(), 0, 0)
				enc.SetBufferWithOffsetAtIndex(layer.WK.Buffer(), 0, 1)
			} else {
				layer := &eng.layers[l]
				enc.SetComputePipelineState(psoGEMV)
				enc.SetBufferWithOffsetAtIndex(eng.normXBuf.Buffer(), 0, 0)
				enc.SetBufferWithOffsetAtIndex(layer.WK.Buffer(), 0, 1)
			}
			enc.SetBufferWithOffsetAtIndex(eng.kBuf.Buffer(), 0, 2)
			metal.SetScalar(enc, dimVal, 3)
			metal.SetScalar(enc, kvDimVal, 4)
			enc.DispatchThreadgroupsThreadsPerThreadgroup(
				metal.Threads(cfg.NumKVHeads*cfg.HeadDim),
				metal.Threads(gemvThreads),
			)

		case "FastGEMV_V":
			if eng.isF16 {
				layer := &eng.layersF16[l]
				enc.SetComputePipelineState(psoGEMVH)
				enc.SetBufferWithOffsetAtIndex(eng.normXBuf.Buffer(), 0, 0)
				enc.SetBufferWithOffsetAtIndex(layer.WV.Buffer(), 0, 1)
			} else {
				layer := &eng.layers[l]
				enc.SetComputePipelineState(psoGEMV)
				enc.SetBufferWithOffsetAtIndex(eng.normXBuf.Buffer(), 0, 0)
				enc.SetBufferWithOffsetAtIndex(layer.WV.Buffer(), 0, 1)
			}
			enc.SetBufferWithOffsetAtIndex(eng.vBuf.Buffer(), 0, 2)
			metal.SetScalar(enc, dimVal, 3)
			metal.SetScalar(enc, kvDimVal, 4)
			enc.DispatchThreadgroupsThreadsPerThreadgroup(
				metal.Threads(cfg.NumKVHeads*cfg.HeadDim),
				metal.Threads(gemvThreads),
			)

		case "ApplyRoPEStoreKV":
			enc.SetComputePipelineState(eng.pso.RoPE)
			enc.SetBufferWithOffsetAtIndex(eng.qBuf.Buffer(), 0, 0)
			enc.SetBufferWithOffsetAtIndex(eng.kBuf.Buffer(), 0, 1)
			enc.SetBufferWithOffsetAtIndex(eng.vBuf.Buffer(), 0, 2)
			enc.SetBufferWithOffsetAtIndex(kCache[l].Buffer(), 0, 3)
			enc.SetBufferWithOffsetAtIndex(vCache[l].Buffer(), 0, 4)
			metal.SetScalar(enc, posVal, 5)
			metal.SetScalar(enc, headDimVal, 6)
			metal.SetScalar(enc, qHeadsVal, 7)
			metal.SetScalar(enc, kvHeadsVal, 8)
			metal.SetScalar(enc, thetaVal, 9)
			enc.DispatchThreadsThreadsPerThreadgroup(
				metal.Threads(cfg.NumHeads*cfg.HeadDim),
				metal.Threads(128),
			)

		case "CausalFlashAttentionDecode":
			enc.SetComputePipelineState(eng.pso.Attn)
			enc.SetBufferWithOffsetAtIndex(eng.qBuf.Buffer(), 0, 0)
			enc.SetBufferWithOffsetAtIndex(kCache[l].Buffer(), 0, 1)
			enc.SetBufferWithOffsetAtIndex(vCache[l].Buffer(), 0, 2)
			enc.SetBufferWithOffsetAtIndex(eng.attnOutBuf.Buffer(), 0, 3)
			metal.SetScalar(enc, int32(pos), 4)
			metal.SetScalar(enc, int32(cfg.MaxSeqLen), 5)
			metal.SetScalar(enc, int32(cfg.NumHeads), 6)
			metal.SetScalar(enc, int32(cfg.NumKVHeads), 7)
			metal.SetScalar(enc, int32(cfg.HeadDim), 8)
			metal.SetScalar(enc, scaleVal, 9)
			enc.DispatchThreadgroupsThreadsPerThreadgroup(
				metal.Threads(cfg.NumHeads),
				metal.Threads(128),
			)

		case "FastGEMVResidual_O":
			if eng.isF16 {
				layer := &eng.layersF16[l]
				enc.SetComputePipelineState(psoGEMVResH)
				enc.SetBufferWithOffsetAtIndex(eng.attnOutBuf.Buffer(), 0, 0)
				enc.SetBufferWithOffsetAtIndex(layer.WO.Buffer(), 0, 1)
			} else {
				layer := &eng.layers[l]
				enc.SetComputePipelineState(psoGEMVRes)
				enc.SetBufferWithOffsetAtIndex(eng.attnOutBuf.Buffer(), 0, 0)
				enc.SetBufferWithOffsetAtIndex(layer.WO.Buffer(), 0, 1)
			}
			enc.SetBufferWithOffsetAtIndex(eng.xBuf.Buffer(), 0, 2)
			metal.SetScalar(enc, dimVal, 3)
			metal.SetScalar(enc, dimVal, 4)
			enc.DispatchThreadgroupsThreadsPerThreadgroup(
				metal.Threads(cfg.Dim),
				metal.Threads(gemvThreads),
			)

		case "ComputeRMSNorm_Post":
			if eng.isF16 {
				layer := &eng.layersF16[l]
				enc.SetComputePipelineState(eng.pso.NormH)
				enc.SetBufferWithOffsetAtIndex(eng.xBuf.Buffer(), 0, 0)
				enc.SetBufferWithOffsetAtIndex(layer.PostNorm.Buffer(), 0, 1)
				enc.SetBufferWithOffsetAtIndex(eng.normXBuf.Buffer(), 0, 2)
			} else {
				layer := &eng.layers[l]
				enc.SetComputePipelineState(eng.pso.Norm)
				enc.SetBufferWithOffsetAtIndex(eng.xBuf.Buffer(), 0, 0)
				enc.SetBufferWithOffsetAtIndex(layer.PostNorm.Buffer(), 0, 1)
				enc.SetBufferWithOffsetAtIndex(eng.normXBuf.Buffer(), 0, 2)
			}
			metal.SetScalar(enc, dimVal, 3)
			metal.SetScalar(enc, epsVal, 4)
			enc.DispatchThreadgroupsThreadsPerThreadgroup(
				metal.Threads(1),
				metal.Threads(128),
			)

		case "FastGEMV_Gate":
			if eng.isF16 {
				layer := &eng.layersF16[l]
				enc.SetComputePipelineState(psoGEMVH)
				enc.SetBufferWithOffsetAtIndex(eng.normXBuf.Buffer(), 0, 0)
				enc.SetBufferWithOffsetAtIndex(layer.WGate.Buffer(), 0, 1)
			} else {
				layer := &eng.layers[l]
				enc.SetComputePipelineState(psoGEMV)
				enc.SetBufferWithOffsetAtIndex(eng.normXBuf.Buffer(), 0, 0)
				enc.SetBufferWithOffsetAtIndex(layer.WGate.Buffer(), 0, 1)
			}
			enc.SetBufferWithOffsetAtIndex(eng.gateBuf.Buffer(), 0, 2)
			metal.SetScalar(enc, dimVal, 3)
			metal.SetScalar(enc, hiddenDimVal, 4)
			enc.DispatchThreadgroupsThreadsPerThreadgroup(
				metal.Threads(cfg.HiddenDim),
				metal.Threads(gemvThreads),
			)

		case "FastGEMV_Up":
			if eng.isF16 {
				layer := &eng.layersF16[l]
				enc.SetComputePipelineState(psoGEMVH)
				enc.SetBufferWithOffsetAtIndex(eng.normXBuf.Buffer(), 0, 0)
				enc.SetBufferWithOffsetAtIndex(layer.WUp.Buffer(), 0, 1)
			} else {
				layer := &eng.layers[l]
				enc.SetComputePipelineState(psoGEMV)
				enc.SetBufferWithOffsetAtIndex(eng.normXBuf.Buffer(), 0, 0)
				enc.SetBufferWithOffsetAtIndex(layer.WUp.Buffer(), 0, 1)
			}
			enc.SetBufferWithOffsetAtIndex(eng.upBuf.Buffer(), 0, 2)
			metal.SetScalar(enc, dimVal, 3)
			metal.SetScalar(enc, hiddenDimVal, 4)
			enc.DispatchThreadgroupsThreadsPerThreadgroup(
				metal.Threads(cfg.HiddenDim),
				metal.Threads(gemvThreads),
			)

		case "FusedSwiGLUDown":
			if eng.isF16 {
				layer := &eng.layersF16[l]
				enc.SetComputePipelineState(eng.pso.SiLUMul)
				enc.SetBufferWithOffsetAtIndex(eng.gateBuf.Buffer(), 0, 0)
				enc.SetBufferWithOffsetAtIndex(eng.upBuf.Buffer(), 0, 1)
				enc.SetBufferWithOffsetAtIndex(eng.ffnActBuf.Buffer(), 0, 2)
				metal.SetScalar(enc, hiddenDimVal, 3)
				enc.DispatchThreadsThreadsPerThreadgroup(
					metal.Threads(cfg.HiddenDim),
					metal.Threads(128),
				)

				enc.SetComputePipelineState(psoGEMVResH)
				enc.SetBufferWithOffsetAtIndex(eng.ffnActBuf.Buffer(), 0, 0)
				enc.SetBufferWithOffsetAtIndex(layer.WDown.Buffer(), 0, 1)
				enc.SetBufferWithOffsetAtIndex(eng.xBuf.Buffer(), 0, 2)
				metal.SetScalar(enc, hiddenDimVal, 3)
				metal.SetScalar(enc, dimVal, 4)
				enc.DispatchThreadgroupsThreadsPerThreadgroup(
					metal.Threads(cfg.Dim),
					metal.Threads(gemvThreads),
				)
			} else {
				layer := &eng.layers[l]
				enc.SetComputePipelineState(eng.pso.FusedSwiGLU)
				enc.SetBufferWithOffsetAtIndex(eng.gateBuf.Buffer(), 0, 0)
				enc.SetBufferWithOffsetAtIndex(eng.upBuf.Buffer(), 0, 1)
				enc.SetBufferWithOffsetAtIndex(layer.WDown.Buffer(), 0, 2)
				enc.SetBufferWithOffsetAtIndex(eng.xBuf.Buffer(), 0, 3)
				metal.SetScalar(enc, dimVal, 4)
				metal.SetScalar(enc, hiddenDimVal, 5)
				enc.DispatchThreadgroupsThreadsPerThreadgroup(
					metal.Threads(cfg.Dim),
					metal.Threads(128),
				)
			}

		case "FastSiLUMul":
			enc.SetComputePipelineState(eng.pso.SiLUMul)
			enc.SetBufferWithOffsetAtIndex(eng.gateBuf.Buffer(), 0, 0)
			enc.SetBufferWithOffsetAtIndex(eng.upBuf.Buffer(), 0, 1)
			enc.SetBufferWithOffsetAtIndex(eng.ffnActBuf.Buffer(), 0, 2)
			metal.SetScalar(enc, hiddenDimVal, 3)
			enc.DispatchThreadsThreadsPerThreadgroup(
				metal.Threads(cfg.HiddenDim),
				metal.Threads(128),
			)

		case "FastGEMVResidual_Down":
			if eng.isF16 {
				layer := &eng.layersF16[l]
				enc.SetComputePipelineState(psoGEMVResH)
				enc.SetBufferWithOffsetAtIndex(eng.ffnActBuf.Buffer(), 0, 0)
				enc.SetBufferWithOffsetAtIndex(layer.WDown.Buffer(), 0, 1)
			} else {
				layer := &eng.layers[l]
				enc.SetComputePipelineState(psoGEMVRes)
				enc.SetBufferWithOffsetAtIndex(eng.ffnActBuf.Buffer(), 0, 0)
				enc.SetBufferWithOffsetAtIndex(layer.WDown.Buffer(), 0, 1)
			}
			enc.SetBufferWithOffsetAtIndex(eng.xBuf.Buffer(), 0, 2)
			metal.SetScalar(enc, hiddenDimVal, 3)
			metal.SetScalar(enc, dimVal, 4)
			enc.DispatchThreadgroupsThreadsPerThreadgroup(
				metal.Threads(cfg.Dim),
				metal.Threads(gemvThreads),
			)

		case "ComputeRMSNorm_Final":
			if isLast {
				if eng.isF16 {
					enc.SetComputePipelineState(eng.pso.NormH)
					enc.SetBufferWithOffsetAtIndex(eng.xBuf.Buffer(), 0, 0)
					enc.SetBufferWithOffsetAtIndex(eng.finalNormWF16.Buffer(), 0, 1)
					enc.SetBufferWithOffsetAtIndex(eng.normXBuf.Buffer(), 0, 2)
				} else {
					enc.SetComputePipelineState(eng.pso.Norm)
					enc.SetBufferWithOffsetAtIndex(eng.xBuf.Buffer(), 0, 0)
					enc.SetBufferWithOffsetAtIndex(eng.finalNormW.Buffer(), 0, 1)
					enc.SetBufferWithOffsetAtIndex(eng.normXBuf.Buffer(), 0, 2)
				}
				metal.SetScalar(enc, dimVal, 3)
				metal.SetScalar(enc, epsVal, 4)
				enc.DispatchThreadgroupsThreadsPerThreadgroup(
					metal.Threads(1),
					metal.Threads(128),
				)
			}

		case "FastGEMV_LMHead":
			if isLast {
				if eng.isF16 {
					enc.SetComputePipelineState(psoGEMVH)
					enc.SetBufferWithOffsetAtIndex(eng.normXBuf.Buffer(), 0, 0)
					enc.SetBufferWithOffsetAtIndex(eng.lmHeadWF16.Buffer(), 0, 1)
					enc.SetBufferWithOffsetAtIndex(targetLogits.Buffer(), 0, 2)
				} else {
					enc.SetComputePipelineState(psoGEMV)
					enc.SetBufferWithOffsetAtIndex(eng.normXBuf.Buffer(), 0, 0)
					enc.SetBufferWithOffsetAtIndex(eng.lmHeadW.Buffer(), 0, 1)
					enc.SetBufferWithOffsetAtIndex(targetLogits.Buffer(), 0, 2)
				}
				metal.SetScalar(enc, dimVal, 3)
				metal.SetScalar(enc, vocabVal, 4)
				enc.DispatchThreadgroupsThreadsPerThreadgroup(
					metal.Threads(cfg.VocabSize),
					metal.Threads(gemvThreads),
				)
			}

		case "ArgmaxSampling":
			if isLast {
				enc.SetComputePipelineState(eng.pso.Sample)
				enc.SetBufferWithOffsetAtIndex(targetLogits.Buffer(), 0, 0)
				enc.SetBufferWithOffsetAtIndex(eng.tokBuf.Buffer(), 0, 1)
				metal.SetScalar(enc, vocabVal, 2)
				enc.DispatchThreadsThreadsPerThreadgroup(
					metal.Threads(1),
					metal.Threads(1),
				)
			}

		default:
			return fmt.Errorf("execute plan: unsupported operation %q", op.Name)
		}
	}
	return nil
}

func encodeForwardStepWithKV(enc apple.MTLComputeCommandEncoder, eng *ModelEngine, cfg Config, kCache, vCache []mempool.Slice[float32], tokenID int32, pos int, isLast bool) error {
	if eng == nil {
		return errors.New("encode forward step: nil engine")
	}
	return encodeForwardStepWithKVAndLogits(enc, eng, cfg, kCache, vCache, eng.logitsBuf, tokenID, pos, isLast)
}

func encodeForwardStepWithKVAndLogits(enc apple.MTLComputeCommandEncoder, eng *ModelEngine, cfg Config, kCache, vCache []mempool.Slice[float32], logitsBuf mempool.Slice[float32], tokenID int32, pos int, isLast bool) error {
	if eng == nil {
		return errors.New("encode forward step: nil engine")
	}
	plan := eng.Plan
	if plan == nil {
		plan = BuildExecutionPlan(cfg)
	}
	return ExecutePlanWithLogits(enc, eng, plan, cfg, kCache, vCache, logitsBuf, tokenID, pos, isLast)
}

// GPUForwardPass executes the 100% GPU-resident forward pass for a single decode token.
func GPUForwardPass(eng *ModelEngine, cfg Config, tokenID int32, pos int) int32 {
	tok, _ := GPUForwardPassKV(eng, cfg, nil, tokenID, pos)
	return tok
}

// GPUPrefillPass executes prompt prefill across all prompt tokens via matrix GEMM prefill.
func GPUPrefillPass(eng *ModelEngine, cfg Config, promptTokens []int32) int32 {
	tok, _ := GPUPrefillPassKV(eng, cfg, nil, promptTokens)
	return tok
}

// GPUForwardPassKV executes the GPU forward pass using an isolated SequenceKVCache.
func GPUForwardPassKV(eng *ModelEngine, cfg Config, skv *SequenceKVCache, tokenID int32, pos int) (int32, error) {
	if eng == nil {
		return 0, errors.New("forward pass: nil engine")
	}
	if pos >= cfg.MaxSeqLen {
		return 0, fmt.Errorf("forward pass: position %d exceeds MaxSeqLen %d", pos, cfg.MaxSeqLen)
	}
	kCache := eng.kCacheBuf
	vCache := eng.vCacheBuf
	if skv != nil && len(skv.KCache) == cfg.NumLayers {
		kCache = skv.KCache
		vCache = skv.VCache
	}

	queue := eng.dev.Queue()
	cb := queue.CommandBuffer()
	enc := cb.ComputeCommandEncoder()
	if err := encodeForwardStepWithKV(enc, eng, cfg, kCache, vCache, tokenID, pos, true); err != nil {
		enc.EndEncoding()
		return 0, fmt.Errorf("encode forward step: %w", err)
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
		Position: pos,
		GPUTime:  gpuTime,
		WallTime: wallTime,
		When:     time.Now(),
	})

	if eng.recorder != nil {
		eng.recorder.Record(fmt.Sprintf("LLaMA_Forward_Pass_%dL", cfg.NumLayers), gpuTime, [3]int{cfg.Dim, cfg.NumLayers, 1}, [3]int{128, 1, 1}, map[string]string{
			"model": cfg.Name,
			"pos":   fmt.Sprintf("%d", pos),
		})
	}

	return eng.tokBuf.HostSlice()[0], nil
}

// GPUPrefillPassKV executes prompt prefill using an isolated SequenceKVCache via matrix GEMM prefill.
func GPUPrefillPassKV(eng *ModelEngine, cfg Config, skv *SequenceKVCache, promptTokens []int32) (int32, error) {
	if len(promptTokens) == 0 {
		return 0, nil
	}
	if eng == nil {
		return 0, errors.New("prefill pass: nil engine")
	}
	if eng.pso.BatchedAttn != nil {
		return GPUBatchedPrefillPassKV(eng, cfg, skv, promptTokens, 0)
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
		if err := encodeForwardStepWithKV(enc, eng, cfg, kCache, vCache, tokID, pos, isLast); err != nil {
			enc.EndEncoding()
			return 0, fmt.Errorf("prefill step pos %d: %w", pos, err)
		}
	}
	enc.EndEncoding()

	cb.Commit()
	cb.WaitUntilCompleted()
	return eng.tokBuf.HostSlice()[0], nil
}

// ensurePrefillArena makes the batched-prefill scratch buffers big enough for
// seqLen tokens, reallocating them only when the sequence outgrows what is
// already there.
func (eng *ModelEngine) ensurePrefillArena(seqLen int, cfg Config) error {
	if eng.prefillMaxSeqLen >= seqLen {
		return nil
	}
	eng.freePrefillArena()

	allocLen := max(seqLen, 128)
	a := metalmem.NewArena(eng.dev)
	eng.prefillXMat = metalmem.AllocIn[float32](a, "xMat", allocLen*cfg.Dim)
	eng.prefillNormXMat = metalmem.AllocIn[float32](a, "normXMat", allocLen*cfg.Dim)
	eng.prefillQMat = metalmem.AllocIn[float32](a, "qMat", allocLen*cfg.NumHeads*cfg.HeadDim)
	eng.prefillKMat = metalmem.AllocIn[float32](a, "kMat", allocLen*cfg.NumKVHeads*cfg.HeadDim)
	eng.prefillVMat = metalmem.AllocIn[float32](a, "vMat", allocLen*cfg.NumKVHeads*cfg.HeadDim)
	eng.prefillAttnOut = metalmem.AllocIn[float32](a, "attnOut", allocLen*cfg.Dim)
	eng.prefillGateMat = metalmem.AllocIn[float32](a, "gateMat", allocLen*cfg.HiddenDim)
	eng.prefillUpMat = metalmem.AllocIn[float32](a, "upMat", allocLen*cfg.HiddenDim)
	eng.prefillFFNAct = metalmem.AllocIn[float32](a, "ffnAct", allocLen*cfg.HiddenDim)
	eng.prefillTokIn = metalmem.AllocIn[int32](a, "tokIn", allocLen)
	if err := a.Err(); err != nil {
		a.Free()
		return fmt.Errorf("alloc prefill arena for %d tokens: %w", allocLen, err)
	}

	eng.prefill = a
	eng.prefillMaxSeqLen = allocLen
	return nil
}

// freePrefillArena releases the batched-prefill scratch buffers.
func (eng *ModelEngine) freePrefillArena() {
	if eng.prefill != nil {
		eng.prefill.Free()
		eng.prefill = nil
	}
	eng.prefillMaxSeqLen = 0
}

func (eng *ModelEngine) Free() {
	if eng == nil {
		return
	}
	darwinEngineMu.Lock()
	if eng.cacheKey != "" && darwinEngineCache[eng.cacheKey] == eng {
		delete(darwinEngineCache, eng.cacheKey)
	}
	darwinEngineMu.Unlock()
	eng.freePrefillArena()
	eng.mem.Free()
}

func encodeBatchedPrefillWithKV(
	enc apple.MTLComputeCommandEncoder,
	eng *ModelEngine,
	cfg Config,
	kCache, vCache []mempool.Slice[float32],
	promptTokens []int32,
	startPos int,
	xMat, normXMat, qMat, kMat, vMat, attnOutMat, gateMat, upMat, ffnActMat mempool.Slice[float32],
	tokInBuf mempool.Slice[int32],
) {
	seqLen := len(promptTokens)
	if seqLen == 0 {
		return
	}

	copy(tokInBuf.HostSlice()[:seqLen], promptTokens)

	seqLenVal := uint32(seqLen)
	startPosVal := uint32(startPos)
	dimVal := uint32(cfg.Dim)
	hiddenDimVal := uint32(cfg.HiddenDim)
	qDimVal := uint32(cfg.NumHeads * cfg.HeadDim)
	kvDimVal := uint32(cfg.NumKVHeads * cfg.HeadDim)
	headDimVal := uint32(cfg.HeadDim)
	qHeadsVal := uint32(cfg.NumHeads)
	kvHeadsVal := uint32(cfg.NumKVHeads)
	thetaVal := cfg.RopeTheta
	epsVal := cfg.Eps
	scaleVal := float32(1.0 / math.Sqrt(float64(cfg.HeadDim)))
	vocabVal := uint32(cfg.VocabSize)
	totalFFNElems := uint32(seqLen * cfg.HiddenDim)

	if eng.isF16 {
		// 1. Batched Embedding Lookup over Half Weights
		enc.SetComputePipelineState(eng.pso.BatchedEmbedH)
		enc.SetBufferWithOffsetAtIndex(eng.embedTableF16.Buffer(), 0, 0)
		enc.SetBufferWithOffsetAtIndex(tokInBuf.Buffer(), 0, 1)
		enc.SetBufferWithOffsetAtIndex(xMat.Buffer(), 0, 2)
		metal.SetScalar(enc, seqLenVal, 3)
		metal.SetScalar(enc, dimVal, 4)
		enc.DispatchThreadsThreadsPerThreadgroup(
			metal.Threads(seqLen*cfg.Dim),
			metal.Threads(128),
		)

		// 2. Transformer Decoder Layers
		for l := 0; l < cfg.NumLayers; l++ {
			layer := &eng.layersF16[l]
			if layer.InputNorm.Len() == 0 {
				continue
			}

			// (a) Batched Input RMSNorm
			enc.SetComputePipelineState(eng.pso.BatchedNormH)
			enc.SetBufferWithOffsetAtIndex(xMat.Buffer(), 0, 0)
			enc.SetBufferWithOffsetAtIndex(layer.InputNorm.Buffer(), 0, 1)
			enc.SetBufferWithOffsetAtIndex(normXMat.Buffer(), 0, 2)
			metal.SetScalar(enc, seqLenVal, 3)
			metal.SetScalar(enc, dimVal, 4)
			metal.SetScalar(enc, epsVal, 5)
			enc.DispatchThreadgroupsThreadsPerThreadgroup(
				metal.Threads(seqLen),
				metal.Threads(128),
			)

			// (b) Batched Q, K, V Matrix Multiplications (GEMM)
			enc.SetComputePipelineState(eng.pso.BatchedGEMMH)
			enc.SetBufferWithOffsetAtIndex(normXMat.Buffer(), 0, 0)
			enc.SetBufferWithOffsetAtIndex(layer.WQ.Buffer(), 0, 1)
			enc.SetBufferWithOffsetAtIndex(qMat.Buffer(), 0, 2)
			metal.SetScalar(enc, seqLenVal, 3)
			metal.SetScalar(enc, dimVal, 4)
			metal.SetScalar(enc, qDimVal, 5)
			enc.DispatchThreadgroupsThreadsPerThreadgroup(
				metal.Threads(seqLen*cfg.NumHeads*cfg.HeadDim),
				metal.Threads(128),
			)

			enc.SetComputePipelineState(eng.pso.BatchedGEMMH)
			enc.SetBufferWithOffsetAtIndex(normXMat.Buffer(), 0, 0)
			enc.SetBufferWithOffsetAtIndex(layer.WK.Buffer(), 0, 1)
			enc.SetBufferWithOffsetAtIndex(kMat.Buffer(), 0, 2)
			metal.SetScalar(enc, seqLenVal, 3)
			metal.SetScalar(enc, dimVal, 4)
			metal.SetScalar(enc, kvDimVal, 5)
			enc.DispatchThreadgroupsThreadsPerThreadgroup(
				metal.Threads(seqLen*cfg.NumKVHeads*cfg.HeadDim),
				metal.Threads(128),
			)

			enc.SetComputePipelineState(eng.pso.BatchedGEMMH)
			enc.SetBufferWithOffsetAtIndex(normXMat.Buffer(), 0, 0)
			enc.SetBufferWithOffsetAtIndex(layer.WV.Buffer(), 0, 1)
			enc.SetBufferWithOffsetAtIndex(vMat.Buffer(), 0, 2)
			metal.SetScalar(enc, seqLenVal, 3)
			metal.SetScalar(enc, dimVal, 4)
			metal.SetScalar(enc, kvDimVal, 5)
			enc.DispatchThreadgroupsThreadsPerThreadgroup(
				metal.Threads(seqLen*cfg.NumKVHeads*cfg.HeadDim),
				metal.Threads(128),
			)

			// (c) Batched RoPE + Store KV cache
			enc.SetComputePipelineState(eng.pso.BatchedRoPE)
			enc.SetBufferWithOffsetAtIndex(qMat.Buffer(), 0, 0)
			enc.SetBufferWithOffsetAtIndex(kMat.Buffer(), 0, 1)
			enc.SetBufferWithOffsetAtIndex(vMat.Buffer(), 0, 2)
			enc.SetBufferWithOffsetAtIndex(kCache[l].Buffer(), 0, 3)
			enc.SetBufferWithOffsetAtIndex(vCache[l].Buffer(), 0, 4)
			metal.SetScalar(enc, seqLenVal, 5)
			metal.SetScalar(enc, startPosVal, 6)
			metal.SetScalar(enc, headDimVal, 7)
			metal.SetScalar(enc, qHeadsVal, 8)
			metal.SetScalar(enc, kvHeadsVal, 9)
			metal.SetScalar(enc, thetaVal, 10)
			enc.DispatchThreadsThreadsPerThreadgroup(
				metal.Threads(seqLen),
				metal.Threads(1),
			)

			// (d) Batched Causal FlashAttention
			enc.SetComputePipelineState(eng.pso.BatchedAttn)
			enc.SetBufferWithOffsetAtIndex(qMat.Buffer(), 0, 0)
			enc.SetBufferWithOffsetAtIndex(kCache[l].Buffer(), 0, 1)
			enc.SetBufferWithOffsetAtIndex(vCache[l].Buffer(), 0, 2)
			enc.SetBufferWithOffsetAtIndex(attnOutMat.Buffer(), 0, 3)
			metal.SetScalar(enc, seqLenVal, 4)
			metal.SetScalar(enc, startPosVal, 5)
			metal.SetScalar(enc, qHeadsVal, 6)
			metal.SetScalar(enc, kvHeadsVal, 7)
			metal.SetScalar(enc, headDimVal, 8)
			metal.SetScalar(enc, scaleVal, 9)
			enc.DispatchThreadgroupsThreadsPerThreadgroup(
				metal.Threads(seqLen*cfg.NumHeads),
				metal.Threads(128),
			)

			// (e) Batched Out Proj + Residual Add
			enc.SetComputePipelineState(eng.pso.BatchedGEMMResH)
			enc.SetBufferWithOffsetAtIndex(attnOutMat.Buffer(), 0, 0)
			enc.SetBufferWithOffsetAtIndex(layer.WO.Buffer(), 0, 1)
			enc.SetBufferWithOffsetAtIndex(xMat.Buffer(), 0, 2)
			metal.SetScalar(enc, seqLenVal, 3)
			metal.SetScalar(enc, dimVal, 4)
			metal.SetScalar(enc, dimVal, 5)
			enc.DispatchThreadgroupsThreadsPerThreadgroup(
				metal.Threads(seqLen*cfg.Dim),
				metal.Threads(128),
			)

			// (f) Batched Post-Attention RMSNorm
			enc.SetComputePipelineState(eng.pso.BatchedNormH)
			enc.SetBufferWithOffsetAtIndex(xMat.Buffer(), 0, 0)
			enc.SetBufferWithOffsetAtIndex(layer.PostNorm.Buffer(), 0, 1)
			enc.SetBufferWithOffsetAtIndex(normXMat.Buffer(), 0, 2)
			metal.SetScalar(enc, seqLenVal, 3)
			metal.SetScalar(enc, dimVal, 4)
			metal.SetScalar(enc, epsVal, 5)
			enc.DispatchThreadgroupsThreadsPerThreadgroup(
				metal.Threads(seqLen),
				metal.Threads(128),
			)

			// (g) Batched SwiGLU FFN: Gate + Up GEMMs
			enc.SetComputePipelineState(eng.pso.BatchedGEMMH)
			enc.SetBufferWithOffsetAtIndex(normXMat.Buffer(), 0, 0)
			enc.SetBufferWithOffsetAtIndex(layer.WGate.Buffer(), 0, 1)
			enc.SetBufferWithOffsetAtIndex(gateMat.Buffer(), 0, 2)
			metal.SetScalar(enc, seqLenVal, 3)
			metal.SetScalar(enc, dimVal, 4)
			metal.SetScalar(enc, hiddenDimVal, 5)
			enc.DispatchThreadgroupsThreadsPerThreadgroup(
				metal.Threads(seqLen*cfg.HiddenDim),
				metal.Threads(128),
			)

			enc.SetComputePipelineState(eng.pso.BatchedGEMMH)
			enc.SetBufferWithOffsetAtIndex(normXMat.Buffer(), 0, 0)
			enc.SetBufferWithOffsetAtIndex(layer.WUp.Buffer(), 0, 1)
			enc.SetBufferWithOffsetAtIndex(upMat.Buffer(), 0, 2)
			metal.SetScalar(enc, seqLenVal, 3)
			metal.SetScalar(enc, dimVal, 4)
			metal.SetScalar(enc, hiddenDimVal, 5)
			enc.DispatchThreadgroupsThreadsPerThreadgroup(
				metal.Threads(seqLen*cfg.HiddenDim),
				metal.Threads(128),
			)

			// Elementwise silu(gate) * up
			enc.SetComputePipelineState(eng.pso.BatchedSiLU)
			enc.SetBufferWithOffsetAtIndex(gateMat.Buffer(), 0, 0)
			enc.SetBufferWithOffsetAtIndex(upMat.Buffer(), 0, 1)
			enc.SetBufferWithOffsetAtIndex(ffnActMat.Buffer(), 0, 2)
			metal.SetScalar(enc, totalFFNElems, 3)
			enc.DispatchThreadsThreadsPerThreadgroup(
				metal.Threads(int(totalFFNElems)),
				metal.Threads(128),
			)

			// Down GEMM + Residual Add: xMat += ffnActMat * WDown^T
			enc.SetComputePipelineState(eng.pso.BatchedGEMMResH)
			enc.SetBufferWithOffsetAtIndex(ffnActMat.Buffer(), 0, 0)
			enc.SetBufferWithOffsetAtIndex(layer.WDown.Buffer(), 0, 1)
			enc.SetBufferWithOffsetAtIndex(xMat.Buffer(), 0, 2)
			metal.SetScalar(enc, seqLenVal, 3)
			metal.SetScalar(enc, hiddenDimVal, 4)
			metal.SetScalar(enc, dimVal, 5)
			enc.DispatchThreadgroupsThreadsPerThreadgroup(
				metal.Threads(seqLen*cfg.Dim),
				metal.Threads(128),
			)
		}

		// 3. Final RMSNorm on the LAST token row (seqLen - 1)
		lastRowOffset := uint((seqLen - 1) * cfg.Dim * 4) // bytes
		enc.SetComputePipelineState(eng.pso.NormH)
		enc.SetBufferWithOffsetAtIndex(xMat.Buffer(), lastRowOffset, 0)
		enc.SetBufferWithOffsetAtIndex(eng.finalNormWF16.Buffer(), 0, 1)
		enc.SetBufferWithOffsetAtIndex(eng.normXBuf.Buffer(), 0, 2)
		metal.SetScalar(enc, dimVal, 3)
		metal.SetScalar(enc, epsVal, 4)
		enc.DispatchThreadgroupsThreadsPerThreadgroup(
			metal.Threads(1),
			metal.Threads(128),
		)

		// 4. LM Head Logits on the last token
		enc.SetComputePipelineState(eng.pso.GEMVH)
		enc.SetBufferWithOffsetAtIndex(eng.normXBuf.Buffer(), 0, 0)
		enc.SetBufferWithOffsetAtIndex(eng.lmHeadWF16.Buffer(), 0, 1)
		enc.SetBufferWithOffsetAtIndex(eng.logitsBuf.Buffer(), 0, 2)
		metal.SetScalar(enc, dimVal, 3)
		metal.SetScalar(enc, vocabVal, 4)
		enc.DispatchThreadgroupsThreadsPerThreadgroup(
			metal.Threads(cfg.VocabSize),
			metal.Threads(128),
		)
	} else {
		// 1. Batched Embedding Lookup
		enc.SetComputePipelineState(eng.pso.BatchedEmbed)
		enc.SetBufferWithOffsetAtIndex(eng.embedTable.Buffer(), 0, 0)
		enc.SetBufferWithOffsetAtIndex(tokInBuf.Buffer(), 0, 1)
		enc.SetBufferWithOffsetAtIndex(xMat.Buffer(), 0, 2)
		metal.SetScalar(enc, seqLenVal, 3)
		metal.SetScalar(enc, dimVal, 4)
		enc.DispatchThreadsThreadsPerThreadgroup(
			metal.Threads(seqLen*cfg.Dim),
			metal.Threads(128),
		)

		// 2. Transformer Decoder Layers
		for l := 0; l < cfg.NumLayers; l++ {
			layer := &eng.layers[l]
			if layer.InputNorm.Len() == 0 {
				continue
			}

			// (a) Batched Input RMSNorm
			enc.SetComputePipelineState(eng.pso.BatchedNorm)
			enc.SetBufferWithOffsetAtIndex(xMat.Buffer(), 0, 0)
			enc.SetBufferWithOffsetAtIndex(layer.InputNorm.Buffer(), 0, 1)
			enc.SetBufferWithOffsetAtIndex(normXMat.Buffer(), 0, 2)
			metal.SetScalar(enc, seqLenVal, 3)
			metal.SetScalar(enc, dimVal, 4)
			metal.SetScalar(enc, epsVal, 5)
			enc.DispatchThreadgroupsThreadsPerThreadgroup(
				metal.Threads(seqLen),
				metal.Threads(128),
			)

			// (b) Batched Q, K, V Matrix Multiplications (GEMM)
			enc.SetComputePipelineState(eng.pso.BatchedGEMM)
			enc.SetBufferWithOffsetAtIndex(normXMat.Buffer(), 0, 0)
			enc.SetBufferWithOffsetAtIndex(layer.WQ.Buffer(), 0, 1)
			enc.SetBufferWithOffsetAtIndex(qMat.Buffer(), 0, 2)
			metal.SetScalar(enc, seqLenVal, 3)
			metal.SetScalar(enc, dimVal, 4)
			metal.SetScalar(enc, qDimVal, 5)
			enc.DispatchThreadgroupsThreadsPerThreadgroup(
				metal.Threads(seqLen*cfg.NumHeads*cfg.HeadDim),
				metal.Threads(128),
			)

			enc.SetComputePipelineState(eng.pso.BatchedGEMM)
			enc.SetBufferWithOffsetAtIndex(normXMat.Buffer(), 0, 0)
			enc.SetBufferWithOffsetAtIndex(layer.WK.Buffer(), 0, 1)
			enc.SetBufferWithOffsetAtIndex(kMat.Buffer(), 0, 2)
			metal.SetScalar(enc, seqLenVal, 3)
			metal.SetScalar(enc, dimVal, 4)
			metal.SetScalar(enc, kvDimVal, 5)
			enc.DispatchThreadgroupsThreadsPerThreadgroup(
				metal.Threads(seqLen*cfg.NumKVHeads*cfg.HeadDim),
				metal.Threads(128),
			)

			enc.SetComputePipelineState(eng.pso.BatchedGEMM)
			enc.SetBufferWithOffsetAtIndex(normXMat.Buffer(), 0, 0)
			enc.SetBufferWithOffsetAtIndex(layer.WV.Buffer(), 0, 1)
			enc.SetBufferWithOffsetAtIndex(vMat.Buffer(), 0, 2)
			metal.SetScalar(enc, seqLenVal, 3)
			metal.SetScalar(enc, dimVal, 4)
			metal.SetScalar(enc, kvDimVal, 5)
			enc.DispatchThreadgroupsThreadsPerThreadgroup(
				metal.Threads(seqLen*cfg.NumKVHeads*cfg.HeadDim),
				metal.Threads(128),
			)

			// (c) Batched RoPE + Store KV cache
			enc.SetComputePipelineState(eng.pso.BatchedRoPE)
			enc.SetBufferWithOffsetAtIndex(qMat.Buffer(), 0, 0)
			enc.SetBufferWithOffsetAtIndex(kMat.Buffer(), 0, 1)
			enc.SetBufferWithOffsetAtIndex(vMat.Buffer(), 0, 2)
			enc.SetBufferWithOffsetAtIndex(kCache[l].Buffer(), 0, 3)
			enc.SetBufferWithOffsetAtIndex(vCache[l].Buffer(), 0, 4)
			metal.SetScalar(enc, seqLenVal, 5)
			metal.SetScalar(enc, startPosVal, 6)
			metal.SetScalar(enc, headDimVal, 7)
			metal.SetScalar(enc, qHeadsVal, 8)
			metal.SetScalar(enc, kvHeadsVal, 9)
			metal.SetScalar(enc, thetaVal, 10)
			enc.DispatchThreadsThreadsPerThreadgroup(
				metal.Threads(seqLen),
				metal.Threads(1),
			)

			// (d) Batched Causal FlashAttention
			enc.SetComputePipelineState(eng.pso.BatchedAttn)
			enc.SetBufferWithOffsetAtIndex(qMat.Buffer(), 0, 0)
			enc.SetBufferWithOffsetAtIndex(kCache[l].Buffer(), 0, 1)
			enc.SetBufferWithOffsetAtIndex(vCache[l].Buffer(), 0, 2)
			enc.SetBufferWithOffsetAtIndex(attnOutMat.Buffer(), 0, 3)
			metal.SetScalar(enc, seqLenVal, 4)
			metal.SetScalar(enc, startPosVal, 5)
			metal.SetScalar(enc, qHeadsVal, 6)
			metal.SetScalar(enc, kvHeadsVal, 7)
			metal.SetScalar(enc, headDimVal, 8)
			metal.SetScalar(enc, scaleVal, 9)
			enc.DispatchThreadgroupsThreadsPerThreadgroup(
				metal.Threads(seqLen*cfg.NumHeads),
				metal.Threads(128),
			)

			// (e) Batched Out Proj + Residual Add
			enc.SetComputePipelineState(eng.pso.BatchedGEMMRes)
			enc.SetBufferWithOffsetAtIndex(attnOutMat.Buffer(), 0, 0)
			enc.SetBufferWithOffsetAtIndex(layer.WO.Buffer(), 0, 1)
			enc.SetBufferWithOffsetAtIndex(xMat.Buffer(), 0, 2)
			metal.SetScalar(enc, seqLenVal, 3)
			metal.SetScalar(enc, dimVal, 4)
			metal.SetScalar(enc, dimVal, 5)
			enc.DispatchThreadgroupsThreadsPerThreadgroup(
				metal.Threads(seqLen*cfg.Dim),
				metal.Threads(128),
			)

			// (f) Batched Post-Attention RMSNorm
			enc.SetComputePipelineState(eng.pso.BatchedNorm)
			enc.SetBufferWithOffsetAtIndex(xMat.Buffer(), 0, 0)
			enc.SetBufferWithOffsetAtIndex(layer.PostNorm.Buffer(), 0, 1)
			enc.SetBufferWithOffsetAtIndex(normXMat.Buffer(), 0, 2)
			metal.SetScalar(enc, seqLenVal, 3)
			metal.SetScalar(enc, dimVal, 4)
			metal.SetScalar(enc, epsVal, 5)
			enc.DispatchThreadgroupsThreadsPerThreadgroup(
				metal.Threads(seqLen),
				metal.Threads(128),
			)

			// (g) Batched SwiGLU FFN: Gate + Up GEMMs
			enc.SetComputePipelineState(eng.pso.BatchedGEMM)
			enc.SetBufferWithOffsetAtIndex(normXMat.Buffer(), 0, 0)
			enc.SetBufferWithOffsetAtIndex(layer.WGate.Buffer(), 0, 1)
			enc.SetBufferWithOffsetAtIndex(gateMat.Buffer(), 0, 2)
			metal.SetScalar(enc, seqLenVal, 3)
			metal.SetScalar(enc, dimVal, 4)
			metal.SetScalar(enc, hiddenDimVal, 5)
			enc.DispatchThreadgroupsThreadsPerThreadgroup(
				metal.Threads(seqLen*cfg.HiddenDim),
				metal.Threads(128),
			)

			enc.SetComputePipelineState(eng.pso.BatchedGEMM)
			enc.SetBufferWithOffsetAtIndex(normXMat.Buffer(), 0, 0)
			enc.SetBufferWithOffsetAtIndex(layer.WUp.Buffer(), 0, 1)
			enc.SetBufferWithOffsetAtIndex(upMat.Buffer(), 0, 2)
			metal.SetScalar(enc, seqLenVal, 3)
			metal.SetScalar(enc, dimVal, 4)
			metal.SetScalar(enc, hiddenDimVal, 5)
			enc.DispatchThreadgroupsThreadsPerThreadgroup(
				metal.Threads(seqLen*cfg.HiddenDim),
				metal.Threads(128),
			)

			// Elementwise silu(gate) * up
			enc.SetComputePipelineState(eng.pso.BatchedSiLU)
			enc.SetBufferWithOffsetAtIndex(gateMat.Buffer(), 0, 0)
			enc.SetBufferWithOffsetAtIndex(upMat.Buffer(), 0, 1)
			enc.SetBufferWithOffsetAtIndex(ffnActMat.Buffer(), 0, 2)
			metal.SetScalar(enc, totalFFNElems, 3)
			enc.DispatchThreadsThreadsPerThreadgroup(
				metal.Threads(int(totalFFNElems)),
				metal.Threads(128),
			)

			// Down GEMM + Residual Add: xMat += ffnActMat * WDown^T
			enc.SetComputePipelineState(eng.pso.BatchedGEMMRes)
			enc.SetBufferWithOffsetAtIndex(ffnActMat.Buffer(), 0, 0)
			enc.SetBufferWithOffsetAtIndex(layer.WDown.Buffer(), 0, 1)
			enc.SetBufferWithOffsetAtIndex(xMat.Buffer(), 0, 2)
			metal.SetScalar(enc, seqLenVal, 3)
			metal.SetScalar(enc, hiddenDimVal, 4)
			metal.SetScalar(enc, dimVal, 5)
			enc.DispatchThreadgroupsThreadsPerThreadgroup(
				metal.Threads(seqLen*cfg.Dim),
				metal.Threads(128),
			)
		}

		// 3. Final RMSNorm on the LAST token row (seqLen - 1)
		lastRowOffset := uint((seqLen - 1) * cfg.Dim * 4) // bytes
		enc.SetComputePipelineState(eng.pso.Norm)
		enc.SetBufferWithOffsetAtIndex(xMat.Buffer(), lastRowOffset, 0)
		enc.SetBufferWithOffsetAtIndex(eng.finalNormW.Buffer(), 0, 1)
		enc.SetBufferWithOffsetAtIndex(eng.normXBuf.Buffer(), 0, 2)
		metal.SetScalar(enc, dimVal, 3)
		metal.SetScalar(enc, epsVal, 4)
		enc.DispatchThreadgroupsThreadsPerThreadgroup(
			metal.Threads(1),
			metal.Threads(128),
		)

		// 4. LM Head Logits on the last token
		enc.SetComputePipelineState(eng.pso.GEMV)
		enc.SetBufferWithOffsetAtIndex(eng.normXBuf.Buffer(), 0, 0)
		enc.SetBufferWithOffsetAtIndex(eng.lmHeadW.Buffer(), 0, 1)
		enc.SetBufferWithOffsetAtIndex(eng.logitsBuf.Buffer(), 0, 2)
		metal.SetScalar(enc, dimVal, 3)
		metal.SetScalar(enc, vocabVal, 4)
		enc.DispatchThreadgroupsThreadsPerThreadgroup(
			metal.Threads(cfg.VocabSize),
			metal.Threads(128),
		)
	}

	// 5. GPU Hardware Argmax Sampling
	enc.SetComputePipelineState(eng.pso.Sample)
	enc.SetBufferWithOffsetAtIndex(eng.logitsBuf.Buffer(), 0, 0)
	enc.SetBufferWithOffsetAtIndex(eng.tokBuf.Buffer(), 0, 1)
	metal.SetScalar(enc, vocabVal, 2)
	enc.DispatchThreadsThreadsPerThreadgroup(
		metal.Threads(1),
		metal.Threads(1),
	)
}

// GPUBatchedPrefillPass executes prompt prefill across all prompt tokens via matrix GEMMs.
func GPUBatchedPrefillPass(eng *ModelEngine, cfg Config, promptTokens []int32) (int32, error) {
	return GPUBatchedPrefillPassKV(eng, cfg, nil, promptTokens, 0)
}

// GPUBatchedPrefillPassKV executes prompt prefill via matrix GEMMs using an isolated SequenceKVCache.
func GPUBatchedPrefillPassKV(eng *ModelEngine, cfg Config, skv *SequenceKVCache, promptTokens []int32, startPos int) (int32, error) {
	seqLen := len(promptTokens)
	if seqLen == 0 {
		return 0, nil
	}
	if eng == nil {
		return 0, errors.New("batched prefill pass: nil engine")
	}

	if err := eng.ensurePrefillArena(seqLen, cfg); err != nil {
		return 0, fmt.Errorf("ensure prefill arena: %w", err)
	}

	dev := eng.dev
	queue := dev.Queue()
	cb := queue.CommandBuffer()
	enc := cb.ComputeCommandEncoder()

	kCache := eng.kCacheBuf
	vCache := eng.vCacheBuf
	if skv != nil && len(skv.KCache) == cfg.NumLayers {
		kCache = skv.KCache
		vCache = skv.VCache
	}

	encodeBatchedPrefillWithKV(enc, eng, cfg, kCache, vCache, promptTokens, startPos,
		eng.prefillXMat, eng.prefillNormXMat, eng.prefillQMat, eng.prefillKMat, eng.prefillVMat,
		eng.prefillAttnOut, eng.prefillGateMat, eng.prefillUpMat, eng.prefillFFNAct, eng.prefillTokIn)
	enc.EndEncoding()

	t0 := time.Now()
	cb.Commit()
	cb.WaitUntilCompleted()
	wallTime := time.Since(t0)
	gpuTime := time.Duration((cb.GPUEndTime() - cb.GPUStartTime()) * float64(time.Second))
	if gpuTime < 0 {
		gpuTime = 0
	}
	if eng.metrics != nil {
		eng.metrics.RecordForward(ForwardMeasurement{
			Position: startPos + seqLen - 1,
			GPUTime:  gpuTime,
			WallTime: wallTime,
			When:     time.Now(),
		})
	}
	if eng.recorder != nil {
		eng.recorder.Record("LLaMA_Batched_Prefill_Pass", gpuTime, [3]int{cfg.Dim, seqLen, 1}, [3]int{128, 1, 1}, map[string]string{
			"model": cfg.Name,
			"len":   fmt.Sprintf("%d", seqLen),
		})
	}

	return eng.tokBuf.HostSlice()[0], nil
}

// GPUBatchDecodeStep executes concurrent single-token decode for multiple active sequences in a single command buffer.
func GPUBatchDecodeStep(eng *ModelEngine, cfg Config, batchKVs []*SequenceKVCache, batchTokens []int32, batchPositions []int) ([]int32, error) {
	batchSize := len(batchTokens)
	if batchSize == 0 {
		return nil, nil
	}
	if len(batchPositions) != batchSize || len(batchKVs) != batchSize {
		return nil, fmt.Errorf("batch size mismatch: tokens=%d, positions=%d, kvs=%d", batchSize, len(batchPositions), len(batchKVs))
	}
	if eng == nil {
		return nil, errors.New("batch decode: nil engine")
	}

	dev := eng.dev
	tokOutBuf, err := mempool.Alloc[int32](dev, batchSize)
	if err != nil {
		return nil, fmt.Errorf("alloc batch token output: %w", err)
	}
	defer tokOutBuf.Free()

	var batchLogits []mempool.Slice[float32]
	if batchSize > 1 {
		batchLogits = make([]mempool.Slice[float32], batchSize)
		for b := 0; b < batchSize; b++ {
			buf, err := mempool.Alloc[float32](dev, cfg.VocabSize)
			if err != nil {
				for i := 0; i < b; i++ {
					batchLogits[i].Free()
				}
				return nil, fmt.Errorf("alloc batch logits %d: %w", b, err)
			}
			defer buf.Free()
			batchLogits[b] = buf
		}
	}

	queue := dev.Queue()
	cb := queue.CommandBuffer()
	enc := cb.ComputeCommandEncoder()

	vocabVal := uint32(cfg.VocabSize)
	for b := 0; b < batchSize; b++ {
		skv := batchKVs[b]
		kCache := eng.kCacheBuf
		vCache := eng.vCacheBuf
		if skv != nil && len(skv.KCache) == cfg.NumLayers {
			kCache = skv.KCache
			vCache = skv.VCache
		}

		logitsTarget := eng.logitsBuf
		if batchLogits != nil {
			logitsTarget = batchLogits[b]
		}

		if err := encodeForwardStepWithKVAndLogits(enc, eng, cfg, kCache, vCache, logitsTarget, batchTokens[b], batchPositions[b], true); err != nil {
			enc.EndEncoding()
			return nil, fmt.Errorf("batch decode sequence %d: %w", b, err)
		}

		outOffset := uint(b * 4)
		enc.SetComputePipelineState(eng.pso.Sample)
		enc.SetBufferWithOffsetAtIndex(logitsTarget.Buffer(), 0, 0)
		enc.SetBufferWithOffsetAtIndex(tokOutBuf.Buffer(), outOffset, 1)
		metal.SetScalar(enc, vocabVal, 2)
		enc.DispatchThreadsThreadsPerThreadgroup(
			metal.Threads(1),
			metal.Threads(1),
		)
		enc.MemoryBarrierWithScope(apple.MTLBarrierScopeBuffers)
	}
	enc.EndEncoding()

	cb.Commit()
	cb.WaitUntilCompleted()

	outTokens := make([]int32, batchSize)
	copy(outTokens, tokOutBuf.HostSlice()[:batchSize])
	return outTokens, nil
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

	loadedTensors := cfg.NumLayers*9 + 2
	if eng.weights != nil && len(eng.weights.Tensors) > 0 {
		loadedTensors = len(eng.weights.Tensors)
	}
	fmt.Printf("[Weights] Ingested %d tensors (%d Dim, %d Vocab, %d Layers)\n", loadedTensors, cfg.Dim, cfg.VocabSize, len(eng.layers))
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

	bytesPerParam := 4.0
	if cfg.Precision == "f16" || eng.isF16 {
		bytesPerParam = 2.0
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
		BytesPerParam:   bytesPerParam,
	}
	devName := "Apple Silicon Metal"
	if eng.dev != nil && eng.dev.Name() != "" {
		devName = eng.dev.Name()
	}
	hwProf := gpuprof.HardwareProfile{
		Name:              devName,
		Source:            "device-query",
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
	if snapshot, matched, ok := eng.prefixes.load(promptTokens); ok && eng.restorePrefixPartial(snapshot, matched, cfg) {
		currentTok = snapshot.current
		prefilled = matched
	}
	if prefilled < len(promptTokens) {
		remaining := promptTokens[prefilled:]
		tok, err := GPUBatchedPrefillPassKV(eng, cfg, nil, remaining, prefilled)
		if err == nil {
			currentTok = tok
		} else {
			for i, tokID := range remaining {
				pos := prefilled + i
				select {
				case <-ctx.Done():
					return ctx.Err()
				default:
				}
				currentTok = GPUForwardPass(eng, cfg, tokID, pos)
			}
		}
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
		if cfg.IsEOS(nextTok) {
			break
		}
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

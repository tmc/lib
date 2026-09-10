//go:build !darwin && !tpu

package main

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"path/filepath"
	"runtime"
	"sync"
	"time"

	"github.com/tmc/lib/apple/x/metal/metalprof"
	"github.com/tmc/lib/examples/llama-infer/gpuprof"
	"github.com/tmc/lib/lm/speculative"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudaattn"
	"github.com/tmc/lib/nvidia/x/cudakv"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudasample"
)

// BatchedQKVResult is unavailable on non-Metal platforms.
type BatchedQKVResult struct {
	Positions                      []int
	Embedding, Normalized, Q, K, V []float32
	Dispatches                     int
}

func RunBatchedQKV(*ModelEngine, Config, int, []int32, []int) (BatchedQKVResult, error) {
	return BatchedQKVResult{}, errors.New("batched qkv: Metal backend required")
}

type BatchedAttentionCache struct{}

func (*BatchedAttentionCache) ImportSequence(int, int, *SequenceKVCache) error {
	return errors.New("import sequence requires darwin")
}

type SamplerConfig struct {
	Method      string
	Temperature float32
}

type BatchedModelTail struct{}

type BatchedModelTailResult struct {
	Normalized []float32
	Logits     []float32
	Tokens     []int32
	Dispatches int
}

func (*ModelEngine) NewBatchedModelTail(Config, int) (*BatchedModelTail, error) {
	return nil, errors.New("batched model tail requires darwin")
}

func (*BatchedModelTail) Free() {}

func (*BatchedModelTail) Run(*ModelEngine, Config, []float32, []SamplerConfig) (BatchedModelTailResult, error) {
	return BatchedModelTailResult{}, errors.New("batched model tail requires darwin")
}

func RunBatchedModelTailFromChain(*ModelEngine, Config, *BatchedModelTail, BatchedLayerChainResult, []SamplerConfig) (BatchedModelTailResult, error) {
	return BatchedModelTailResult{}, errors.New("batched model tail requires darwin")
}

type BatchedAttentionResult struct {
	Q, K, Attention []float32
	Positions, Rows []int
	Dispatches      int
}
type BatchedLayerResult struct {
	Hidden          []float32
	Positions, Rows []int
	Dispatches      int
}
type BatchedLayerChainResult struct {
	Hidden          []float32
	Positions, Rows []int
	Dispatches      int
	DeviceBytes     int64
}

type SegmentedLayerChainResult struct {
	Hidden          []float32
	Positions, Rows []int
	Dispatches      int
	DeviceBytes     int64
}

func RunBatchedLayerChain(*ModelEngine, Config, *BatchedAttentionCache, []int32, []int, []int) (BatchedLayerChainResult, error) {
	return BatchedLayerChainResult{}, errors.New("batched layer chain: Metal backend required")
}

func RunSegmentedLayerChain(context.Context, *ModelEngine, Config, *BatchedAttentionCache, PackedPrefill) (SegmentedLayerChainResult, error) {
	return SegmentedLayerChainResult{}, errors.New("segmented layer chain: Metal backend required")
}

func RunSegmentedModelTailFromChain(*ModelEngine, Config, *BatchedModelTail, SegmentedLayerChainResult, []SamplerConfig) (BatchedModelTailResult, error) {
	return BatchedModelTailResult{}, errors.New("segmented model tail requires darwin")
}

func RunBatchedLayer(*ModelEngine, Config, int, *BatchedAttentionCache, []int32, []int, []int) (BatchedLayerResult, error) {
	return BatchedLayerResult{}, errors.New("batched layer: Metal backend required")
}

func (*ModelEngine) NewBatchedAttentionCache(Config, int) (*BatchedAttentionCache, error) {
	return nil, errors.New("batched attention: Metal backend required")
}
func (*BatchedAttentionCache) Free() {}
func RunBatchedAttention(*ModelEngine, Config, int, *BatchedAttentionCache, []float32, []float32, []float32, []int, []int) (BatchedAttentionResult, error) {
	return BatchedAttentionResult{}, errors.New("batched attention: Metal backend required")
}

func RunSegmentedAttention(context.Context, *ModelEngine, Config, int, *BatchedAttentionCache, PackedPrefill, []float32, []float32, []float32) (BatchedAttentionResult, error) {
	return BatchedAttentionResult{}, errors.New("segmented attention: Metal backend required")
}

func init() {
	runtime.LockOSThread()
}

// ModelEngine maintains cached CUDA allocations, JIT modules, and device contexts on Linux.
type ModelEngine struct {
	dev            *cuda.Device
	stream         *cuda.Stream
	weights        *GPUModelWeights
	scratch        *GPUScratchpad
	prefillScratch *GPUBatchScratchpad
	modules        *GPUModules
	kvCache        *cudakv.Cache
	// paged replaces kvCache and every SequenceKVCache when the engine was
	// compiled with LLAMA_INFER_PAGED_KV=1; pagedMain is then the single-stream
	// sequence kvCache would have been. Both are nil on a dense engine.
	paged     *pagedKVRuntime
	pagedMain *pagedSequence
	// fp8 and fp8Main are the same pair for LLAMA_INFER_FP8_KV=1: the E4M3
	// kernels and scratch, and the single-stream cache. Both nil otherwise.
	fp8             *fp8KVRuntime
	fp8Main         *cudakv.FP8Cache
	attnFn          *boundedKernel
	sampleFn        *cuda.Kernel
	mCfg            ModelConfig
	fuseActive      bool
	tokenizer       Tokenizer
	recorder        *metalprof.FlightRecorder
	metrics         *RuntimeMetrics
	drafter         *speculative.NgramDrafter
	egraphSats      int
	egraphKnown     bool
	generateMu      sync.Mutex
	Plan            *ExecutionPlan
	consumption     CheckpointReceipt
	memory          *cudamem.Tracker
	benchmarkSerial bool
	tracer          *cudaTracer
}

// CheckpointReceipt returns the validated tensor-consumption account.
func (eng *ModelEngine) CheckpointReceipt() CheckpointReceipt {
	if eng == nil {
		return CheckpointReceipt{}
	}
	return eng.consumption
}

// closeTracer flushes and closes the Perfetto trace, if one was opened. It is
// safe to call on an engine that was never tracing.
func (eng *ModelEngine) closeTracer() error {
	if eng == nil {
		return nil
	}
	err := eng.tracer.close()
	eng.tracer = nil
	return err
}

// FlightRecorder returns nil on CUDA targets until unified event recorder is attached.
func (eng *ModelEngine) FlightRecorder() *metalprof.FlightRecorder {
	return nil
}

// IsF16 reports whether the engine is executing over FP16 weights.
//
// It reads the dtype the kernels were actually bound for, not the dtype the
// plan asked for. Those can only differ if something went wrong, but reporting
// the request would make that difference invisible, which is the one thing this
// predicate must not do.
func (eng *ModelEngine) IsF16() bool {
	if eng == nil || eng.modules == nil {
		return false
	}
	return eng.modules.WeightDType() == WeightF16
}

// WeightReport returns the engine's actual weight storage and dispatch.
func (eng *ModelEngine) WeightReport() WeightReport {
	if eng == nil || eng.modules == nil {
		return WeightReport{}
	}
	d := eng.modules.WeightDType()
	r := WeightReport{
		DType:   d.String(),
		PerElem: d.Bytes(),
		Kernels: weightKernelNames(d),
	}
	if eng.weights != nil {
		r.Storage = eng.weights.DType().String()
		r.Bytes = eng.weights.WeightBytes()
	}
	return r
}

func (eng *ModelEngine) precisionTelemetry() []TensorPrecision {
	return nil
}

// fusionActive reports whether the engine dispatches the fused e-graph kernels.
func (eng *ModelEngine) fusionActive() bool {
	return eng != nil && eng.fuseActive
}

func (eng *ModelEngine) prefixCacheMetrics() PrefixCacheMetrics {
	return PrefixCacheMetrics{}
}

func (eng *ModelEngine) prefixCacheAvailable() bool {
	return false
}

// ReadLogits returns a copy of the final logits computed in the last forward pass.
func (eng *ModelEngine) ReadLogits() ([]float32, error) {
	if eng == nil {
		return nil, errors.New("read logits: nil engine")
	}
	if eng.scratch == nil || eng.scratch.Logits.Len() == 0 {
		return nil, errors.New("read logits: empty logits buffer")
	}
	var res []float32
	err := eng.dev.WithContext(func() error {
		if eng.stream != nil {
			if err := eng.stream.Synchronize(context.Background()); err != nil {
				return fmt.Errorf("read logits: stream synchronize: %w", err)
			}
		}
		res = make([]float32, eng.scratch.Logits.Len())
		if err := eng.scratch.Logits.CopyToHost(res); err != nil {
			return fmt.Errorf("read logits: copy to host: %w", err)
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	return res, nil
}

// ResetKV zeroes out the entire KV cache buffer on device and resets position.
func (eng *ModelEngine) ResetKV() error {
	if eng == nil {
		return nil
	}
	if eng.pagedMain != nil {
		return eng.pagedMain.reset()
	}
	if eng.fp8Main != nil {
		// Rows are rewritten in position order and attention reads only
		// positions up to the current one, so there is no state to clear.
		return nil
	}
	if eng.kvCache == nil {
		return nil
	}
	eng.kvCache.Reset()
	return nil
}

// MutateWeightForTest alters a single weight element in GPU memory and returns a restore function.
func (eng *ModelEngine) MutateWeightForTest(layerIndex, elemIndex int, delta float32) (func(), error) {
	if eng == nil || eng.weights == nil {
		return nil, errors.New("mutate weight: nil engine or weights")
	}
	var targetBuf WeightSlice
	if layerIndex < 0 {
		targetBuf = eng.weights.EmbedTokens
	} else {
		if len(eng.weights.Layers) <= layerIndex {
			return nil, errors.New("mutate weight: invalid layer index")
		}
		targetBuf = eng.weights.Layers[layerIndex].WQ
	}
	if targetBuf.Len() <= elemIndex {
		return nil, errors.New("mutate weight: invalid element index")
	}
	hostSlice := make([]float32, targetBuf.Len())
	var orig float32
	err := eng.dev.WithContext(func() error {
		if err := targetBuf.CopyToHost(hostSlice); err != nil {
			return fmt.Errorf("mutate weight read: %w", err)
		}
		orig = hostSlice[elemIndex]
		hostSlice[elemIndex] += delta
		if err := targetBuf.CopyFromHost(hostSlice); err != nil {
			return fmt.Errorf("mutate weight write: %w", err)
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	return func() {
		_ = eng.dev.WithContext(func() error {
			hostSlice[elemIndex] = orig
			return targetBuf.CopyFromHost(hostSlice)
		})
	}, nil
}

var (
	cudaEngineCache = make(map[cudaEngineKey]*ModelEngine)
	cudaEngineMu    sync.Mutex
)

type cudaEngineKey struct {
	name       string
	modelPath  string
	dim        int
	hiddenDim  int
	vocabSize  int
	maxSeqLen  int
	precision  string
	fuseEGraph bool
	// pagedKV is the environment switch at compile time, so a process that
	// compiles one config both ways holds two engines rather than one.
	pagedKV bool
	fp8KV   bool
}

func cudaEngineCacheKey(cfg Config) cudaEngineKey {
	return cudaEngineKey{
		name:       cfg.Name,
		modelPath:  cfg.ModelPath,
		dim:        cfg.Dim,
		hiddenDim:  cfg.HiddenDim,
		vocabSize:  cfg.VocabSize,
		maxSeqLen:  cfg.MaxSeqLen,
		precision:  cfg.Precision,
		fuseEGraph: cfg.FuseEGraph,
		pagedKV:    pagedKVEnabled(),
		fp8KV:      fp8KVEnabled(),
	}
}

// GetOrCompileEngine is the universal entrypoint for obtaining the GPU engine.
func GetOrCompileEngine(cfg Config) (*ModelEngine, error) {
	return GetOrCompileCudaEngine(cfg)
}

// GetOrCompileCudaEngine builds or reuses the compiled CUDA engine per model config.
func GetOrCompileCudaEngine(cfg Config) (_ *ModelEngine, resultErr error) {
	cudaEngineMu.Lock()
	defer cudaEngineMu.Unlock()

	key := cudaEngineCacheKey(cfg)
	if eng, ok := cudaEngineCache[key]; ok {
		return eng, nil
	}
	if key.pagedKV && cudaGraphEnabled() {
		return nil, errors.New("LLAMA_INFER_PAGED_KV=1 and LLAMA_INFER_CUDA_GRAPH=1 are exclusive: the paged step is not graph-captured")
	}
	if key.fp8KV && key.pagedKV {
		return nil, errors.New("LLAMA_INFER_FP8_KV=1 and LLAMA_INFER_PAGED_KV=1 are exclusive: the block pool holds float32")
	}
	if key.fp8KV && cudaGraphEnabled() {
		return nil, errors.New("LLAMA_INFER_FP8_KV=1 and LLAMA_INFER_CUDA_GRAPH=1 are exclusive: the FP8 step is not graph-captured")
	}

	dev, err := cuda.NewDevice(0)
	if err != nil {
		return nil, fmt.Errorf("cuda device: %w", err)
	}

	stream, err := dev.NewStream()
	if err != nil {
		return nil, finishCUDAInit(fmt.Errorf("cuda stream: %w", err), dev.Close)
	}
	var (
		gpuWeights *GPUModelWeights
		scratch    *GPUScratchpad
		modules    *GPUModules
		attnMod    *cuda.Module
		sampleMod  *cuda.Module
		kvCache    *cudakv.Cache
		paged      *pagedKVRuntime
		pagedMain  *pagedSequence
		fp8        *fp8KVRuntime
		fp8Main    *cudakv.FP8Cache
		committed  bool
	)
	defer func() {
		if committed {
			return
		}
		resultErr = finishCUDAInit(resultErr,
			func() error {
				if kvCache != nil {
					return kvCache.Close()
				}
				var errs []error
				if fp8Main != nil {
					errs = append(errs, fp8Main.Close())
				}
				return errors.Join(append(errs, fp8.Close(), paged.Close())...)
			},
			sampleMod.Close, attnMod.Close,
			func() error {
				if modules != nil {
					return modules.Close()
				}
				return nil
			},
			func() error {
				if scratch != nil {
					scratch.Free()
				}
				return nil
			},
			func() error {
				if gpuWeights != nil {
					gpuWeights.Free()
				}
				return nil
			},
			stream.Destroy, dev.Close,
		)
	}()

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
		FuseEGraph:        cfg.FuseEGraph,
	}
	if mCfg.HeadDim() == 0 && cfg.HeadDim > 0 {
		mCfg.ExplicitHeadDim = cfg.HeadDim
	}
	if mCfg.MaxSeqLen == 0 {
		mCfg.MaxSeqLen = 2048
	}
	if mCfg.RmsNormEps == 0 {
		mCfg.RmsNormEps = 1e-5
	}
	if mCfg.RopeTheta == 0 {
		mCfg.RopeTheta = 10000.0
	}

	// A checkpoint is loaded in the dtypes it declares and widened at the
	// upload boundary; synthetic test weights are already float32 and skip
	// that step. Either way the device receives float32 today.
	var weights *ModelWeights
	var retained *RetainedModel
	var tokenizer Tokenizer
	if cfg.ModelPath != "" {
		loadedW, loadedCfg, err := LoadRetainedModel(cfg.ModelPath)
		if err != nil {
			return nil, fmt.Errorf("load model %q: %w", cfg.ModelPath, err)
		}
		mCfg = *loadedCfg
		mCfg.FuseEGraph = cfg.FuseEGraph
		if cfg.MaxSeqLen > 0 && cfg.MaxSeqLen < mCfg.MaxSeqLen {
			mCfg.MaxSeqLen = cfg.MaxSeqLen
		}
		retained = loadedW
		tokenizer, err = LoadHFTokenizer(filepath.Join(cfg.ModelPath, "tokenizer.json"))
		if err != nil {
			return nil, fmt.Errorf("load tokenizer: %w", err)
		}
	} else if cfg.TestMode {
		weights = GenerateDeterministicWeights(&mCfg)
		tokenizer = NewByteTokenizer()
	} else {
		return nil, errors.New("model path required: synthetic test weights require explicit TestMode=true")
	}

	// The device weight dtype is selected explicitly and never inferred. A
	// checkpoint stored in half precision still uploads as float32 unless the
	// caller asked for a half mode, so no run silently changes precision.
	weightDType, err := ParseWeightDType(cfg.Precision)
	if err != nil {
		return nil, fmt.Errorf("weight precision: %w", err)
	}

	memory := new(cudamem.Tracker)
	if retained != nil {
		gpuWeights, err = UploadRetainedModelToGPUAs(dev, memory, retained, &mCfg, weightDType)
	} else {
		gpuWeights, err = UploadModelToGPUAs(dev, memory, weights, &mCfg, weightDType)
	}
	if err != nil {
		return nil, fmt.Errorf("upload weights to gpu: %w", err)
	}

	scratch, err = NewGPUScratchpad(dev, memory, &mCfg)
	if err != nil {
		return nil, fmt.Errorf("alloc gpu scratchpad: %w", err)
	}

	modules, err = CompileGPUModulesAs(dev, weightDType)
	if err != nil {
		return nil, fmt.Errorf("compile forward modules (%s weights): %w", weightDType, err)
	}
	// The uploaded storage and the bound kernels must agree, or the kernels
	// would read the weights at the wrong width. Checking the two independently
	// recorded dtypes rather than the single requested one means a mismatch is
	// caught here instead of producing quiet garbage.
	if got, want := modules.WeightDType(), gpuWeights.DType(); got != want {
		return nil, fmt.Errorf("kernels bound for %s weights but model uploaded as %s", got, want)
	}
	// Both flag-named outputs are per engine, and the cache holds one engine
	// per config, so a second config in one process gets its own file rather
	// than truncating the first. The lock is held, so the count is stable.
	if *kernelReportPath != "" {
		if err := writeKernelReport(engineOutputPath(*kernelReportPath, len(cudaEngineCache)), modules); err != nil {
			return nil, fmt.Errorf("kernel report: %w", err)
		}
	}

	major, minor, err := dev.ComputeCapability()
	if err != nil {
		return nil, fmt.Errorf("compute capability: %w", err)
	}

	attnModCompiled, err := cudaattn.Compile(major, minor)
	if err != nil {
		return nil, fmt.Errorf("compile attention: %w", err)
	}
	attnMod, err = loadCUDAProgram(dev, attnModCompiled)
	if err != nil {
		return nil, fmt.Errorf("load attention module: %w", err)
	}
	attnRaw, err := attnMod.Function("CausalFlashAttentionDecode")
	if err != nil {
		return nil, fmt.Errorf("get attention fn: %w", err)
	}
	attnFn, err := newBoundedKernel(dev, attnRaw)
	if err != nil {
		return nil, fmt.Errorf("bound attention fn: %w", err)
	}

	sampleModCompiled, err := cudasample.Compile(major, minor)
	if err != nil {
		return nil, fmt.Errorf("compile sampler: %w", err)
	}
	sampleMod, err = loadCUDAProgram(dev, sampleModCompiled)
	if err != nil {
		return nil, fmt.Errorf("load sample module: %w", err)
	}
	sampleFn, err := sampleMod.Function("ArgmaxLogits")
	if err != nil {
		return nil, fmt.Errorf("get sample fn: %w", err)
	}

	if key.pagedKV {
		// UNQUALIFIED: see paged_kv_cuda.go. One block pool stands in for
		// the single-stream cache and every per-sequence cache, sized for
		// the serving sequences plus the single-stream one. A step carries
		// at most one row per serving sequence or one per prefill row.
		seqs, err := pagedKVSequences()
		if err != nil {
			return nil, err
		}
		paged, err = newPagedKVRuntime(dev, memory, defaultPagedKVConfig(mCfg.MaxSeqLen, seqs+1),
			mCfg.NumHiddenLayers, mCfg.NumAttentionHeads, mCfg.NumKeyValueHeads, mCfg.HeadDim(), mCfg.MaxSeqLen, max(seqs, cudaPrefillRows))
		if err != nil {
			return nil, fmt.Errorf("create paged kv cache: %w", err)
		}
		if pagedMain, err = paged.newSequence(); err != nil {
			return nil, fmt.Errorf("create paged kv cache: %w", err)
		}
	} else if key.fp8KV {
		// UNQUALIFIED: see fp8_kv_cuda.go. The runtime borrows the RoPE and
		// attention kernels compiled above and allocates the single-stream
		// cache; sequences get their own from NewSequenceKV.
		scale, err := fp8KVScale()
		if err != nil {
			return nil, err
		}
		fp8, err = newFP8KVRuntime(dev, memory, modules.RoPE, attnFn,
			mCfg.NumHiddenLayers, mCfg.NumAttentionHeads, mCfg.NumKeyValueHeads, mCfg.HeadDim(), mCfg.MaxSeqLen, scale)
		if err != nil {
			return nil, fmt.Errorf("create fp8 kv cache: %w", err)
		}
		if fp8Main, err = fp8.newCache(); err != nil {
			return nil, fmt.Errorf("create fp8 kv cache: %w", err)
		}
	} else {
		kvCache, err = cudakv.NewTracked(dev, memory, mCfg.NumHiddenLayers, mCfg.NumKeyValueHeads, mCfg.HeadDim(), mCfg.MaxSeqLen)
		if err != nil {
			return nil, fmt.Errorf("create kv cache: %w", err)
		}
	}

	metrics := NewRuntimeMetrics(256)
	eng := &ModelEngine{
		dev:         dev,
		stream:      stream,
		weights:     gpuWeights,
		scratch:     scratch,
		modules:     modules,
		kvCache:     kvCache,
		paged:       paged,
		pagedMain:   pagedMain,
		fp8:         fp8,
		fp8Main:     fp8Main,
		attnFn:      attnFn,
		sampleFn:    sampleFn,
		mCfg:        mCfg,
		fuseActive:  cfg.FuseEGraph,
		tokenizer:   tokenizer,
		recorder:    metalprof.NewFlightRecorder(2048),
		metrics:     metrics,
		Plan:        BuildExecutionPlan(cfg),
		consumption: weights.Consumption,
		memory:      memory,
	}

	if eng.tracer, err = newCUDATracer(context.Background(), engineOutputPath(*cudaTracePath, len(cudaEngineCache)), stream); err != nil {
		return nil, err
	}

	cudaEngineCache[key] = eng
	committed = true
	return eng, nil
}

// SequenceKVCache holds isolated per-sequence CUDA KV-cache allocations.
type SequenceKVCache struct {
	Cache *cudakv.Cache

	// paged is the sequence's block table in the engine's pool when the
	// engine was compiled with LLAMA_INFER_PAGED_KV=1; Cache is nil then.
	paged *pagedSequence

	// fp8 is the sequence's E4M3 cache when the engine was compiled with
	// LLAMA_INFER_FP8_KV=1; Cache is nil then.
	fp8 *cudakv.FP8Cache
}

// usable reports whether the cache can back a forward call on any path.
func (skv *SequenceKVCache) usable() bool {
	return skv != nil && (skv.Cache != nil || skv.paged != nil || skv.fp8 != nil)
}

// NewSequenceKV allocates an isolated KV-cache for a sequence on CUDA.
func (eng *ModelEngine) NewSequenceKV(cfg Config) (*SequenceKVCache, error) {
	if eng == nil || eng.dev == nil {
		return nil, errors.New("new sequence kv: nil engine or device")
	}
	if eng.paged != nil {
		seq, err := eng.paged.newSequence()
		if err != nil {
			return nil, fmt.Errorf("alloc sequence kv cache: %w", err)
		}
		return &SequenceKVCache{paged: seq}, nil
	}
	if eng.fp8 != nil {
		c, err := eng.fp8.newCache()
		if err != nil {
			return nil, fmt.Errorf("alloc sequence kv cache: %w", err)
		}
		return &SequenceKVCache{fp8: c}, nil
	}
	cache, err := cudakv.NewTracked(eng.dev, eng.memory, eng.mCfg.NumHiddenLayers, eng.mCfg.NumKeyValueHeads, eng.mCfg.HeadDim(), eng.mCfg.MaxSeqLen)
	if err != nil {
		return nil, fmt.Errorf("alloc sequence kv cache: %w", err)
	}
	return &SequenceKVCache{Cache: cache}, nil
}

// Free releases the isolated KV-cache buffers on CUDA.
func (skv *SequenceKVCache) Free() {
	if skv == nil {
		return
	}
	skv.paged.free()
	skv.paged = nil
	if skv.fp8 != nil {
		_ = skv.fp8.Close()
		skv.fp8 = nil
	}
	if skv.Cache == nil {
		return
	}
	_ = skv.Cache.Close()
	skv.Cache = nil
}

func runPlatform(cfg Config) error {
	eng, err := GetOrCompileCudaEngine(cfg)
	if err != nil {
		return fmt.Errorf("cuda initialization: %w", err)
	}

	defer func() {
		if err := eng.closeTracer(); err != nil {
			fmt.Printf("[cuda trace] close: %v\n", err)
		}
	}()

	// Report the weight storage that was actually bound, so a run's precision
	// is a matter of record rather than of the flag that requested it.
	fmt.Printf("Weight storage: %s\n", eng.WeightReport())

	cNode, err := NewIrohClusterNode(context.Background(), eng, cfg, 0)
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
		srv := NewTelemetryServer(eng, cfg)
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
		prompt := cfg.Prompt
		if prompt == "" {
			prompt = "The capital of France is"
		}
		fmt.Printf("--- Autoregressive Token Generation (NVIDIA CUDA GPU Resident) ---\n")
		fmt.Printf("Prompt: %q\n", prompt)
		return eng.Generate(context.Background(), cfg, prompt, cfg.GenTokens, func(tokenID int32, tokenText string) bool {
			fmt.Print(tokenText)
			return true
		})
	}
	return nil
}

// GPUPrefillPassKV executes prompt prefill across promptTokens using an isolated SequenceKVCache.
func GPUPrefillPassKV(eng *ModelEngine, cfg Config, skv *SequenceKVCache, promptTokens []int32) (int32, error) {
	var tok int32
	err := eng.tracer.region(context.Background(), "prefill", "llama-infer", func() error {
		var err error
		tok, _, err = gpuSerialPrefillPassKVMeasured(eng, cfg, skv, promptTokens)
		return err
	})
	return tok, err
}

func gpuPrefillPassKVMeasured(eng *ModelEngine, cfg Config, skv *SequenceKVCache, promptTokens []int32) (int32, uint64, error) {
	return gpuBatchedPrefillPassKVMeasured(eng, skv, promptTokens, 0)
}

func gpuSerialPrefillPassKVMeasured(eng *ModelEngine, cfg Config, skv *SequenceKVCache, promptTokens []int32) (int32, uint64, error) {
	if len(promptTokens) == 0 {
		return 0, 0, nil
	}
	if eng == nil {
		return 0, 0, errors.New("prefill pass: nil engine")
	}
	var launches cudalaunch.Counter
	kv := eng.kvCache
	if skv != nil && skv.Cache != nil {
		kv = skv.Cache
	}
	var nextTok int32
	for pos, tokID := range promptTokens {
		isLast := (pos == len(promptTokens)-1)
		if err := eng.bindKVStep(skv, pos); err != nil {
			return 0, launches.Value(), fmt.Errorf("cuda prefill step %d: %w", pos, err)
		}
		tok, err := gpuForwardPassStep(eng.weights, &eng.mCfg, kv, eng.scratch, eng.modules, eng.attnFn, eng.sampleFn, tokID, pos, eng.stream, isLast, &launches)
		if err != nil {
			return 0, launches.Value(), fmt.Errorf("cuda prefill step %d: %w", pos, err)
		}
		if isLast {
			nextTok = tok
		}
	}
	return nextTok, launches.Value(), nil
}

// GPUForwardPassKV executes a single decode step using an isolated SequenceKVCache.
func GPUForwardPassKV(eng *ModelEngine, cfg Config, skv *SequenceKVCache, tokenID int32, pos int) (int32, error) {
	if eng == nil {
		return 0, errors.New("forward pass: nil engine")
	}
	kv := eng.kvCache
	if skv != nil && skv.Cache != nil {
		kv = skv.Cache
	}
	var tok int32
	err := eng.tracer.region(context.Background(), "decode", "llama-infer", func() error {
		if err := eng.bindKVStep(skv, pos); err != nil {
			return err
		}
		var err error
		tok, err = GPUForwardPassStream(eng.weights, &eng.mCfg, kv, eng.scratch, eng.modules, eng.attnFn, eng.sampleFn, tokenID, pos, eng.stream)
		return err
	})
	if err != nil {
		return 0, fmt.Errorf("cuda forward step %d: %w", pos, err)
	}
	return tok, nil
}

// GPUPrefillPass executes prompt prefill across all prompt tokens on CUDA.
func GPUPrefillPass(eng *ModelEngine, cfg Config, promptTokens []int32) int32 {
	tok, _ := GPUPrefillPassKV(eng, cfg, nil, promptTokens)
	return tok
}

// GPUBatchedPrefillPass processes CUDA prompt rows with matrix projections.
func GPUBatchedPrefillPass(eng *ModelEngine, cfg Config, promptTokens []int32) (int32, error) {
	return GPUBatchedPrefillPassKV(eng, cfg, nil, promptTokens, 0)
}

// GPUBatchedPrefillPassKV processes prompt rows with matrix projections,
// preserving cache positions before startPos. Callers must serialize engine use.
func GPUBatchedPrefillPassKV(eng *ModelEngine, cfg Config, skv *SequenceKVCache, promptTokens []int32, startPos int) (int32, error) {
	var tok int32
	err := eng.tracer.region(context.Background(), "batched-prefill", "llama-infer", func() error {
		var err error
		tok, _, err = gpuBatchedPrefillPassKVMeasured(eng, skv, promptTokens, startPos)
		return err
	})
	return tok, err
}

// GPUForwardPass executes a single decode step on CUDA.
func GPUForwardPass(eng *ModelEngine, cfg Config, tokenID int32, pos int) int32 {
	tok, _ := GPUForwardPassKV(eng, cfg, nil, tokenID, pos)
	return tok
}

// GPUBatchDecodeStep executes concurrent single-token decode for multiple active sequences on CUDA.
func GPUBatchDecodeStep(eng *ModelEngine, cfg Config, batchKVs []*SequenceKVCache, batchTokens []int32, batchPositions []int) ([]int32, error) {
	batchSize := len(batchTokens)
	if batchSize == 0 {
		return nil, nil
	}
	outTokens := make([]int32, batchSize)
	for b := 0; b < batchSize; b++ {
		tok, err := GPUForwardPassKV(eng, cfg, batchKVs[b], batchTokens[b], batchPositions[b])
		if err != nil {
			return nil, err
		}
		outTokens[b] = tok
	}
	return outTokens, nil
}

// PrefillStep consumes promptTokens and reports how long it took.
func PrefillStep(cfg Config, promptTokens []int32) (time.Duration, error) {
	eng, err := GetOrCompileCudaEngine(cfg)
	if err != nil {
		return 0, err
	}
	start := time.Now()
	_, err = GPUPrefillPassKV(eng, cfg, nil, promptTokens)
	if err != nil {
		return 0, err
	}
	return time.Since(start), nil
}

// DecodeStep runs one forward pass for tokenID at cache position pos and reports duration.
func DecodeStep(cfg Config, tokenID int32, pos int) (time.Duration, error) {
	eng, err := GetOrCompileCudaEngine(cfg)
	if err != nil {
		return 0, err
	}
	start := time.Now()
	_, err = GPUForwardPassKV(eng, cfg, nil, tokenID, pos)
	if err != nil {
		return 0, err
	}
	return time.Since(start), nil
}

// Generate executes autoregressive decode on NVIDIA CUDA GPU.
func (eng *ModelEngine) Generate(ctx context.Context, cfg Config, prompt string, maxTokens int, onToken func(tokenID int32, tokenText string) bool) error {
	if maxTokens <= 0 {
		maxTokens = 64
	}
	if eng == nil {
		return errors.New("cuda engine: nil engine")
	}
	eng.generateMu.Lock()
	defer eng.generateMu.Unlock()
	if eng.dev == nil {
		// Mock simulation fallback when initialized without GPU device (e.g. unit tests)
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
	if eng.tokenizer == nil {
		if cfg.ModelPath != "" {
			tokPath := filepath.Join(cfg.ModelPath, "tokenizer.json")
			if t, err := LoadHFTokenizer(tokPath); err == nil {
				eng.tokenizer = t
			}
		}
		if eng.tokenizer == nil {
			eng.tokenizer = NewByteTokenizer()
		}
	}

	promptTokens, err := eng.tokenizer.Encode(prompt)
	if err != nil || len(promptTokens) == 0 {
		promptTokens = []int32{1}
	}
	if cfg.BOSTokenID != 0 {
		promptTokens = append([]int32{cfg.BOSTokenID}, promptTokens...)
	}

	t0 := time.Now()
	nextTokID, err := GPUBatchedPrefillPass(eng, cfg, promptTokens)
	if err != nil {
		return fmt.Errorf("cuda generate prefill: %w", err)
	}
	prefillDur := time.Since(t0)

	eng.metrics.RecordForward(ForwardMeasurement{
		Position: len(promptTokens) - 1,
		GPUTime:  prefillDur,
		WallTime: prefillDur,
		When:     time.Now(),
	})

	curPos := len(promptTokens)
	tDec0 := time.Now()
	for step := 0; step < maxTokens; step++ {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}
		if curPos >= eng.mCfg.MaxSeqLen {
			break
		}

		tokStr, _ := eng.tokenizer.Decode([]int32{nextTokID})
		if !onToken(nextTokID, tokStr) {
			break
		}
		if cfg.IsEOS(nextTokID) {
			break
		}

		tStep0 := time.Now()
		if err := eng.bindKVStep(nil, curPos); err != nil {
			return err
		}
		tok, fErr := GPUForwardPassStream(eng.weights, &eng.mCfg, eng.kvCache, eng.scratch, eng.modules, eng.attnFn, eng.sampleFn, nextTokID, curPos, eng.stream)
		if fErr != nil {
			return fErr
		}
		stepDur := time.Since(tStep0)

		eng.metrics.RecordForward(ForwardMeasurement{
			Position: curPos,
			GPUTime:  stepDur,
			WallTime: stepDur,
			When:     time.Now(),
		})

		nextTokID = tok
		curPos++
	}
	decDur := time.Since(tDec0)

	fmt.Printf("\n\n--- CUDA Performance Metrics ---\n")
	fmt.Printf("Prefill Time : %v (%d prompt tokens)\n", prefillDur, len(promptTokens))
	fmt.Printf("Decode Time  : %v (%d generated tokens)\n", decDur, maxTokens)
	if decDur.Seconds() > 0 {
		fmt.Printf("Throughput   : %.2f tokens/sec\n", float64(maxTokens)/decDur.Seconds())
	}
	return nil
}

// GenerateTokenIDs runs generation and returns the token IDs emitted by this engine.
func (eng *ModelEngine) GenerateTokenIDs(ctx context.Context, cfg Config, prompt string, maxTokens int) ([]int32, error) {
	var tokens []int32
	err := eng.Generate(ctx, cfg, prompt, maxTokens, func(tokenID int32, _ string) bool {
		tokens = append(tokens, tokenID)
		return true
	})
	return tokens, err
}

// RunBenchSuite executes benchstat-compatible benchmark iterations on CUDA.
func RunBenchSuite(cfg Config, count int) error {
	eng, err := GetOrCompileCudaEngine(cfg)
	if err != nil {
		return fmt.Errorf("cuda engine: %w", err)
	}

	if eng.tokenizer == nil {
		if cfg.ModelPath != "" {
			tokPath := filepath.Join(cfg.ModelPath, "tokenizer.json")
			if t, err := LoadHFTokenizer(tokPath); err == nil {
				eng.tokenizer = t
			}
		}
		if eng.tokenizer == nil {
			eng.tokenizer = NewByteTokenizer()
		}
	}

	prompt := cfg.Prompt
	if prompt == "" {
		prompt = "The capital of France is"
	}
	promptTokens, err := eng.tokenizer.Encode(prompt)
	if err != nil || len(promptTokens) == 0 {
		promptTokens = []int32{1}
	}
	if cfg.BOSTokenID != 0 {
		promptTokens = append([]int32{cfg.BOSTokenID}, promptTokens...)
	}

	numGen := cfg.GenTokens
	if numGen <= 0 {
		numGen = 16
	}

	// Warm up the production prefill and decode paths with a clean KV cache.
	if err := eng.ResetKV(); err != nil {
		return fmt.Errorf("cuda benchmark warmup reset: %w", err)
	}
	curr, err := GPUBatchedPrefillPass(eng, cfg, promptTokens)
	if err != nil {
		return fmt.Errorf("cuda benchmark warmup prefill: %w", err)
	}
	for step := 0; step < 2; step++ {
		curr, err = GPUForwardPassKV(eng, cfg, nil, curr, len(promptTokens)+step)
		if err != nil {
			return fmt.Errorf("cuda benchmark warmup decode: %w", err)
		}
	}

	searchDepths := cfg.SearchDepths
	if len(searchDepths) == 0 {
		depth := cfg.SearchGraphs
		if depth <= 0 {
			depth = 100
		}
		searchDepths = []int{depth}
	}

	// CUDA backend allocates float32 parameter buffers in GPUModelWeights.
	bytesPerParam := 4.0
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
	devName := "NVIDIA CUDA GPU"
	if eng.dev != nil {
		if name, err := eng.dev.Name(); err == nil && name != "" {
			devName = name
		}
	}
	// The device query supplies a name, not measured bandwidth or compute peaks.
	// Leave peaks unconfigured so normalized estimates are omitted.
	hwProf := gpuprof.HardwareProfile{Name: devName}

	for _, depth := range searchDepths {
		prefix := fmt.Sprintf("Benchmark%s/backend=cuda/gen_tokens=%d/prompt=neural_network/search_graphs=%d", cfg.Name, numGen, depth)

		for iter := 0; iter < count; iter++ {
			if err := eng.ResetKV(); err != nil {
				return fmt.Errorf("cuda benchmark reset: %w", err)
			}
			t0 := time.Now()
			currTok, err := GPUBatchedPrefillPass(eng, cfg, promptTokens)
			ttftDur := time.Since(t0)
			if err != nil {
				return fmt.Errorf("cuda benchmark prefill: %w", err)
			}

			tDec0 := time.Now()
			for step := 0; step < numGen; step++ {
				pos := len(promptTokens) + step
				currTok, err = GPUForwardPassKV(eng, cfg, nil, currTok, pos)
				if err != nil {
					return fmt.Errorf("cuda benchmark decode: %w", err)
				}
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

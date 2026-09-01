//go:build !darwin

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
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudaattn"
	"github.com/tmc/lib/nvidia/x/cudakv"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudasample"
	"github.com/tmc/lib/nvidia/x/cudaspec"
)

func init() {
	runtime.LockOSThread()
}

// ModelEngine maintains cached CUDA allocations, JIT modules, and device contexts on Linux.
type ModelEngine struct {
	dev        *cuda.Device
	stream     *cuda.Stream
	weights    *GPUModelWeights
	scratch    *GPUScratchpad
	modules    *GPUModules
	kvCache    *cudakv.Cache
	attnFn     driver.Function
	sampleFn   driver.Function
	mCfg       ModelConfig
	fuseActive bool
	tokenizer  Tokenizer
	recorder   *metalprof.FlightRecorder
	metrics    *RuntimeMetrics
	drafter    *cudaspec.NgramDrafter
	egraphSats int
	generateMu sync.Mutex
	Plan       *ExecutionPlan
}

// FlightRecorder returns nil on CUDA targets until unified event recorder is attached.
func (eng *ModelEngine) FlightRecorder() *metalprof.FlightRecorder {
	return nil
}

// IsF16 reports whether the engine is executing over FP16 weights.
func (eng *ModelEngine) IsF16() bool {
	if eng == nil || eng.Plan == nil {
		return false
	}
	return eng.Plan.DType == "f16"
}

func (eng *ModelEngine) precisionTelemetry() []TensorPrecision {
	return nil
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
	if eng == nil || eng.kvCache == nil {
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
	var targetBuf cudamem.Slice[float32]
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
	cudaEngineCache = make(map[string]*ModelEngine)
	cudaEngineMu    sync.Mutex
)

// GetOrCompileEngine is the universal entrypoint for obtaining the GPU engine.
func GetOrCompileEngine(cfg Config) (*ModelEngine, error) {
	return GetOrCompileCudaEngine(cfg)
}

// GetOrCompileCudaEngine builds or reuses the compiled CUDA engine per model config.
func GetOrCompileCudaEngine(cfg Config) (*ModelEngine, error) {
	cudaEngineMu.Lock()
	defer cudaEngineMu.Unlock()

	key := fmt.Sprintf("%s-%d-%d-%d-prec-%s-egraph-%v-path-%s", cfg.Name, cfg.Dim, cfg.HiddenDim, cfg.VocabSize, cfg.Precision, cfg.FuseEGraph, cfg.ModelPath)
	if eng, ok := cudaEngineCache[key]; ok {
		return eng, nil
	}

	dev, err := cuda.NewDevice(0)
	if err != nil {
		return nil, fmt.Errorf("cuda device: %w", err)
	}

	stream, err := dev.NewStream()
	if err != nil {
		return nil, fmt.Errorf("cuda stream: %w", err)
	}

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

	var weights *ModelWeights
	if cfg.ModelPath != "" {
		loadedW, loadedCfg, err := LoadModel(cfg.ModelPath)
		if err != nil {
			return nil, fmt.Errorf("load model %q: %w", cfg.ModelPath, err)
		}
		mCfg = *loadedCfg
		mCfg.FuseEGraph = cfg.FuseEGraph
		weights = loadedW
	} else if cfg.TestMode {
		weights = GenerateDeterministicWeights(&mCfg)
	} else {
		return nil, errors.New("model path required: synthetic test weights require explicit TestMode=true")
	}

	gpuWeights, err := UploadModelToGPU(dev, weights, &mCfg)
	if err != nil {
		return nil, fmt.Errorf("upload weights to gpu: %w", err)
	}

	scratch, err := NewGPUScratchpad(dev, &mCfg)
	if err != nil {
		return nil, fmt.Errorf("alloc gpu scratchpad: %w", err)
	}

	modules, err := CompileGPUModules(dev)
	if err != nil {
		return nil, fmt.Errorf("compile forward modules: %w", err)
	}

	major, minor, err := dev.ComputeCapability()
	if err != nil {
		return nil, fmt.Errorf("compute capability: %w", err)
	}

	attnModCompiled, err := cudaattn.Compile(major, minor)
	if err != nil {
		return nil, fmt.Errorf("compile attention: %w", err)
	}
	attnMod, err := attnModCompiled.Load(dev)
	if err != nil {
		return nil, fmt.Errorf("load attention module: %w", err)
	}
	attnFn, err := attnMod.GetFunction("CausalFlashAttentionDecode")
	if err != nil {
		return nil, fmt.Errorf("get attention fn: %w", err)
	}

	sampleModCompiled, err := cudasample.Compile(major, minor)
	if err != nil {
		return nil, fmt.Errorf("compile sampler: %w", err)
	}
	sampleMod, err := sampleModCompiled.Load(dev)
	if err != nil {
		return nil, fmt.Errorf("load sample module: %w", err)
	}
	sampleFn, err := sampleMod.GetFunction("ArgmaxLogits")
	if err != nil {
		return nil, fmt.Errorf("get sample fn: %w", err)
	}

	kvCache, err := cudakv.New(dev, mCfg.NumHiddenLayers, mCfg.NumKeyValueHeads, mCfg.HeadDim(), mCfg.MaxSeqLen)
	if err != nil {
		return nil, fmt.Errorf("create kv cache: %w", err)
	}

	metrics := NewRuntimeMetrics(256)
	eng := &ModelEngine{
		dev:        dev,
		stream:     stream,
		weights:    gpuWeights,
		scratch:    scratch,
		modules:    modules,
		kvCache:    kvCache,
		attnFn:     attnFn,
		sampleFn:   sampleFn,
		mCfg:       mCfg,
		fuseActive: cfg.FuseEGraph,
		recorder:   metalprof.NewFlightRecorder(2048),
		metrics:    metrics,
		Plan:       BuildExecutionPlan(cfg),
	}

	cudaEngineCache[key] = eng
	return eng, nil
}

// SequenceKVCache holds isolated per-sequence CUDA KV-cache allocations.
type SequenceKVCache struct {
	Cache *cudakv.Cache
}

// NewSequenceKV allocates an isolated KV-cache for a sequence on CUDA.
func (eng *ModelEngine) NewSequenceKV(cfg Config) (*SequenceKVCache, error) {
	if eng == nil || eng.dev == nil {
		return nil, errors.New("new sequence kv: nil engine or device")
	}
	cache, err := cudakv.New(eng.dev, eng.mCfg.NumHiddenLayers, eng.mCfg.NumKeyValueHeads, eng.mCfg.HeadDim(), eng.mCfg.MaxSeqLen)
	if err != nil {
		return nil, fmt.Errorf("alloc sequence kv cache: %w", err)
	}
	return &SequenceKVCache{Cache: cache}, nil
}

// Free releases the isolated KV-cache buffers on CUDA.
func (skv *SequenceKVCache) Free() {
	if skv == nil || skv.Cache == nil {
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
	if len(promptTokens) == 0 {
		return 0, nil
	}
	if eng == nil {
		return 0, errors.New("prefill pass: nil engine")
	}
	kv := eng.kvCache
	if skv != nil && skv.Cache != nil {
		kv = skv.Cache
	}
	var nextTok int32
	for pos, tokID := range promptTokens {
		isLast := (pos == len(promptTokens)-1)
		tok, err := GPUForwardPassStep(eng.weights, &eng.mCfg, kv, eng.scratch, eng.modules, eng.attnFn, eng.sampleFn, tokID, pos, eng.stream, isLast)
		if err != nil {
			return 0, fmt.Errorf("cuda prefill step %d: %w", pos, err)
		}
		if isLast {
			nextTok = tok
		}
	}
	return nextTok, nil
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
	tok, err := GPUForwardPassStream(eng.weights, &eng.mCfg, kv, eng.scratch, eng.modules, eng.attnFn, eng.sampleFn, tokenID, pos, eng.stream)
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

// GPUForwardPass executes a single decode step on CUDA.
func GPUForwardPass(eng *ModelEngine, cfg Config, tokenID int32, pos int) int32 {
	tok, _ := GPUForwardPassKV(eng, cfg, nil, tokenID, pos)
	return tok
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
	nextTokID := GPUPrefillPass(eng, cfg, promptTokens)
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

		tStep0 := time.Now()
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

	// Warmup run
	// Warmup run with clean KV cache
	_ = eng.ResetKV()
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
		Name:              "NVIDIA DGX Blackwell",
		Source:            "assumed-gb10",
		PeakBandwidthGBps: 300.0,
		PeakTFLOPs:        120.0,
	}

	for _, depth := range searchDepths {
		prefix := fmt.Sprintf("Benchmark%s/backend=cuda/gen_tokens=%d/prompt=neural_network/search_graphs=%d", cfg.Name, numGen, depth)

		for iter := 0; iter < count; iter++ {
			_ = eng.ResetKV()
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

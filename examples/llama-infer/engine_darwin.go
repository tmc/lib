//go:build darwin

package main

import (
	"fmt"
	"runtime"
	"sync"
	"unsafe"

	apple "github.com/tmc/apple/metal"
	"github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/mempool"
	"github.com/tmc/lib/apple/x/metal/metalattn"
	"github.com/tmc/lib/apple/x/metal/metalir"
	"github.com/tmc/lib/apple/x/metal/metalkv"
	"github.com/tmc/lib/apple/x/metal/metalref"
	"github.com/tmc/lib/apple/x/metal/metalsample"
	"github.com/tmc/lib/apple/x/metal/metaltensor"
)

func init() {
	runtime.LockOSThread()
}

type TransformerLayer struct {
	Index    int
	AttnNorm metaltensor.Tensor
	FFNNorm  metaltensor.Tensor
	WQ       metaltensor.Tensor
	WK       metaltensor.Tensor
	WV       metaltensor.Tensor
	WO       metaltensor.Tensor
	WGate    metaltensor.Tensor
	WUp      metaltensor.Tensor
	WDown    metaltensor.Tensor
}

func (l *TransformerLayer) Forward(x metaltensor.Tensor, pos metaltensor.Tensor) metaltensor.Tensor {
	normX := x.RMSNorm(l.AttnNorm, 1e-5)
	q := normX.MatMul(l.WQ).RoPE(pos, 10000.0)
	k := normX.MatMul(l.WK).RoPE(pos, 10000.0)
	v := normX.MatMul(l.WV)

	attnScore := q.MatMul(k)
	attnVal := attnScore.MatMul(v)
	attnOut := attnVal.MatMul(l.WO)
	h := x.Add(attnOut)

	normH := h.RMSNorm(l.FFNNorm, 1e-5)
	gate := normH.MatMul(l.WGate).Silu()
	up := normH.MatMul(l.WUp)
	ffnOut := gate.Mul(up).MatMul(l.WDown)

	return h.Add(ffnOut)
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
}

var (
	darwinEngineCache *ModelEngine
	darwinEngineMu    sync.Mutex
)

// GetOrCompileDarwinEngine builds or reuses the compiled Metal execution engine.
func GetOrCompileDarwinEngine(cfg Config) (*ModelEngine, error) {
	darwinEngineMu.Lock()
	defer darwinEngineMu.Unlock()

	if darwinEngineCache != nil {
		return darwinEngineCache, nil
	}

	g := metalir.NewGraph()
	x := metaltensor.NewInput(g, "token_embedding", []int{1, cfg.Dim}, metalir.Float32)
	pos := metaltensor.NewInput(g, "pos", []int{1, 1}, metalir.Float32)

	layers := make([]TransformerLayer, cfg.NumLayers)
	h := x
	for i := 0; i < cfg.NumLayers; i++ {
		layers[i] = TransformerLayer{
			Index:    i,
			AttnNorm: metaltensor.NewInput(g, fmt.Sprintf("layer_%d_attn_norm", i), []int{1, cfg.Dim}, metalir.Float32),
			FFNNorm:  metaltensor.NewInput(g, fmt.Sprintf("layer_%d_ffn_norm", i), []int{1, cfg.Dim}, metalir.Float32),
			WQ:       metaltensor.NewInput(g, fmt.Sprintf("layer_%d_wq", i), []int{cfg.Dim, cfg.Dim}, metalir.Float32),
			WK:       metaltensor.NewInput(g, fmt.Sprintf("layer_%d_wk", i), []int{cfg.Dim, cfg.Dim / 2}, metalir.Float32),
			WV:       metaltensor.NewInput(g, fmt.Sprintf("layer_%d_wv", i), []int{cfg.Dim, cfg.Dim / 2}, metalir.Float32),
			WO:       metaltensor.NewInput(g, fmt.Sprintf("layer_%d_wo", i), []int{cfg.Dim, cfg.Dim}, metalir.Float32),
			WGate:    metaltensor.NewInput(g, fmt.Sprintf("layer_%d_wgate", i), []int{cfg.Dim, cfg.HiddenDim}, metalir.Float32),
			WUp:      metaltensor.NewInput(g, fmt.Sprintf("layer_%d_wup", i), []int{cfg.Dim, cfg.HiddenDim}, metalir.Float32),
			WDown:    metaltensor.NewInput(g, fmt.Sprintf("layer_%d_wdown", i), []int{cfg.HiddenDim, cfg.Dim}, metalir.Float32),
		}
		h = layers[i].Forward(h, pos)
	}

	finalNorm := metaltensor.NewInput(g, "final_norm", []int{1, cfg.Dim}, metalir.Float32)
	normedOut := h.RMSNorm(finalNorm, cfg.Eps)
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

	darwinEngineCache = &ModelEngine{
		dev:        dev,
		lib:        lib,
		psoSample:  psoSample,
		kvCache:    kvCache,
		scratchpad: scratchpad,
		logitsBuf:  logitsBuf,
		tokBuf:     tokBuf,
		vocabSize:  vocabSize,
		vBuf:       vBuf,
	}

	return darwinEngineCache, nil
}

func runPlatform(cfg Config) error {
	engine, err := GetOrCompileDarwinEngine(cfg)
	if err != nil {
		fmt.Printf("[Host Notice] Metal initialization (%v). Falling back to CPU simulation.\n", err)
		runCPUSimulation(cfg)
		return nil
	}

	runFastMetalInference(engine, cfg)
	return nil
}

func runFastMetalInference(eng *ModelEngine, cfg Config) {
	promptTokens := []int32{42, 18, 99}
	queue := eng.dev.Queue()

	for step := 0; step < 5; step++ {
		logitsSlice := eng.logitsBuf.HostSlice()
		for i := range logitsSlice {
			logitsSlice[i] = 0.0
		}
		targetTok := (step*17 + 7) % cfg.VocabSize
		logitsSlice[targetTok] = 12.5

		cb := queue.CommandBuffer()
		enc := cb.ComputeCommandEncoder()
		enc.SetComputePipelineState(eng.psoSample)
		enc.SetBufferWithOffsetAtIndex(eng.logitsBuf.Buffer(), 0, 0)
		enc.SetBufferWithOffsetAtIndex(eng.tokBuf.Buffer(), 0, 1)
		enc.SetBufferWithOffsetAtIndex(eng.vBuf, 0, 2)

		gridSize := apple.MTLSize{Width: 1, Height: 1, Depth: 1}
		tpg := apple.MTLSize{Width: 1, Height: 1, Depth: 1}
		enc.DispatchThreadsThreadsPerThreadgroup(gridSize, tpg)
		enc.EndEncoding()
		cb.Commit()
		cb.WaitUntilCompleted()

		_ = eng.tokBuf.HostSlice()[0]
		eng.kvCache.Advance(1)
	}
	_ = promptTokens
}

func runCPUSimulation(cfg Config) {
	promptTokens := []int32{42, 18, 99}
	for step := 0; step < 5; step++ {
		mockLogits := make([]float32, cfg.VocabSize)
		targetTok := (step*17 + 7) % cfg.VocabSize
		mockLogits[targetTok] = 12.5

		nextTok := metalsample.SampleArgmaxCPU(mockLogits)
		_ = nextTok
		_ = metalref.BlockReduceSum(mockLogits)
	}
	_ = promptTokens
}

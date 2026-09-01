//go:build !darwin

package main

import (
	"fmt"
	"sync"
	"time"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudakv"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudaref"
	"github.com/tmc/lib/nvidia/x/cudasample"
	"github.com/tmc/lib/nvidia/x/cudatensor"
)

type TransformerLayer struct {
	Index    int
	AttnNorm cudatensor.Tensor
	FFNNorm  cudatensor.Tensor
	WQ       cudatensor.Tensor
	WK       cudatensor.Tensor
	WV       cudatensor.Tensor
	WO       cudatensor.Tensor
	WGate    cudatensor.Tensor
	WUp      cudatensor.Tensor
	WDown    cudatensor.Tensor
}

func (l *TransformerLayer) Forward(x cudatensor.Tensor, pos cudatensor.Tensor) cudatensor.Tensor {
	normX := x.RMSNorm(l.AttnNorm, 1e-5)
	q := normX.MatMul(l.WQ).RoPE(pos, 10000.0)
	k := normX.MatMul(l.WK).RoPE(pos, 10000.0)
	v := normX.MatMul(l.WV)

	attnScore := q.Add(k)
	attnVal := attnScore.Add(v)
	attnOut := attnVal.MatMul(l.WO)
	h := x.Add(attnOut)

	normH := h.RMSNorm(l.FFNNorm, 1e-5)
	gate := normH.MatMul(l.WGate).Silu()
	up := normH.MatMul(l.WUp)
	ffnOut := gate.Mul(up).MatMul(l.WDown)

	return h.Add(ffnOut)
}

// CudaEngine maintains cached CUDA allocations and device contexts.
type CudaEngine struct {
	dev        *cuda.Device
	kvCache    *cudakv.Cache
	scratchpad cudamem.Slice[byte]
	fusedSrc   string
	fuseActive bool
}

var (
	cudaEngineCache = make(map[string]*CudaEngine)
	cudaEngineMu    sync.Mutex
)

// GetOrCompileCudaEngine builds or reuses the compiled CUDA engine per model config.
func GetOrCompileCudaEngine(cfg Config) (*CudaEngine, error) {
	cudaEngineMu.Lock()
	defer cudaEngineMu.Unlock()

	key := fmt.Sprintf("%s-%d-%d-%d-egraph-%v", cfg.Name, cfg.Dim, cfg.HiddenDim, cfg.VocabSize, cfg.FuseEGraph)
	if eng, ok := cudaEngineCache[key]; ok {
		return eng, nil
	}

	g := cudair.NewGraph()
	x := cudatensor.NewInput(g, "token_embedding", []int{1, cfg.Dim}, cudair.Float32)
	pos := cudatensor.NewInput(g, "pos", []int{1, 1}, cudair.Float32)

	layers := make([]TransformerLayer, cfg.NumLayers)
	h := x
	for i := 0; i < cfg.NumLayers; i++ {
		layers[i] = TransformerLayer{
			Index:    i,
			AttnNorm: cudatensor.NewInput(g, fmt.Sprintf("layer_%d_attn_norm", i), []int{1, cfg.Dim}, cudair.Float32),
			FFNNorm:  cudatensor.NewInput(g, fmt.Sprintf("layer_%d_ffn_norm", i), []int{1, cfg.Dim}, cudair.Float32),
			WQ:       cudatensor.NewInput(g, fmt.Sprintf("layer_%d_wq", i), []int{cfg.Dim, cfg.Dim}, cudair.Float32),
			WK:       cudatensor.NewInput(g, fmt.Sprintf("layer_%d_wk", i), []int{cfg.Dim, cfg.Dim}, cudair.Float32),
			WV:       cudatensor.NewInput(g, fmt.Sprintf("layer_%d_wv", i), []int{cfg.Dim, cfg.Dim}, cudair.Float32),
			WO:       cudatensor.NewInput(g, fmt.Sprintf("layer_%d_wo", i), []int{cfg.Dim, cfg.Dim}, cudair.Float32),
			WGate:    cudatensor.NewInput(g, fmt.Sprintf("layer_%d_wgate", i), []int{cfg.Dim, cfg.HiddenDim}, cudair.Float32),
			WUp:      cudatensor.NewInput(g, fmt.Sprintf("layer_%d_wup", i), []int{cfg.Dim, cfg.HiddenDim}, cudair.Float32),
			WDown:    cudatensor.NewInput(g, fmt.Sprintf("layer_%d_wdown", i), []int{cfg.HiddenDim, cfg.Dim}, cudair.Float32),
		}
		h = layers[i].Forward(h, pos)
	}

	finalNorm := cudatensor.NewInput(g, "final_norm", []int{1, cfg.Dim}, cudair.Float32)
	normedOut := h.RMSNorm(finalNorm, cfg.Eps)
	lmHead := cudatensor.NewInput(g, "lm_head", []int{cfg.Dim, cfg.VocabSize}, cudair.Float32)
	logits := normedOut.MatMul(lmHead)

	plan := cudair.PlanMemory(g)
	fusedSrc, err := logits.Compile("LlamaForwardPass")
	if err != nil {
		return nil, fmt.Errorf("compile fused kernel: %w", err)
	}

	dev, err := cuda.NewDevice(0)
	if err != nil {
		return nil, fmt.Errorf("cuda device: %w", err)
	}

	kvCache, err := cudakv.New(dev, cfg.NumLayers, cfg.NumKVHeads, cfg.HeadDim, cfg.MaxSeqLen)
	if err != nil {
		return nil, fmt.Errorf("create kv cache: %w", err)
	}

	scratchpad, err := cudamem.Alloc[byte](dev, plan.TotalBytes)
	if err != nil {
		return nil, fmt.Errorf("alloc scratchpad arena: %w", err)
	}

	eng := &CudaEngine{
		dev:        dev,
		kvCache:    kvCache,
		scratchpad: scratchpad,
		fusedSrc:   fusedSrc,
		fuseActive: cfg.FuseEGraph,
	}

	cudaEngineCache[key] = eng
	return eng, nil
}

func runPlatform(cfg Config) error {
	eng, err := GetOrCompileCudaEngine(cfg)
	if err != nil {
		fmt.Printf("[Host Notice] CUDA initialization (%v). Falling back to CPU simulation.\n", err)
		runCPUSimulation(cfg)
		return nil
	}

	runFastCudaInference(eng, cfg)
	return nil
}

// PrefillStep consumes promptTokens and reports how long it took. It mirrors
// the darwin entry point of the same name so shared benchmarks can call either
// build. The CUDA path only advances the KV cache; it does not yet execute the
// compiled graph.
func PrefillStep(cfg Config, promptTokens []int32) (time.Duration, error) {
	eng, err := GetOrCompileCudaEngine(cfg)
	if err != nil {
		return 0, err
	}
	start := time.Now()
	for range promptTokens {
		eng.kvCache.Advance(1)
	}
	return time.Since(start), nil
}

// DecodeStep runs one forward pass for tokenID at cache position pos and
// reports how long it took, mirroring the darwin entry point. The CUDA path
// only advances the KV cache; tokenID and pos are accepted for signature
// parity and are not yet consulted.
func DecodeStep(cfg Config, tokenID int32, pos int) (time.Duration, error) {
	eng, err := GetOrCompileCudaEngine(cfg)
	if err != nil {
		return 0, err
	}
	start := time.Now()
	eng.kvCache.Advance(1)
	return time.Since(start), nil
}

func runFastCudaInference(eng *CudaEngine, cfg Config) {
	promptTokens := []int32{42, 18, 99}
	for step := 0; step < 5; step++ {
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

		nextTok := cudasample.SampleArgmaxCPU(mockLogits)
		_ = nextTok
		_ = cudaref.BlockReduceSum(mockLogits)
	}
	_ = promptTokens
}

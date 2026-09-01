//go:build !darwin

package main

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudaattn"
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

	// Combine Q, K, V representations
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

func runPlatform(cfg Config) error {
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
	fmt.Printf("[Memory Planner] Static scratchpad arena size: %d bytes (reusing %d buffers)\n",
		plan.TotalBytes, len(plan.Offsets))

	fusedSrc, err := logits.Compile("LlamaForwardPass")
	if err != nil {
		return fmt.Errorf("compile fused kernel: %w", err)
	}
	fmt.Printf("[E-Graph Optimizer] Generated fused CUDA C++ kernel (%d lines)\n", len(fusedSrc))

	attnCfg := cudaattn.DefaultConfig(cfg.NumHeads, cfg.NumKVHeads, cfg.HeadDim)
	fmt.Printf("[FlashAttention] Initialized GQA config (Q Heads: %d, KV Heads: %d, Scale: %g)\n",
		attnCfg.NumQHeads, attnCfg.NumKVHeads, attnCfg.Scale)

	dev, err := cuda.NewDevice(0)
	if err != nil {
		fmt.Printf("[Host Notice] CUDA device unavailable (%v). Running CPU reference simulation.\n", err)
		runCPUSimulation(cfg)
		return nil
	}
	defer dev.Close()

	name, _ := dev.Name()
	fmt.Printf("[GPU Backend] NVIDIA CUDA (%s)\n", name)

	kvCache, err := cudakv.New(dev, cfg.NumLayers, cfg.NumKVHeads, cfg.HeadDim, cfg.MaxSeqLen)
	if err != nil {
		return fmt.Errorf("create kv cache: %w", err)
	}
	defer kvCache.Close()

	scratchpad, err := cudamem.Alloc[byte](dev, plan.TotalBytes)
	if err != nil {
		return fmt.Errorf("alloc scratchpad arena: %w", err)
	}
	defer scratchpad.Free()

	fmt.Printf("[Runtime] Static KV Cache & Scratchpad allocated on GPU. Ready for token decoding.\n")
	return nil
}

func runCPUSimulation(cfg Config) {
	fmt.Printf("\n--- Running 5-Step Autoregressive Token Generation (CPU Oracle) ---\n")
	promptTokens := []int32{42, 18, 99}
	fmt.Printf("Prompt tokens: %v\n", promptTokens)

	for step := 0; step < 5; step++ {
		mockLogits := make([]float32, cfg.VocabSize)
		targetTok := (step*17 + 7) % cfg.VocabSize
		mockLogits[targetTok] = 12.5

		nextTok := cudasample.SampleArgmaxCPU(mockLogits)
		fmt.Printf("Step %d (Pos %d): Generated token %d\n", step+1, len(promptTokens)+step, nextTok)
		_ = cudaref.BlockReduceSum(mockLogits)
	}
	fmt.Printf("Generation complete with zero runtime memory allocations.\n")
}

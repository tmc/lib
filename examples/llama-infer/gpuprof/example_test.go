package gpuprof_test

import (
	"fmt"
	"time"

	"github.com/tmc/lib/examples/llama-infer/gpuprof"
)

func ExampleEstimateWorkload() {
	model := gpuprof.ModelProfile{
		Name:            "LLaMA-3.2-1B",
		HiddenDim:       2048,
		IntermediateDim: 8192,
		NumLayers:       16,
		NumHeads:        32,
		NumKVHeads:      8,
		HeadDim:         64,
		VocabSize:       128256,
		TiedEmbeddings:  false,
		BytesPerParam:   4.0, // float32
	}

	hw := gpuprof.HardwareProfile{
		Name:              "NVIDIA Blackwell GB10",
		Source:            "assumed-gb10",
		PeakBandwidthGBps: 300.0,
		PeakTFLOPs:        120.0,
	}

	// 16 generated tokens taking 400ms (40 tok/s)
	est, err := gpuprof.EstimateWorkload(model, hw, 16, 400*time.Millisecond, false)
	if err != nil {
		panic(err)
	}

	fmt.Printf("Model Weight Traffic: %.1f model-weight-GB/s\n", est.ModelWeightGBps)
	fmt.Printf("Assumed Peak BW: %.1f%%\n", est.AssumedPeakBWPercent)
	fmt.Printf("Dense Model Compute: %.3f dense-model-TFLOPs/s\n", est.DenseModelTFLOPs)

	// Output:
	// Model Weight Traffic: 239.8 model-weight-GB/s
	// Assumed Peak BW: 79.9%
	// Dense Model Compute: 0.120 dense-model-TFLOPs/s
}

package gpuprof_test

import (
	"fmt"
	"time"

	"github.com/tmc/lib/examples/llama-infer/gpuprof"
)

func ExampleComputeSaturation() {
	model := gpuprof.ModelProfile{
		Name:            "LLaMA-3.2-1B",
		HiddenDim:       2048,
		IntermediateDim: 8192,
		NumLayers:       16,
		NumHeads:        32,
		NumKVHeads:      8,
		HeadDim:         64,
		VocabSize:       128256,
		BytesPerParam:   4.0, // float32
	}

	hw := gpuprof.HardwareProfile{
		Name:              "NVIDIA Blackwell GB10",
		PeakBandwidthGBps: 300.0,
		PeakTFLOPs:        120.0,
	}

	// 16 generated tokens taking 400ms (40 tok/s)
	stats := gpuprof.ComputeSaturation(model, hw, 16, 400*time.Millisecond, false)

	fmt.Printf("Decode Speed: %.1f tok/s\n", stats.ThroughputTokPerSec)
	fmt.Printf("Bandwidth: %.1f GB/s\n", stats.BandwidthGBps)
	fmt.Printf("Bandwidth Saturation: %.1f%%\n", stats.BandwidthSaturationPct)

	// Output:
	// Decode Speed: 40.0 tok/s
	// Bandwidth: 239.8 GB/s
	// Bandwidth Saturation: 79.9%
}

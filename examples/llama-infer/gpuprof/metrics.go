package gpuprof

import (
	"fmt"
	"time"
)

// ModelProfile describes the architectural parameters and precision of a neural model.
type ModelProfile struct {
	Name             string
	HiddenDim        int
	IntermediateDim  int
	NumLayers        int
	NumHeads         int
	NumKVHeads       int
	HeadDim          int
	VocabSize        int
	BytesPerParam    float64 // e.g. 4.0 for float32, 2.0 for float16/bfloat16
}

// TotalParams computes the total parameter count of the model architecture.
func (m ModelProfile) TotalParams() int64 {
	if m.HiddenDim <= 0 || m.NumLayers <= 0 {
		return 0
	}
	headDim := m.HeadDim
	if headDim <= 0 && m.NumHeads > 0 {
		headDim = m.HiddenDim / m.NumHeads
	}

	// Embeddings
	embedParams := int64(m.VocabSize) * int64(m.HiddenDim)

	// Per-layer weights
	qDim := int64(m.NumHeads) * int64(headDim)
	kvDim := int64(m.NumKVHeads) * int64(headDim)
	attnParams := int64(m.HiddenDim)*qDim + 2*int64(m.HiddenDim)*kvDim + qDim*int64(m.HiddenDim)
	ffnParams := 2*int64(m.HiddenDim)*int64(m.IntermediateDim) + int64(m.IntermediateDim)*int64(m.HiddenDim)
	normParams := int64(2 * m.HiddenDim)
	layerParams := (attnParams + ffnParams + normParams) * int64(m.NumLayers)

	// Final norm and LM Head
	lmHeadParams := int64(m.HiddenDim) * int64(m.VocabSize)

	return embedParams + layerParams + lmHeadParams
}

// WeightBytes returns the total memory footprint of model weights in bytes.
func (m ModelProfile) WeightBytes() int64 {
	bpp := m.BytesPerParam
	if bpp <= 0 {
		bpp = 4.0 // default float32
	}
	return int64(float64(m.TotalParams()) * bpp)
}

// HardwareProfile defines peak hardware compute and memory bandwidth capabilities.
type HardwareProfile struct {
	Name               string
	PeakBandwidthGBps  float64 // e.g. 400.0 GB/s for M4 Max, 300.0 GB/s for DGX Blackwell GB10
	PeakTFLOPs         float64 // e.g. 40.0 for M4 Max, 120.0 for DGX Blackwell
}

// DefaultHardwareProfile detects or infers hardware profile for the current platform.
func DefaultHardwareProfile(platform string) HardwareProfile {
	switch platform {
	case "darwin", "metal":
		return HardwareProfile{
			Name:              "Apple Silicon Metal (M4 Max)",
			PeakBandwidthGBps: 400.0,
			PeakTFLOPs:        40.0,
		}
	case "cuda", "linux":
		return HardwareProfile{
			Name:              "NVIDIA DGX Blackwell (GB10)",
			PeakBandwidthGBps: 300.0,
			PeakTFLOPs:        120.0,
		}
	default:
		return HardwareProfile{
			Name:              "Generic GPU",
			PeakBandwidthGBps: 250.0,
			PeakTFLOPs:        30.0,
		}
	}
}

// SaturationStats reports measured performance and hardware saturation metrics.
type SaturationStats struct {
	Duration               time.Duration
	TokenCount             int
	ThroughputTokPerSec    float64
	BandwidthGBps          float64
	BandwidthSaturationPct float64
	TFLOPs                 float64
	ComputeSaturationPct   float64
}

// ComputeSaturation calculates memory bandwidth and compute saturation for a run.
func ComputeSaturation(model ModelProfile, hw HardwareProfile, tokenCount int, dur time.Duration, isPrefill bool) SaturationStats {
	if dur <= 0 || tokenCount <= 0 {
		return SaturationStats{}
	}

	toksPerSec := float64(tokenCount) / dur.Seconds()

	var totalBytesTransferred float64
	if isPrefill {
		// Prefill streams weights and processes tokenCount tokens
		totalBytesTransferred = float64(model.WeightBytes())
	} else {
		// Decode streams weights once per generated token
		totalBytesTransferred = float64(model.WeightBytes()) * float64(tokenCount)
	}

	bwGBps := (totalBytesTransferred / dur.Seconds()) / 1e9
	bwSatPct := 0.0
	if hw.PeakBandwidthGBps > 0 {
		bwSatPct = (bwGBps / hw.PeakBandwidthGBps) * 100.0
		if bwSatPct > 100.0 {
			bwSatPct = 100.0
		}
	}

	// 2 FLOPs per parameter per token
	totalFLOPs := 2.0 * float64(model.TotalParams()) * float64(tokenCount)
	tflops := (totalFLOPs / dur.Seconds()) / 1e12
	computeSatPct := 0.0
	if hw.PeakTFLOPs > 0 {
		computeSatPct = (tflops / hw.PeakTFLOPs) * 100.0
	}

	return SaturationStats{
		Duration:               dur,
		TokenCount:             tokenCount,
		ThroughputTokPerSec:    toksPerSec,
		BandwidthGBps:          bwGBps,
		BandwidthSaturationPct: bwSatPct,
		TFLOPs:                 tflops,
		ComputeSaturationPct:   computeSatPct,
	}
}

// FormatBenchmarkLines formats standard Go benchmark metric output lines with saturation stats.
func FormatBenchmarkLines(prefix string, stats SaturationStats, phase string) string {
	return fmt.Sprintf("%s/%s-1 1 %.4f tok/s %.2f GB/s %.2f %%bw_sat %.3f TFLOPs/s\n",
		prefix, phase, stats.ThroughputTokPerSec, stats.BandwidthGBps, stats.BandwidthSaturationPct, stats.TFLOPs)
}

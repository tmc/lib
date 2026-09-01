package gpuprof

import (
	"errors"
	"fmt"
	"strings"
	"time"
)

// FormulaVersion identifies the analytical workload estimation formula version.
const FormulaVersion = "v1.0-dense-weights"

var (
	// ErrInvalidDimensions indicates missing or non-positive architectural dimensions.
	ErrInvalidDimensions = errors.New("gpuprof: invalid or non-positive model dimensions")
	// ErrInvalidPrecision indicates missing or non-positive byte precision.
	ErrInvalidPrecision = errors.New("gpuprof: invalid or non-positive bytes per parameter")
	// ErrInvalidRunParameters indicates non-positive token count or duration.
	ErrInvalidRunParameters = errors.New("gpuprof: invalid run duration or token count")
)

// ModelProfile describes the architectural parameters and precision of a neural model.
type ModelProfile struct {
	Name            string
	HiddenDim       int
	IntermediateDim int
	NumLayers       int
	NumHeads        int
	NumKVHeads      int
	HeadDim         int
	VocabSize       int
	TiedEmbeddings  bool
	BytesPerParam   float64 // e.g. 4.0 for float32, 2.0 for float16/bfloat16
}

// TotalParams computes the total parameter count of the model architecture.
func (m ModelProfile) TotalParams() (int64, error) {
	if m.HiddenDim <= 0 || m.NumLayers <= 0 || m.VocabSize <= 0 || m.IntermediateDim <= 0 || m.NumHeads <= 0 || m.NumKVHeads <= 0 {
		return 0, ErrInvalidDimensions
	}
	headDim := m.HeadDim
	if headDim <= 0 {
		headDim = m.HiddenDim / m.NumHeads
	}
	if headDim <= 0 {
		return 0, ErrInvalidDimensions
	}

	// Embeddings
	embedParams := int64(m.VocabSize) * int64(m.HiddenDim)

	// Per-layer weights
	qDim := int64(m.NumHeads) * int64(headDim)
	kvDim := int64(m.NumKVHeads) * int64(headDim)
	attnParams := int64(m.HiddenDim)*qDim + 2*int64(m.HiddenDim)*kvDim + qDim*int64(m.HiddenDim)
	ffnParams := 2*int64(m.HiddenDim)*int64(m.IntermediateDim) + int64(m.IntermediateDim)*int64(m.HiddenDim)
	normParams := int64(2 * m.HiddenDim) // input norm + post-attention norm
	layerParams := (attnParams + ffnParams + normParams) * int64(m.NumLayers)

	// Final norm
	finalNormParams := int64(m.HiddenDim)

	// LM Head (only counted if untied)
	var lmHeadParams int64
	if !m.TiedEmbeddings {
		lmHeadParams = int64(m.HiddenDim) * int64(m.VocabSize)
	}

	return embedParams + layerParams + finalNormParams + lmHeadParams, nil
}

// WeightBytes returns the total memory footprint of model weights in bytes.
func (m ModelProfile) WeightBytes() (int64, error) {
	params, err := m.TotalParams()
	if err != nil {
		return 0, err
	}
	if m.BytesPerParam <= 0 {
		return 0, ErrInvalidPrecision
	}
	return int64(float64(params) * m.BytesPerParam), nil
}

// HardwareProfile defines peak hardware compute and memory bandwidth capabilities with explicit provenance.
type HardwareProfile struct {
	Name              string
	Source            string  // e.g. "assumed-m4-max", "assumed-gb10", "device-query", "unspecified"
	PeakBandwidthGBps float64 // Peak DRAM bandwidth in GB/s, 0 if unconfigured
	PeakTFLOPs        float64 // Peak dense compute in TFLOPs, 0 if unconfigured
}

// Observation records measured run durations, token counts, and verified hardware counter data.
type Observation struct {
	Duration            time.Duration
	TokenCount          int
	ThroughputTokPerSec float64
	DeviceDuration      time.Duration      // 0 if unmeasured on device
	HardwareCounters    map[string]float64 // nil or empty if unmeasured
	CounterProvenance   string             // Provenance of hardware counters, or "UNMEASURED"
}

// WorkloadEstimate contains analytical estimates of weight traffic and arithmetic intensity.
type WorkloadEstimate struct {
	ModelWeightGBps           float64 // labeled model-weight-GB/s
	AssumedPeakBWPercent      float64 // labeled assumed-peak-bw-pct
	DenseModelTFLOPs          float64 // labeled dense-model-TFLOPs/s
	AssumedPeakComputePercent float64 // labeled assumed-peak-compute-pct
	FormulaVersion            string
	HardwareSource            string
}

// EstimateWorkload calculates analytical memory traffic and FLOP rates from model architecture.
func EstimateWorkload(model ModelProfile, hw HardwareProfile, tokenCount int, dur time.Duration, isPrefill bool) (WorkloadEstimate, error) {
	if dur <= 0 || tokenCount <= 0 {
		return WorkloadEstimate{}, ErrInvalidRunParameters
	}

	weightBytes, err := model.WeightBytes()
	if err != nil {
		return WorkloadEstimate{}, err
	}
	totalParams, err := model.TotalParams()
	if err != nil {
		return WorkloadEstimate{}, err
	}

	var totalBytesTransferred float64
	if isPrefill {
		// Prefill streams weights once across all prompt tokens
		totalBytesTransferred = float64(weightBytes)
	} else {
		// Decode streams weights once per generated token
		totalBytesTransferred = float64(weightBytes) * float64(tokenCount)
	}

	bwGBps := (totalBytesTransferred / dur.Seconds()) / 1e9
	bwSatPct := 0.0
	if hw.PeakBandwidthGBps > 0 {
		bwSatPct = (bwGBps / hw.PeakBandwidthGBps) * 100.0
		if bwSatPct > 100.0 {
			bwSatPct = 100.0
		}
	}

	// 2 FLOPs per parameter per token (dense model forward pass)
	totalFLOPs := 2.0 * float64(totalParams) * float64(tokenCount)
	tflops := (totalFLOPs / dur.Seconds()) / 1e12
	computeSatPct := 0.0
	if hw.PeakTFLOPs > 0 {
		computeSatPct = (tflops / hw.PeakTFLOPs) * 100.0
	}

	hwSrc := hw.Source
	if hwSrc == "" {
		hwSrc = "unspecified"
	}

	return WorkloadEstimate{
		ModelWeightGBps:           bwGBps,
		AssumedPeakBWPercent:      bwSatPct,
		DenseModelTFLOPs:          tflops,
		AssumedPeakComputePercent: computeSatPct,
		FormulaVersion:            FormulaVersion,
		HardwareSource:            hwSrc,
	}, nil
}

// FormatBenchmarkMetricLine formats a single Go benchmark result line parseable by benchstat.
func FormatBenchmarkMetricLine(prefix string, phase string, nsPerOp int64, obs Observation, est *WorkloadEstimate) string {
	var b strings.Builder
	fmt.Fprintf(&b, "%s/%s-1 1 %d ns/op", prefix, phase, nsPerOp)

	if obs.ThroughputTokPerSec > 0 {
		fmt.Fprintf(&b, " %.4f tok/s", obs.ThroughputTokPerSec)
	}

	if est != nil {
		if est.ModelWeightGBps > 0 {
			fmt.Fprintf(&b, " %.2f model-weight-GB/s", est.ModelWeightGBps)
		}
		if est.AssumedPeakBWPercent > 0 {
			fmt.Fprintf(&b, " %.2f assumed-peak-bw-pct", est.AssumedPeakBWPercent)
		}
		if est.DenseModelTFLOPs > 0 {
			fmt.Fprintf(&b, " %.3f dense-model-TFLOPs/s", est.DenseModelTFLOPs)
		}
	}

	b.WriteString("\n")
	return b.String()
}

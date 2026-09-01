package gpuprof

import (
	"strings"
	"testing"
	"time"
)

func TestModelProfileTotalParams(t *testing.T) {
	tests := []struct {
		name        string
		profile     ModelProfile
		wantMin     int64
		wantMax     int64
		expectError bool
	}{
		{
			name: "Llama-3.2-1B-Untied",
			profile: ModelProfile{
				Name:            "Llama-3.2-1B",
				HiddenDim:       2048,
				IntermediateDim: 8192,
				NumLayers:       16,
				NumHeads:        32,
				NumKVHeads:      8,
				HeadDim:         64,
				VocabSize:       128256,
				TiedEmbeddings:  false,
				BytesPerParam:   4.0,
			},
			wantMin: 1_230_000_000,
			wantMax: 1_500_000_000,
		},
		{
			name: "Llama-3.2-1B-Tied",
			profile: ModelProfile{
				Name:            "Llama-3.2-1B-Tied",
				HiddenDim:       2048,
				IntermediateDim: 8192,
				NumLayers:       16,
				NumHeads:        32,
				NumKVHeads:      8,
				HeadDim:         64,
				VocabSize:       128256,
				TiedEmbeddings:  true,
				BytesPerParam:   4.0,
			},
			wantMin: 950_000_000,
			wantMax: 1_250_000_000,
		},
		{
			name: "Zero-Layers-Invalid",
			profile: ModelProfile{
				HiddenDim: 2048,
				NumLayers: 0,
			},
			expectError: true,
		},
		{
			name: "Negative-Dim-Invalid",
			profile: ModelProfile{
				HiddenDim: -2048,
				NumLayers: 16,
			},
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			params, err := tt.profile.TotalParams()
			if tt.expectError {
				if err == nil {
					t.Fatalf("expected error, got nil with params=%d", params)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if params < tt.wantMin || params > tt.wantMax {
				t.Errorf("expected params in [%d, %d], got %d", tt.wantMin, tt.wantMax, params)
			}
		})
	}
}

func TestModelProfileWeightBytes(t *testing.T) {
	base := ModelProfile{
		HiddenDim:       1024,
		IntermediateDim: 2048,
		NumLayers:       4,
		NumHeads:        8,
		NumKVHeads:      4,
		HeadDim:         128,
		VocabSize:       32000,
	}

	// FP32
	base.BytesPerParam = 4.0
	b32, err := base.WeightBytes()
	if err != nil {
		t.Fatalf("FP32 WeightBytes: %v", err)
	}

	// FP16
	base.BytesPerParam = 2.0
	b16, err := base.WeightBytes()
	if err != nil {
		t.Fatalf("FP16 WeightBytes: %v", err)
	}

	if b32 != 2*b16 {
		t.Errorf("expected FP32 (%d) to be exactly 2x FP16 (%d)", b32, b16)
	}

	// Invalid precision <= 0
	base.BytesPerParam = 0
	_, err = base.WeightBytes()
	if err == nil {
		t.Errorf("expected error for zero BytesPerParam")
	}
}

func TestEstimateWorkload(t *testing.T) {
	model := ModelProfile{
		Name:            "Llama-1B",
		HiddenDim:       2048,
		IntermediateDim: 8192,
		NumLayers:       16,
		NumHeads:        32,
		NumKVHeads:      8,
		HeadDim:         64,
		VocabSize:       128256,
		TiedEmbeddings:  false,
		BytesPerParam:   4.0,
	}

	hw := HardwareProfile{
		Name:              "Test GPU",
		Source:            "test-suite",
		PeakBandwidthGBps: 300.0,
		PeakTFLOPs:        100.0,
	}

	// Test Decode
	estDec, err := EstimateWorkload(model, hw, 16, 400*time.Millisecond, false)
	if err != nil {
		t.Fatalf("EstimateWorkload decode: %v", err)
	}
	if estDec.ModelWeightGBps <= 0 {
		t.Errorf("expected positive ModelWeightGBps, got %f", estDec.ModelWeightGBps)
	}
	if estDec.DenseModelTFLOPs <= 0 {
		t.Errorf("expected positive DenseModelTFLOPs, got %f", estDec.DenseModelTFLOPs)
	}
	if estDec.AssumedPeakBWPercent <= 0 || estDec.AssumedPeakBWPercent > 100 {
		t.Errorf("unexpected AssumedPeakBWPercent: %f", estDec.AssumedPeakBWPercent)
	}

	// Test Prefill (should stream weights once, not 16 times)
	estPref, err := EstimateWorkload(model, hw, 16, 400*time.Millisecond, true)
	if err != nil {
		t.Fatalf("EstimateWorkload prefill: %v", err)
	}
	if estPref.ModelWeightGBps*16 != estDec.ModelWeightGBps {
		t.Errorf("expected decode weight traffic (16 tokens) to be 16x prefill weight traffic: pref=%f dec=%f",
			estPref.ModelWeightGBps, estDec.ModelWeightGBps)
	}

	// Invalid Duration / Tokens
	_, err = EstimateWorkload(model, hw, 0, 400*time.Millisecond, false)
	if err == nil {
		t.Errorf("expected error for tokenCount=0")
	}
	_, err = EstimateWorkload(model, hw, 16, 0, false)
	if err == nil {
		t.Errorf("expected error for duration=0")
	}
}

func TestFormatBenchmarkMetricLine(t *testing.T) {
	obs := Observation{
		Duration:            25 * time.Millisecond,
		TokenCount:          1,
		ThroughputTokPerSec: 40.0,
	}
	est := &WorkloadEstimate{
		ModelWeightGBps:      200.0,
		AssumedPeakBWPercent: 66.6,
		DenseModelTFLOPs:     0.120,
	}

	line := FormatBenchmarkMetricLine("BenchmarkTest", "decode", 25000000, obs, est)

	if !strings.Contains(line, "25000000 ns/op") {
		t.Errorf("expected ns/op in %q", line)
	}
	if !strings.Contains(line, "40.0000 tok/s") {
		t.Errorf("expected tok/s in %q", line)
	}
	if !strings.Contains(line, "200.00 model-weight-GB/s") {
		t.Errorf("expected model-weight-GB/s in %q", line)
	}
	if !strings.Contains(line, "66.60 assumed-peak-bw-pct") {
		t.Errorf("expected assumed-peak-bw-pct in %q", line)
	}
	if !strings.Contains(line, "0.120 dense-model-TFLOPs/s") {
		t.Errorf("expected dense-model-TFLOPs/s in %q", line)
	}
}

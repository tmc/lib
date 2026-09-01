package gpuprof

import (
	"strings"
	"testing"
)

func TestModelProfileParams(t *testing.T) {
	model := ModelProfile{
		Name:            "Test-1B",
		HiddenDim:       2048,
		IntermediateDim: 8192,
		NumLayers:       16,
		NumHeads:        32,
		NumKVHeads:      8,
		HeadDim:         64,
		VocabSize:       128256,
		BytesPerParam:   4.0,
	}

	params := model.TotalParams()
	if params < 1_000_000_000 || params > 1_500_000_000 {
		t.Errorf("expected ~1.23B parameters, got %d", params)
	}

	weightBytes := model.WeightBytes()
	if weightBytes < 4_000_000_000 || weightBytes > 6_000_000_000 {
		t.Errorf("expected ~4.9GB for FP32 weights, got %d", weightBytes)
	}
}

func TestFormatBenchmarkLines(t *testing.T) {
	stats := SaturationStats{
		ThroughputTokPerSec:    40.5,
		BandwidthGBps:          198.2,
		BandwidthSaturationPct: 66.1,
		TFLOPs:                 0.098,
	}

	line := FormatBenchmarkLines("BenchmarkTest", stats, "decode")
	if !strings.Contains(line, "40.5000 tok/s") {
		t.Errorf("expected tok/s in output line, got: %s", line)
	}
	if !strings.Contains(line, "198.20 GB/s") {
		t.Errorf("expected GB/s in output line, got: %s", line)
	}
	if !strings.Contains(line, "66.10 %bw_sat") {
		t.Errorf("expected %%bw_sat in output line, got: %s", line)
	}
}

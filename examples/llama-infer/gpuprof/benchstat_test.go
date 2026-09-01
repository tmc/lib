package gpuprof

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestBenchstatParseability(t *testing.T) {
	// Generate representative multi-iteration benchmark output
	var buf bytes.Buffer

	model := ModelProfile{
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
	}

	hw := HardwareProfile{
		Name:              "Reference GPU",
		Source:            "benchstat-test",
		PeakBandwidthGBps: 300.0,
		PeakTFLOPs:        100.0,
	}

	for i := 0; i < 5; i++ {
		dur := 25*time.Millisecond + time.Duration(i)*time.Millisecond
		obs := Observation{
			Duration:            dur,
			TokenCount:          1,
			ThroughputTokPerSec: float64(time.Second) / float64(dur),
		}
		est, err := EstimateWorkload(model, hw, 1, dur, false)
		if err != nil {
			t.Fatalf("EstimateWorkload: %v", err)
		}
		line := FormatBenchmarkMetricLine("BenchmarkLlamaInfer/backend=test/decode", "step", dur.Nanoseconds(), obs, &est)
		buf.WriteString(line)
	}

	output := buf.String()
	t.Logf("Generated benchmark output:\n%s", output)

	// Check line structure: must have whitespace-separated pairs of <value> <unit>
	for _, line := range strings.Split(strings.TrimSpace(output), "\n") {
		fields := strings.Fields(line)
		if len(fields) < 4 {
			t.Errorf("expected at least 4 fields per line, got %d: %q", len(fields), line)
		}
		// Field 0: Benchmark name
		// Field 1: Iteration count (e.g. 1)
		// Field 2+: Value Unit pairs
		metricPairs := fields[2:]
		if len(metricPairs)%2 != 0 {
			t.Errorf("fields after iteration must come in value-unit pairs: %v", metricPairs)
		}
	}

	// If benchstat tool is installed in PATH, execute it on a temp file to ensure zero parser errors
	benchstatPath, err := exec.LookPath("benchstat")
	if err == nil && benchstatPath != "" {
		tmpFile := filepath.Join(t.TempDir(), "bench.txt")
		if err := os.WriteFile(tmpFile, []byte(output), 0644); err != nil {
			t.Fatalf("write temp bench file: %v", err)
		}
		cmd := exec.Command(benchstatPath, tmpFile)
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("benchstat failed to parse benchmark lines (%v): %s", err, string(out))
		}
		t.Logf("benchstat output:\n%s", string(out))
	}
}

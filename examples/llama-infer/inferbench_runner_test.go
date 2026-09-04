package main

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/examples/internal/continuousbatch"
	"github.com/tmc/lib/examples/internal/inferbench"
)

type rejectingTokenizer struct{}

func (rejectingTokenizer) Encode(string) ([]int32, error) { panic("tokenizer called") }
func (rejectingTokenizer) Decode([]int32) (string, error) { return "", nil }

func TestPrefillTokensPreservesExplicitIDs(t *testing.T) {
	state := &continuousbatch.State{Prompt: "must not be tokenized", PromptTokenIDs: []int32{7, 8}}
	got, err := prefillTokens(Config{BOSTokenID: 1}, rejectingTokenizer{}, state)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 || got[0] != 7 || got[1] != 8 {
		t.Fatalf("tokens = %v", got)
	}
	state.PromptTokenIDs[0] = 99
	if got[0] != 7 {
		t.Fatal("explicit token IDs were not copied")
	}
}

func TestBenchmarkIdentityRejectsSkippedHardware(t *testing.T) {
	dir := t.TempDir()
	model := filepath.Join(dir, "model")
	if err := os.Mkdir(model, 0700); err != nil {
		t.Fatal(err)
	}
	tokenizer := filepath.Join(model, "tokenizer.json")
	config := filepath.Join(model, "config.json")
	source := filepath.Join(dir, "source")
	proof := filepath.Join(dir, "proof.json")
	for path, data := range map[string]string{tokenizer: "tokenizer", config: "config", source: "source"} {
		if err := os.WriteFile(path, []byte(data), 0600); err != nil {
			t.Fatal(err)
		}
	}
	proofJSON := `{"backend":"` + benchmarkBackend() + `","device_id":"gpu0","captured_at":"` + time.Now().UTC().Format(time.RFC3339Nano) + `","command":["probe"],"output":"skipped","hardware_verified":false,"device_resident_kv":true,"cpu_fallback":false}`
	if err := os.WriteFile(proof, []byte(proofJSON), 0600); err != nil {
		t.Fatal(err)
	}
	exe, _ := os.Executable()
	spec := inferbenchSpec{TokenizerPath: tokenizer, ConfigPath: config, SourcePath: source, DeviceEvidencePath: proof}
	var err error
	spec.ExpectedBinarySHA256, err = inferbench.HashPath(exe)
	if err != nil {
		t.Fatal(err)
	}
	spec.ExpectedModelSHA256, _ = inferbench.HashPath(model)
	spec.ExpectedTokenizerSHA256, _ = inferbench.HashPath(tokenizer)
	spec.ExpectedConfigSHA256, _ = inferbench.HashPath(config)
	spec.ExpectedSourceSHA256, _ = inferbench.HashPath(source)
	_, _, err = benchmarkIdentity(Config{ModelPath: model}, spec)
	if err == nil || !strings.Contains(err.Error(), "hardware qualification refused") {
		t.Fatalf("error = %v", err)
	}
}

func TestUnqualifiedHostLaunchesZeroTimedArms(t *testing.T) {
	dir := t.TempDir()
	model := filepath.Join(dir, "model")
	if err := os.Mkdir(model, 0700); err != nil {
		t.Fatal(err)
	}
	tokenizer := filepath.Join(model, "tokenizer.json")
	config := filepath.Join(model, "config.json")
	source := filepath.Join(dir, "source")
	nullControl := filepath.Join(dir, "null-control.json")
	for path, data := range map[string]string{tokenizer: "tokenizer", config: "config", source: "source", nullControl: "[]"} {
		if err := os.WriteFile(path, []byte(data), 0600); err != nil {
			t.Fatal(err)
		}
	}
	hostname, err := os.Hostname()
	if err != nil {
		t.Fatal(err)
	}
	proof := filepath.Join(dir, "proof.json")
	if err := inferbench.WriteJSONAtomic(proof, deviceEvidence{Backend: benchmarkBackend(), DeviceID: "gpu0", Hostname: hostname, CapturedAt: time.Now(), Command: []string{"null-control"}, Output: "noisy", HardwareVerified: true, DeviceResidentKV: true, NullControlPath: nullControl, NullControlSamples: 20, NullControlP95: 3 * time.Millisecond, NullControlLimit: time.Millisecond}); err != nil {
		t.Fatal(err)
	}
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	spec := inferbenchSpec{Arm: "llama-infer", Output: filepath.Join(dir, "arm.json"), Lease: filepath.Join(dir, "runner.lock"), SourcePath: source, TokenizerPath: tokenizer, ConfigPath: config, DeviceEvidencePath: proof, SampleIDs: []string{"sample-1"}, Workload: inferbench.Workload{PromptTokenIDs: [][]int32{{1}}, ContextTokens: 1, GeneratedTokens: 2, Concurrency: 1, WarmupRuns: 1, Sampler: "greedy-argmax", CacheState: "cold", OutputMode: inferbench.ThroughputOutputMode}}
	spec.ExpectedBinarySHA256, _ = inferbench.HashPath(exe)
	spec.ExpectedModelSHA256, _ = inferbench.HashPath(model)
	spec.ExpectedTokenizerSHA256, _ = inferbench.HashPath(tokenizer)
	spec.ExpectedConfigSHA256, _ = inferbench.HashPath(config)
	spec.ExpectedSourceSHA256, _ = inferbench.HashPath(source)
	specPath := filepath.Join(dir, "spec.json")
	if err := inferbench.WriteJSONAtomic(specPath, spec); err != nil {
		t.Fatal(err)
	}
	launches := 0
	err = runInferbenchWithEngine(t.Context(), Config{ModelPath: model}, specPath, func(Config) (*ModelEngine, error) {
		launches++
		return nil, errors.New("engine launched")
	})
	if err == nil || !strings.Contains(err.Error(), "hardware qualification refused") {
		t.Fatalf("error = %v", err)
	}
	if launches != 0 {
		t.Fatalf("timed arm launches = %d, want 0", launches)
	}
}

func TestPhysicalTelemetryCapability(t *testing.T) {
	for _, batch := range []int{1, 2, 4} {
		t.Run(fmt.Sprintf("B%d", batch), func(t *testing.T) {
			err := benchmarkPhysicalTelemetry()
			if benchmarkBackend() == "metal" && err != nil {
				t.Fatal(err)
			}
			if benchmarkBackend() != "metal" && (err == nil || !strings.Contains(err.Error(), "exact dispatch and tier-byte counters")) {
				t.Fatalf("error = %v", err)
			}
		})
	}
}

func TestValidateBenchmarkArms(t *testing.T) {
	tests := []struct {
		name string
		spec inferbenchSpec
		ok   bool
	}{
		{"single", inferbenchSpec{Arm: "llama-infer", Workload: inferbench.Workload{OutputMode: inferbench.ThroughputOutputMode}}, true},
		{"single observed", inferbenchSpec{Arm: "llama-infer-observed-output", Workload: inferbench.Workload{OutputMode: inferbench.ObservedOutputMode}}, true},
		{"ordinary arm observed work", inferbenchSpec{Arm: "llama-infer", Workload: inferbench.Workload{OutputMode: inferbench.ObservedOutputMode}}, false},
		{"zero seed", inferbenchSpec{Arms: []string{"llama-infer", "llama-infer-unfused-projections"}, RawDir: "raw", Workload: inferbench.Workload{OutputMode: inferbench.ThroughputOutputMode}}, false},
		{"duplicate", inferbenchSpec{Arms: []string{"llama-infer", "llama-infer"}, RandomSeed: 1, RawDir: "raw", Workload: inferbench.Workload{OutputMode: inferbench.ThroughputOutputMode}}, false},
		{"single multiarm", inferbenchSpec{Arms: []string{"llama-infer"}, RandomSeed: 1, RawDir: "raw", Workload: inferbench.Workload{OutputMode: inferbench.ThroughputOutputMode}}, false},
		{"mixed modes", inferbenchSpec{Arm: "llama-infer", Arms: []string{"llama-infer", "llama-infer-unfused-projections"}, RandomSeed: 1, RawDir: "raw", Workload: inferbench.Workload{OutputMode: inferbench.ThroughputOutputMode}}, false},
	}
	if benchmarkBackend() == "metal" {
		tests = append(tests, struct {
			name string
			spec inferbenchSpec
			ok   bool
		}{"campaign", inferbenchSpec{Arms: []string{"llama-infer", "llama-infer-unfused-projections"}, RandomSeed: 1, RawDir: "raw", SampleIDs: benchmarkSampleIDs(20), Workload: inferbench.Workload{OutputMode: inferbench.ThroughputOutputMode}}, true})
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			err := validateBenchmarkArms(test.spec)
			if (err == nil) != test.ok {
				t.Fatalf("error = %v, want success %t", err, test.ok)
			}
		})
	}
}

func benchmarkSampleIDs(n int) []string {
	ids := make([]string, n)
	for i := range ids {
		ids[i] = fmt.Sprintf("sample-%d", i)
	}
	return ids
}

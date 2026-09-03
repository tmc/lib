package main

import (
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
	tokenizer := filepath.Join(dir, "tokenizer")
	source := filepath.Join(dir, "source")
	proof := filepath.Join(dir, "proof.json")
	for path, data := range map[string]string{model: "model", tokenizer: "tokenizer", source: "source"} {
		if err := os.WriteFile(path, []byte(data), 0600); err != nil {
			t.Fatal(err)
		}
	}
	proofJSON := `{"backend":"` + benchmarkBackend() + `","device_id":"gpu0","captured_at":"` + time.Now().UTC().Format(time.RFC3339Nano) + `","command":["probe"],"output":"skipped","hardware_verified":false,"device_resident_kv":true,"cpu_fallback":false}`
	if err := os.WriteFile(proof, []byte(proofJSON), 0600); err != nil {
		t.Fatal(err)
	}
	exe, _ := os.Executable()
	spec := inferbenchSpec{TokenizerPath: tokenizer, SourcePath: source, DeviceEvidencePath: proof}
	var err error
	spec.ExpectedBinarySHA256, err = inferbench.HashPath(exe)
	if err != nil {
		t.Fatal(err)
	}
	spec.ExpectedModelSHA256, _ = inferbench.HashPath(model)
	spec.ExpectedTokenizerSHA256, _ = inferbench.HashPath(tokenizer)
	spec.ExpectedSourceSHA256, _ = inferbench.HashPath(source)
	_, _, err = benchmarkIdentity(Config{ModelPath: model}, spec)
	if err == nil || !strings.Contains(err.Error(), "hardware qualification refused") {
		t.Fatalf("error = %v", err)
	}
}

package main

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"time"
)

// TestPrefillCapture retains exact batch and teacher-forced continuation logits.
// It does not compare captures or certify parity. Each selected case runs twice:
// a warm pass, then a reset and captured pass, all on one runtime.
func TestPrefillCapture(t *testing.T) {
	if os.Getenv("DS4_PREFILL_GATE") == "" {
		t.Skip("DS4_PREFILL_GATE is not set")
	}
	if os.Getenv("DS4_PREFILL_GATE") != "1" {
		t.Fatal("DS4_PREFILL_GATE must be 1")
	}
	required := func(name string) string {
		t.Helper()
		value := os.Getenv(name)
		if value == "" {
			t.Fatalf("%s is required", name)
		}
		return value
	}
	model := required("DS4_MODEL")
	backend := required("DS4_PREFILL_BACKEND")
	if backend != "metal" && backend != "cuda" {
		t.Fatal("DS4_PREFILL_BACKEND must be metal or cuda")
	}
	if _, err := selectBackend(backend); err != nil {
		t.Fatal(err)
	}
	inputPath := required("DS4_PREFILL_TOKENS")
	dir := required("DS4_PREFILL_DIR")
	continuation, err := strconv.Atoi(required("DS4_PREFILL_CONTINUATION"))
	const expectedVocab = 129280
	if err != nil || continuation < 0 || continuation >= expectedVocab {
		t.Fatal("DS4_PREFILL_CONTINUATION must be a token ID in 0..129279")
	}
	var cases []int
	seen := make(map[int]bool)
	for _, value := range strings.Split(required("DS4_PREFILL_CASES"), ",") {
		n, err := strconv.Atoi(value)
		if err != nil || (n != 2 && n != 64 && n != 2048) || seen[n] {
			t.Fatal("DS4_PREFILL_CASES must contain distinct comma-separated values from 2,64,2048")
		}
		seen[n] = true
		cases = append(cases, n)
	}
	input, err := os.ReadFile(inputPath)
	if err != nil {
		t.Fatal(err)
	}
	var tokens []int
	if err := json.Unmarshal(input, &tokens); err != nil {
		t.Fatal(err)
	}
	if len(tokens) < 2048 {
		t.Fatal("DS4_PREFILL_TOKENS must contain a JSON array of at least 2048 token IDs")
	}
	for _, token := range tokens {
		if token < 0 || token >= expectedVocab {
			t.Fatal("input token ID outside 0..129279")
		}
	}
	model, err = filepath.Abs(model)
	if err != nil {
		t.Fatal(err)
	}
	info, err := os.Stat(model)
	if err != nil {
		t.Fatal(err)
	}
	if !info.Mode().IsRegular() {
		t.Fatal("model must be a regular file")
	}
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	hashFile := func(path string) string {
		t.Helper()
		f, err := os.Open(path)
		if err != nil {
			t.Fatal(err)
		}
		h := sha256.New()
		_, hashErr := io.Copy(h, f)
		closeErr := f.Close()
		if hashErr != nil || closeErr != nil {
			t.Fatalf("hash %s: read=%v close=%v", path, hashErr, closeErr)
		}
		return fmt.Sprintf("%x", h.Sum(nil))
	}
	executableHash := hashFile(executable)
	modelHash := hashFile(model)
	host, err := os.Hostname()
	if err != nil {
		t.Fatal(err)
	}
	// Mkdir deliberately rejects existing directories, including empty ones.
	if err := os.Mkdir(dir, 0700); err != nil {
		t.Fatal(err)
	}
	write := func(name string, data []byte) {
		t.Helper()
		if err := os.WriteFile(filepath.Join(dir, name), data, 0600); err != nil {
			t.Fatal(err)
		}
	}
	writeJSON := func(name string, value any) {
		t.Helper()
		data, err := json.MarshalIndent(value, "", "  ")
		if err != nil {
			t.Fatal(err)
		}
		write(name, append(data, '\n'))
	}
	write("tokens.json", input)
	writeJSON("identity.json", map[string]any{
		"schema": 1, "backend": backend, "context": 2049,
		"cases": cases, "continuation_token": continuation,
		"tokens_sha256": fmt.Sprintf("%x", sha256.Sum256(input)),
		"model_path":    model, "model_bytes": info.Size(), "model_mtime": info.ModTime().UTC(),
		"model_sha256":      modelHash,
		"executable_sha256": executableHash,
		"go_version":        runtime.Version(), "goos": runtime.GOOS, "goarch": runtime.GOARCH,
		"host": host, "started": time.Now().UTC(),
		"lifecycle":  "one runtime; each case: reset, warm batch and fixed continuation, reset, captured batch and same continuation; close before COMPLETE",
		"comparison": "exact little-endian float32 bytes at both captures; compare externally; no token-zero hash claim",
	})
	r, closer, err := openSessionBackend(config{model: model, backend: backend, context: 2049, log: io.Discard})
	if err != nil {
		t.Fatal(err)
	}
	closed := false
	defer func() {
		if !closed {
			if err := closer.Close(); err != nil {
				t.Errorf("close runtime: %v", err)
			}
		}
	}()
	batch, ok := r.(batchEvaluator)
	if !ok {
		t.Fatal("backend does not implement EvalBatch")
	}
	capture := func(name string, retain bool) {
		t.Helper()
		logits := r.Logits()
		if len(logits) != expectedVocab {
			t.Fatalf("%s: vocabulary = %d, want %d", name, len(logits), expectedVocab)
		}
		data := make([]byte, 4*len(logits))
		for i, value := range logits {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				t.Fatalf("%s: non-finite logit %d", name, i)
			}
			binary.LittleEndian.PutUint32(data[4*i:], math.Float32bits(value))
		}
		if retain {
			write(name+".f32le", data)
			writeJSON(name+".json", map[string]any{"vocab": expectedVocab, "bytes": len(data), "sha256": fmt.Sprintf("%x", sha256.Sum256(data))})
		}
	}
	for _, n := range cases {
		for pass := 0; pass < 2; pass++ {
			if err := r.Reset(); err != nil {
				t.Fatalf("case %d pass %d reset: %v", n, pass, err)
			}
			if err := batch.EvalBatch(tokens[:n], 0); err != nil {
				t.Fatalf("case %d pass %d prefill: %v", n, pass, err)
			}
			capture(fmt.Sprintf("n%d-prefill", n), pass == 1)
			if err := r.Eval(continuation, n); err != nil {
				t.Fatalf("case %d pass %d continuation: %v", n, pass, err)
			}
			capture(fmt.Sprintf("n%d-continuation", n), pass == 1)
		}
	}
	err = closer.Close()
	closed = true
	if err != nil {
		t.Fatal(err)
	}
	write("COMPLETE", []byte("runtime closed; all selected captures written\n"))
}

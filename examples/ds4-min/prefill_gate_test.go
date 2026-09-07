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
	cases, err := parsePrefillCases(required("DS4_PREFILL_CASES"))
	if err != nil {
		t.Fatal(err)
	}
	input, err := os.ReadFile(inputPath)
	if err != nil {
		t.Fatal(err)
	}
	tokens, err := parsePrefillTokens(input)
	if err != nil {
		t.Fatal(err)
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

// Decode pointers so JSON null cannot silently become token zero.
func parsePrefillTokens(input []byte) ([]int, error) {
	var values []*int
	if err := json.Unmarshal(input, &values); err != nil {
		return nil, fmt.Errorf("prefill token corpus: %w", err)
	}
	if len(values) < 2048 {
		return nil, fmt.Errorf("prefill token corpus: got %d IDs, want at least 2048", len(values))
	}
	tokens := make([]int, len(values))
	for i, value := range values {
		if value == nil {
			return nil, fmt.Errorf("prefill token %d: null is not a token ID", i)
		}
		if *value < 0 || *value >= 129280 {
			return nil, fmt.Errorf("prefill token %d: ID outside 0..129279", i)
		}
		tokens[i] = *value
	}
	return tokens, nil
}

func TestParsePrefillTokens(t *testing.T) {
	for _, tt := range []struct {
		name, first string
		count       int
		wantError   bool
	}{
		{"zero", "0", 2048, false},
		{"maximum", "129279", 2048, false},
		{"null", "null", 2048, true},
		{"fraction", "0.5", 2048, true},
		{"boolean", "true", 2048, true},
		{"negative", "-1", 2048, true},
		{"out of range", "129280", 2048, true},
		{"short", "0", 2047, true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			input := []byte("[" + tt.first + strings.Repeat(",0", tt.count-1) + "]")
			got, err := parsePrefillTokens(input)
			if (err != nil) != tt.wantError {
				t.Fatalf("parse error = %v, want error %t", err, tt.wantError)
			}
			if err == nil && len(got) != tt.count {
				t.Fatalf("got %d tokens, want %d", len(got), tt.count)
			}
			if err == nil && tt.name == "maximum" && got[0] != 129279 {
				t.Fatalf("first token = %d, want 129279", got[0])
			}
		})
	}
}

func parsePrefillCases(input string) ([]int, error) {
	var cases []int
	seen := make(map[int]bool)
	for _, value := range strings.Split(input, ",") {
		n, err := strconv.Atoi(value)
		if err != nil || (n != 2 && n != 64 && n != 65 && n != 66 && n != 128 && n != 129 && n != 2048) || seen[n] {
			return nil, fmt.Errorf("DS4_PREFILL_CASES must contain distinct comma-separated values from 2,64,65,66,128,129,2048")
		}
		seen[n] = true
		cases = append(cases, n)
	}
	return cases, nil
}

func TestParsePrefillCases(t *testing.T) {
	for _, tt := range []struct {
		input string
		want  string
	}{
		{"2,64,2048", "2,64,2048"},
		{"128", "128"}, {"129", "129"}, {"128,129", "128,129"},
		{"129,128", "129,128"}, {"128,128", ""}, {"129,129", ""},
		{"127", ""}, {"130", ""},
		{"65", "65"}, {"66", "66"}, {"64,65,66", "64,65,66"},
		{"66,65", "66,65"}, {"2,64,65,66,2048", "2,64,65,66,2048"},
		{"", ""}, {"65,65", ""}, {"66,66", ""}, {"63", ""},
		{"67", ""}, {"64,", ""}, {"65.0", ""}, {"true", ""},
	} {
		t.Run(tt.input, func(t *testing.T) {
			got, err := parsePrefillCases(tt.input)
			if tt.want == "" {
				if err == nil {
					t.Fatalf("parsePrefillCases(%q) = %v, want error", tt.input, got)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			var values []string
			for _, n := range got {
				values = append(values, strconv.Itoa(n))
			}
			if strings.Join(values, ",") != tt.want {
				t.Fatalf("got %v, want %s", got, tt.want)
			}
		})
	}
}

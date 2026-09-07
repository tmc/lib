//go:build linux

package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"testing"
	"time"
)

const cudaPrefillProgressSchema = "ds4-prefill-progress/v1"
const cudaPrefillProgressStep = 128

type cudaPrefillProgressRecord struct {
	Schema          string `json:"schema"`
	CompletedTokens int    `json:"completed_tokens"`
	ElapsedNS       int64  `json:"elapsed_ns"`
	TotalTokens     int    `json:"total_tokens"`
}

type cudaPrefillProgressWriter struct {
	f     *os.File
	start time.Time
	last  cudaPrefillProgressRecord
}

func newCUDAPrefillProgressWriter(path string) (*cudaPrefillProgressWriter, error) {
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		return nil, err
	}
	w := &cudaPrefillProgressWriter{f: f}
	if err := w.write(0, 0); err != nil {
		f.Close()
		return nil, err
	}
	return w, nil
}

func (w *cudaPrefillProgressWriter) mark(completed int) error {
	if completed != oracleWorkloadPrefill && completed%cudaPrefillProgressStep != 0 {
		return nil
	}
	return w.write(completed, time.Since(w.start).Nanoseconds())
}

func (w *cudaPrefillProgressWriter) write(completed int, elapsed int64) error {
	record := cudaPrefillProgressRecord{
		Schema:          cudaPrefillProgressSchema,
		CompletedTokens: completed,
		ElapsedNS:       elapsed,
		TotalTokens:     oracleWorkloadPrefill,
	}
	if err := validateCUDAPrefillProgressRecord(w.last, record); err != nil {
		return err
	}
	data, err := json.Marshal(record)
	if err != nil {
		return err
	}
	data = append(data, '\n')
	if _, err := w.f.Write(data); err != nil {
		return err
	}
	if err := w.f.Sync(); err != nil {
		return err
	}
	w.last = record
	return nil
}

func (w *cudaPrefillProgressWriter) close() error { return w.f.Close() }

func (w *cudaPrefillProgressWriter) finalize() error {
	if w.last.CompletedTokens != oracleWorkloadPrefill {
		return fmt.Errorf("cuda prefill progress: completed %d tokens, want %d", w.last.CompletedTokens, oracleWorkloadPrefill)
	}
	return w.close()
}

func validateCUDAPrefillProgressRecord(previous, next cudaPrefillProgressRecord) error {
	if next.Schema != cudaPrefillProgressSchema || next.TotalTokens != oracleWorkloadPrefill {
		return fmt.Errorf("cuda prefill progress: invalid identity")
	}
	if previous.Schema == "" {
		if next.CompletedTokens != 0 || next.ElapsedNS != 0 {
			return fmt.Errorf("cuda prefill progress: invalid initial record")
		}
		return nil
	}
	if next.CompletedTokens != previous.CompletedTokens+cudaPrefillProgressStep || next.CompletedTokens > next.TotalTokens {
		return fmt.Errorf("cuda prefill progress: token step %d after %d", next.CompletedTokens, previous.CompletedTokens)
	}
	if next.ElapsedNS <= previous.ElapsedNS {
		return fmt.Errorf("cuda prefill progress: elapsed time did not increase")
	}
	return nil
}

func TestCUDAPrefillProgress(t *testing.T) {
	if os.Getenv("DS4_CUDA_PREFILL_PROGRESS") == "" {
		t.Skip("DS4_CUDA_PREFILL_PROGRESS is not set")
	}
	if os.Getenv("DS4_CUDA_PREFILL_PROGRESS") != "1" {
		t.Fatal("DS4_CUDA_PREFILL_PROGRESS must be 1")
	}
	required := func(key string) string {
		t.Helper()
		value := os.Getenv(key)
		if value == "" {
			t.Fatalf("%s is required", key)
		}
		return value
	}
	model := required("DS4_MODEL")
	tokensPath := required("DS4_PREFILL_TOKENS")
	progressPath := required("DS4_CUDA_PREFILL_PROGRESS_OUT")
	input, err := os.ReadFile(tokensPath)
	if err != nil {
		t.Fatal(err)
	}
	tokens, err := parsePrefillTokens(input)
	if err != nil {
		t.Fatal(err)
	}
	tokens = tokens[:oracleWorkloadPrefill]
	if got := fmt.Sprintf("%x", sha256.Sum256(oracleWorkloadTokenBytes(tokens))); got != oracleWorkloadTokenHash {
		t.Fatal("token prefix does not match registered oracle workload")
	}
	backend, closer, err := openSessionBackend(config{model: model, backend: "cuda", context: oracleWorkloadContext, log: io.Discard})
	if err != nil {
		t.Fatal(err)
	}
	closed := false
	defer func() {
		if !closed {
			closer.Close()
		}
	}()
	runtime, ok := backend.(*cudaRuntime)
	if !ok {
		t.Fatal("CUDA prefill progress opened a non-CUDA runtime")
	}
	w, err := newCUDAPrefillProgressWriter(progressPath)
	if err != nil {
		t.Fatal(err)
	}
	writerClosed := false
	defer func() {
		if !writerClosed {
			w.close()
		}
	}()
	runtime.batchProgress = w.mark
	w.start = time.Now()
	if err := runtime.EvalBatch(tokens, 0); err != nil {
		t.Fatal(err)
	}
	runtime.batchProgress = nil
	if err := w.finalize(); err != nil {
		t.Fatal(err)
	}
	writerClosed = true
	if err := closer.Close(); err != nil {
		t.Fatal(err)
	}
	closed = true
}

func TestCUDAPrefillProgressWriter(t *testing.T) {
	path := t.TempDir() + "/progress.jsonl"
	w, err := newCUDAPrefillProgressWriter(path)
	if err != nil {
		t.Fatal(err)
	}
	w.start = time.Now().Add(-time.Second)
	if err := w.mark(8); err != nil {
		t.Fatal(err)
	}
	if err := w.mark(128); err != nil {
		t.Fatal(err)
	}
	if err := w.close(); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var records []cudaPrefillProgressRecord
	for _, line := range bytes.Split(bytes.TrimSpace(data), []byte{'\n'}) {
		var record cudaPrefillProgressRecord
		if err := json.Unmarshal(line, &record); err != nil {
			t.Fatal(err)
		}
		records = append(records, record)
	}
	if len(records) != 2 || records[0].CompletedTokens != 0 || records[1].CompletedTokens != 128 {
		t.Fatalf("records = %+v", records)
	}
}

func TestCUDAPrefillProgressRequiresCompletion(t *testing.T) {
	w, err := newCUDAPrefillProgressWriter(t.TempDir() + "/progress.jsonl")
	if err != nil {
		t.Fatal(err)
	}
	if err := w.finalize(); err == nil {
		t.Fatal("initial-only progress finalized")
	}
	if err := w.close(); err != nil {
		t.Fatal(err)
	}
	complete, err := newCUDAPrefillProgressWriter(t.TempDir() + "/progress.jsonl")
	if err != nil {
		t.Fatal(err)
	}
	complete.start = time.Now().Add(-time.Second)
	for completed := cudaPrefillProgressStep; completed <= oracleWorkloadPrefill; completed += cudaPrefillProgressStep {
		if err := complete.mark(completed); err != nil {
			t.Fatal(err)
		}
	}
	if err := complete.finalize(); err != nil {
		t.Fatalf("complete progress did not finalize: %v", err)
	}
}

func TestValidateCUDAPrefillProgressRecord(t *testing.T) {
	initial := cudaPrefillProgressRecord{Schema: cudaPrefillProgressSchema, TotalTokens: oracleWorkloadPrefill}
	next := cudaPrefillProgressRecord{Schema: cudaPrefillProgressSchema, CompletedTokens: 128, ElapsedNS: 1, TotalTokens: oracleWorkloadPrefill}
	if err := validateCUDAPrefillProgressRecord(cudaPrefillProgressRecord{}, initial); err != nil {
		t.Fatal(err)
	}
	if err := validateCUDAPrefillProgressRecord(initial, next); err != nil {
		t.Fatal(err)
	}
	for _, tt := range []struct {
		name string
		prev cudaPrefillProgressRecord
		next cudaPrefillProgressRecord
	}{
		{name: "wrong initial position", next: cudaPrefillProgressRecord{Schema: cudaPrefillProgressSchema, CompletedTokens: 128, TotalTokens: oracleWorkloadPrefill}},
		{name: "wrong schema", prev: initial, next: cudaPrefillProgressRecord{Schema: "other", CompletedTokens: 128, ElapsedNS: 1, TotalTokens: oracleWorkloadPrefill}},
		{name: "wrong total", prev: initial, next: cudaPrefillProgressRecord{Schema: cudaPrefillProgressSchema, CompletedTokens: 128, ElapsedNS: 1, TotalTokens: 64}},
		{name: "gap", prev: initial, next: cudaPrefillProgressRecord{Schema: cudaPrefillProgressSchema, CompletedTokens: 256, ElapsedNS: 1, TotalTokens: oracleWorkloadPrefill}},
		{name: "duplicate", prev: next, next: next},
		{name: "nonmonotonic elapsed", prev: next, next: cudaPrefillProgressRecord{Schema: cudaPrefillProgressSchema, CompletedTokens: 256, ElapsedNS: 1, TotalTokens: oracleWorkloadPrefill}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if err := validateCUDAPrefillProgressRecord(tt.prev, tt.next); err == nil {
				t.Fatal("invalid progress record passed")
			}
		})
	}
}

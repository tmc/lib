//go:build linux

package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// Two complete 43-layer evaluations of token0 at position0, separated by Reset.
// This is a repeatability diagnostic, not oracle or token-equivalence evidence.
func TestCUDAObserveSingleTokenRepeat(t *testing.T) {
	path, dir := os.Getenv("DS4_MODEL"), os.Getenv("DS4_OBSERVE_DIR")
	if dir == "" {
		t.Skip("DS4_OBSERVE_DIR is not set")
	}
	if path == "" {
		t.Fatal("DS4_MODEL is required")
	}
	cudaObserveDirectory(t, dir)
	r, err := newCUDARuntime(path, 1024)
	if err != nil {
		t.Fatal(err)
	}
	closed := false
	defer func() {
		if !closed {
			if err := r.Close(); err != nil {
				t.Fatalf("close runtime: %v", err)
			}
		}
	}()
	// newCUDARuntime already validates the pinned model profile. Retain its
	// actual digests as an observation identity in addition to that admission gate.
	digests, err := profileHashes(r.gguf)
	if err != nil {
		t.Fatal(err)
	}
	canonical, err := filepath.Abs(path)
	if err != nil {
		t.Fatal(err)
	}
	canonical, err = filepath.EvalSymlinks(canonical)
	if err != nil {
		t.Fatal(err)
	}
	info, err := os.Stat(canonical)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("model canonical path: %s", canonical)
	t.Logf("metadata=%s tensors=%s quantization=%s tokenizer=%s", digests.metadata, digests.tensors, digests.quant, digests.tokenizer)
	cudaObserveJSON(t, dir, "identity.json", map[string]any{
		"model": canonical, "size": info.Size(), "metadata": digests.metadata, "tensors": digests.tensors, "quantization": digests.quant, "tokenizer": digests.tokenizer,
	})
	cudaObserveJSON(t, dir, "input.json", map[string]any{"token": 0, "position": 0, "context": 1024, "iterations": 2, "reset_between": true, "vocab": 129280})
	var first []byte
	hashes := make([]string, 0, 2)
	for iteration := range 2 {
		if iteration != 0 {
			if err := r.Reset(); err != nil {
				t.Fatalf("reset: %v", err)
			}
		}
		// The frozen runtime names its single-token API Eval(token, position).
		if err := cudaRepeatObservedEval(r, t, dir, iteration); err != nil {
			t.Fatalf("iteration %d: %v", iteration, err)
		}
		logits := r.Logits()
		if len(logits) != 129280 {
			t.Fatalf("iteration %d: logits length %d", iteration, len(logits))
		}
		raw := make([]byte, len(logits)*4)
		for i, v := range logits {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatalf("iteration %d: nonfinite logit[%d]=%08x", iteration, i, math.Float32bits(v))
			}
			binary.LittleEndian.PutUint32(raw[i*4:], math.Float32bits(v))
		}
		name := fmt.Sprintf("logits-%02d.bin", iteration)
		cudaObserveWrite(t, dir, name, raw)
		hash := fmt.Sprintf("%x", sha256.Sum256(raw))
		hashes = append(hashes, hash)
		cudaObserveJSON(t, dir, fmt.Sprintf("logits-%02d.json", iteration), map[string]any{"iteration": iteration, "vocab": len(logits), "sha256": hash, "format": "little-endian float32", "logits": logits})
		t.Logf("iteration=%d full_finite_logits=%d sha256=%s", iteration, len(logits), hash)
		if iteration == 0 {
			first = raw
		}
	}
	// Read the retained second array rather than r.Logits after runtime closure.
	second, err := os.ReadFile(filepath.Join(dir, "logits-01.bin"))
	if err != nil {
		t.Fatal(err)
	}
	equal := bytes.Equal(first, second)
	result := map[string]any{"exact_bits": equal, "iterations": 2, "logit_hashes": hashes, "scope": "same-runtime token0 position0 repeat after Reset; no oracle equivalence claim"}
	if !equal {
		changed, firstIndex := 0, -1
		var maxDelta, sum float64
		for i := 0; i < len(first); i += 4 {
			a, b := binary.LittleEndian.Uint32(first[i:]), binary.LittleEndian.Uint32(second[i:])
			delta := float64(math.Float32frombits(a)) - float64(math.Float32frombits(b))
			maxDelta = max(maxDelta, math.Abs(delta))
			sum += delta * delta
			if a != b {
				changed++
				if firstIndex < 0 {
					firstIndex = i / 4
					result["first_index"] = firstIndex
					result["first_bits"] = fmt.Sprintf("%08x", a)
					result["second_bits"] = fmt.Sprintf("%08x", b)
					result["first_value"] = math.Float32frombits(a)
					result["second_value"] = math.Float32frombits(b)
				}
			}
		}
		result["changed_logits"] = changed
		result["max_absolute_delta"] = maxDelta
		result["rms_delta"] = math.Sqrt(sum / 129280)
	}
	closed = true
	if err := r.Close(); err != nil {
		t.Fatalf("close runtime: %v", err)
	}
	cudaObserveJSON(t, dir, "result.json", result)
	if !equal {
		t.Fatalf("single-token Reset repeat differs: %v", result)
	}
}

// Diagnostic clone with host observations and an explicit prefill phase.
// Historical decode-route source and receipts remain preserved separately.
func cudaRepeatObservedEval(r *cudaRuntime, t *testing.T, dir string, iteration int) (err error) {
	token, position, logits := 0, 0, true
	if r == nil || r.arena == nil {
		return errors.New("cuda runtime: closed")
	}
	if r.failed != nil {
		return fmt.Errorf("cuda runtime: reset required: %w", r.failed)
	}
	if position != r.pos || position < 0 || position >= r.ctx {
		return fmt.Errorf("cuda runtime: position %d does not follow %d", position, r.pos)
	}
	if token < 0 || uint64(token) >= r.model.output.embedding.dimensions[1] {
		return fmt.Errorf("cuda runtime: token %d out of range", token)
	}
	defer func() {
		if err != nil {
			r.failed = err
		}
	}()
	for stream := range 4 {
		if err := r.encodeEmbeddingRow(r.model.output.embedding, uint32(token), r.state.hc[0].Slice(stream*cudaWidth, (stream+1)*cudaWidth)); err != nil {
			return err
		}
	}
	cudaRepeatSnapshot(t, r, dir, iteration, -1)
	for layer := range r.model.layers {
		if err := r.evalLayer(layer, token, position); err != nil {
			return fmt.Errorf("cuda runtime: layer %d: %w", layer, err)
		}
		cudaRepeatSnapshot(t, r, dir, iteration, layer)
	}
	if logits {
		if err := r.hcMath.pre(r.state.hc[0], r.model.output.hcFn, r.model.output.hcScale, r.model.output.hcBase, r.model.output.norm, r.state.outputMixed, true); err != nil {
			return err
		}
	}
	if logits {
		if err := r.encodeQ80Matvec(r.model.output.weight, r.state.outputMixed.Slice(4096, 8192), r.state.logits); err != nil {
			return err
		}
	}
	if err := r.device.DefaultStream().Synchronize(context.Background()); err != nil {
		return err
	}
	if logits {
		if err := r.state.logits.CopyToHost(r.logits); err != nil {
			return err
		}
		for i, v := range r.logits {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				return fmt.Errorf("cuda runtime: non-finite logit %d", i)
			}
		}
	}
	r.pos++
	return nil
}

func cudaRepeatSnapshot(t *testing.T, r *cudaRuntime, dir string, iteration, layer int) {
	t.Helper()
	// CopyToHost synchronizes observation; this can perturb a scheduling defect.
	for stream := range 2 {
		if layer < 0 && stream != 0 {
			continue
		}
		values := make([]float32, r.state.hc[stream].Len())
		if err := r.state.hc[stream].CopyToHost(values); err != nil {
			t.Fatal(err)
		}
		raw := make([]byte, len(values)*4)
		nonfinite := 0
		for i, v := range values {
			binary.LittleEndian.PutUint32(raw[4*i:], math.Float32bits(v))
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				nonfinite++
			}
		}
		name := fmt.Sprintf("iteration-%02d-layer-%02d-hc%d.bin", iteration, layer, stream)
		cudaObserveWrite(t, dir, name, raw)
		t.Logf("snapshot=%s values=%d nonfinite=%d sha256=%x", name, len(values), nonfinite, sha256.Sum256(raw))
		if nonfinite != 0 {
			t.Fatalf("first observed nonfinite boundary: %s", name)
		}
	}
}

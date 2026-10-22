//go:build ds4cpumodel && ds4cpuactualmodel

package dsv4cpu

import (
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"testing"
	"time"
)

func writeCPUForwardOutput(path string, data []byte) error {
	f, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	n, writeErr := f.Write(data)
	if writeErr == nil && n != len(data) {
		writeErr = io.ErrShortWrite
	}
	return errors.Join(writeErr, f.Close())
}

func checkCPUForwardReference(raw []byte) error {
	const digest = "7f25b3c5f7d855a5b2ccf1aff8549e4b46f6733eb53853ba5619a4d762cb492d"
	if len(raw) != 129280*4 || fmt.Sprintf("%x", sha256.Sum256(raw)) != digest {
		return errors.New("reference identity differs from retained scalar C0")
	}
	return nil
}

// This is an actual asset gate, never part of the host-only test closure.
func TestFlashRealModelOneToken(t *testing.T) {
	path, reference, output := os.Getenv("DS4_CPU_MODEL"), os.Getenv("DS4_CPU_REFERENCE"), os.Getenv("DS4_CPU_OUTPUT")
	if path == "" || reference == "" || output == "" {
		t.Fatal("DS4_CPU_MODEL, DS4_CPU_REFERENCE and DS4_CPU_OUTPUT are required")
	}
	wantBytes, err := os.ReadFile(reference)
	if err != nil {
		t.Fatal(err)
	}
	if err := checkCPUForwardReference(wantBytes); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(output); err == nil || !os.IsNotExist(err) {
		t.Fatal("output receipt must not already exist", err)
	}
	const digest = "ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0"
	b, err := openFlashBackend(path, digest)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := b.Close(); err != nil {
			t.Errorf("close model: %v", err)
		}
	}()
	s, err := b.NewState(256)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := s.Close(); err != nil {
			t.Errorf("close state: %v", err)
		}
	}()
	start := time.Now()
	if err := b.Eval(s, 128000, 0); err != nil {
		t.Fatal(err)
	}
	t.Logf("token=128000 context=256 position=0 forward_seconds=%.9f", time.Since(start).Seconds())
	got := b.Logits(s)
	if len(got) != 129280 {
		t.Fatal("candidate shape differs")
	}
	gotBytes := make([]byte, len(got)*4)
	for i, v := range got {
		binary.LittleEndian.PutUint32(gotBytes[i*4:], math.Float32bits(v))
	}
	if err := writeCPUForwardOutput(output, gotBytes); err != nil {
		t.Fatal(err)
	}
	violations, ai, bi := 0, 0, 0
	maxDelta, maxRelative, squaredError := 0.0, 0.0, 0.0
	maxA, maxB := math.Inf(-1), math.Inf(-1)
	for i, v := range got {
		w := math.Float32frombits(binary.LittleEndian.Uint32(wantBytes[i*4:]))
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || math.IsNaN(float64(w)) || math.IsInf(float64(w), 0) {
			t.Fatalf("nonfinite reference/candidate at %d", i)
		}
		if float64(w) > maxA {
			maxA, ai = float64(w), i
		}
		if float64(v) > maxB {
			maxB, bi = float64(v), i
		}
		d := math.Abs(float64(v) - float64(w))
		squaredError += d * d
		relative := 0.0
		if d != 0 {
			relative = d / math.Abs(float64(w))
		}
		if relative > maxRelative {
			maxRelative = relative
		}
		if d > maxDelta {
			maxDelta = d
		}
		if d > 1e-4+1e-4*math.Abs(float64(w)) {
			violations++
		}
	}
	t.Logf("vocab=%d reference_argmax=%d candidate_argmax=%d violations=%d max_abs=%.9g max_relative=%.9g rms=%.9g qualification=NO_VERDICT_control_out_of_bound", len(got), ai, bi, violations, maxDelta, maxRelative, math.Sqrt(squaredError/float64(len(got))))
	if ai != bi || violations != 0 {
		t.Fatal("actual model F5 numerical gate failed")
	}
}

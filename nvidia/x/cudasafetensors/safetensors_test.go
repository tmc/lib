package cudasafetensors_test

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudasafetensors"
)

func createSampleSafetensors(t *testing.T) string {
	headerJSON := `{"weight":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}}`
	headerBytes := []byte(headerJSON)
	headerLen := uint64(len(headerBytes))

	data := make([]byte, 8+headerLen+16)
	binary.LittleEndian.PutUint64(data[:8], headerLen)
	copy(data[8:8+headerLen], headerBytes)

	// Float values: [1.0, 2.0, 3.0, 4.0]
	floats := []float32{1.0, 2.0, 3.0, 4.0}
	for i, f := range floats {
		binary.LittleEndian.PutUint32(data[8+headerLen+uint64(i*4):], math.Float32bits(f))
	}

	tmpDir := t.TempDir()
	path := filepath.Join(tmpDir, "model.safetensors")
	if err := os.WriteFile(path, data, 0644); err != nil {
		t.Fatalf("write file: %v", err)
	}
	return path
}

func TestOpen(t *testing.T) {
	path := createSampleSafetensors(t)
	w, err := cudasafetensors.Open(path)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}
	defer w.Close()

	if len(w.Tensors) != 1 {
		t.Fatalf("expected 1 tensor, got %d", len(w.Tensors))
	}

	info, exists := w.Tensors["weight"]
	if !exists {
		t.Fatal("expected tensor 'weight'")
	}

	if info.DType != "F32" || len(info.Shape) != 2 || info.Shape[0] != 2 || info.Shape[1] != 2 {
		t.Errorf("unexpected tensor info: %+v", info)
	}

	rawBytes, _, err := w.TensorBytes("weight")
	if err != nil {
		t.Fatalf("TensorBytes failed: %v", err)
	}
	if len(rawBytes) != 16 {
		t.Errorf("expected 16 bytes, got %d", len(rawBytes))
	}
}

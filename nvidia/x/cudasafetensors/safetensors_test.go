package cudasafetensors_test

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/tmc/lib/nvidia/x/cuda"
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

func writeSafetensors(t *testing.T, header string, data []byte) string {
	t.Helper()
	b := make([]byte, 8+len(header)+len(data))
	binary.LittleEndian.PutUint64(b, uint64(len(header)))
	copy(b[8:], header)
	copy(b[8+len(header):], data)
	path := filepath.Join(t.TempDir(), "model.safetensors")
	if err := os.WriteFile(path, b, 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestOpenRejectsMalformedTensorHeaders(t *testing.T) {
	tests := []struct {
		name   string
		header string
		data   []byte
	}{
		{"malformed entry", `{"weight":1}`, nil},
		{"missing offsets", `{"weight":{"dtype":"F32","shape":[1]}}`, make([]byte, 4)},
		{"unknown dtype", `{"weight":{"dtype":"X4","shape":[1],"data_offsets":[0,1]}}`, make([]byte, 1)},
		{"negative shape", `{"weight":{"dtype":"F32","shape":[-1],"data_offsets":[0,4]}}`, make([]byte, 4)},
		{"wrong byte count", `{"weight":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}}`, make([]byte, 4)},
		{"overlap", `{"a":{"dtype":"U8","shape":[2],"data_offsets":[0,2]},"b":{"dtype":"U8","shape":[2],"data_offsets":[1,3]}}`, make([]byte, 3)},
		{"hole", `{"weight":{"dtype":"U8","shape":[1],"data_offsets":[1,2]}}`, make([]byte, 2)},
		{"trailing data", `{"weight":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}`, make([]byte, 2)},
		{"invalid metadata", `{"__metadata__":1}`, nil},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if w, err := cudasafetensors.Open(writeSafetensors(t, tt.header, tt.data)); err == nil {
				w.Close()
				t.Fatal("Open succeeded")
			}
		})
	}
}

func TestOpenRejectsOversizedHeader(t *testing.T) {
	path := filepath.Join(t.TempDir(), "model.safetensors")
	var b [8]byte
	binary.LittleEndian.PutUint64(b[:], 100<<20+1)
	if err := os.WriteFile(path, b[:], 0o600); err != nil {
		t.Fatal(err)
	}
	if w, err := cudasafetensors.Open(path); err == nil {
		w.Close()
		t.Fatal("Open succeeded")
	}
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

func TestOpenSharded(t *testing.T) {
	tmpDir := t.TempDir()
	shard1Path := filepath.Join(tmpDir, "model-00001.safetensors")

	// Write shard 1
	headerJSON := `{"layer1.weight":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}}`
	headerBytes := []byte(headerJSON)
	headerLen := uint64(len(headerBytes))
	data := make([]byte, 8+headerLen+16)
	binary.LittleEndian.PutUint64(data[:8], headerLen)
	copy(data[8:8+headerLen], headerBytes)
	for i := 0; i < 4; i++ {
		binary.LittleEndian.PutUint32(data[8+headerLen+uint64(i*4):], math.Float32bits(float32(i+1)))
	}
	if err := os.WriteFile(shard1Path, data, 0644); err != nil {
		t.Fatalf("write shard: %v", err)
	}

	// Write index JSON
	indexJSON := `{"metadata":{},"weight_map":{"layer1.weight":"model-00001.safetensors"}}`
	if err := os.WriteFile(filepath.Join(tmpDir, "model.safetensors.index.json"), []byte(indexJSON), 0644); err != nil {
		t.Fatalf("write index: %v", err)
	}

	sw, err := cudasafetensors.OpenSharded(tmpDir)
	if err != nil {
		t.Fatalf("OpenSharded failed: %v", err)
	}
	defer sw.Close()

	if _, exists := sw.Tensors["layer1.weight"]; !exists {
		t.Fatal("expected layer1.weight in sharded weights")
	}

	rawBytes, _, err := sw.TensorBytes("layer1.weight")
	if err != nil {
		t.Fatalf("TensorBytes failed: %v", err)
	}
	if len(rawBytes) != 16 {
		t.Errorf("expected 16 bytes, got %d", len(rawBytes))
	}
}

func TestOpenShardedRejectsInvalidIndex(t *testing.T) {
	tests := []struct {
		name  string
		index string
	}{
		{"empty map", `{"weight_map":{}}`},
		{"path traversal", `{"weight_map":{"weight":"../model.safetensors"}}`},
		{"missing tensor", `{"weight_map":{"other":"model.safetensors"}}`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dir := t.TempDir()
			valid := `{"weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}`
			source := writeSafetensors(t, valid, make([]byte, 4))
			data, err := os.ReadFile(source)
			if err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(dir, "model.safetensors"), data, 0o600); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(dir, "model.safetensors.index.json"), []byte(tt.index), 0o600); err != nil {
				t.Fatal(err)
			}
			if w, err := cudasafetensors.OpenSharded(dir); err == nil {
				w.Close()
				t.Fatal("OpenSharded succeeded")
			}
		})
	}
}

func TestLoadSliceCUDA(t *testing.T) {
	dev, err := cuda.NewDevice(0)
	if err != nil {
		t.Skipf("CUDA unavailable: %v", err)
	}
	defer dev.Close()

	w, err := cudasafetensors.Open(createSampleSafetensors(t))
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	s, err := cudasafetensors.LoadSlice[float32](w, "weight", dev)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Free()
	got := make([]float32, 4)
	if err := s.CopyToHost(got); err != nil {
		t.Fatal(err)
	}
	want := []float32{1, 2, 3, 4}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("element %d = %v, want %v", i, got[i], want[i])
		}
	}
}

type tensorProvider struct {
	data []byte
	info cudasafetensors.TensorInfo
}

func (p tensorProvider) TensorBytes(string) ([]byte, cudasafetensors.TensorInfo, error) {
	return p.data, p.info, nil
}

func TestLoadSliceRejectsIncompatibleStorage(t *testing.T) {
	p := tensorProvider{data: make([]byte, 2), info: cudasafetensors.TensorInfo{Name: "weight", DType: "F16", Shape: []int{1}, OffsetEnd: 2}}
	if s, err := cudasafetensors.LoadSlice[float32](p, "weight", nil); err == nil {
		s.Free()
		t.Fatal("LoadSlice[float32] accepted F16 storage")
	}
}

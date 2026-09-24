package cudasafetensors_test

import (
	"bytes"
	"encoding/binary"
	"math"
	"path/filepath"
	"slices"
	"testing"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudasafetensors"
	"github.com/tmc/lib/nvidia/x/safetensors"
)

func float32Bytes(v ...float32) []byte {
	var b []byte
	for _, f := range v {
		b = binary.LittleEndian.AppendUint32(b, math.Float32bits(f))
	}
	return b
}

// newFile returns an in-memory safetensors file holding tensors.
func newFile(t *testing.T, tensors ...safetensors.Tensor) *safetensors.File {
	t.Helper()
	var buf bytes.Buffer
	if err := safetensors.Write(&buf, tensors, nil); err != nil {
		t.Fatal(err)
	}
	f, err := safetensors.NewFile(bytes.NewReader(buf.Bytes()), int64(buf.Len()))
	if err != nil {
		t.Fatal(err)
	}
	return f
}

func TestLoadSliceRejects(t *testing.T) {
	f := newFile(t,
		safetensors.Tensor{Name: "half", Dtype: safetensors.F16, Shape: []int{1}, Data: make([]byte, 2)},
		safetensors.Tensor{Name: "empty", Dtype: safetensors.F32, Shape: []int{0}, Data: nil},
	)
	tests := []struct {
		name string
		load func() (func() error, error)
	}{
		{"float32 from F16", func() (func() error, error) {
			s, err := cudasafetensors.LoadSlice[float32](f, "half", nil)
			return s.Free, err
		}},
		{"int16 from F16", func() (func() error, error) {
			s, err := cudasafetensors.LoadSlice[int16](f, "half", nil)
			return s.Free, err
		}},
		{"struct from F16", func() (func() error, error) {
			s, err := cudasafetensors.LoadSlice[struct{ a, b byte }](f, "half", nil)
			return s.Free, err
		}},
		{"no elements", func() (func() error, error) {
			s, err := cudasafetensors.LoadSlice[float32](f, "empty", nil)
			return s.Free, err
		}},
		{"missing tensor", func() (func() error, error) {
			s, err := cudasafetensors.LoadSlice[float32](f, "missing", nil)
			return s.Free, err
		}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			free, err := tt.load()
			if err == nil {
				free()
				t.Fatal("LoadSlice succeeded")
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

	dir := t.TempDir()
	want := []float32{1, 2, 3, 4}
	w := safetensors.Tensor{Name: "weight", Dtype: safetensors.F32, Shape: []int{2, 2}, Data: float32Bytes(want...)}
	if err := safetensors.WriteFile(filepath.Join(dir, "model.safetensors"), []safetensors.Tensor{w}, nil); err != nil {
		t.Fatal(err)
	}
	f, err := safetensors.OpenSharded(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	s, err := cudasafetensors.LoadSlice[float32](f, "weight", dev)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Free()
	got := make([]float32, len(want))
	if err := s.CopyToHost(got); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(got, want) {
		t.Fatalf("device holds %v, want %v", got, want)
	}
}

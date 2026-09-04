//go:build darwin

package metalsafetensors

import (
	"encoding/binary"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/apple/x/metal"
)

func writeFixture(t *testing.T, header any, data []byte) string {
	t.Helper()
	b, err := json.Marshal(header)
	if err != nil {
		t.Fatal(err)
	}
	p := filepath.Join(t.TempDir(), "weights.safetensors")
	f, err := os.Create(p)
	if err != nil {
		t.Fatal(err)
	}
	if err := binary.Write(f, binary.LittleEndian, uint64(len(b))); err != nil {
		t.Fatal(err)
	}
	if _, err := f.Write(b); err != nil {
		t.Fatal(err)
	}
	if _, err := f.Write(data); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	return p
}

func tensor(dtype string, shape []int, offsets ...uint64) map[string]any {
	return map[string]any{"dtype": dtype, "shape": shape, "data_offsets": offsets}
}

func TestOpenRejectsInvalidLayout(t *testing.T) {
	cases := []struct {
		name   string
		header any
		data   []byte
		want   string
	}{
		{"offset count", map[string]any{"x": tensor("F32", []int{1}, 0)}, make([]byte, 4), "invalid data offsets"},
		{"reverse", map[string]any{"x": tensor("F32", []int{1}, 4, 0)}, make([]byte, 4), "invalid data offsets"},
		{"unsupported dtype", map[string]any{"x": tensor("I32", []int{1}, 0, 4)}, make([]byte, 4), "unsupported dtype"},
		{"negative shape", map[string]any{"x": tensor("F32", []int{-1}, 0, 4)}, make([]byte, 4), "invalid shape"},
		{"wrong byte count", map[string]any{"x": tensor("F32", []int{2}, 0, 4)}, make([]byte, 4), "byte length"},
		{"past file", map[string]any{"x": tensor("F32", []int{1}, 0, 4)}, make([]byte, 3), "exceeds file"},
		{"gap", map[string]any{"x": tensor("F32", []int{1}, 1, 5)}, make([]byte, 5), "not contiguous"},
		{"overlap", map[string]any{"x": tensor("F32", []int{1}, 0, 4), "y": tensor("F32", []int{1}, 2, 6)}, make([]byte, 6), "not contiguous"},
		{"trailing bytes", map[string]any{"x": tensor("F32", []int{1}, 0, 4)}, make([]byte, 5), "file contains"},
		{"unknown field", map[string]any{"x": map[string]any{"dtype": "F32", "shape": []int{1}, "data_offsets": []int{0, 4}, "surprise": true}}, make([]byte, 4), "unknown field"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			w, err := Open(writeFixture(t, tc.header, tc.data))
			if w != nil {
				w.Close()
			}
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("Open error = %v, want %q", err, tc.want)
			}
		})
	}
}

func TestOpenRejectsInvalidHeaderLength(t *testing.T) {
	for _, n := range []uint64{0, maxHeaderSize + 1} {
		p := filepath.Join(t.TempDir(), "weights.safetensors")
		f, err := os.Create(p)
		if err != nil {
			t.Fatal(err)
		}
		if err := binary.Write(f, binary.LittleEndian, n); err != nil {
			t.Fatal(err)
		}
		f.Close()
		if w, err := Open(p); err == nil {
			w.Close()
			t.Fatalf("Open header length %d succeeded", n)
		}
	}
}

func TestTensorMetalSliceOwnsCopiedData(t *testing.T) {
	p := writeFixture(t, map[string]any{"x": tensor("F32", []int{2}, 0, 8)}, []byte{0, 0, 128, 63, 0, 0, 0, 64})
	w, err := Open(p)
	if err != nil {
		t.Fatal(err)
	}
	dev, err := metal.NewDevice()
	if err != nil {
		w.Close()
		t.Skipf("Metal device unavailable: %v", err)
	}
	defer dev.Close()
	s, err := w.TensorMetalSlice(dev, "x")
	if err != nil {
		w.Close()
		t.Fatal(err)
	}
	defer s.Free()
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
	got := s.HostSlice()
	if len(got) != 2 || got[0] != 1 || got[1] != 2 {
		t.Fatalf("Metal copy after close = %v", got)
	}
	t.Logf("device=%q values=%v", dev.Name(), got)
}

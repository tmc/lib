// SPDX-License-Identifier: Apache-2.0

package cudakv

import (
	"bytes"
	"math"
	"os"
	"path/filepath"
	"testing"
	"testing/iotest"
)

func TestSSDOffloaderCreation(t *testing.T) {
	offloader, err := NewSSDOffloader(t.TempDir())
	if err != nil {
		t.Fatalf("NewSSDOffloader failed: %v", err)
	}
	defer offloader.Close()
}

// TestOffloadRoundTripFile writes the layer encoding through a real file and
// reads it back, which is the property EvictLayer and RestoreLayer rely on.
func TestOffloadRoundTripFile(t *testing.T) {
	k := []float32{0, 1, -1, 0.5, float32(math.Pi), -1e30, 1e-30}
	v := []float32{2, 3, 4}

	path := filepath.Join(t.TempDir(), "cuda_layer_0.bin")
	f, err := os.Create(path)
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	if err := writeFloats(f, k); err != nil {
		t.Fatalf("writeFloats K: %v", err)
	}
	if err := writeFloats(f, v); err != nil {
		t.Fatalf("writeFloats V: %v", err)
	}
	if err := f.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	fi, err := os.Stat(path)
	if err != nil {
		t.Fatalf("Stat: %v", err)
	}
	if want := int64(len(k)+len(v)) * 4; fi.Size() != want {
		t.Fatalf("file size = %d, want %d", fi.Size(), want)
	}

	r, err := os.Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer r.Close()

	gotK := make([]float32, len(k))
	gotV := make([]float32, len(v))
	if err := readFloats(r, gotK); err != nil {
		t.Fatalf("readFloats K: %v", err)
	}
	if err := readFloats(r, gotV); err != nil {
		t.Fatalf("readFloats V: %v", err)
	}

	for i := range k {
		if gotK[i] != k[i] {
			t.Errorf("K[%d] = %v, want %v", i, gotK[i], k[i])
		}
	}
	for i := range v {
		if gotV[i] != v[i] {
			t.Errorf("V[%d] = %v, want %v", i, gotV[i], v[i])
		}
	}
}

// TestReadFloatsShortReader is the negative control: a reader that dribbles out
// one byte per call must still fill the destination, and a truncated stream must
// report an error rather than leave the tail of the buffer silently zeroed.
func TestReadFloatsShortReader(t *testing.T) {
	want := []float32{1, 2, 3, 4}
	var buf bytes.Buffer
	if err := writeFloats(&buf, want); err != nil {
		t.Fatalf("writeFloats: %v", err)
	}
	encoded := buf.Bytes()

	got := make([]float32, len(want))
	if err := readFloats(iotest.OneByteReader(bytes.NewReader(encoded)), got); err != nil {
		t.Fatalf("readFloats from a one-byte reader: %v", err)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("got[%d] = %v, want %v", i, got[i], want[i])
		}
	}

	truncated := make([]float32, len(want))
	err := readFloats(bytes.NewReader(encoded[:len(encoded)-1]), truncated)
	if err == nil {
		t.Error("readFloats accepted a truncated stream, want an error")
	}
}

func TestSSDOffloaderCloseRemovesFiles(t *testing.T) {
	dir := t.TempDir()
	o, err := NewSSDOffloader(dir)
	if err != nil {
		t.Fatalf("NewSSDOffloader: %v", err)
	}

	path := filepath.Join(dir, "cuda_layer_7.bin")
	if err := os.WriteFile(path, []byte{0, 0, 0, 0}, 0o600); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	o.files[7] = path

	if err := o.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Errorf("spilled layer file still present after Close (err=%v)", err)
	}
	if len(o.files) != 0 {
		t.Errorf("files = %v, want empty after Close", o.files)
	}
	if err := o.Close(); err != nil {
		t.Errorf("second Close: %v", err)
	}
}

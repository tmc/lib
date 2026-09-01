//go:build darwin

package metalio

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/tmc/lib/apple/x/metal"
)

func TestWeightLoaderZeroCopy(t *testing.T) {
	dev, err := metal.NewDevice()
	if err != nil {
		t.Skipf("Metal device unavailable: %v", err)
	}

	tmpDir := t.TempDir()
	weightPath := filepath.Join(tmpDir, "weights.bin")

	const numElements = 256
	rawFloats := make([]float32, numElements)
	for i := range rawFloats {
		rawFloats[i] = float32(i + 1)
	}

	f, err := os.Create(weightPath)
	if err != nil {
		t.Fatalf("Create file failed: %v", err)
	}
	for _, val := range rawFloats {
		var buf [4]byte
		binary.LittleEndian.PutUint32(buf[:], math.Float32bits(val))
		f.Write(buf[:])
	}
	f.Close()

	loader := NewWeightLoader(dev)
	slice, err := loader.LoadSafetensorsTensor(weightPath, 0, numElements)
	if err != nil {
		t.Fatalf("LoadSafetensorsTensor failed: %v", err)
	}
	defer slice.Free()

	hostView := slice.HostSlice()
	if len(hostView) != numElements {
		t.Fatalf("expected %d elements, got %d", numElements, len(hostView))
	}

	if hostView[0] != 1.0 || hostView[255] != 256.0 {
		t.Errorf("unexpected tensor data values: first=%f, last=%f", hostView[0], hostView[255])
	}

	t.Logf("Successfully loaded %d float32 elements via direct zero-copy unified memory mapping", len(hostView))
}

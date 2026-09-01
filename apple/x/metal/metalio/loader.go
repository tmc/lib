//go:build darwin

package metalio

import (
	"fmt"
	"os"
	"syscall"
	"unsafe"

	apple "github.com/tmc/apple/metal"
	"github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/mempool"
)

// WeightLoader streams model weights directly from disk into shared unified memory.
type WeightLoader struct {
	dev *metal.Device
}

// NewWeightLoader creates a weight loader for the given Metal device.
func NewWeightLoader(dev *metal.Device) *WeightLoader {
	return &WeightLoader{dev: dev}
}

// LoadSafetensorsTensor maps a tensor directly from a file into a zero-copy MTLBuffer.
func (l *WeightLoader) LoadSafetensorsTensor(path string, offset int64, numElements int) (mempool.Slice[float32], error) {
	f, err := os.Open(path)
	if err != nil {
		return mempool.Slice[float32]{}, fmt.Errorf("metalio: open weight file: %w", err)
	}
	defer f.Close()

	return mempool.AllocMMapZeroCopy[float32](l.dev, f, offset, numElements)
}

// StreamToBuffer reads file bytes into a pre-allocated Metal buffer using direct memory mapping.
func (l *WeightLoader) StreamToBuffer(buf apple.MTLBuffer, f *os.File, offset int64, length int) error {
	if buf.GetID() == 0 {
		return fmt.Errorf("metalio: invalid nil buffer")
	}

	data, err := syscall.Mmap(int(f.Fd()), offset, length, syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return fmt.Errorf("metalio: mmap error: %w", err)
	}
	defer syscall.Munmap(data)

	dest := unsafe.Slice((*byte)(buf.Contents()), length)
	copy(dest, data)
	return nil
}

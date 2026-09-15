package mempool

import (
	"os"

	applemetal "github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/metalmem"
)

// Slice aliases metalmem.Slice[T] for backward compatibility.
type Slice[T any] = metalmem.Slice[T]

// Alloc allocates a typed Slice[T] via metalmem.
func Alloc[T any](dev *applemetal.Device, length int) (Slice[T], error) {
	return metalmem.Alloc[T](dev, length)
}

// AllocFromSlice allocates and copies data via metalmem.
func AllocFromSlice[T any](dev *applemetal.Device, data []T) (Slice[T], error) {
	return metalmem.AllocFromSlice[T](dev, data)
}

// AllocMMapZeroCopy creates a zero-copy buffer via metalmem.
func AllocMMapZeroCopy[T any](dev *applemetal.Device, f *os.File, offset int64, length int) (Slice[T], error) {
	return metalmem.AllocMMapZeroCopy[T](dev, f, offset, length)
}

// Pool aliases metalmem.Pool.
type Pool = metalmem.Pool

// NewPool creates an allocation pool via metalmem.
func NewPool(dev *applemetal.Device) *Pool {
	return metalmem.NewPool(dev)
}

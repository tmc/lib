package mempool

import (
	"os"

	applemetal "github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/metalmem"
)

// Slice aliases [metalmem.Slice].
type Slice[T any] = metalmem.Slice[T]

// Alloc allocates a typed Slice[T] via metalmem.
func Alloc[T any](dev *applemetal.Device, length int) (Slice[T], error) {
	return metalmem.Alloc[T](dev, length)
}

// AllocFromSlice allocates and copies data via metalmem.
func AllocFromSlice[T any](dev *applemetal.Device, data []T) (Slice[T], error) {
	return metalmem.AllocFromSlice[T](dev, data)
}

// AllocMMapZeroCopy creates a buffer over a file mapping via metalmem.
func AllocMMapZeroCopy[T any](dev *applemetal.Device, f *os.File, offset int64, length int) (Slice[T], error) {
	return metalmem.AllocMMapZeroCopy[T](dev, f, offset, length)
}

// Arena aliases [metalmem.Arena].
type Arena = metalmem.Arena

// NewArena creates an allocation arena via metalmem.
func NewArena(dev *applemetal.Device) *Arena {
	return metalmem.NewArena(dev)
}

// AllocIn allocates a Slice[T] from an arena via metalmem.
func AllocIn[T any](a *Arena, name string, length int) Slice[T] {
	return metalmem.AllocIn[T](a, name, length)
}

// UploadIn allocates a Slice[T] holding a copy of data from an arena via metalmem.
func UploadIn[T any](a *Arena, name string, data []T) Slice[T] {
	return metalmem.UploadIn[T](a, name, data)
}

// Pool aliases [metalmem.Pool].
type Pool = metalmem.Pool

// NewPool creates an allocation pool via metalmem.
func NewPool(dev *applemetal.Device) *Pool {
	return metalmem.NewPool(dev)
}

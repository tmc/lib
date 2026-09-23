//go:build !windows

package moonep

import (
	"fmt"
	"os"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
)

// MoonEP's device kernels are CUTLASS CuTe-DSL JIT programs; they cannot be
// built by a Go toolchain. The helpers here load cubins exported once from
// the Python side and launch them. Exporting: run the MoonEP Python package
// with CUTE_DSL_CACHE_DIR set, exercise the target (S, K, E, R, H, ...)
// configuration once, and collect the cached cubins; each kernel is
// specialized on those constexprs, so a cubin is only valid for the exact
// Layout it was compiled for.

// LoadModule loads a cubin (or fatbin/PTX) from path into the current
// context.
func LoadModule(path string) (driver.Module, error) {
	if _, err := os.Stat(path); err != nil {
		return 0, fmt.Errorf("moonep: kernel module: %w", err)
	}
	m, err := driver.NewModuleLoad(path)
	if err != nil {
		return 0, fmt.Errorf("moonep: cuModuleLoad(%s): %w", path, err)
	}
	return m, nil
}

// LoadModuleData loads a cubin image from memory (e.g. a go:embed asset)
// into the current context.
func LoadModuleData(image []byte) (driver.Module, error) {
	if len(image) == 0 {
		return 0, fmt.Errorf("moonep: empty kernel image")
	}
	m, err := driver.NewModuleLoadData(unsafe.Pointer(&image[0]))
	if err != nil {
		return 0, fmt.Errorf("moonep: cuModuleLoadData: %w", err)
	}
	return m, nil
}

// Params packs kernel arguments for launch: each Add stores one argument
// value and Ptrs returns the pointer-to-argument array the driver expects.
// The zero value is ready to use.
type Params struct {
	vals []unsafe.Pointer
}

// AddPtr appends a device-pointer argument.
func (p *Params) AddPtr(d driver.DevicePtr) *Params {
	v := new(driver.DevicePtr)
	*v = d
	p.vals = append(p.vals, unsafe.Pointer(v))
	return p
}

// AddInt32 appends an int32 argument.
func (p *Params) AddInt32(n int32) *Params {
	v := new(int32)
	*v = n
	p.vals = append(p.vals, unsafe.Pointer(v))
	return p
}

// Ptrs returns the kernel parameter array for the driver launch APIs.
func (p *Params) Ptrs() []unsafe.Pointer {
	return p.vals
}

// LaunchCooperative launches fn as a cooperative grid (the launch mode all
// MoonEP kernels use: their software grid barrier requires every block to be
// resident) with gridDim×blockDim threads and sharedMem bytes of dynamic
// shared memory on stream.
func LaunchCooperative(fn driver.Function, gridDim, blockDim, sharedMem uint32, stream driver.Stream, params *Params) error {
	if err := fn.LaunchCooperativeKernel(
		gridDim, 1, 1, blockDim, 1, 1, sharedMem, stream, params.Ptrs(),
	); err != nil {
		return fmt.Errorf("moonep: cooperative launch: %w", err)
	}
	return nil
}

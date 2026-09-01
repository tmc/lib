package metalx

import (
	"fmt"
	"sync"

	apple "github.com/tmc/apple/metal"
	"github.com/tmc/lib/apple/x/metal"
)

var (
	cacheMu  sync.RWMutex
	psoCache = make(map[string]apple.MTLComputePipelineState)
)

// CompileLibrary compiles MSL source code into an MTLLibrary.
func CompileLibrary(dev *metal.Device, source string) (apple.MTLLibrary, error) {
	if dev == nil || dev.Raw().GetID() == 0 {
		return nil, fmt.Errorf("metalx: nil or uninitialized Metal device")
	}
	lib, err := dev.Raw().NewLibraryWithSourceOptionsError(source, apple.MTLCompileOptions{})
	if err != nil {
		return nil, fmt.Errorf("metalx: compile MSL library: %w", err)
	}
	return lib, nil
}

// Compile compiles an MSL source string and constructs a compute pipeline state for a named kernel.
func Compile(dev *metal.Device, source, kernelName string) (apple.MTLComputePipelineState, error) {
	cacheKey := source + "::" + kernelName

	cacheMu.RLock()
	pso, ok := psoCache[cacheKey]
	cacheMu.RUnlock()
	if ok {
		return pso, nil
	}

	lib, err := CompileLibrary(dev, source)
	if err != nil {
		return nil, err
	}

	fn := lib.NewFunctionWithName(kernelName)
	if fn.GetID() == 0 {
		return nil, fmt.Errorf("metalx: kernel %q not found in compiled library", kernelName)
	}

	newPso, err := dev.Raw().NewComputePipelineStateWithFunctionError(fn)
	if err != nil {
		return nil, fmt.Errorf("metalx: create compute pipeline %q: %w", kernelName, err)
	}

	cacheMu.Lock()
	psoCache[cacheKey] = newPso
	cacheMu.Unlock()

	return newPso, nil
}

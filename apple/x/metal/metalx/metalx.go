package metalx

import (
	"errors"
	"fmt"
	"sync"

	apple "github.com/tmc/apple/metal"
	"github.com/tmc/apple/objectivec"
	"github.com/tmc/lib/apple/x/metal"
)

// ErrClosed is returned when Compile is called after Compiler.Close.
var ErrClosed = errors.New("metalx: compiler is closed")

// Compiler owns compute pipelines compiled for one Metal device. Pipelines
// returned by Compile are borrowed and remain valid until Close. The device
// must outlive the Compiler. The zero value is not usable. Compiler methods
// are safe for concurrent use.
type Compiler struct {
	mu     sync.Mutex
	dev    *metal.Device
	cache  map[string]apple.MTLComputePipelineState
	closed bool
}

// NewCompiler returns a pipeline compiler bound to dev.
func NewCompiler(dev *metal.Device) (*Compiler, error) {
	if dev == nil || dev.Raw() == nil || dev.Raw().GetID() == 0 {
		return nil, errors.New("metalx: nil or uninitialized Metal device")
	}
	return &Compiler{dev: dev, cache: make(map[string]apple.MTLComputePipelineState)}, nil
}

// Close releases cached pipelines and prevents new compilation. It is
// idempotent and waits for an in-progress Compile call. Callers must finish
// submitted work using borrowed pipelines before calling Close.
func (c *Compiler) Close() error {
	if c == nil {
		return nil
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return nil
	}
	for _, pso := range c.cache {
		objectivec.ObjectFromID(pso.GetID()).Release()
	}
	c.closed = true
	c.cache = nil
	c.dev = nil
	return nil
}

// CompileLibrary compiles MSL source code into a library owned by the caller.
// The caller must release the library after its functions and pipelines have
// been created. Compiler handles this ownership automatically for Compile.
func CompileLibrary(dev *metal.Device, source string) (apple.MTLLibrary, error) {
	if dev == nil {
		return nil, fmt.Errorf("metalx: nil or uninitialized Metal device")
	}
	lib, err := dev.CompileLibrary(source)
	if err != nil {
		return nil, fmt.Errorf("metalx: compile MSL library: %w", err)
	}
	return lib, nil
}

// Compile compiles source and returns the named compute pipeline. Cache
// identity is scoped to the Compiler's Metal device.
func (c *Compiler) Compile(source, kernelName string) (apple.MTLComputePipelineState, error) {
	if c == nil {
		return nil, errors.New("metalx: nil compiler")
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return nil, ErrClosed
	}
	if c.dev == nil || c.dev.Raw() == nil || c.dev.Raw().GetID() == 0 {
		return nil, errors.New("metalx: uninitialized compiler")
	}
	cacheKey := source + "::" + kernelName
	if pso := c.cache[cacheKey]; pso != nil {
		return pso, nil
	}

	lib, err := CompileLibrary(c.dev, source)
	if err != nil {
		return nil, err
	}
	defer objectivec.ObjectFromID(lib.GetID()).Release()

	newPso, err := c.dev.NewComputePipeline(lib, kernelName)
	if err != nil {
		return nil, fmt.Errorf("metalx: create compute pipeline %q: %w", kernelName, err)
	}

	c.cache[cacheKey] = newPso
	return newPso, nil
}

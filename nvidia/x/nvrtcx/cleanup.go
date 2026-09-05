package nvrtcx

import (
	"errors"
	"fmt"
	"github.com/tmc/lib/nvidia/driver"
	"sync"
)

// programCleanupError retains the program on a failed destruction. It exposes
// io.Closer through errors.As so cleanup can be retried without recompilation.
type programCleanupError struct {
	mu      sync.Mutex
	err     error
	destroy func() error
}

func (e *programCleanupError) Error() string { return e.err.Error() }
func (e *programCleanupError) Unwrap() error { return e.err }
func (e *programCleanupError) Close() error {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.destroy == nil {
		return nil
	}
	if err := e.destroy(); err != nil {
		return fmt.Errorf("nvrtcx: destroy program: %w", err)
	}
	e.destroy = nil
	return nil
}

func finishProgram(result *CompiledModule, compileErr error, destroy func() error) (*CompiledModule, error) {
	if err := destroy(); err != nil {
		return nil, &programCleanupError{
			err:     errors.Join(compileErr, fmt.Errorf("nvrtcx: destroy program: %w", err)),
			destroy: destroy,
		}
	}
	return result, compileErr
}

// moduleCleanupError owns a module whose unload must be retried.
type moduleCleanupError struct {
	mu     sync.Mutex
	err    error
	unload func() error
}

func (e *moduleCleanupError) Error() string { return e.err.Error() }
func (e *moduleCleanupError) Unwrap() error { return e.err }
func (e *moduleCleanupError) Close() error {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.unload == nil {
		return nil
	}
	if err := e.unload(); err != nil {
		return fmt.Errorf("nvrtcx: unload module: %w", err)
	}
	e.unload = nil
	return nil
}

func finishModuleLoad(mod driver.Module, err error, unload func() error) (driver.Module, error) {
	if err == nil {
		return mod, nil
	}
	err = fmt.Errorf("nvrtcx: load module: %w", err)
	if mod == 0 {
		return 0, err
	}
	if cleanupErr := unload(); cleanupErr != nil {
		return 0, &moduleCleanupError{err: errors.Join(err, fmt.Errorf("nvrtcx: unload module: %w", cleanupErr)), unload: unload}
	}
	return 0, err
}

package nvrtcx

import (
	"errors"
	"fmt"
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

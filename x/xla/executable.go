package xla

import (
	"context"
	"errors"
	"fmt"

	"github.com/tmc/lib/openxla"
)

// Executable represents a compiled XLA program ready for execution.
type Executable struct {
	engine  *Engine
	wrapper *openxla.ExecutableWrapper
}

// Execute runs the program on a single device with the given arguments.
// Cancellation before execution prevents dispatch. Cancellation during execution
// is reported after PJRT completes and the outputs have been released; it does
// not interrupt native execution. The executable, engine and input buffers must
// remain open until Execute returns. Shared inputs are never donated to PJRT.
func (e *Executable) Execute(ctx context.Context, args []*Buffer) ([]*Buffer, error) {
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("xla: context error before execution: %w", err)
	}

	if e == nil || e.wrapper == nil {
		return nil, fmt.Errorf("xla: executable is closed")
	}
	rawArgs := make([]*openxla.BufferWrapper, len(args))
	var nonDonatable []int64
	for i, a := range args {
		if a == nil || a.wrapper == nil {
			return nil, fmt.Errorf("xla: nil buffer at argument index %d", i)
		}
		rawArgs[i] = a.wrapper
		if a.IsShared() {
			nonDonatable = append(nonDonatable, int64(i))
		}
	}

	return e.execute(ctx, rawArgs, nonDonatable)
}

func (e *Executable) execute(ctx context.Context, args []*openxla.BufferWrapper, nonDonatable []int64) (outputs []*Buffer, err error) {
	raw, err := e.wrapper.ExecuteWithNonDonatableInputs([][]*openxla.BufferWrapper{args}, nonDonatable)
	if err != nil {
		return nil, fmt.Errorf("xla: execute: %w", err)
	}
	defer func() {
		if err == nil {
			return
		}
		for _, device := range raw {
			for _, out := range device {
				if out != nil {
					err = errors.Join(err, out.Destroy())
				}
			}
		}
	}()
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("xla: execution cancelled: %w", err)
	}
	if len(raw) != 1 {
		return nil, fmt.Errorf("xla: expected 1 device output, got %d", len(raw))
	}
	outputs = make([]*Buffer, len(raw[0]))
	for i, out := range raw[0] {
		typ, dims, err := out.Shape()
		if err != nil {
			return nil, fmt.Errorf("xla: output %d shape: %w", i, err)
		}
		if _, err := bufferSize(typ, dims); err != nil {
			return nil, fmt.Errorf("xla: output %d: %w", i, err)
		}
		outputs[i] = &Buffer{wrapper: out, dims: dims, elemTyp: typ}
	}
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("xla: execution cancelled: %w", err)
	}
	return outputs, nil
}

// Close destroys the executable and frees its resources.
func (e *Executable) Close() error {
	if e == nil || e.wrapper == nil {
		return nil
	}
	resource := e.wrapper
	e.wrapper = nil
	if err := resource.Destroy(); err != nil {
		return fmt.Errorf("xla: close executable: %w", err)
	}
	return nil
}

// CompiledMemoryStats returns the compiler's estimates of this program's memory
// requirements. Values are plugin-dependent and are not live allocator counters.
// The executable and its engine must remain open during the call.
func (e *Executable) CompiledMemoryStats() (openxla.CompiledMemoryStats, error) {
	if e == nil || e.wrapper == nil || e.engine == nil || e.engine.client == nil {
		return openxla.CompiledMemoryStats{}, fmt.Errorf("xla: executable is closed")
	}
	stats, err := e.wrapper.CompiledMemoryStats()
	if err != nil {
		return openxla.CompiledMemoryStats{}, fmt.Errorf("xla: compiled memory statistics: %w", err)
	}
	return stats, nil
}

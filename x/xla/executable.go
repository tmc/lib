package xla

import (
	"context"
	"fmt"

	"github.com/tmc/lib/openxla"
)

// Executable represents a compiled XLA program ready for execution.
type Executable struct {
	engine  *Engine
	wrapper *openxla.ExecutableWrapper
}

// Execute runs the program on a single device with the given arguments.
func (e *Executable) Execute(ctx context.Context, args []*Buffer) ([]*Buffer, error) {
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("xla: context error before execution: %w", err)
	}

	// Validate arguments
	// (For a robust implementation, we would check types/shapes against e.wrapper's expected inputs)
	rawArgs := make([]*openxla.BufferWrapper, len(args))
	for i, a := range args {
		if a == nil || a.wrapper == nil {
			return nil, fmt.Errorf("xla: nil buffer at argument index %d", i)
		}
		rawArgs[i] = a.wrapper
	}

	// We wrap in a channel to support context cancellation/deadlines
	type result struct {
		outputs []*Buffer
		err     error
	}
	ch := make(chan result, 1)

	go func() {
		// Run on a single device (inputLists is [1][]*BufferWrapper)
		rawOuts, err := e.wrapper.Execute([][]*openxla.BufferWrapper{rawArgs})
		if err != nil {
			ch <- result{nil, fmt.Errorf("xla: execute failed: %w", err)}
			return
		}

		if len(rawOuts) != 1 {
			ch <- result{nil, fmt.Errorf("xla: expected 1 device output, got %d", len(rawOuts))}
			return
		}

		outputs := make([]*Buffer, len(rawOuts[0]))
		for i, out := range rawOuts[0] {
			outputs[i] = &Buffer{
				wrapper: out,
				// Ideally we would fetch shape and type from the executable outputs
			}
		}
		ch <- result{outputs, nil}
	}()

	select {
	case <-ctx.Done():
		return nil, fmt.Errorf("xla: execution cancelled: %w", ctx.Err())
	case res := <-ch:
		return res.outputs, res.err
	}
}

// Close destroys the executable and frees its resources.
func (e *Executable) Close() error {
	if e.wrapper != nil {
		if err := e.wrapper.Destroy(); err != nil {
			return fmt.Errorf("xla: failed to close executable: %w", err)
		}
		e.wrapper = nil
	}
	return nil
}

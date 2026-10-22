package dsv4server

import (
	"context"
	"errors"
	"fmt"

	"github.com/tmc/lib/examples/ds4-min/dsv4tools"
)

var (
	ErrToolExecutorMissing = errors.New("tool executor is not configured")
	ErrToolReplayMissing   = errors.New("tool replay entry is not available")
)

// ToolExecutor is the optional host-side execution seam. It receives parsed
// calls with owned argument bytes; the server does not invent an executor.
type ToolExecutor interface {
	Execute(context.Context, dsv4tools.Call) (string, error)
}

// ToolReplay supplies an exact previously sampled DSML wrapper. dsv4tools.Memory
// implements this interface. Replay is separate from execution so callers can
// preserve the original sampled bytes before parsing them again.
type ToolReplay interface {
	Replay(string) (string, bool)
}

// ToolExecution contains the parsed calls and one output per call. Content is
// the non-tool text returned by dsv4tools.Parse. Calls and Arguments preserve
// source order and JSON argument bytes after validation.
type ToolExecution struct {
	Content string
	Calls   []dsv4tools.Call
	Outputs []string
}

// ExecuteToolText parses generated DSML and invokes each call in source order.
// Plain text is returned without requiring an executor; generated tool calls
// fail closed when no executor is supplied.
func ExecuteToolText(ctx context.Context, text string, executor ToolExecutor) (ToolExecution, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	content, calls, err := dsv4tools.Parse(text)
	if err != nil {
		return ToolExecution{}, fmt.Errorf("parse generated tools: %w", err)
	}
	result := ToolExecution{Content: content, Calls: append([]dsv4tools.Call(nil), calls...)}
	if len(calls) == 0 {
		return result, nil
	}
	if executor == nil {
		return ToolExecution{}, ErrToolExecutorMissing
	}
	result.Outputs = make([]string, 0, len(calls))
	for _, call := range calls {
		if err := ctx.Err(); err != nil {
			return ToolExecution{}, err
		}
		output, err := executor.Execute(ctx, call)
		if err != nil {
			return ToolExecution{}, fmt.Errorf("execute tool %q: %w", call.Name, err)
		}
		result.Outputs = append(result.Outputs, output)
	}
	return result, nil
}

// ReplayToolText fetches exact sampled DSML by id, then runs the same parser
// and executor path as ExecuteToolText. A missing replay entry is an error.
func ReplayToolText(ctx context.Context, replay ToolReplay, id string, executor ToolExecutor) (ToolExecution, error) {
	if replay == nil {
		return ToolExecution{}, ErrToolReplayMissing
	}
	text, ok := replay.Replay(id)
	if !ok {
		return ToolExecution{}, ErrToolReplayMissing
	}
	return ExecuteToolText(ctx, text, executor)
}

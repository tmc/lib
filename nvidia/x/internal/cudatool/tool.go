package cudatool

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"os/exec"
	"path/filepath"
	"sync"
	"time"
)

// Output retains the executable, arguments, and both output streams, even on error.
type Output struct {
	Path           string
	Args           []string
	Stdout         []byte
	Stderr         []byte
	ExitCode       int // -1 if no process exit status was available
	Duration       time.Duration
	Truncated      bool
	MaxOutputBytes int64
	Timeout        time.Duration
}

// Limits bounds one tool invocation. Zero fields use 256 MiB aggregate stdout
// and stderr and a 60-second timeout. A shorter caller deadline takes precedence.
type Limits struct {
	MaxOutputBytes int64
	Timeout        time.Duration
}

// ErrOutputLimit means output exceeded the retained byte budget. Truncated
// output must not be used to qualify a successful parse or inspection.
var ErrOutputLimit = errors.New("tool output limit exceeded")

// Run resolves path before changing directories. Cancellation is observable with errors.Is.
func Run(ctx context.Context, path, dir string, stdin io.Reader, args ...string) (Output, error) {
	return RunWithLimits(ctx, Limits{}, path, dir, stdin, args...)
}

// RunWithLimits runs a tool, retaining at most the configured output bytes.
// Budget exhaustion cancels and reaps the direct child. The returned error
// matches ErrOutputLimit, and timeout/cancellation remain observable with Is.
func RunWithLimits(ctx context.Context, limits Limits, path, dir string, stdin io.Reader, args ...string) (Output, error) {
	r := Output{Args: append([]string(nil), args...), ExitCode: -1}
	if limits.MaxOutputBytes < 0 || limits.Timeout < 0 {
		return r, fmt.Errorf("negative tool limit")
	}
	if limits.MaxOutputBytes == 0 {
		limits.MaxOutputBytes = 256 << 20
	}
	if limits.Timeout == 0 {
		limits.Timeout = time.Minute
	}
	r.MaxOutputBytes, r.Timeout = limits.MaxOutputBytes, limits.Timeout
	ctx, stop := context.WithTimeout(ctx, limits.Timeout)
	defer stop()
	if err := ctx.Err(); err != nil {
		return r, err
	}
	p, err := exec.LookPath(lookupCUDA(path))
	if err != nil {
		return r, fmt.Errorf("find %s: %w", path, err)
	}
	r.Path, err = filepath.Abs(p)
	if err != nil {
		return r, err
	}
	cmd := exec.CommandContext(ctx, r.Path, args...)
	cmd.Dir, cmd.Stdin = dir, stdin
	cmd.WaitDelay = time.Second
	var stdout, stderr bytes.Buffer
	budget := &outputBudget{remaining: limits.MaxOutputBytes, cancel: stop}
	cmd.Stdout, cmd.Stderr = limitedOutput{budget, &stdout}, limitedOutput{budget, &stderr}
	started := time.Now()
	err = cmd.Run()
	r.Duration = time.Since(started)
	if cmd.ProcessState != nil {
		r.ExitCode = cmd.ProcessState.ExitCode()
	}
	r.Truncated = budget.truncated
	r.Stdout, r.Stderr = stdout.Bytes(), stderr.Bytes()
	if r.Truncated {
		err = errors.Join(err, ErrOutputLimit)
	}
	if err != nil {
		return r, fmt.Errorf("run %s: %w: %s", r.Path, errors.Join(err, ctx.Err()), r.Stderr)
	}
	return r, nil
}

type outputBudget struct {
	mu        sync.Mutex
	remaining int64
	truncated bool
	cancel    context.CancelFunc
}

type limitedOutput struct {
	budget *outputBudget
	buffer *bytes.Buffer
}

func (w limitedOutput) Write(p []byte) (int, error) {
	w.budget.mu.Lock()
	defer w.budget.mu.Unlock()
	n := len(p)
	if int64(n) > w.budget.remaining {
		n = int(w.budget.remaining)
		w.budget.truncated = true
		w.budget.cancel()
	}
	_, _ = w.buffer.Write(p[:n])
	w.budget.remaining -= int64(n)
	// Drain in-flight pipe writes while cancellation terminates the process.
	// Returning the full input length keeps os/exec's copy goroutines running.
	return len(p), nil
}

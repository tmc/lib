package cudatool

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"os/exec"
	"path/filepath"
	"time"
)

// Output retains the executable, arguments, and both output streams, even on error.
type Output struct {
	Path   string
	Args   []string
	Stdout []byte
	Stderr []byte
}

// Run resolves path before changing directories. Cancellation is observable with errors.Is.
func Run(ctx context.Context, path, dir string, stdin io.Reader, args ...string) (Output, error) {
	r := Output{Args: append([]string(nil), args...)}
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
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	err = cmd.Run()
	r.Stdout, r.Stderr = stdout.Bytes(), stderr.Bytes()
	if err != nil {
		return r, fmt.Errorf("run %s: %w: %s", r.Path, errors.Join(err, ctx.Err()), r.Stderr)
	}
	return r, nil
}

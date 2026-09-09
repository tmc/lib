// Package cudatool runs CUDA binary utilities without a shell.
package cudatool

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
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

// cudaToolDirs are the default CUDA toolkit locations searched for a bare tool
// name that is not on PATH.
var cudaToolDirs = []string{
	"/usr/local/cuda/bin",
	"/usr/local/cuda-13.0/bin",
	"/usr/local/cuda-12.0/bin",
}

// lookupCUDA returns an executable in a default CUDA toolkit directory when
// name is a bare tool name. Otherwise it returns name unchanged, leaving
// resolution to exec.LookPath.
func lookupCUDA(name string) string {
	if filepath.IsAbs(name) || strings.ContainsRune(name, filepath.Separator) {
		return name
	}
	for _, dir := range cudaToolDirs {
		p := filepath.Join(dir, name)
		if fi, err := os.Stat(p); err == nil && !fi.IsDir() && fi.Mode()&0111 != 0 {
			return p
		}
	}
	return name
}

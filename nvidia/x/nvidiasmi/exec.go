package nvidiasmi

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io/fs"
	"os/exec"
	"strings"
)

// command is the nvidia-smi program run by Query, QueryTopology,
// and Reset. Tests replace it with a fake.
var command = "nvidia-smi"

// Errors matched by [Error] through [errors.Is], one per exit status
// documented in the RETURN VALUE section of nvidia-smi(1)
// (https://docs.nvidia.com/deploy/nvidia-smi/index.html). An Error
// with exit status 4 also matches [fs.ErrPermission].
var (
	ErrInvalidArgument = errors.New("nvidia-smi: invalid argument or flag")             // 2
	ErrNotSupported    = errors.New("nvidia-smi: operation not available on device")    // 3
	ErrNotFound        = errors.New("nvidia-smi: object not found")                     // 6
	ErrDriverNotLoaded = errors.New("nvidia-smi: NVIDIA driver is not loaded")          // 9
	ErrGPULost         = errors.New("nvidia-smi: GPU has fallen off the bus")           // 15
	ErrUnknown         = errors.New("nvidia-smi: other error or internal driver error") // 255
)

// An Error reports that nvidia-smi exited with a nonzero status.
// nvidia-smi writes some failure reasons to standard output rather
// than standard error, so both are kept.
type Error struct {
	Args   []string // arguments after "nvidia-smi"
	Code   int      // exit status
	Stdout []byte
	Stderr []byte
}

func (e *Error) Error() string {
	msg := bytes.TrimSpace(e.Stderr)
	if len(msg) == 0 {
		msg = bytes.TrimSpace(e.Stdout)
	}
	s := fmt.Sprintf("nvidia-smi %s: exit status %d", strings.Join(e.Args, " "), e.Code)
	if len(msg) > 0 {
		s += ": " + strings.Join(strings.Fields(string(msg)), " ")
	}
	return s
}

// Is reports whether target is the error for e's exit status.
func (e *Error) Is(target error) bool {
	switch target {
	case ErrInvalidArgument:
		return e.Code == 2
	case ErrNotSupported:
		return e.Code == 3
	case fs.ErrPermission:
		return e.Code == 4
	case ErrNotFound:
		return e.Code == 6
	case ErrDriverNotLoaded:
		return e.Code == 9
	case ErrGPULost:
		return e.Code == 15
	case ErrUnknown:
		return e.Code == 255
	}
	return false
}

// run runs nvidia-smi with args and returns its standard output.
// A nonzero exit status is returned as an *Error.
func run(ctx context.Context, args ...string) ([]byte, error) {
	var stdout, stderr bytes.Buffer
	cmd := exec.CommandContext(ctx, command, args...)
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	err := cmd.Run()
	var ee *exec.ExitError
	if errors.As(err, &ee) && ee.Exited() {
		return nil, &Error{
			Args:   args,
			Code:   ee.ExitCode(),
			Stdout: stdout.Bytes(),
			Stderr: stderr.Bytes(),
		}
	}
	if err != nil {
		return nil, fmt.Errorf("run nvidia-smi: %w", err)
	}
	return stdout.Bytes(), nil
}

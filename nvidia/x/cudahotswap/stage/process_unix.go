//go:build linux || darwin

package stage

import (
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"syscall"
	"time"
)

func configureCompiler(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	cmd.Cancel = func() error {
		if cmd.Process == nil {
			return os.ErrProcessDone
		}
		err := syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL)
		if errors.Is(err, syscall.ESRCH) {
			return os.ErrProcessDone
		}
		return err
	}
	cmd.WaitDelay = 3 * time.Second
}

// WaitDelay closes inherited pipes but does not stop a surviving child. End
// the owned process group even when its leader exited successfully.
func finishCompiler(cmd *exec.Cmd) error {
	if cmd.Process == nil {
		return nil
	}
	err := syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL)
	if errors.Is(err, syscall.ESRCH) {
		return nil
	}
	return err
}

const maxCompilerImage = 64 << 20

func readCompilerImage(path string) ([]byte, error) {
	fd, err := syscall.Open(path, syscall.O_RDONLY|syscall.O_NONBLOCK|syscall.O_NOFOLLOW, 0)
	if err != nil {
		return nil, err
	}
	f := os.NewFile(uintptr(fd), path)
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		return nil, err
	}
	if !info.Mode().IsRegular() || info.Size() > maxCompilerImage {
		return nil, fmt.Errorf("compiler output must be a regular file of at most %d bytes", maxCompilerImage)
	}
	data, err := io.ReadAll(io.LimitReader(f, maxCompilerImage+1))
	if err != nil {
		return nil, err
	}
	if len(data) > maxCompilerImage {
		return nil, fmt.Errorf("compiler output exceeds %d bytes", maxCompilerImage)
	}
	return data, nil
}

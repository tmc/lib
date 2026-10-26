//go:build unix

package main

import (
	"errors"
	"os/exec"
	"syscall"
)

// processGroup is the group a tool is started in; 0 makes a new one. Tests
// set it to a group the process may not join, to exercise the fallback.
var processGroup = 0

func setProcessGroup(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true, Pgid: processGroup}
	cmd.Cancel = func() error {
		return syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL)
	}
}

// isGroupError reports whether a start failure is setpgid being refused.
func isGroupError(err error) bool {
	return errors.Is(err, syscall.EPERM) || errors.Is(err, syscall.EACCES)
}

//go:build unix

package main

import (
	"os/exec"
	"syscall"
)

// setProcessGroup makes the timeout kill the command's whole process group,
// so a test binary that go test started dies with it.
func setProcessGroup(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	cmd.Cancel = func() error { return syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL) }
}

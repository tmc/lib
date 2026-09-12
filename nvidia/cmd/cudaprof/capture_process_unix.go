//go:build !windows

package main

import (
	"os"
	"os/exec"
	"syscall"
)

func configureCaptureProcess(cmd *exec.Cmd) error {
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	return nil
}

func signalCaptureProcess(p *os.Process, sig syscall.Signal) { _ = syscall.Kill(-p.Pid, sig) }

func killCaptureProcess(p *os.Process) { _ = syscall.Kill(-p.Pid, syscall.SIGKILL) }

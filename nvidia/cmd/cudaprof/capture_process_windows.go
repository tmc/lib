//go:build windows

package main

import (
	"os"
	"os/exec"
	"syscall"
)

func configureCaptureProcess(cmd *exec.Cmd) error { return nil }

func signalCaptureProcess(p *os.Process, sig syscall.Signal) { _ = p.Signal(os.Interrupt) }

func killCaptureProcess(p *os.Process) { _ = p.Kill() }

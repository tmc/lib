//go:build !windows

package main

import (
	"os"
	"syscall"
)

func signalCaptureTestProcess() { _ = syscall.Kill(os.Getpid(), syscall.SIGTERM) }

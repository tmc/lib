//go:build windows

package main

import "os"

func signalCaptureTestProcess() {
	p, err := os.FindProcess(os.Getpid())
	if err == nil {
		_ = p.Signal(os.Interrupt)
	}
}

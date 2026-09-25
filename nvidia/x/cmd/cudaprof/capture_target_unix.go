//go:build unix

package main

import (
	"syscall"

	"golang.org/x/sys/unix"
)

// signalName maps a signal as captureResult.TargetSignal records it, its
// description such as "segmentation fault", to its name, such as SIGSEGV.
// A description it does not know is returned unchanged.
func signalName(desc string) string {
	for sig := syscall.Signal(1); sig < 65; sig++ {
		if sig.String() == desc {
			if name := unix.SignalName(sig); name != "" {
				return name
			}
		}
	}
	return desc
}

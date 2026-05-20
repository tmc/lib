//go:build windows

package loader

import (
	"runtime"
	"syscall"
)

var ErrUnsupported = errUnsupported{}

type errUnsupported struct{}

func (errUnsupported) Error() string { return "vulkan: unsupported platform" }

func open(name string) (uintptr, error) {
	dll, err := syscall.LoadDLL(name)
	if err != nil {
		return 0, err
	}
	return uintptr(dll.Handle), nil
}

func runtimeGOOS() string { return runtime.GOOS }

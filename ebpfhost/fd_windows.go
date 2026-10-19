//go:build windows

package ebpfhost

import (
	"errors"
	"syscall"
)

type traceFD syscall.Handle

func openTracePipe(string) (traceFD, error) {
	return traceFD(syscall.InvalidHandle), errors.New("tracefs collector requires linux/amd64 or linux/arm64")
}

func readTracePipe(fd traceFD, p []byte) (int, error) {
	return syscall.Read(syscall.Handle(fd), p)
}

func closeTracePipe(fd traceFD) error {
	return syscall.Close(syscall.Handle(fd))
}

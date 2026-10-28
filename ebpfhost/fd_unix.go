//go:build !windows

package ebpfhost

import "syscall"

type traceFD int

func openTracePipe(path string) (traceFD, error) {
	fd, err := syscall.Open(path, syscall.O_RDONLY|syscall.O_NONBLOCK|syscall.O_CLOEXEC, 0)
	return traceFD(fd), err
}

func readTracePipe(fd traceFD, p []byte) (int, error) {
	return syscall.Read(int(fd), p)
}

func closeTracePipe(fd traceFD) error {
	return syscall.Close(int(fd))
}

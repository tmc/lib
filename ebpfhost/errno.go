package ebpfhost

import "syscall"

func errWouldBlock() error { return syscall.EAGAIN }

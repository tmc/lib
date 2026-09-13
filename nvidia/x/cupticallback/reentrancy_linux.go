//go:build linux

package cupticallback

import "syscall"

func currentThreadID() uint64 {
	tid, _, _ := syscall.RawSyscall(syscall.SYS_GETTID, 0, 0, 0)
	return uint64(tid)
}

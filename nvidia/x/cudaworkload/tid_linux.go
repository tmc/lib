//go:build linux

package cudaworkload

import "golang.org/x/sys/unix"

func currentTID() uint64 {
	return uint64(unix.Gettid())
}

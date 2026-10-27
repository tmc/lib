//go:build unix

package gpuhealth

import (
	"io/fs"
	"syscall"

	"golang.org/x/sys/unix"
)

func deviceNumbers(fi fs.FileInfo) (major, minor uint32) {
	st, ok := fi.Sys().(*syscall.Stat_t)
	if !ok {
		return 0, 0
	}
	return unix.Major(uint64(st.Rdev)), unix.Minor(uint64(st.Rdev))
}

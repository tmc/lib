//go:build linux

package cudatracehost

import (
	"os"
	"syscall"
)

func sameMappedFile(info os.FileInfo, mappedInode uint64, mappedDev string) bool {
	if info == nil {
		return false
	}
	stat, ok := info.Sys().(*syscall.Stat_t)
	if !ok || stat == nil {
		return false
	}
	return sameMappedIdentity(stat.Ino, uint64(stat.Dev), mappedInode, mappedDev)
}

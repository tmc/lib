package cudacache

import (
	"os"
	"syscall"
)

func unixFileIdentity(fi os.FileInfo) (dev, ino uint64, ctimeNanos int64, ok bool) {
	st, ok := fi.Sys().(*syscall.Stat_t)
	if !ok {
		return 0, 0, 0, false
	}
	return uint64(st.Dev), st.Ino, st.Ctimespec.Sec*1e9 + st.Ctimespec.Nsec, true
}

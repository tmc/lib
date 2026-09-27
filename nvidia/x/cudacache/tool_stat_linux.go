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
	return uint64(st.Dev), uint64(st.Ino), int64(st.Ctim.Sec)*1e9 + int64(st.Ctim.Nsec), true
}

//go:build !linux && !darwin

package cudacache

import "os"

func unixFileIdentity(fi os.FileInfo) (dev, ino uint64, ctimeNanos int64, ok bool) {
	return 0, 0, 0, false
}

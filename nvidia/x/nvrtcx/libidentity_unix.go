//go:build linux || darwin

package nvrtcx

import (
	"fmt"
	"os"
	"syscall"
)

func fileSerial(fi os.FileInfo) string {
	st, ok := fi.Sys().(*syscall.Stat_t)
	if !ok {
		return ""
	}
	return fmt.Sprintf(" dev=%d ino=%d", st.Dev, st.Ino)
}

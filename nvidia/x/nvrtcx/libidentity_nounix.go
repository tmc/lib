//go:build !linux && !darwin

package nvrtcx

import "os"

func fileSerial(fi os.FileInfo) string { return "" }

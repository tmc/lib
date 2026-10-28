//go:build !linux

package cudatracehost

import "os"

func sameMappedFile(os.FileInfo, uint64, string) bool {
	return false
}

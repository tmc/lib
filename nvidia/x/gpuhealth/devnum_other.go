//go:build !unix

package gpuhealth

import "io/fs"

func deviceNumbers(fs.FileInfo) (major, minor uint32) { return 0, 0 }

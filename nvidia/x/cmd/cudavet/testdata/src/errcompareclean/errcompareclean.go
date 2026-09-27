// Package errcompareclean does not import cufile, so its errno and fs
// comparisons are left to other checkers.
package errcompareclean

import (
	"io/fs"
	"os"
	"syscall"
)

func missing(name string) bool {
	_, err := os.Stat(name)
	return err == fs.ErrNotExist || err == syscall.ENOENT
}

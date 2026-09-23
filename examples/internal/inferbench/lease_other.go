//go:build !aix && !darwin && !dragonfly && !freebsd && !linux && !netbsd && !openbsd && !solaris

package inferbench

import (
	"errors"
	"os"
)

func lockFile(*os.File) error   { return errors.New("file locking is unsupported") }
func unlockFile(*os.File) error { return nil }

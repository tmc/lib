//go:build aix || darwin || dragonfly || freebsd || linux || netbsd || openbsd || solaris

package inferbench

import (
	"fmt"
	"os"
	"syscall"
)

func lockFile(f *os.File) error {
	if err := syscall.Flock(int(f.Fd()), syscall.LOCK_EX|syscall.LOCK_NB); err != nil {
		return fmt.Errorf("lock: %w", err)
	}
	return nil
}

func unlockFile(f *os.File) error {
	if err := syscall.Flock(int(f.Fd()), syscall.LOCK_UN); err != nil {
		return fmt.Errorf("unlock: %w", err)
	}
	return nil
}

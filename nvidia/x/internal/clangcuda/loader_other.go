//go:build !aix && !android && !darwin && !dragonfly && !freebsd && !hurd && !illumos && !ios && !linux && !netbsd && !openbsd && !solaris && !windows

package clangcuda

import "fmt"

func findLibclang() (string, error) {
	return "", fmt.Errorf("clangcuda: libclang discovery is unavailable on this platform")
}

func loadLibrary(string) (uintptr, error) {
	return 0, fmt.Errorf("clangcuda: libclang loading is unavailable on this platform")
}

func closeLibrary(uintptr) error { return nil }

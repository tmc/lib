//go:build windows

package clangcuda

import (
	"fmt"
	"os"
	"os/exec"
	"strings"

	"golang.org/x/sys/windows"
)

func findLibclang() (string, error) {
	if path := os.Getenv("LIBCLANG_PATH"); path != "" {
		return path, nil
	}
	if clang, err := exec.LookPath("clang"); err == nil {
		for _, name := range []string{"libclang.dll", "clang.dll"} {
			if out, err := exec.Command(clang, "-print-file-name="+name).Output(); err == nil {
				path := strings.TrimSpace(string(out))
				if path != "" && path != name {
					return path, nil
				}
			}
		}
	}
	return "", fmt.Errorf("clangcuda: find libclang.dll")
}

func loadLibrary(path string) (uintptr, error) {
	handle, err := windows.LoadLibrary(path)
	return uintptr(handle), err
}

func closeLibrary(handle uintptr) error {
	return windows.FreeLibrary(windows.Handle(handle))
}

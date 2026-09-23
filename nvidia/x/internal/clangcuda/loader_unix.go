//go:build aix || android || darwin || dragonfly || freebsd || hurd || illumos || ios || linux || netbsd || openbsd || solaris

package clangcuda

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/ebitengine/purego"
)

func findLibclang() (string, error) {
	if path := os.Getenv("LIBCLANG_PATH"); path != "" {
		if info, err := os.Stat(path); err == nil && !info.IsDir() {
			return path, nil
		}
	}
	clang, err := exec.LookPath("clang")
	if err != nil {
		return "", fmt.Errorf("clangcuda: find clang: %w", err)
	}
	for _, name := range []string{"libclang.so", "libclang.dylib"} {
		if out, err := exec.Command(clang, "-print-file-name="+name).Output(); err == nil {
			path := strings.TrimSpace(string(out))
			if path != "" && path != name {
				if _, err := os.Stat(path); err == nil {
					return path, nil
				}
			}
		}
	}
	if llvmConfig, err := exec.LookPath("llvm-config"); err == nil {
		if out, err := exec.Command(llvmConfig, "--libdir").Output(); err == nil {
			dir := strings.TrimSpace(string(out))
			for _, name := range []string{"libclang.so", "libclang.dylib"} {
				path := filepath.Join(dir, name)
				if _, err := os.Stat(path); err == nil {
					return path, nil
				}
			}
		}
	}
	return "", fmt.Errorf("clangcuda: find libclang for %s", clang)
}

func loadLibrary(path string) (uintptr, error) {
	return purego.Dlopen(path, purego.RTLD_LAZY|purego.RTLD_LOCAL)
}

func closeLibrary(handle uintptr) error { return purego.Dlclose(handle) }

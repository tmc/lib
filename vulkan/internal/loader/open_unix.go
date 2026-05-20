//go:build darwin || linux || freebsd || netbsd

package loader

import (
	"runtime"

	"github.com/ebitengine/purego"
)

var ErrUnsupported = errUnsupported{}

type errUnsupported struct{}

func (errUnsupported) Error() string { return "vulkan: unsupported platform" }

func open(name string) (uintptr, error) {
	return purego.Dlopen(name, purego.RTLD_NOW|purego.RTLD_GLOBAL)
}

func runtimeGOOS() string { return runtime.GOOS }

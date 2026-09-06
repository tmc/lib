//go:build !windows

package nvtx

import "github.com/ebitengine/purego"

func openDynamicLibrary(path string) (uintptr, error) {
	return purego.Dlopen(path, purego.RTLD_NOW|purego.RTLD_GLOBAL)
}

func lookupLibrarySymbol(handle uintptr, name string) (uintptr, error) {
	return purego.Dlsym(handle, name)
}

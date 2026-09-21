//go:build !windows

package nvtx

import "github.com/ebitengine/purego"

func openDynamicLibrary(path string) (uintptr, error) {
	// Resolve provider dependencies before initInjection calls its entry point.
	// A load error leaves instrumentation disabled. Keep this distinct from
	// the generated driver loader's lazy relocation policy.
	return purego.Dlopen(path, purego.RTLD_NOW|purego.RTLD_GLOBAL)
}

func lookupLibrarySymbol(handle uintptr, name string) (uintptr, error) {
	return purego.Dlsym(handle, name)
}

//go:build linux || darwin

package stage

import (
	"fmt"

	"github.com/ebitengine/purego"
)

func openLibrary(path string) (uintptr, error) {
	return purego.Dlopen(path, purego.RTLD_NOW|purego.RTLD_LOCAL)
}

func closeLibrary(handle uintptr) error { return purego.Dlclose(handle) }

func bindEntry(handle uintptr, name string, target any) error {
	if _, err := entryType(target); err != nil {
		return err
	}
	address, err := purego.Dlsym(handle, name)
	if err != nil || address == 0 {
		return fmt.Errorf("bind stage entry %s: missing symbol: %v", name, err)
	}
	purego.RegisterFunc(target, address)
	return nil
}

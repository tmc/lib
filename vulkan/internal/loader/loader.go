package loader

import (
	"fmt"
	"os"

	"github.com/ebitengine/purego"
)

const EnvLibraryPath = "VULKAN_LIBRARY_PATH"

// Open loads the Vulkan loader. An explicit path overrides VULKAN_LIBRARY_PATH.
func Open(path string) (uintptr, error) {
	if path != "" {
		return open(path)
	}
	if env := os.Getenv(EnvLibraryPath); env != "" {
		return open(env)
	}
	var last error
	for _, name := range defaultLibraryNames() {
		lib, err := open(name)
		if err == nil {
			return lib, nil
		}
		last = err
	}
	if last != nil {
		return 0, fmt.Errorf("vulkan: open loader: %w", last)
	}
	return 0, ErrUnsupported
}

// Register binds name in lib to fn.
func Register(lib uintptr, fn any, name string) (err error) {
	defer func() {
		if v := recover(); v != nil {
			err = fmt.Errorf("vulkan: register %s: %v", name, v)
		}
	}()
	purego.RegisterLibFunc(fn, lib, name)
	return nil
}

func defaultLibraryNames() []string {
	switch runtimeGOOS() {
	case "linux":
		return []string{"libvulkan.so.1", "libvulkan.so"}
	case "windows":
		return []string{"vulkan-1.dll"}
	case "darwin":
		return []string{"libvulkan.1.dylib", "libvulkan.dylib"}
	default:
		return nil
	}
}

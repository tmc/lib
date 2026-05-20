package abi

import (
	"fmt"
	"unsafe"

	"github.com/tmc/lib/vulkan/internal/loader"
)

var (
	CreateInstance           func(*InstanceCreateInfo, unsafe.Pointer, *Instance) Result
	DestroyInstance          func(Instance, unsafe.Pointer)
	EnumeratePhysicalDevices func(Instance, *uint32, unsafe.Pointer) Result
)

// Register binds Vulkan entry points from lib.
func Register(lib uintptr) error {
	for name, fn := range map[string]any{
		"vkCreateInstance":           &CreateInstance,
		"vkDestroyInstance":          &DestroyInstance,
		"vkEnumeratePhysicalDevices": &EnumeratePhysicalDevices,
	} {
		if err := loader.Register(lib, fn, name); err != nil {
			return fmt.Errorf("register %s: %w", name, err)
		}
	}
	return nil
}

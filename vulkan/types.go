package vulkan

// ApplicationInfo describes the application creating a Vulkan instance.
type ApplicationInfo struct {
	ApplicationName    string
	ApplicationVersion uint32
	EngineName         string
	EngineVersion      uint32
	APIVersion         uint32
}

// InstanceCreateInfo describes a Vulkan instance to create.
//
// The generated wrapper sets the Vulkan sType fields automatically.
type InstanceCreateInfo struct {
	Flags                 InstanceCreateFlags
	ApplicationInfo       *ApplicationInfo
	EnabledLayerNames     []string
	EnabledExtensionNames []string
}

// InstanceCreateFlags contains Vulkan instance creation flags.
type InstanceCreateFlags uint32

// Instance is an owned Vulkan instance handle.
type Instance struct {
	handle uintptr
	closed bool
}

// PhysicalDevice is a borrowed Vulkan physical-device handle.
type PhysicalDevice struct {
	handle uintptr
}

// Valid reports whether d is a non-zero physical-device handle.
func (d PhysicalDevice) Valid() bool {
	return d.handle != 0
}

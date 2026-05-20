package abi

import "unsafe"

type Result int32
type StructureType int32
type Instance uintptr
type PhysicalDevice uintptr
type InstanceCreateFlags uint32

const (
	StructureTypeApplicationInfo    StructureType = 0
	StructureTypeInstanceCreateInfo StructureType = 1
)

type ApplicationInfo struct {
	SType              StructureType
	PNext              unsafe.Pointer
	PApplicationName   *byte
	ApplicationVersion uint32
	PEngineName        *byte
	EngineVersion      uint32
	APIVersion         uint32
}

type InstanceCreateInfo struct {
	SType                   StructureType
	PNext                   unsafe.Pointer
	Flags                   InstanceCreateFlags
	PApplicationInfo        *ApplicationInfo
	EnabledLayerCount       uint32
	PpEnabledLayerNames     unsafe.Pointer
	EnabledExtensionCount   uint32
	PpEnabledExtensionNames unsafe.Pointer
}

// Registry structure values verified from vk.xml:
// VkApplicationInfo uses VK_STRUCTURE_TYPE_APPLICATION_INFO.
// VkInstanceCreateInfo uses VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO.

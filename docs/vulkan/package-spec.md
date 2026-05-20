# Package Spec: vulkan

## Summary

Package `vulkan` provides a generated pure-Go first slice of the Vulkan API.
It is generated from Khronos `vk.xml` and is intentionally limited to
loader, instance lifecycle, and physical-device discovery.

The selected command profile is small. `vk.xml` supplies result values,
structure-type values, pointer depth, array/null-terminated length metadata,
handles, nested structs, and ABI function signatures.

## Public API

- `type Result int32`
- `type ApplicationInfo struct`
- `type InstanceCreateInfo struct`
- `type Instance struct`
- `type PhysicalDevice struct`
- `func Load() error`
- `func CreateInstance(*InstanceCreateInfo) (*Instance, error)`
- `func (*Instance) Close() error`
- `func (*Instance) EnumeratePhysicalDevices() ([]PhysicalDevice, error)`
- `func MakeAPIVersion(variant, major, minor, patch uint32) uint32`

The public create-info types do not expose Vulkan `sType` fields. The generated
wrapper fills them from registry metadata.

## Semantic Sources

- Khronos `vk.xml`: canonical registry input.
- `github.com/bbredesen/go-vk`: reference for Go-style wrappers over vk.xml.
- `github.com/vulkan-go/vulkan`: reference for broad coverage and cgo pitfalls.
- `github.com/gogpu/wgpu`: reference for pure-Go GPU backend pressure.

Semantic sources are not ABI inputs and are not linked into the package.

## Deferred

- WSI and platform surface packages.
- Logical devices, queues, command pools, command buffers, memory, and pipelines.
- SPIR-V, shader modules, validation layers, debug callbacks, pNext chains, and unions.

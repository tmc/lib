package vulkan

const (
	APIVersion10 = uint32(1 << 22)
)

// MakeAPIVersion returns a Vulkan API version number.
func MakeAPIVersion(variant, major, minor, patch uint32) uint32 {
	return (variant << 29) | (major << 22) | (minor << 12) | patch
}

// APIVersionMajor returns the major component of a Vulkan API version.
func APIVersionMajor(v uint32) uint32 {
	return (v >> 22) & 0x7f
}

// APIVersionMinor returns the minor component of a Vulkan API version.
func APIVersionMinor(v uint32) uint32 {
	return (v >> 12) & 0x3ff
}

// APIVersionPatch returns the patch component of a Vulkan API version.
func APIVersionPatch(v uint32) uint32 {
	return v & 0xfff
}

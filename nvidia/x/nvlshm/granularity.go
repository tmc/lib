package nvlshm

import (
	"fmt"
	"math"

	"github.com/tmc/lib/nvidia/driver"
)

// allocProp returns the allocation properties used for every nvlshm chunk:
// pinned device memory on dev, exportable as a POSIX file descriptor.
func allocProp(dev driver.Device) driver.MemAllocationProp {
	return driver.MemAllocationProp{
		Type:                 driver.MemAllocationTypePinned,
		RequestedHandleTypes: driver.MemHandleTypePosixFileDescriptor,
		Location: driver.MemLocation{
			Type: driver.MemLocationTypeDevice,
			Id:   int32(dev),
		},
	}
}

// Granularity returns the recommended VMM allocation granularity in bytes
// for chunks on dev. Chunk sizes passed to the alloc functions must be a
// multiple of it.
func Granularity(dev driver.Device) (uintptr, error) {
	prop := allocProp(dev)
	g, err := driver.MemGetAllocationGranularity(&prop, driver.MemAllocGranularityRecommended)
	if err != nil {
		return 0, fmt.Errorf("cuMemGetAllocationGranularity: %w", err)
	}
	return uintptr(g), nil
}

// MulticastGranularity returns the recommended multicast granularity in
// bytes for a group of numDevices devices. Multicast bind sizes must be a
// multiple of it; chunks for [DistAllocMulticast] must be padded to
// max(Granularity, MulticastGranularity).
func MulticastGranularity(numDevices int) (uintptr, error) {
	prop := driver.MulticastObjectProp{
		NumDevices:  uint32(numDevices),
		HandleTypes: uint64(driver.MemHandleTypePosixFileDescriptor),
	}
	g, err := driver.MulticastGetGranularity(&prop, driver.MulticastGranularityRecommended)
	if err != nil {
		return 0, fmt.Errorf("cuMulticastGetGranularity: %w", err)
	}
	return uintptr(g), nil
}

// MulticastSupported reports whether dev supports NVSwitch multicast.
func MulticastSupported(dev driver.Device) (bool, error) {
	v, err := dev.GetAttribute(driver.DevAttrMulticastSupported)
	if err != nil {
		return false, fmt.Errorf("cuDeviceGetAttribute(MULTICAST_SUPPORTED): %w", err)
	}
	return v != 0, nil
}

// PadTo rounds n up to the next multiple of granularity.
func PadTo(n, granularity uintptr) (uintptr, error) {
	if granularity == 0 {
		return 0, fmt.Errorf("nvlshm: zero granularity")
	}
	if n > uintptr(math.MaxUint)-granularity+1 {
		return 0, fmt.Errorf("nvlshm: padded size overflows uintptr")
	}
	return (n + granularity - 1) / granularity * granularity, nil
}

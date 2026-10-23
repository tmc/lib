package nvml_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/nvml"
)

func ExampleDevice_ComputeRunningProcessesInto() {
	// The caller owns the storage and its length is the only capacity used.
	// An empty buffer is refused before any library or driver call, so this
	// example needs neither; it shows no device data.
	var buf []nvml.ProcessInfo
	n, err := nvml.Device(0).ComputeRunningProcessesInto(buf)
	fmt.Println(n, err)
	// Output: 0 query compute processes: empty buffer
}

func ExampleDevice_GraphicsRunningProcessesInto() {
	// A zero-length buffer is refused before any library or driver call, so
	// this example needs neither; it shows no device data.
	buf := []nvml.ProcessInfo{}
	n, err := nvml.Device(0).GraphicsRunningProcessesInto(buf)
	fmt.Println(n, err)
	// Output: 0 query graphics processes: empty buffer
}

func ExampleDevice_MPSComputeRunningProcessesInto() {
	// Capacity beyond the length is not used: the length is zero, so the
	// buffer is refused before any library or driver call. This example needs
	// neither and shows no device data.
	buf := make([]nvml.ProcessInfo, 0, 64)
	n, err := nvml.Device(0).MPSComputeRunningProcessesInto(buf)
	fmt.Println(n, err)
	// Output: 0 query mps compute processes: empty buffer
}

// ExampleDevice_ComputeRunningProcessesInto_device requires an NVML driver
// and a device. It makes one bounded call; if more than 64 processes hold a
// compute context, it reports an error instead of a truncated list.
func ExampleDevice_ComputeRunningProcessesInto_device() {
	if err := nvml.Init(); err != nil {
		fmt.Println("NVML unavailable:", err)
		return
	}
	defer nvml.Shutdown()

	device, err := nvml.NewDeviceByIndex(0)
	if err != nil {
		fmt.Println(err)
		return
	}
	buf := make([]nvml.ProcessInfo, 64)
	n, err := device.ComputeRunningProcessesInto(buf)
	if err != nil {
		fmt.Println(err)
		return
	}
	for _, p := range buf[:n] {
		fmt.Println(p.Pid)
	}
}

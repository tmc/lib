package nvml_test

import (
	"errors"
	"fmt"

	"github.com/tmc/lib/nvidia/nvml"
)

func ExampleIsUnsupported() {
	err := errors.Join(nvml.ErrNoPermission, &nvml.SymbolError{Symbol: "nvmlNewQuery"})
	fmt.Println(nvml.IsUnsupported(err))
	fmt.Println(nvml.IsUnsupported(nvml.ErrGpuIsLost))
	// Output:
	// true
	// false
}

func ExampleDevice_FieldValuesFor() {
	// Empty requests require neither a loaded driver nor a device.
	values, err := nvml.Device(0).FieldValuesFor()
	fmt.Println(len(values), err)
	// Output: 0 <nil>
}

func ExampleGpmMetrics() {
	metrics, err := nvml.GpmMetrics(0, 0)
	fmt.Println(len(metrics), err)
	// Output: 0 <nil>
}

// ExampleDevice_GpmSupported requires an initialized NVML driver and device.
func ExampleDevice_GpmSupported() {
	device, err := nvml.NewDeviceByIndex(0)
	if err != nil {
		return
	}
	supported, err := device.GpmSupported()
	fmt.Println(supported, err)
}

// ExampleDevice_FabricInfo requires an initialized NVML driver and device.
func ExampleDevice_FabricInfo() {
	device, err := nvml.NewDeviceByIndex(0)
	if err != nil {
		return
	}
	info, err := device.FabricInfo()
	fmt.Println(info.State, info.Status, err)
}

// ExampleDevice_ProcessesUtilization requires an initialized NVML driver.
func ExampleDevice_ProcessesUtilization() {
	device, err := nvml.NewDeviceByIndex(0)
	if err != nil {
		return
	}
	samples, err := device.ProcessesUtilization(0)
	fmt.Println(len(samples), err)
}

// ExampleDevice_GpuInstanceList requires an initialized MIG device.
func ExampleDevice_GpuInstanceList() {
	device, err := nvml.NewDeviceByIndex(0)
	if err != nil {
		return
	}
	profile, err := device.GpuInstanceProfileInfo(0)
	if err != nil {
		return
	}
	instances, err := device.GpuInstanceList(profile.Id)
	fmt.Println(len(instances), err)
}

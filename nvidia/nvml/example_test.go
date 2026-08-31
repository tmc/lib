package nvml_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/nvml"
)

// Example enumerates the GPUs on the system and prints basic telemetry
// for each. It returns early when the NVML library is not available.
func Example() {
	if err := nvml.Init(); err != nil {
		fmt.Println("NVML unavailable:", err)
		return
	}
	defer nvml.Shutdown()

	count, err := nvml.DeviceGetCount()
	if err != nil {
		fmt.Println("device count:", err)
		return
	}
	for i := uint32(0); i < count; i++ {
		dev, err := nvml.NewDeviceByIndex(i)
		if err != nil {
			continue
		}
		name, _ := dev.GetName()
		mem, _ := dev.GetMemoryInfo()
		util, _ := dev.GetUtilizationRates()
		fmt.Printf("gpu %d: %s, %d/%d MiB used, %d%% busy\n",
			i, name, mem.Used>>20, mem.Total>>20, util.Gpu)
	}
}

// ExampleDevice_GetComputeRunningProcesses lists the compute processes on
// a device. Passing the previous result back in reuses its backing array,
// so a polling loop settles into zero allocations.
func ExampleDevice_GetComputeRunningProcesses() {
	if err := nvml.Init(); err != nil {
		fmt.Println("NVML unavailable:", err)
		return
	}
	defer nvml.Shutdown()

	dev, err := nvml.NewDeviceByIndex(0)
	if err != nil {
		fmt.Println("no device:", err)
		return
	}
	var procs []nvml.ProcessInfo
	for i := 0; i < 3; i++ {
		procs, err = dev.GetComputeRunningProcesses(procs)
		if err != nil {
			return
		}
		fmt.Println("processes:", len(procs))
	}
}

// Nvml-info prints basic NVIDIA management information without changing device state.
package main

import (
	"fmt"
	"log"

	"github.com/tmc/lib/nvidia/nvml"
)

func main() {
	if err := nvml.Init(); err != nil {
		log.Fatal(err)
	}
	defer nvml.Shutdown()
	count, err := nvml.DeviceGetCount()
	if err != nil {
		log.Fatal(err)
	}
	for i := uint32(0); i < count; i++ {
		dev, err := nvml.NewDeviceByIndex(i)
		if err != nil {
			log.Fatal(err)
		}
		name, err := dev.GetName()
		if err != nil {
			log.Fatal(err)
		}
		fmt.Printf("device %d: %s\n", i, name)
	}
}

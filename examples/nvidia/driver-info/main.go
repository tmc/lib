// Driver-info prints identity and capability information from the CUDA Driver API.
package main

import (
	"fmt"
	"log"

	"github.com/tmc/lib/nvidia/driver"
)

func main() {
	if err := driver.Init(0); err != nil {
		log.Fatal(err)
	}
	version, err := driver.DriverGetVersion()
	if err != nil {
		log.Fatal(err)
	}
	count, err := driver.DeviceGetCount()
	if err != nil {
		log.Fatal(err)
	}
	for i := 0; i < count; i++ {
		dev, err := driver.NewDeviceGet(i)
		if err != nil {
			log.Fatal(err)
		}
		name, err := dev.GetName()
		if err != nil {
			log.Fatal(err)
		}
		major, err := dev.GetAttribute(driver.DevAttrComputeCapabilityMajor)
		if err != nil {
			log.Fatal(err)
		}
		minor, err := dev.GetAttribute(driver.DevAttrComputeCapabilityMinor)
		if err != nil {
			log.Fatal(err)
		}
		fmt.Printf("CUDA %d: device %d: %s (sm_%d%d)\n", version, i, name, major, minor)
	}
}

// Nvlshm-capabilities reports VMM alignment and multicast support for a GPU.
package main

import (
	"fmt"
	"log"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/nvlshm"
)

func main() {
	dev, err := cuda.NewDevice(0)
	if err != nil {
		log.Fatal(err)
	}
	defer dev.Close()
	if err := dev.WithContext(func() error {
		granularity, err := nvlshm.Granularity(dev.Raw())
		if err != nil {
			return err
		}
		multicast, err := nvlshm.MulticastSupported(dev.Raw())
		if err != nil {
			return err
		}
		fmt.Printf("VMM granularity=%d multicast=%t\n", granularity, multicast)
		return nil
	}); err != nil {
		log.Fatal(err)
	}
}

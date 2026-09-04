// Cusolver-version creates a cuSOLVER handle and prints its version.
package main

import (
	"fmt"
	"log"

	"github.com/tmc/lib/nvidia/cusolver"
	"github.com/tmc/lib/nvidia/x/cuda"
)

func main() {
	dev, err := cuda.NewDevice(0)
	if err != nil {
		log.Fatal(err)
	}
	defer dev.Close()
	if err := dev.WithContext(func() error {
		h, err := cusolver.Create()
		if err != nil {
			return err
		}
		defer h.Destroy()
		version, err := cusolver.GetVersion()
		if err != nil {
			return err
		}
		fmt.Printf("cuSOLVER %d\n", version)
		return nil
	}); err != nil {
		log.Fatal(err)
	}
}

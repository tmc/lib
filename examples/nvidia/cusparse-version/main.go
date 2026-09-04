// Cusparse-version creates a cuSPARSE handle and prints its version.
package main

import (
	"fmt"
	"log"

	"github.com/tmc/lib/nvidia/cusparse"
	"github.com/tmc/lib/nvidia/x/cuda"
)

func main() {
	dev, err := cuda.NewDevice(0)
	if err != nil {
		log.Fatal(err)
	}
	defer dev.Close()
	if err := dev.WithContext(func() error {
		h, err := cusparse.Create()
		if err != nil {
			return err
		}
		defer h.Destroy()
		version, err := h.GetVersion()
		if err != nil {
			return err
		}
		fmt.Printf("cuSPARSE %d\n", version)
		return nil
	}); err != nil {
		log.Fatal(err)
	}
}

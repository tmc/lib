// Cudnn-tensor creates a cuDNN handle and tensor descriptor.
package main

import (
	"fmt"
	"log"

	"github.com/tmc/lib/nvidia/cudnn"
	"github.com/tmc/lib/nvidia/x/cuda"
)

func main() {
	dev, err := cuda.NewDevice(0)
	if err != nil {
		log.Fatal(err)
	}
	defer dev.Close()
	if err := dev.WithContext(func() error {
		h, err := cudnn.Create()
		if err != nil {
			return err
		}
		defer h.Destroy()
		desc, err := cudnn.CreateTensorDescriptor()
		if err != nil {
			return err
		}
		defer desc.Destroy()
		fmt.Println("created a cuDNN tensor descriptor")
		return nil
	}); err != nil {
		log.Fatal(err)
	}
}

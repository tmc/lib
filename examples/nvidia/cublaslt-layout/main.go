// Cublaslt-layout creates a cuBLASLt matrix layout descriptor.
package main

import (
	"fmt"
	"log"

	"github.com/tmc/lib/nvidia/cublaslt"
	"github.com/tmc/lib/nvidia/x/cuda"
)

func main() {
	dev, err := cuda.NewDevice(0)
	if err != nil {
		log.Fatal(err)
	}
	defer dev.Close()
	if err := dev.WithContext(func() error {
		h, err := cublaslt.Create()
		if err != nil {
			return err
		}
		defer h.Destroy()
		layout, err := cublaslt.NewMatrixLayout(cublaslt.R32F, 2, 2, 2)
		if err != nil {
			return err
		}
		defer layout.Destroy()
		fmt.Println("created a 2x2 FP32 layout")
		return nil
	}); err != nil {
		log.Fatal(err)
	}
}

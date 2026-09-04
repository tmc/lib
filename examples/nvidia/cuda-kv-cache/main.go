// Cuda-kv-cache allocates and advances a small transformer KV cache.
package main

import (
	"fmt"
	"log"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudakv"
)

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	dev, err := cuda.NewDevice(0)
	if err != nil {
		return err
	}
	defer dev.Close()

	return dev.WithContext(func() error {
		cache, err := cudakv.New(dev, 2, 4, 64, 128)
		if err != nil {
			return err
		}
		defer cache.Close()

		layer, err := cache.Layer(0)
		if err != nil {
			return err
		}
		cache.Advance(3)
		fmt.Printf("allocated %d K elements and %d V elements; decode position %d\n", layer.K.Len(), layer.V.Len(), cache.Pos())
		return nil
	})
}

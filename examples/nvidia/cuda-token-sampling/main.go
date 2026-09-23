// Cuda-token-sampling compiles the GPU argmax sampler and checks its CPU oracle.
package main

import (
	"fmt"
	"log"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudasample"
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
	major, minor, err := dev.ComputeCapability()
	if err != nil {
		return err
	}
	mod, err := cudasample.Compile(major, minor)
	if err != nil {
		return err
	}

	logits := []float32{-3, 0.25, 2.5, 1.75}
	index := cudasample.SampleArgmaxCPU(logits)
	fmt.Printf("compiled argmax sampler (%d bytes PTX); CPU oracle chose token %d\n", len(mod.PTX()), index)
	return nil
}

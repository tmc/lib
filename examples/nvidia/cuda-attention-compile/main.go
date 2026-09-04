// Cuda-attention-compile builds the CUDA attention decode kernel for the first GPU.
package main

import (
	"fmt"
	"log"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudaattn"
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
	mod, err := cudaattn.Compile(major, minor)
	if err != nil {
		return err
	}

	cfg := cudaattn.DefaultConfig(32, 8, 128)
	fmt.Printf("compiled GQA decode for sm_%d%d: heads=%d/%d dim=%d scale=%.6f (%d bytes PTX)\n",
		major, minor, cfg.NumQHeads, cfg.NumKVHeads, cfg.HeadDim, cfg.Scale, len(mod.PTX()))
	return nil
}

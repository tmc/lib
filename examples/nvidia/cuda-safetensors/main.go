// Cuda-safetensors creates a small safetensors file and loads it into GPU memory.
package main

import (
	"encoding/binary"
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudasafetensors"
)

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	dir, err := tempDir()
	if err != nil {
		return err
	}
	defer os.RemoveAll(dir)

	path := filepath.Join(dir, "weights.safetensors")
	if err := writeWeights(path, []float32{1, 2, 3, 4}); err != nil {
		return err
	}
	weights, err := cudasafetensors.Open(path)
	if err != nil {
		return err
	}
	defer weights.Close()

	dev, err := cuda.NewDevice(0)
	if err != nil {
		return err
	}
	defer dev.Close()

	return dev.WithContext(func() error {
		gpu, err := cudasafetensors.LoadSlice[float32](weights, "weight", dev)
		if err != nil {
			return err
		}
		defer gpu.Free()

		got := make([]float32, gpu.Len())
		if err := gpu.CopyToHost(got); err != nil {
			return err
		}
		fmt.Printf("loaded %q %v to %d bytes of GPU memory\n", "weight", got, gpu.SizeBytes())
		return nil
	})
}

func tempDir() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	return os.MkdirTemp(filepath.Join(home, "tmp"), "lib-cuda-safetensors-")
}

func writeWeights(path string, values []float32) error {
	header := []byte(`{"weight":{"dtype":"F32","shape":[4],"data_offsets":[0,16]}}`)
	data := make([]byte, 8+len(header)+len(values)*4)
	binary.LittleEndian.PutUint64(data, uint64(len(header)))
	copy(data[8:], header)
	for i, value := range values {
		binary.LittleEndian.PutUint32(data[8+len(header)+4*i:], math.Float32bits(value))
	}
	return os.WriteFile(path, data, 0666)
}

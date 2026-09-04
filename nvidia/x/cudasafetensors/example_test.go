package cudasafetensors_test

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/tmc/lib/nvidia/x/cudasafetensors"
)

func ExampleWeights_TensorBytes() {
	home, err := os.UserHomeDir()
	if err != nil {
		panic(err)
	}
	dir, err := os.MkdirTemp(filepath.Join(home, "tmp"), "cudasafetensors-example-")
	if err != nil {
		panic(err)
	}
	defer os.RemoveAll(dir)

	header := []byte(`{"weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}`)
	file := make([]byte, 8+len(header)+4)
	binary.LittleEndian.PutUint64(file, uint64(len(header)))
	copy(file[8:], header)
	binary.LittleEndian.PutUint32(file[8+len(header):], math.Float32bits(1.5))
	path := filepath.Join(dir, "weights.safetensors")
	if err := os.WriteFile(path, file, 0o600); err != nil {
		panic(err)
	}

	weights, err := cudasafetensors.Open(path)
	if err != nil {
		panic(err)
	}
	defer weights.Close()
	raw, info, err := weights.TensorBytes("weight")
	if err != nil {
		panic(err)
	}
	fmt.Println(info.DType, info.Shape, math.Float32frombits(binary.LittleEndian.Uint32(raw)))
	// Output: F32 [1] 1.5
}

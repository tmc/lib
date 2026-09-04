//go:build darwin

package metalsafetensors_test

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"

	"github.com/tmc/lib/apple/x/metal/metalsafetensors"
)

func Example() {
	f, err := os.CreateTemp("", "weights-*.safetensors")
	if err != nil {
		fmt.Println(err)
		return
	}
	defer os.Remove(f.Name())

	header := []byte(`{"x":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}`)
	if err := binary.Write(f, binary.LittleEndian, uint64(len(header))); err != nil {
		fmt.Println(err)
		return
	}
	if _, err := f.Write(header); err != nil {
		fmt.Println(err)
		return
	}
	for _, x := range []float32{1, 2} {
		if err := binary.Write(f, binary.LittleEndian, math.Float32bits(x)); err != nil {
			fmt.Println(err)
			return
		}
	}
	if err := f.Close(); err != nil {
		fmt.Println(err)
		return
	}

	w, err := metalsafetensors.Open(f.Name())
	if err != nil {
		fmt.Println(err)
		return
	}
	defer w.Close()
	x, err := w.TensorFloat32("x")
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(x)
	// Output: [1 2]
}

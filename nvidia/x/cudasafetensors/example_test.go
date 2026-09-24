package cudasafetensors_test

import (
	"bytes"
	"fmt"
	"log"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudasafetensors"
	"github.com/tmc/lib/nvidia/x/safetensors"
)

func Example() {
	f, err := safetensors.OpenSharded("/path/to/checkpoint")
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()
	dev, err := cuda.NewDevice(0)
	if err != nil {
		log.Fatal(err)
	}
	defer dev.Close()

	w, err := cudasafetensors.LoadSlice[uint16](f, "model.embed_tokens.weight", dev)
	if err != nil {
		log.Fatal(err)
	}
	defer w.Free()
}

// LoadSlice refuses an element type that does not hold the tensor's dtype
// bit for bit, before it touches the device.
func ExampleLoadSlice_dtype() {
	var buf bytes.Buffer
	w := safetensors.Tensor{Name: "w", Dtype: safetensors.BF16, Shape: []int{2}, Data: make([]byte, 4)}
	if err := safetensors.Write(&buf, []safetensors.Tensor{w}, nil); err != nil {
		log.Fatal(err)
	}
	f, err := safetensors.NewFile(bytes.NewReader(buf.Bytes()), int64(buf.Len()))
	if err != nil {
		log.Fatal(err)
	}
	_, err = cudasafetensors.LoadSlice[float32](f, "w", nil)
	fmt.Println(err)
	// Output: cudasafetensors: tensor "w" dtype BF16 is incompatible with float32 storage
}

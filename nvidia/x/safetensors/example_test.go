package safetensors_test

import (
	"bytes"
	"fmt"
	"log"

	"github.com/tmc/lib/nvidia/x/safetensors"
)

func Example() {
	data, err := safetensors.EncodeFloat32(safetensors.BF16, []float32{1, 0.5, -3})
	if err != nil {
		log.Fatal(err)
	}
	var buf bytes.Buffer
	w := safetensors.Tensor{Name: "head.bias", Dtype: safetensors.BF16, Shape: []int{3}, Data: data}
	if err := safetensors.Write(&buf, []safetensors.Tensor{w}, map[string]string{"format": "pt"}); err != nil {
		log.Fatal(err)
	}

	f, err := safetensors.NewFile(bytes.NewReader(buf.Bytes()), int64(buf.Len()))
	if err != nil {
		log.Fatal(err)
	}
	t, err := f.Tensor("head.bias")
	if err != nil {
		log.Fatal(err)
	}
	vals, err := safetensors.DecodeFloat32(t.Dtype, t.Data)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(t.Name, t.Dtype, t.Shape, vals, f.Metadata["format"])
	// Output: head.bias BF16 [3] [1 0.5 -3] pt
}

func ExampleFloat32ToFloat16() {
	h := safetensors.Float32ToFloat16(0.1)
	fmt.Printf("%#04x %v\n", h, safetensors.Float16ToFloat32(h))
	// Output: 0x2e66 0.099975586
}

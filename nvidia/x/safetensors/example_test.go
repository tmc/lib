package safetensors_test

import (
	"bytes"
	"fmt"
	"io"
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

func ExampleFile_TensorReader() {
	data, err := safetensors.EncodeFloat32(safetensors.F32, []float32{1, 2, 3, 4})
	if err != nil {
		log.Fatal(err)
	}
	var buf bytes.Buffer
	w := safetensors.Tensor{Name: "w", Dtype: safetensors.F32, Shape: []int{4}, Data: data}
	if err := safetensors.Write(&buf, []safetensors.Tensor{w}, nil); err != nil {
		log.Fatal(err)
	}
	f, err := safetensors.NewFile(bytes.NewReader(buf.Bytes()), int64(buf.Len()))
	if err != nil {
		log.Fatal(err)
	}

	// Read the tensor 8 bytes, two elements, at a time.
	r, err := f.TensorReader("w")
	if err != nil {
		log.Fatal(err)
	}
	chunk := make([]byte, 8)
	for {
		n, err := io.ReadFull(r, chunk)
		if err == io.EOF {
			break
		}
		if err != nil {
			log.Fatal(err)
		}
		vals, err := safetensors.DecodeFloat32(safetensors.F32, chunk[:n])
		if err != nil {
			log.Fatal(err)
		}
		fmt.Println(vals)
	}
	// Output:
	// [1 2]
	// [3 4]
}

package gguf

import (
	"bytes"
	"fmt"
)

func Example() {
	b := fixture() // a small in-memory GGUF file
	g, err := Read(bytes.NewReader(b), int64(len(b)))
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(g.Metadata["general.architecture"].Value)
	for _, t := range g.Tensors {
		n, err := t.Bytes()
		if err != nil {
			fmt.Println(err)
			return
		}
		fmt.Println(t.Name, t.Dimensions, g.DataOffset+t.Offset, n)
	}
	// Output:
	// deepseek4
	// token_embd.weight [4 8] 192 128
}

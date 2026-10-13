package tileir_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/tileir"
)

func ExampleModule_EncodeBytecode() {
	m := &tileir.Module{Functions: []*tileir.Function{{Name: "one", Entry: true,
		Operations: []*tileir.Operation{
			{OpCode: "cuda_tile.constant", Results: []string{"%one"},
				TypeVal: tileir.NewTile([]int64{4}, tileir.Float32), Attributes: "{value = 1.0}"},
			{OpCode: "cuda_tile.return"},
		},
	}}}
	data, err := m.EncodeBytecode()
	if err != nil {
		fmt.Println(err)
		return
	}
	bm, err := tileir.DecodeBytecodeBytes(data)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(bm.Functions[0].Name, len(bm.Functions[0].Operations))
	// Output: one 2
}

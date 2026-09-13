package cuptisass_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudasass"
	"github.com/tmc/lib/nvidia/x/cuptisass"
)

func ExampleOutputGranularity_String() {
	fmt.Println(cuptisass.GranularityGPU)
	fmt.Println(cuptisass.GranularitySM)
	fmt.Println(cuptisass.GranularitySMSP)
	// Output:
	// GPU
	// SM
	// SMSP
}

func ExampleInstructionCensus() {
	dis := &cudasass.Disassembly{
		Functions: []*cudasass.Function{
			{
				Name: "VectorAdd",
				Instructions: []cudasass.Instruction{
					{Address: 0x00, Opcode: "MOV"},
					{Address: 0x10, Opcode: "IADD3"},
					{Address: 0x20, Opcode: "RET"},
				},
			},
		},
	}

	census := cuptisass.InstructionCensus(dis)
	fmt.Println(census["VectorAdd"])
	// Output:
	// 3
}

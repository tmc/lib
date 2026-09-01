package anemil_test

import (
	"fmt"
	"strings"

	"github.com/tmc/lib/apple/x/ane/anemil"
)

func Example() {
	prog := anemil.NewProgram()
	prog.AddFunction(anemil.Function{
		Name:      "forward",
		InputName: "in",
		InputType: anemil.TensorType{DType: anemil.FP32, Shape: []int{1, 128, 1, 1}},
		OutputName: "out",
		Statements: []anemil.Statement{
			{
				OutputVar:  "out",
				OutputType: anemil.TensorType{DType: anemil.FP16, Shape: []int{1, 128, 1, 1}},
				Op:         "cast",
				Attributes: map[string]string{"dtype": `string("fp16")`, "x": "in"},
			},
		},
	})

	milText := prog.Serialize()
	hasHeader := strings.Contains(milText, "program(1.3)")
	fmt.Printf("Serialized MIL Valid: %t\n", hasHeader)

	// Output:
	// Serialized MIL Valid: true
}

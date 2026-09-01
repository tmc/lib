package anemil

import (
	"strings"
	"testing"
)

func TestMILProgramSerialization(t *testing.T) {
	prog := NewProgram()
	fn := Function{
		Name:      "main",
		InputName: "x",
		InputType: TensorType{DType: FP32, Shape: []int{1, 64, 1, 1}},
		OutputName: "y",
		Statements: []Statement{
			{
				OutputVar:  "y",
				OutputType: TensorType{DType: FP16, Shape: []int{1, 64, 1, 1}},
				Op:         "cast",
				Attributes: map[string]string{
					"dtype": `string("fp16")`,
					"x":     "x",
				},
			},
		},
	}
	prog.AddFunction(fn)

	out := prog.Serialize()
	if !strings.Contains(out, "program(1.3)") {
		t.Errorf("missing program version")
	}
	if !strings.Contains(out, "func main<ios18>") {
		t.Errorf("missing function header")
	}
	if !strings.Contains(out, "tensor<fp16, [1, 64, 1, 1]> y = cast") {
		t.Errorf("missing cast statement")
	}
}

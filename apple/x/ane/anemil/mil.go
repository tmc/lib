package anemil

import (
	"fmt"
	"strings"
)

// DType represents supported ANE data types.
type DType string

const (
	FP32 DType = "fp32"
	FP16 DType = "fp16"
	INT8 DType = "int8"
	INT4 DType = "int4"
)

// TensorType represents an ANE tensor shape and element type (e.g. [1, C, 1, S]).
type TensorType struct {
	DType DType
	Shape []int
}

// String formats the tensor type in MIL syntax: tensor<fp16, [1, 64, 1, 1]>.
func (t TensorType) String() string {
	var shapeStrs []string
	for _, dim := range t.Shape {
		shapeStrs = append(shapeStrs, fmt.Sprintf("%d", dim))
	}
	return fmt.Sprintf("tensor<%s, [%s]>", t.DType, strings.Join(shapeStrs, ", "))
}

// Statement represents a single SSA instruction in a MIL function.
type Statement struct {
	OutputVar  string
	OutputType TensorType
	Op         string
	Attributes map[string]string
}

// Function represents a single MIL function definition.
type Function struct {
	Name       string
	InputName  string
	InputType  TensorType
	OutputName string
	Statements []Statement
}

// Program represents an entire MIL program.
type Program struct {
	Version   string
	Functions []Function
}

// NewProgram initializes a new MIL program targeting iOS 18 / macOS 15+ ANE runtime.
func NewProgram() *Program {
	return &Program{
		Version:   "1.3",
		Functions: make([]Function, 0),
	}
}

// AddFunction appends a new function to the program.
func (p *Program) AddFunction(fn Function) {
	p.Functions = append(p.Functions, fn)
}

// Serialize serializes the program to standard MIL textual IR.
func (p *Program) Serialize() string {
	var sb strings.Builder

	sb.WriteString(fmt.Sprintf("program(%s)\n", p.Version))
	sb.WriteString("[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}})]\n")
	sb.WriteString("{\n")

	for _, fn := range p.Functions {
		sb.WriteString(fmt.Sprintf("    func %s<ios18>(%s %s) {\n", fn.Name, fn.InputType.String(), fn.InputName))
		for _, stmt := range fn.Statements {
			var attrStrs []string
			for k, v := range stmt.Attributes {
				attrStrs = append(attrStrs, fmt.Sprintf("%s = %s", k, v))
			}
			sb.WriteString(fmt.Sprintf("        %s %s = %s(%s);\n",
				stmt.OutputType.String(), stmt.OutputVar, stmt.Op, strings.Join(attrStrs, ", ")))
		}
		sb.WriteString(fmt.Sprintf("    } -> (%s);\n", fn.OutputName))
	}

	sb.WriteString("}\n")
	return sb.String()
}

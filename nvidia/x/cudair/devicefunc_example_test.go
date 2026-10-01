package cudair_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudair"
)

type exampleLayer struct {
	name     string
	inWidth  int
	outWidth int
}

func (l *exampleLayer) Name() string                  { return l.name }
func (l *exampleLayer) InputWidth() int               { return l.inWidth }
func (l *exampleLayer) OutputWidth() int              { return l.outWidth }
func (l *exampleLayer) PaddedOutputWidth() int        { return l.outWidth }
func (l *exampleLayer) FwdCtxBytes() (int, error)     { return 0, nil }
func (l *exampleLayer) FwdCtxAlignedPerElement() bool { return true }
func (l *exampleLayer) EmitForward(name string) (string, error) {
	return cudair.EmitFromBody(l, name, "\treturn input;")
}

func ExampleEmitFromBody() {
	layer := &exampleLayer{name: "linear", inWidth: 4, outWidth: 4}
	code, err := cudair.EmitFromBody(layer, "linear_eval", "\treturn input;")
	if err != nil {
		fmt.Printf("err: %v\n", err)
		return
	}
	fmt.Printf("%t\n", len(code) > 0)
	// Output:
	// true
}

func ExampleEmitTree() {
	layer := &exampleLayer{name: "linear", inWidth: 8, outWidth: 8}
	unit, err := cudair.EmitTree(layer, "eval_root")
	if err != nil {
		fmt.Printf("err: %v\n", err)
		return
	}
	fmt.Printf("symbols: %v\n", unit.Symbols)
	// Output:
	// symbols: [eval_root]
}

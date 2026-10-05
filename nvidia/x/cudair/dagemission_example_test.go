package cudair

import "fmt"

type dagExampleLeaf struct{}

func (dagExampleLeaf) InputWidth() int               { return 1 }
func (dagExampleLeaf) OutputWidth() int              { return 1 }
func (dagExampleLeaf) PaddedOutputWidth() int        { return 1 }
func (dagExampleLeaf) FwdCtxBytes() (int, error)     { return 0, nil }
func (dagExampleLeaf) FwdCtxAlignedPerElement() bool { return false }
func (dagExampleLeaf) DAGKey() string                { return "input:fp32:1" }
func (dagExampleLeaf) DAGSymbol() string             { return "leaf" }
func (dagExampleLeaf) Dependencies() []DAGEmitter    { return nil }
func (n dagExampleLeaf) EmitForward(name string) (string, error) {
	return EmitFromBody(n, name, "return input;")
}
func (n dagExampleLeaf) EmitForwardWithDependencies(name string, children []ChildSymbol) (string, error) {
	return n.EmitForward(name)
}

func ExampleDAGEmitter() {
	u, err := EmitTree(dagExampleLeaf{}, "leaf")
	fmt.Println(u.Symbols, u.FwdCtxBytes, err)
	// Output: [leaf] 0 <nil>
}

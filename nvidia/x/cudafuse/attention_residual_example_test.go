package cudafuse_test

import (
	"fmt"
	"github.com/tmc/lib/nvidia/x/cudafuse"
)

func ExampleAttentionResidualSource() {
	c := cudafuse.AttentionResidualConfig{Candidates: 3, Rows: 2, Width: 33, Epsilon: 1e-5}
	s, err := cudafuse.AttentionResidualSource(c, cudafuse.AttentionResidualComposed)
	fmt.Println(len(s), err)
	// Output: 4 <nil>
}

func ExampleAttentionResidualSchema() {
	s, err := cudafuse.AttentionResidualSchema()
	fmt.Println(len(s.Params), err)
	// Output: 7 <nil>
}

func ExampleAttentionResidualExecution_Close() {
	var e *cudafuse.AttentionResidualExecution
	fmt.Println(e.Close())
	// Output: <nil>
}

func ExampleCompileAttentionResidual() {
	_, err := cudafuse.CompileAttentionResidual(cudafuse.AttentionResidualConfig{}, cudafuse.AttentionResidualFused, 12, 1)
	fmt.Println(err != nil)
	// Output: true
}

func ExampleLoadAttentionResidual() {
	_, err := cudafuse.LoadAttentionResidual(nil, cudafuse.AttentionResidualConfig{Candidates: 3, Rows: 2, Width: 33, Epsilon: 1e-5}, cudafuse.AttentionResidualFused, nil)
	fmt.Println(err != nil)
	// Output: true
}

func ExampleNewAttentionResidualSnapshots() {
	_, err := cudafuse.NewAttentionResidualSnapshots(nil, 2, 33, nil)
	fmt.Println(err != nil)
	// Output: true
}

func ExampleAttentionResidual_Close() {
	var x *cudafuse.AttentionResidual
	fmt.Println(x.Close())
	// Output: <nil>
}

func ExampleAttentionResidualExecution_Stats() {
	var e *cudafuse.AttentionResidualExecution
	a, b := e.Stats()
	fmt.Println(a, b)
	// Output: 0 0
}

func ExampleAttentionResidualExecution_Wait() {
	var e *cudafuse.AttentionResidualExecution
	fmt.Println(e.Wait(nil))
	// Output: <nil>
}

func ExampleAttentionResidualSnapshots_Close() {
	var s *cudafuse.AttentionResidualSnapshots
	fmt.Println(s.Close())
	// Output: <nil>
}

func ExampleAttentionResidual_Execute() {
	var x *cudafuse.AttentionResidual
	in := cudafuse.AttentionResidualInputs{}
	_, err := x.Execute(nil, nil, in, in.Values)
	fmt.Println(err != nil)
	// Output: true
}

func ExampleAttentionResidualSnapshots_Execute() {
	var s *cudafuse.AttentionResidualSnapshots
	empty := cudafuse.AttentionResidualInputs{}.Values
	_, err := s.Execute(nil, nil, nil, empty, empty, empty, empty)
	fmt.Println(err != nil)
	// Output: true
}

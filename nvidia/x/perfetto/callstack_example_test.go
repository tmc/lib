package perfetto_test

import (
	"bytes"
	"fmt"
	"github.com/tmc/lib/nvidia/x/perfetto"
)

func ExampleWriter_InternCallstack() {
	var b bytes.Buffer
	w := perfetto.NewWriter(&b)
	id, _ := w.InternCallstack([]perfetto.StackFrame{{Function: "main.work"}})
	fmt.Println(id)
	// Output: 1
}
func ExampleWriter_WriteSliceBeginWithCallstack() {
	var b bytes.Buffer
	w := perfetto.NewWriter(&b)
	id, _ := w.InternCallstack([]perfetto.StackFrame{{Function: "main.work"}})
	err := w.WriteSliceBeginWithCallstack(1, 100, "launch", "go", nil, id)
	fmt.Println(err)
	// Output: <nil>
}
func ExampleWriter_WriteInstantWithCallstack() {
	var b bytes.Buffer
	w := perfetto.NewWriter(&b)
	id, _ := w.InternCallstack([]perfetto.StackFrame{{Function: "main.work"}})
	err := w.WriteInstantWithCallstack(1, 100, "CPU sample", "go", nil, id)
	fmt.Println(err)
	// Output: <nil>
}

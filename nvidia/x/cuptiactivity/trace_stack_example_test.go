package cuptiactivity_test

import (
	"bytes"
	"fmt"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/x/cuptiactivity"
	"github.com/tmc/lib/nvidia/x/perfetto"
)

func ExampleWriteRecordWithCallstack() {
	var trace bytes.Buffer
	w := perfetto.NewWriter(&trace)
	id, err := w.InternCallstack([]perfetto.StackFrame{{Function: "main.launch", File: "main.go", Line: 10}})
	if err != nil {
		fmt.Println(err)
		return
	}
	r := cuptiactivity.Record{Kind: cupti.ActivityKindDriver, Name: "cuLaunchKernel", Start: 100, End: 110, Complete: true}
	fmt.Println(cuptiactivity.WriteRecordWithCallstack(w, 1, 100, r, id))
	// Output: <nil>
}

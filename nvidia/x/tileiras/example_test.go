package tileiras_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/tileiras"
)

func ExampleParseRemarksYAML() {
	yamlData := []byte(`--- !Failure
Pass:            Memory
Name:            RemarkMemoryTmaLoadFailed
DebugLoc:        { File: 'kernel.cu', Line: 14, Column: 5 }
Function:        matrix
Args:
  - RemarkId:    5
  - Remark:      Failed to select TMA load instruction
  - Reason:      non-contiguous stride along dim 1
...
`)

	remarks, err := tileiras.ParseRemarksYAML(yamlData)
	if err != nil {
		panic(err)
	}

	for _, rmk := range remarks {
		if rmk.IsTMAFailure() {
			fmt.Printf("TMA Bottleneck at line %d: %s (Reason: %s)\n", rmk.Line, rmk.Remark, rmk.Reason)
		}
	}

	// Output:
	// TMA Bottleneck at line 14: Failed to select TMA load instruction (Reason: non-contiguous stride along dim 1)
}

package advise_test

import (
	"fmt"
	"log"

	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaprof/advise"
)

func ExampleAnalyze() {
	// A capture carrying only the compiler's remarks: a TMA load the
	// compiler could not select.
	cap := &cudacapture.Capture{
		Files: map[string][]byte{
			"remarks.yaml": []byte(`--- !Failure
Pass:            Memory
Name:            RemarkMemoryTmaLoadFailed
DebugLoc:        { File: 'matrix.cu', Line: 14, Column: 5 }
Function:        matrix
Args:
  - Remark:      Failed to select TMA load instruction
  - Reason:      non-contiguous stride along dim 1
...
`),
		},
	}
	report, err := advise.Analyze(cap)
	if err != nil {
		log.Fatal(err)
	}
	for _, item := range report.Items {
		fmt.Printf("[%s] %s: %s\n", item.Severity, item.Category, item.Finding)
	}
	// Output:
	// [Cost] Memory: Failed to select TMA load instruction (reason: non-contiguous stride along dim 1)
}

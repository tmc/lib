package cudatest_test

import (
	"fmt"
	"strings"
	"testing/fstest"

	"github.com/tmc/lib/nvidia/x/cudatest"
)

func ExampleValidateQualification() {
	revision := strings.Repeat("a", 40)
	matrix := cudatest.QualificationMatrix{Version: 1, SourceRevision: revision}
	for _, kind := range []string{"copy", "address-space", "dsmem", "cupti-range", "cupti-pc"} {
		matrix.Rows = append(matrix.Rows, cudatest.QualificationRow{Kind: kind, Status: "unmeasured", Reason: "named native controls have not executed"})
	}
	fmt.Println(cudatest.ValidateQualification(matrix, revision, fstest.MapFS{}, fstest.MapFS{}), matrix.Complete)
	// Output: <nil> false
}

func ExampleReadQualification() {
	_, err := cudatest.ReadQualification(strings.NewReader(`{"version":1,"version":1}`))
	fmt.Println(err != nil)
	// Output: true
}

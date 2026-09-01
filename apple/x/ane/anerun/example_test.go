package anerun_test

import (
	"fmt"

	"github.com/tmc/lib/apple/x/ane/anerun"
)

func Example() {
	runner := anerun.NewRunner("/path/to/model.mlmodelc")
	fmt.Printf("ANE Runner initialized for: %s\n", runner.ModelPath)

	// Output:
	// ANE Runner initialized for: /path/to/model.mlmodelc
}

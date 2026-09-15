package cuptibench_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cuptibench"
)

func ExampleDiagnostics() {
	d := cuptibench.Diagnostics{Image: []byte{1, 2}}
	fmt.Println(len(d.Image), d.Trace == nil)
	// Output: 2 true
}

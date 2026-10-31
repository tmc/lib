package nvtelemetry_test

import (
	"bytes"
	"fmt"

	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

func ExampleOperatorCPUContext() {
	var c nvtelemetry.OperatorCPUContext
	err := nvtelemetry.WriteOperatorCPUContext(&bytes.Buffer{}, c, nvtelemetry.WorkloadLimits{})
	fmt.Println(err != nil)
	// Output: true
}

func ExampleWriteOperatorCPUContext() {
	c, _ := cpuContextFixture()
	var out bytes.Buffer
	err := nvtelemetry.WriteOperatorCPUContext(&out, c, nvtelemetry.WorkloadLimits{})
	fmt.Println(out.Len() > 0, err)
	// Output: true <nil>
}

func ExampleReadOperatorCPUContext() {
	c, source := cpuContextFixture()
	var out bytes.Buffer
	_ = nvtelemetry.WriteOperatorCPUContext(&out, c, nvtelemetry.WorkloadLimits{})
	_, got, err := nvtelemetry.ReadOperatorCPUContext(&out, source, nvtelemetry.WorkloadLimits{})
	fmt.Println(got.RunID, got.TraceSource.Version, err)
	// Output: run 2.10.0 <nil>
}

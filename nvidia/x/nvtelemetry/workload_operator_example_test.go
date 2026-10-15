package nvtelemetry_test

import (
	"bytes"
	"fmt"

	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

func ExampleWriteOperatorContext() {
	// Constructed application metadata; no framework or device is invoked.
	c := operatorContextFixture()
	var raw bytes.Buffer
	err := nvtelemetry.WriteOperatorContext(&raw, c, nvtelemetry.WorkloadLimits{})
	fmt.Println(err, raw.Len() > 0)
	// Output: <nil> true
}

func ExampleReadOperatorContext() {
	var raw bytes.Buffer
	if err := nvtelemetry.WriteOperatorContext(&raw, operatorContextFixture(), nvtelemetry.WorkloadLimits{}); err != nil {
		panic(err)
	}
	export, context, err := nvtelemetry.ReadOperatorContext(&raw, operatorSource(), nvtelemetry.WorkloadLimits{})
	fmt.Println(err, context.SourceClass, len(context.Groups), len(export.RawSHA256))
	// Output: <nil> synthetic 0 64
}

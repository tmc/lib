package cudasearch

import "fmt"

func ExampleMeasurementContext() {
	// These values version caller-known facts; the evaluator does not attest them.
	e := &ExecutableEvaluator{Generation: 3, MeasurementContext: MeasurementContext{
		TargetIdentity: "device-uuid/driver-v1/toolchain-v2/context-4",
		ResetIdentity:  "reset-state-v3",
	}}
	fmt.Println(e.Generation, e.MeasurementContext.ResetIdentity)
	// Output: 3 reset-state-v3
}

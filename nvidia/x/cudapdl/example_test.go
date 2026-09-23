package cudapdl_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudapdl"
)

func ExampleCheckEligibility() {
	producer := cudapdl.KernelDesc{
		Name:                "rms_norm",
		HasLaunchDependents: true,
	}
	consumer := cudapdl.KernelDesc{
		Name:    "gemv_projection",
		HasWait: true,
	}

	el := cudapdl.CheckEligibility("sm_90", producer, consumer)
	fmt.Printf("sm_90 eligible: %v (policy: %s)\n", el.Eligible, el.Policy)

	// Architecture below sm_90
	elLegacy := cudapdl.CheckEligibility("sm_80", producer, consumer)
	fmt.Printf("sm_80 eligible: %v (policy: %s)\n", elLegacy.Eligible, elLegacy.Policy)

	// Output:
	// sm_90 eligible: true (policy: PolicyProgrammatic)
	// sm_80 eligible: false (policy: PolicyStandardFallback)
}

func ExamplePlanAdapter() {
	adapter := cudapdl.NewPlanAdapter("sm_100a")

	prod := cudapdl.KernelDesc{
		Name:                "gemv_stage1",
		HasLaunchDependents: true,
	}
	cons := cudapdl.KernelDesc{
		Name:    "gemv_stage2",
		HasWait: true,
	}

	el := adapter.Evaluate(prod, cons)
	fmt.Printf("policy: %s\n", el.Policy)

	// Output:
	// policy: PolicyProgrammatic
}

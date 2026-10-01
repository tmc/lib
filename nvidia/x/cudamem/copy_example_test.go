package cudamem_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudalayout"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

func ExamplePlanCopy() {
	layout, err := cudalayout.ParseLayout("4:1")
	if err != nil {
		fmt.Println(err)
		return
	}
	plan, err := cudamem.PlanCopy(layout, layout)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(plan.VectorWidth, plan.CollapseFactor)
	// Output: 4 1
}

func ExampleCopyPlan() {
	src, err := cudalayout.ParseLayout("(5,7):(7,1)")
	if err != nil {
		fmt.Println(err)
		return
	}
	dst, err := cudalayout.ParseLayout("(7,5):(5,1)")
	if err != nil {
		fmt.Println(err)
		return
	}
	plan, err := cudamem.PlanCopy(src, dst)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(plan.CommonDomain, plan.SourceResidual, plan.DestinationResidual)
	fmt.Println(plan.VectorWidth, plan.VectorReason)
	// Output:
	// (1):(0) (5,7):(7,1) (7,5):(5,1)
	// 1 no unit-stride source run over destination inverse
}

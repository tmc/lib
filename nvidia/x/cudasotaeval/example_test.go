package cudasotaeval_test

import (
	"context"
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudasotaeval"
)

func Example() {
	proto := cudasotaeval.DefaultProtocol(cudasotaeval.PhasePilot)
	cfg := cudasotaeval.DefaultEvalConfig()
	cfg.ExerciseStress = false

	report, err := cudasotaeval.RunEvaluation(context.Background(), proto, cfg)
	if err != nil {
		fmt.Printf("evaluation error: %v\n", err)
		return
	}

	fmt.Printf("Passed: %t\n", report.Passed)
	fmt.Printf("Exact joins: %d\n", report.Attribution.ExactJoins)
	fmt.Printf("Incorrect joins: %d\n", report.Attribution.IncorrectJoins)
	// Output:
	// Passed: false
	// Exact joins: 20
	// Incorrect joins: 0
}

func ExampleDefaultProtocol() {
	proto := cudasotaeval.DefaultProtocol(cudasotaeval.PhaseQualification)
	fmt.Printf("Name: %s\n", proto.Name)
	fmt.Printf("Phase: %s\n", proto.Phase)
	fmt.Printf("Max incorrect joins: %d\n", proto.Budgets.MaxIncorrectJoins)
	// Output:
	// Name: cudasotaeval-submission-delay-v1
	// Phase: qualification
	// Max incorrect joins: 0
}

func ExampleFormulateClaim() {
	proto := cudasotaeval.DefaultProtocol(cudasotaeval.PhasePilot)
	cfg := cudasotaeval.DefaultEvalConfig()
	cfg.ExerciseStress = false

	report, err := cudasotaeval.RunEvaluation(context.Background(), proto, cfg)
	if err != nil {
		fmt.Printf("err: %v\n", err)
		return
	}

	claim, err := cudasotaeval.FormulateClaim(report)
	if err != nil {
		fmt.Printf("err: %v\n", err)
		return
	}

	fmt.Printf("Has Task: %t\n", claim.Task != "")
	fmt.Printf("Has Alternatives: %t\n", len(claim.ComparedWith) > 0)
	fmt.Printf("Has Limits: %t\n", len(claim.Limits) > 0)
	// Output:
	// err: cudasotaeval: qualification evidence incomplete
}

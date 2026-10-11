package cudadiag_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudadiag"
)

func Example() {
	// Build a deterministic synthetic host delay fixture
	fixture := cudadiag.BuildSyntheticFixture(cudadiag.InjectionHostDelay)

	// Diagnose without providing the injection label to the engine
	result := cudadiag.Diagnose(fixture.Workload, fixture.OpID, fixture.Input)

	fmt.Printf("Admitted: %t\n", result.Admitted)
	fmt.Printf("Mechanism: %s\n", result.Mechanism)
	fmt.Printf("Primary Finding Rule: %s\n", result.PrimaryFinding.Rule)
	fmt.Printf("Primary Finding Class: %s\n", result.PrimaryFinding.Class)

	// Output:
	// Admitted: true
	// Mechanism: host_submission_delay
	// Primary Finding Rule: host_submission_delay
	// Primary Finding Class: observed
}

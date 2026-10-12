package cudadiag_test

import (
	"context"
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudadiag"
)

func ExampleComposeHealthCapacity() {
	in, limits, err := healthCapacityFixture(cudadiag.InjectionHostDelay)
	if err != nil {
		fmt.Println("fixture error")
		return
	}
	incident, err := cudadiag.ComposeHealthCapacity(context.Background(), in, limits)
	if err != nil {
		fmt.Println("composition error")
		return
	}
	fmt.Println(incident.Impact.Status, incident.ImpactClass)
	// Output: admitted associated
}

func ExampleComposeHealthCapacity_withheldWorkloadLabels() {
	in, limits, err := healthCapacityFixture(cudadiag.InjectionHostDelay)
	if err != nil {
		fmt.Println("fixture error")
		return
	}
	in.Telemetry.Workload.Labels = map[string]string{"review": "before"}
	incident, err := cudadiag.ComposeHealthCapacity(context.Background(), in, limits)
	if err != nil {
		fmt.Println("composition error")
		return
	}
	in.Telemetry.Workload.Labels["review"] = "after"
	fmt.Println(incident.Diagnosis.Diagnosis.Workload.Labels["review"])
	// Output: before
}

package cudaflightrecorder_test

import (
	"context"
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudaflightrecorder"
)

func ExampleEscalationCoordinator_TriggerAsync() {
	recorder, err := cudaflightrecorder.New(cudaflightrecorder.Config{Capacity: 100, MaxQueuedJobs: 2})
	if err != nil {
		fmt.Println(err)
		return
	}
	coordinator, err := cudaflightrecorder.NewEscalationCoordinator(recorder, cudaflightrecorder.QualificationBudgets())
	if err != nil {
		fmt.Println(err)
		return
	}
	result, err := coordinator.TriggerAsync(context.Background(), cudaflightrecorder.TriggerRequest{Reason: "latency"})
	if err != nil {
		fmt.Println(err)
		return
	}
	// Queue admission succeeds, but this coordinator has not been armed.
	completion := <-result
	fmt.Println(completion.Receipt.Outcome, completion.Receipt.RefusalReason)
	coordinator.Stop()
	fmt.Println(coordinator.Wait(context.Background()))
	// Output:
	// refused disarmed
	// <nil>
}

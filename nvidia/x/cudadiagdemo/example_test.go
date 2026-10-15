package cudadiagdemo_test

import (
	"context"
	"fmt"
	"time"

	"github.com/tmc/lib/nvidia/x/cudadiagdemo"
)

func Example() {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	cfg := cudadiagdemo.DemonstrationConfig{
		NumRequests:       4,
		Concurrency:       2,
		HostDelay:         5 * time.Millisecond,
		DeviceDelay:       200 * time.Microsecond,
		SkipIntervention:  true,
		ExerciseCorruptEv: false,
	}

	result, err := cudadiagdemo.RunDemonstration(ctx, cfg)
	if err != nil {
		fmt.Printf("demonstration error: %v\n", err)
		return
	}

	fmt.Println("Synthetic:", result.Synthetic)
	fmt.Println("Separate pre-entry metric:", result.HostDelayDiagnosis.TimeAccounting.PreEntryDuration != nil)
	fmt.Println("Scheduler capture:", result.DiagnosticInputs["host"].Sched.ProbePresent)

	// Output:
	// Synthetic: true
	// Separate pre-entry metric: true
	// Scheduler capture: false
}

func ExampleServiceFixture() {
	ctx := context.Background()
	cfg := cudadiagdemo.DemonstrationConfig{
		NumRequests:       2,
		Concurrency:       1,
		HostDelay:         1 * time.Millisecond,
		DeviceDelay:       100 * time.Microsecond,
		SkipIntervention:  true,
		ExerciseCorruptEv: false,
	}

	res, err := cudadiagdemo.RunDemonstration(ctx, cfg)
	if err != nil {
		fmt.Printf("error: %v\n", err)
		return
	}

	fmt.Printf("Attribution Precision: %.2f\n", res.AttributionAccuracy)

	// Output:
	// Attribution Precision: 1.00
}

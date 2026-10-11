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
		SkipIntervention:  false,
		ExerciseCorruptEv: false,
	}

	result, err := cudadiagdemo.RunDemonstration(ctx, cfg)
	if err != nil {
		fmt.Printf("demonstration error: %v\n", err)
		return
	}

	fmt.Printf("Baseline: %s (admitted: %t)\n",
		result.BaselineDiagnosis.Mechanism, result.BaselineDiagnosis.Admitted)
	fmt.Printf("Host Delay: %s (admitted: %t)\n",
		result.HostDelayDiagnosis.Mechanism, result.HostDelayDiagnosis.Admitted)
	fmt.Printf("Device Bottleneck: %s (admitted: %t)\n",
		result.DeviceBottleneckDiag.Mechanism, result.DeviceBottleneckDiag.Admitted)
	fmt.Printf("Outputs verified: %t\n", result.TrialExecution.OutputsVerified)

	// Output:
	// Baseline: no_fault (admitted: true)
	// Host Delay: host_submission_delay (admitted: true)
	// Device Bottleneck: competing_device_bottleneck (admitted: true)
	// Outputs verified: true
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

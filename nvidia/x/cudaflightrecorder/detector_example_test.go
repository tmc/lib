package cudaflightrecorder_test

import (
	"context"
	"fmt"
	"github.com/tmc/lib/nvidia/x/cudaflightrecorder"
	"time"
)

func ExampleNewDetector() {
	d, err := cudaflightrecorder.NewDetector(cudaflightrecorder.DetectorConfig{
		Mode: "rolling", Window: 3, Warmup: 3, Delta: 5, MaxGap: time.Second,
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	recorder, err := cudaflightrecorder.New(cudaflightrecorder.Config{Capacity: 10})
	if err != nil {
		fmt.Println(err)
		return
	}
	coordinator, err := cudaflightrecorder.NewEscalationCoordinator(recorder, cudaflightrecorder.DefaultBudgets())
	if err != nil {
		fmt.Println(err)
		return
	}
	defer coordinator.Stop()
	for i, v := range []float64{10, 10, 10, 30} {
		result := d.Observe(cudaflightrecorder.DetectorObservation{
			Metric: "latency", Unit: "ms", Epoch: "process-1", Clock: "mono-1",
			Timestamp: int64(i), Value: v, Available: true,
		})
		if result.Anomaly {
			// The existing coordinator independently decides capture admission.
			request := cudaflightrecorder.TriggerRequest{Reason: result.Reason,
				MetricName: "latency", MetricValue: v, Threshold: result.Threshold}
			fmt.Println(request.Reason, request.Threshold)
			receipt, _ := coordinator.Trigger(context.Background(), request)
			// No capture is admitted by this disarmed coordinator.
			fmt.Println(receipt.Outcome, receipt.RefusalReason)
		}
	}
	// Output:
	// high_anomaly 15
	// refused disarmed
}

func ExampleDetector_Observe() {
	d, _ := cudaflightrecorder.NewDetector(cudaflightrecorder.DetectorConfig{
		Mode: "ewma", Alpha: .25, Warmup: 3, Delta: 5, MaxGap: time.Second,
	})
	result := d.Observe(cudaflightrecorder.DetectorObservation{Available: false})
	fmt.Println(result.Ready, result.Reason)
	// Output: false missing
}

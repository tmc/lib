package cudaflightrecorder_test

import (
	"bytes"
	"context"
	"fmt"
	"log"
	"time"

	"github.com/tmc/lib/nvidia/x/cudaflightrecorder"
	"github.com/tmc/lib/nvidia/x/cuptipcsampling"
	"github.com/tmc/lib/nvidia/x/cuptipmsampling"
)

func Example() {
	rec, err := cudaflightrecorder.New(cudaflightrecorder.Config{
		Window:   5 * time.Second,
		Capacity: 1024,
	})
	if err != nil {
		log.Fatal(err)
	}

	if err := rec.Start(); err != nil {
		log.Fatal(err)
	}
	defer rec.Stop()

	// Record an execution observation
	rec.Push(cudaflightrecorder.Record{
		TimestampNs:   time.Now().UnixNano(),
		KernelName:    "ds4_moe_expert_dispatch",
		CorrelationID: 42,
		DurationNs:    8500,
	})

	// Preserve recent observations at a workload boundary.
	var buf bytes.Buffer
	manifest, err := rec.Snapshot(context.Background(), &buf, "workload_boundary")
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Snapshot reason: %s (records: %d)\n", manifest.Reason, manifest.GPURecords)

	// Output:
	// Snapshot reason: workload_boundary (records: 1)
}

func ExampleRecorder_Snapshot_withGoTrace() {
	rec, err := cudaflightrecorder.New(cudaflightrecorder.Config{
		Window:         5 * time.Second,
		Capacity:       1024,
		GoTraceEnabled: true,
	})
	if err != nil {
		log.Fatal(err)
	}

	if err := rec.Start(); err != nil {
		log.Fatal(err)
	}
	defer rec.Stop()

	rec.SetGoTraceProvider(func(ctx context.Context) ([]byte, error) {
		return []byte("go 1.22 trace payload"), nil
	})

	// Record a GPU execution observation
	now := time.Now()
	rec.Push(cudaflightrecorder.Record{
		TimestampNs:   now.UnixNano(),
		KernelName:    "flash_attention_fwd",
		CorrelationID: 101,
		DurationNs:    15000,
	})
	time.Sleep(time.Millisecond)

	// Take a multi-modal snapshot containing both Go trace and GPU records
	var buf bytes.Buffer
	manifest, err := rec.Snapshot(context.Background(), &buf, "latency_spike")
	if err != nil {
		log.Fatal(err)
	}

	bundle, err := cudaflightrecorder.ReadBundle(&buf)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Snapshot reason: %s\n", bundle.Manifest.Reason)
	fmt.Printf("GPU records: %d\n", bundle.Manifest.GPURecords)
	fmt.Printf("Has go.trace: %t\n", len(bundle.Files["go.trace"]) > 0)

	// manifest.OverlapReliable reports whether the Go and GPU spans were
	// aligned with a clock uncertainty small enough to trust the overlap, and a
	// caller gates its own conclusions on it:
	//
	//	if manifest.OverlapReliable {
	//		// compare the Go and GPU spans
	//	}
	//
	// The example neither prints nor checks its value. That value depends on
	// whether the host could calibrate against CUPTI, so any assertion on it
	// here would be live on one kind of host and dead on the other. The rules
	// behind it are tested directly and host-independently, by
	// TestAlignSpansUncertaintyInputs and TestClockUncertaintyPrecedence.
	//
	// What does hold on every host is that the manifest Snapshot returns and
	// the one inside the bundle are the same manifest, so that is what the
	// example reports. It is printed rather than checked with log.Fatal:
	// os.Exit from an example aborts the test binary and takes every other
	// result in the package with it.
	fmt.Printf("Bundle manifest agrees: %t\n",
		bundle.Manifest.OverlapReliable == manifest.OverlapReliable &&
			bundle.Manifest.ClockUncertaintyNs == manifest.ClockUncertaintyNs &&
			bundle.Manifest.ClockUncertaintySource == manifest.ClockUncertaintySource)

	// Output:
	// Snapshot reason: latency_spike
	// GPU records: 1
	// Has go.trace: true
	// Bundle manifest agrees: true
}

func ExampleRecorder_PushSASSMetric() {
	rec, err := cudaflightrecorder.New(cudaflightrecorder.Config{
		Window:   5 * time.Second,
		Capacity: 100,
	})
	if err != nil {
		log.Fatal(err)
	}
	if err := rec.Start(); err != nil {
		log.Fatal(err)
	}
	defer rec.Stop()

	sm := cuptipcsampling.SASSMetric{
		PC:           0x100,
		KernelName:   "gemm_wmma",
		TotalSamples: 200,
		Selected:     40,
		Dominant: &cuptipcsampling.PipelineStall{
			Name:    "long_scoreboard",
			Percent: 80.0,
		},
	}
	rec.PushSASSMetric(time.Now().UnixNano(), 10, sm)
	fmt.Println("Pushed SASS metric for:", sm.KernelName)
	// Output:
	// Pushed SASS metric for: gemm_wmma
}

func ExampleRecorder_PushPMSample() {
	rec, err := cudaflightrecorder.New(cudaflightrecorder.Config{
		Window:   5 * time.Second,
		Capacity: 100,
	})
	if err != nil {
		log.Fatal(err)
	}
	if err := rec.Start(); err != nil {
		log.Fatal(err)
	}
	defer rec.Stop()

	sample := cuptipmsampling.Sample{
		Index:          0,
		StartTimestamp: 1000,
		EndTimestamp:   2000,
		DurationNs:     1000,
		Values:         map[string]float64{"sm_busy": 95.0},
	}
	rec.PushPMSample(sample)
	fmt.Printf("Pushed PM sample: duration=%dns\n", sample.DurationNs)
	// Output:
	// Pushed PM sample: duration=1000ns
}

func ExampleEscalationCoordinator() {
	rec, err := cudaflightrecorder.New(cudaflightrecorder.Config{
		Window:   5 * time.Second,
		Capacity: 100,
	})
	if err != nil {
		log.Fatal(err)
	}
	if err := rec.Start(); err != nil {
		log.Fatal(err)
	}
	defer rec.Stop()

	budgets := cudaflightrecorder.QualificationBudgets()
	budgets.HysteresisCount = 1

	coord, err := cudaflightrecorder.NewEscalationCoordinator(rec, budgets, cudaflightrecorder.WithResourceSampler(&cudaflightrecorder.FakeResourceSampler{}))
	if err != nil {
		log.Fatal(err)
	}
	if err := coord.Arm(); err != nil {
		log.Fatal(err)
	}
	defer coord.Stop()

	receipt, err := coord.Trigger(context.Background(), cudaflightrecorder.TriggerRequest{
		TenantID:    "prod-tenant",
		Reason:      "latency_regression",
		MetricName:  "gpu_latency_ms",
		MetricValue: 125.0,
		Threshold:   100.0,
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Incident outcome: %s (state: %s)\n", receipt.Outcome, receipt.State)
	// Output:
	// Incident outcome: captured (state: cooldown)
}

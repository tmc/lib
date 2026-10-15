package cudaflightrecorder_test

import (
	"context"
	"fmt"
	"strings"
	"time"

	fr "github.com/tmc/lib/nvidia/x/cudaflightrecorder"
	nt "github.com/tmc/lib/nvidia/x/nvtelemetry"
)

// This example provider performs no native acquisition or trace production.
type exampleWorkloadCapture struct{}

func (exampleWorkloadCapture) Capture(context.Context, *fr.Recorder, fr.TriggerRequest, func()) (fr.CaptureReport, error) {
	return fr.CaptureReport{Provider: "host_fake", Closed: true}, nil
}

func newWorkloadExample() (*fr.WorkloadTrigger, func(), error) {
	recorder, err := fr.New(fr.Config{Capacity: 8})
	if err != nil {
		return nil, nil, err
	}
	if err := recorder.Start(); err != nil {
		return nil, nil, err
	}
	budgets := fr.DefaultBudgets()
	budgets.HysteresisCount = 1
	coordinator, err := fr.NewEscalationCoordinator(recorder, budgets,
		fr.WithClock(fr.NewFakeClock(time.Unix(1, 0))),
		fr.WithResourceSampler(&fr.FakeResourceSampler{}),
		fr.WithCUPTINegotiator(&fr.FakeCUPTINegotiator{}),
		fr.WithIncidentCapture(exampleWorkloadCapture{}))
	if err != nil {
		recorder.Stop()
		return nil, nil, err
	}
	cleanup := func() { coordinator.Stop(); recorder.Stop() }
	if err := coordinator.Arm(); err != nil {
		cleanup()
		return nil, nil, err
	}
	bridge, err := fr.NewWorkloadTrigger(fr.WorkloadTriggerConfig{
		Provider: "triton", Version: "25.06", Metric: "nv_inference_queue_duration_us", Unit: "us",
		Mode: "counter_delta", TimeSource: "collector_observation",
		Labels: map[string]string{"model": "example", "version": "1"}, TenantID: "host-example",
		WindowDuration: time.Second, StaticThreshold: 20,
		Detector: fr.DetectorConfig{Mode: "rolling", Window: 2, Warmup: 1, Delta: 5, MaxGap: time.Second},
	}, coordinator)
	if err != nil {
		cleanup()
		return nil, nil, err
	}
	return bridge, cleanup, nil
}

func observeWorkloadExample(bridge *fr.WorkloadTrigger, index int64, before, after string) (fr.WorkloadTriggerResult, error) {
	ctx := context.Background()
	source := nt.WorkloadSource{Provider: "triton", Version: "25.06", RunID: "constructed-example", Phase: "steady", ClockDomain: "example-clock", ClockEpochID: "example-epoch"}
	read := func(value string, when int64) (nt.WorkloadExport, error) {
		raw := "# TYPE nv_inference_queue_duration_us counter\nnv_inference_queue_duration_us{model=\"example\",version=\"1\"} " + value + "\n"
		e, err := nt.ReadServingExposition(ctx, strings.NewReader(raw), source, nt.WorkloadLimits{})
		if err != nil {
			return e, err
		}
		return nt.StampWorkloadSnapshot(e, when, source.ClockDomain, source.ClockEpochID)
	}
	window := fr.WorkloadTriggerWindow{StartNanos: index * int64(time.Second), EndNanos: (index + 1) * int64(time.Second), ClockDomain: source.ClockDomain, ClockEpochID: source.ClockEpochID}
	a, err := read(before, window.StartNanos)
	if err != nil {
		return fr.WorkloadTriggerResult{}, err
	}
	b, err := read(after, window.EndNanos)
	if err != nil {
		return fr.WorkloadTriggerResult{}, err
	}
	return bridge.Observe(ctx, a, b, window)
}

func ExampleNewWorkloadTrigger() {
	bridge, cleanup, err := newWorkloadExample()
	if err != nil {
		fmt.Println(err)
		return
	}
	defer cleanup()
	result, err := observeWorkloadExample(bridge, 0, "0", "10")
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(result.Decision.Reason, result.Incident == nil)
	// Output: warmup true
}

func ExampleWorkloadTrigger_Observe() {
	bridge, cleanup, err := newWorkloadExample()
	if err != nil {
		fmt.Println(err)
		return
	}
	defer cleanup()
	if _, err := observeWorkloadExample(bridge, 0, "0", "10"); err != nil {
		fmt.Println(err)
		return
	}
	result, err := observeWorkloadExample(bridge, 1, "10", "40")
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(result.Decision.Reason, result.ExactValue, result.StaticExceeded)
	fmt.Println(result.Incident.Capture.Provider, "native startup:", result.Incident.Capture.Started)
	// Output:
	// high_anomaly 30 true
	// host_fake native startup: false
}

package nvtelemetry_test

import (
	"bytes"
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

func exampleCapturePolicy() nvtelemetry.CapturePolicy {
	return nvtelemetry.CapturePolicy{
		Version: 1, SourceID: "nvml", AllowedDevices: []string{"GPU-0"}, MaxReadings: 32, MaxSampleBytes: 4096,
		Collector: nvtelemetry.CollectorConfig{IntervalNanos: int64(time.Millisecond), MaxRecordBudget: 20, MaxByteBudget: 32768, MaxBufferedRecords: 4, BudgetPolicy: nvtelemetry.BudgetPolicyFail},
	}
}

func ExampleCapturePolicy_Validate() {
	p := exampleCapturePolicy()
	fmt.Println(p.Validate())
	p.AllowedDevices = nil
	fmt.Println(p.Validate() != nil)
	// Output:
	// <nil>
	// true
}

func ExampleReadCapturePolicy() {
	input := `{"version":1,"source_id":"nvml","allowed_devices":["GPU-0"],"max_readings":32,"max_sample_bytes":4096,"collector":{"interval_nanos":1000000,"max_record_budget":20,"max_byte_budget":32768,"max_buffered_records":4,"budget_policy":"fail"}}`
	p, err := nvtelemetry.ReadCapturePolicy(strings.NewReader(input), 4096)
	if err != nil {
		panic(err)
	}
	fmt.Println(p.SourceID, p.AllowedDevices[0])
	// Output: nvml GPU-0
}

func ExampleCaptureRegistry_NewSampler() {
	var registry nvtelemetry.CaptureRegistry
	source := nvtelemetry.NewFakeSampler(nvtelemetry.DeviceMetadata{UUID: "GPU-0"})
	s, err := registry.NewSampler(source, nvtelemetry.NewFakeClock(1, 1), exampleCapturePolicy())
	if err != nil {
		panic(err)
	}
	defer s.Close(context.Background())
	_, err = registry.NewSampler(source, nvtelemetry.NewFakeClock(1, 1), exampleCapturePolicy())
	fmt.Println(err != nil)
	// Output: true
}

func ExampleGovernedSampler() {
	p := exampleCapturePolicy()
	clock := nvtelemetry.NewFakeClock(1, 1)
	var registry nvtelemetry.CaptureRegistry
	s, err := registry.NewSampler(nvtelemetry.NewFakeSampler(nvtelemetry.DeviceMetadata{UUID: "GPU-0"}), clock, p)
	if err != nil {
		panic(err)
	}
	defer s.Close(context.Background())
	var raw bytes.Buffer
	c := nvtelemetry.NewCollector("governed", s, clock, p.Collector, nvtelemetry.NewWriter(&raw, p.Collector))
	if err := c.InitStream(); err != nil {
		panic(err)
	}
	clock.Advance(time.Millisecond)
	if err := c.Step(context.Background()); err != nil {
		panic(err)
	}
	clock.Advance(time.Millisecond)
	if _, err := c.Stop(); err != nil {
		panic(err)
	}
	m, _, samples, err := nvtelemetry.ReadStreamStrict(&raw, int64(p.Collector.MaxByteBudget))
	if err != nil {
		panic(err)
	}
	fmt.Println(m.IsComplete, len(samples), s.Status().SuccessfulQueries)
	// Output: true 1 1
}

func ExampleGovernedSampler_Sample() {
	var registry nvtelemetry.CaptureRegistry
	s, err := registry.NewSampler(nvtelemetry.NewFakeSampler(nvtelemetry.DeviceMetadata{UUID: "GPU-0"}), nvtelemetry.NewFakeClock(1, 1), exampleCapturePolicy())
	if err != nil {
		panic(err)
	}
	defer s.Close(context.Background())
	readings, err := s.Sample(context.Background())
	fmt.Println(len(readings), err)
	// Output: 1 <nil>
}

func ExampleGovernedSampler_Devices() {
	var registry nvtelemetry.CaptureRegistry
	s, err := registry.NewSampler(nvtelemetry.NewFakeSampler(nvtelemetry.DeviceMetadata{UUID: "GPU-0"}), nvtelemetry.NewFakeClock(1, 1), exampleCapturePolicy())
	if err != nil {
		panic(err)
	}
	defer s.Close(context.Background())
	fmt.Println(s.Devices()[0].UUID)
	// Output: GPU-0
}

func ExampleGovernedSampler_Status() {
	var registry nvtelemetry.CaptureRegistry
	s, err := registry.NewSampler(nvtelemetry.NewFakeSampler(nvtelemetry.DeviceMetadata{UUID: "GPU-0"}), nvtelemetry.NewFakeClock(1, 1), exampleCapturePolicy())
	if err != nil {
		panic(err)
	}
	defer s.Close(context.Background())
	fmt.Println(s.Status().HasLastSuccess)
	// Output: false
}

func ExampleGovernedSampler_Ready() {
	var registry nvtelemetry.CaptureRegistry
	s, err := registry.NewSampler(nvtelemetry.NewFakeSampler(nvtelemetry.DeviceMetadata{UUID: "GPU-0"}), nvtelemetry.NewFakeClock(1, 1), exampleCapturePolicy())
	if err != nil {
		panic(err)
	}
	defer s.Close(context.Background())
	if _, err := s.Sample(context.Background()); err != nil {
		panic(err)
	}
	fmt.Println(s.Ready(1, 0))
	// Output: <nil>
}

func ExampleGovernedSampler_Close() {
	var registry nvtelemetry.CaptureRegistry
	s, err := registry.NewSampler(nvtelemetry.NewFakeSampler(nvtelemetry.DeviceMetadata{UUID: "GPU-0"}), nvtelemetry.NewFakeClock(1, 1), exampleCapturePolicy())
	if err != nil {
		panic(err)
	}
	fmt.Println(s.Close(context.Background()), s.Status().Closed)
	// Output: <nil> true
}

func ExampleSafeCaptureLabels() {
	labels, err := nvtelemetry.SafeCaptureLabels(map[string]string{"job_id": "123", "api_key": "do-not-store"}, []string{"job_id"})
	fmt.Println(labels["job_id"], len(labels), err)
	// Output: 123 1 <nil>
}

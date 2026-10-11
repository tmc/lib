package nvtelemetry_test

import (
	"bytes"
	"context"
	"fmt"

	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

func exampleHealth() (nvtelemetry.HealthSnapshot, nvtelemetry.HealthLimits) {
	limits := nvtelemetry.HealthLimits{MaxBytes: 4096, MaxDevices: 1, MaxSources: 1, MaxObservations: 4, MaxEdges: 1}
	s := nvtelemetry.HealthSnapshot{
		Version: 1, NodeID: "node-0", TopologyEpoch: "topology-1", ClockDomain: "synthetic", ClockEpoch: "fixture-1", AtNanos: 10,
		DeviceStates: map[string]string{"GPU-0": "present"},
		Sources:      []nvtelemetry.HealthSource{{ID: "nvml", Kind: "nvml", Version: "fixture-v1", State: "available", HasLastSuccess: true, LastSuccessNanos: 10}},
		Observations: []nvtelemetry.HealthObservation{{SourceID: "nvml", Class: "raw", Reading: nvtelemetry.Reading{
			Name: nvtelemetry.MetricTemperatureC, Unit: nvtelemetry.UnitCelsius, Backend: "nvml", DeviceUUID: "GPU-0", Evidence: nvtelemetry.EvidenceMeasured, Value: 31, TimestampMonotonic: 10,
		}}},
		Complete: true,
	}
	return s, limits
}

func ExampleWriteHealthSnapshot() {
	s, limits := exampleHealth()
	var b bytes.Buffer
	err := nvtelemetry.WriteHealthSnapshot(&b, s, limits)
	fmt.Println(err, b.Len() > 0)
	// Output: <nil> true
}

func ExampleReadHealthSnapshot() {
	s, limits := exampleHealth()
	var b bytes.Buffer
	if err := nvtelemetry.WriteHealthSnapshot(&b, s, limits); err != nil {
		panic(err)
	}
	got, err := nvtelemetry.ReadHealthSnapshot(&b, limits)
	if err != nil {
		panic(err)
	}
	fmt.Println(got.DeviceStates["GPU-0"], got.Observations[0].Reading.Value)
	fmt.Println(got.Observations[0].Reading.HasSourceTimestamp)
	// Output:
	// present 31
	// false
}

func ExampleHealthSnapshot_Validate() {
	s, limits := exampleHealth()
	fmt.Println(s.Validate(limits))
	s.Observations[0].Reading.Unit = "fahrenheit"
	fmt.Println(s.Validate(limits) != nil)
	// Output:
	// <nil>
	// true
}

func ExampleHealthSource_QueryStale() {
	s := nvtelemetry.HealthSource{HasLastSuccess: true, LastSuccessNanos: 10}
	fmt.Println(s.QueryStale(11, 5), s.QueryStale(16, 5))
	// Output: false true
}

func ExampleNewNVMLHealthReader() {
	_, err := nvtelemetry.NewNVMLHealthReader(0, nvtelemetry.NewFakeClock(1, 1), "node", "epoch", nvtelemetry.MIGScopeUnknown, nil)
	fmt.Println(err != nil)
	// Output: true
}

func ExampleNVMLHealthReader_Read() {
	var reader nvtelemetry.NVMLHealthReader
	_, err := reader.Read(context.Background())
	fmt.Println(err != nil)
	// Output: true
}

func ExampleNewNVMLXIDReader() {
	_, err := nvtelemetry.NewNVMLXIDReader(0, nvtelemetry.NewFakeClock(1, 1))
	fmt.Println(err != nil)
	// Output: true
}

func ExampleNVMLXIDReader_Poll() {
	var reader nvtelemetry.NVMLXIDReader
	_, err := reader.Poll(context.Background(), 4)
	fmt.Println(err != nil)
	// Output: true
}

func ExampleNVMLXIDReader_Close() {
	var reader *nvtelemetry.NVMLXIDReader
	fmt.Println(reader.Close())
	// Output: <nil>
}

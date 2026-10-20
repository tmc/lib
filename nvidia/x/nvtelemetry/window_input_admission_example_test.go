package nvtelemetry_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

func ExampleAggregateWindowStrict_units() {
	samples := []nvtelemetry.Sample{{Readings: []nvtelemetry.Reading{{
		Name: nvtelemetry.MetricPowerUsageMW, Unit: "W", DeviceUUID: "GPU-example",
		Value: 1, Evidence: nvtelemetry.EvidenceMeasured,
	}}}}
	_, _, err := nvtelemetry.AggregateWindowStrict("measurement", 0, 1, samples, nvtelemetry.WindowOptions{})
	fmt.Println(err != nil)
	// Output: true
}

func ExampleIntegratePowerToEnergy_timeOrder() {
	points := []nvtelemetry.SamplePoint{
		{Timestamp: 0, Value: 1000, Available: true},
		{Timestamp: 1e9, Value: 1000, Available: true},
		{Timestamp: 0, Value: 1000, Available: true},
		{Timestamp: 1e9, Value: 1000, Available: true},
	}
	energy := nvtelemetry.IntegratePowerToEnergy("measurement", "GPU-example", points, 0, 1e9, nvtelemetry.WindowOptions{})
	fmt.Println(energy.Valid, energy.Reason)
	// Output: false non_increasing_sample_time
}

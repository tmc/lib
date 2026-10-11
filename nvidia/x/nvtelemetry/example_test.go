package nvtelemetry_test

import (
	"context"
	"fmt"

	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

func ExampleFakeSampler() {
	sampler := nvtelemetry.NewFakeSampler(nvtelemetry.DeviceMetadata{
		UUID:  "GPU-EXAMPLE-1",
		Index: 0,
		Model: "NVIDIA L4",
	})

	sampler.Enqueue([]nvtelemetry.Reading{
		{
			Name:       nvtelemetry.MetricGPUUtilization,
			Unit:       nvtelemetry.UnitPercent,
			Backend:    "fake",
			DeviceUUID: "GPU-EXAMPLE-1",
			Evidence:   nvtelemetry.EvidenceMeasured,
			Value:      78.5,
		},
	})

	readings, err := sampler.Sample(context.Background())
	if err != nil {
		fmt.Printf("error: %v\n", err)
		return
	}

	for _, r := range readings {
		fmt.Printf("%s: %.1f %s (%s)\n", r.Name, r.Value, r.Unit, r.Evidence)
	}

	// Output:
	// gpu_utilization_pct: 78.5 percent (measured)
}

package nvtelemetry_test

import (
	"context"
	"errors"
	"fmt"

	"github.com/tmc/lib/nvidia/nvml"
	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

// unsupportedQuery stands in for a device whose driver does not support the
// running-process queries. It reports no device data.
type unsupportedQuery struct{}

func (unsupportedQuery) ComputeRunningProcessesInto([]nvml.ProcessInfo) (int, error) {
	return 0, nvml.ErrNotSupported
}

func (unsupportedQuery) GraphicsRunningProcessesInto([]nvml.ProcessInfo) (int, error) {
	return 0, nvml.ErrNotSupported
}

func (unsupportedQuery) MPSComputeRunningProcessesInto([]nvml.ProcessInfo) (int, error) {
	return 0, nvml.ErrNotSupported
}

func ExampleQueryNVMLProcesses() {
	const uuid = "GPU-11111111-2222-3333-4444-555555555555"
	sel := nvtelemetry.NVMLProcessSelection{
		UUIDs:        []string{uuid},
		Sources:      []string{"nvml_compute_v3"},
		MaxPerCall:   64,
		MaxRecords:   65537,
		BootID:       "boot-1",
		PIDNamespace: "pid:[4026531836]",
		Snapshot: nvtelemetry.EntitySnapshot{
			SourceRevision: "fixture-nvml-1",
			NodeID:         "node-1",
			ClockDomain:    "synthetic_unix_utc",
			ClockEpoch:     "fixture-epoch-1",
			StartNanos:     100,
			EndNanos:       200,
		},
	}
	devices := []nvtelemetry.NVMLProcessDevice{{Device: nvtelemetry.DeviceMetadata{UUID: uuid}, Query: unsupportedQuery{}}}

	// A cap above 65536 is refused before any query.
	_, err := nvtelemetry.QueryNVMLProcesses(context.Background(), sel, devices)
	fmt.Println(err)

	// An unsupported query is a failure, not an empty list.
	sel.MaxRecords = 64
	facts, err := nvtelemetry.QueryNVMLProcesses(context.Background(), sel, devices)
	fmt.Println(len(facts), errors.Is(err, nvml.ErrNotSupported), nvml.IsUnsupported(err))
	// Output:
	// nvml processes: record caps must be in [1, 65536]
	// 0 true true
}

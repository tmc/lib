package nvtelemetry_test

import (
	"bytes"
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

func ExampleWriteTrainingWorkloadOTLP() {
	// Declared example input; the regression uses a retained real SDK derivative.
	source := nvtelemetry.WorkloadSource{Provider: "wandb", Version: "0.19.11", RunID: "example", ClockDomain: "unix_utc"}
	e, err := nvtelemetry.ReadWandBHistory(strings.NewReader(`{"_timestamp":1700000000,"_step":0,"loss":1}`), source, nvtelemetry.WorkloadLimits{})
	if err != nil {
		panic(err)
	}
	var out bytes.Buffer
	err = nvtelemetry.WriteTrainingWorkloadOTLP(&out, e, nvtelemetry.TrainingWorkloadOTLPOptions{UnixClockProvenance: "explicit example caller declaration; no synchronization proof"})
	fmt.Println(err, strings.Contains(out.String(), `"gauge"`), strings.Contains(out.String(), `"unit":""`))
	// Output:
	// <nil> true true
}

func ExampleTrainingWorkloadOTLPOptions() {
	options := nvtelemetry.TrainingWorkloadOTLPOptions{
		UnixClockProvenance: "caller-owned SDK timestamp contract; no synchronization bound",
		Limits:              nvtelemetry.WorkloadLimits{MaxMetrics: 10},
	}
	fmt.Println(options.Limits.MaxMetrics, len(options.UnitEvidence))
	// Output:
	// 10 0
}

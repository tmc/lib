package nvtelemetry_test

import (
	"bytes"
	"fmt"

	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

func ExampleWriteTrainingPhaseContext() {
	// Constructed application context; no optimizer outcome is invented.
	c := phaseContextFixture()
	var raw bytes.Buffer
	err := nvtelemetry.WriteTrainingPhaseContext(&raw, c, nvtelemetry.WorkloadLimits{})
	_, read, err2 := nvtelemetry.ReadTrainingPhaseContext(&raw, nvtelemetry.WorkloadSource{Provider: "application", Version: "training-context-v2", RunID: "run"}, nvtelemetry.WorkloadLimits{})
	fmt.Println(err, err2, read.Phases[0].Phase)
	// Output: <nil> <nil> forward
}

func ExampleReadTrainingPhaseContext() {
	raw := `{"version":2,"run_id":"run","policies":[],"phases":[]}`
	_, c, err := nvtelemetry.ReadTrainingPhaseContext(bytes.NewBufferString(raw), nvtelemetry.WorkloadSource{Provider: "application", Version: "training-context-v2", RunID: "run"}, nvtelemetry.WorkloadLimits{})
	fmt.Println(c.Version, len(c.Phases), err)
	// Output: 2 0 <nil>
}

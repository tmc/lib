package cudadiageval_test

import (
	"encoding/json"
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudadiageval"
)

func ExampleObserverPhasePlan() {
	// A real registration requires four complete candidate/comparator off/on
	// entries and original configuration artifacts before acquisition.
	plan := cudadiageval.ObserverPhasePlan{Version: 1}
	fmt.Println(plan.Version, len(plan.Configurations))
	// Output: 1 0
}

func ExampleObserverPhaseConfiguration() {
	config := cudadiageval.ObserverPhaseConfiguration{Version: 1, Arm: "candidate", Phase: "observer-off"}
	fmt.Println(config.Arm, config.Phase)
	// Output: candidate observer-off
}

func ExampleObserverPhaseSettings() {
	settings := cudadiageval.ObserverPhaseSettings{Mode: "off"}
	fmt.Println(settings.Mode, settings.DutyCycle, settings.SamplingRateHz)
	// Output: off 0 0
}

func ExampleObserverPhaseAcquisition() {
	// Record must name canonical acquired configuration bytes. A reference alone
	// never establishes that a tool actually used those settings.
	acquired := cudadiageval.ObserverPhaseAcquisition{Version: 1}
	raw, _ := json.Marshal(acquired)
	fmt.Println(string(raw))
	// Output: {"version":1,"record":{"path":"","sha256":""}}
}

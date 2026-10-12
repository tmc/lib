package nvtelemetry_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

func ExampleSamplingCoverage() {
	observed, eligible := uint64(0), uint64(100)
	coverage := nvtelemetry.SamplingCoverage{
		Status:      "measured",
		Basis:       "time",
		Scope:       "activity-union/eligible-window-v1",
		Numerator:   &nvtelemetry.Number{Uint: &observed},
		Denominator: &nvtelemetry.Number{Uint: &eligible},
		Unit:        "ns",
	}
	fmt.Printf("%s %s %d/%d %s\n", coverage.Status, coverage.Basis, *coverage.Numerator.Uint, *coverage.Denominator.Uint, coverage.Unit)
	// Output: measured time 0/100 ns
}

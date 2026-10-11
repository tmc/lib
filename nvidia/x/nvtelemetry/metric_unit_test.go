package nvtelemetry_test

import (
	"fmt"
	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

func ExampleLookupMetricUnit() {
	unit, ok := nvtelemetry.LookupMetricUnit(nvtelemetry.MetricPowerUsageMW)
	fmt.Println(unit, ok)
	// Output: milliwatts true
}

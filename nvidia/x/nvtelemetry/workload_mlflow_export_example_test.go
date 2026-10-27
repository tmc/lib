package nvtelemetry_test

import (
	"bytes"
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

func ExampleWriteTrainingWorkloadOTLP_mlflow() {
	// Declared example; the regression uses an actual SDK file-store response.
	source := nvtelemetry.WorkloadSource{Provider: "mlflow", Version: "2.22.0", RunID: "example", ClockDomain: "unix_utc"}
	e, err := nvtelemetry.ReadMLflowHistory(strings.NewReader(`{"metrics":[{"key":"loss","value":0.5,"timestamp":1700000000000,"step":0}]}`), source, nvtelemetry.WorkloadLimits{})
	if err != nil {
		panic(err)
	}
	var out bytes.Buffer
	err = nvtelemetry.WriteTrainingWorkloadOTLP(&out, e, nvtelemetry.TrainingWorkloadOTLPOptions{UnixClockProvenance: "caller-declared historical Unix milliseconds, no synchronization proof"})
	fmt.Println(err, strings.Contains(out.String(), `"timeUnixNano":"1700000000000000000"`), strings.Contains(out.String(), "supplied_page_only"))
	// Output:
	// <nil> true true
}

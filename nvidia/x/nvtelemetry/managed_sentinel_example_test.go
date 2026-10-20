package nvtelemetry_test

import (
	"context"
	"fmt"
	telemetry "github.com/tmc/lib/nvidia/x/nvtelemetry"
	"strings"
)

func ExampleReadManaged_floatingSentinel() {
	raw := "# TYPE DCGM_FI_DEV_POWER_USAGE gauge\nDCGM_FI_DEV_POWER_USAGE{UUID=\"GPU-a\"} 140737488355344\n"
	sample, err := telemetry.ReadManaged(context.Background(), strings.NewReader(raw), telemetry.ManagedImport{SourceID: "protocol-control", Producer: "dcgm-exporter", ProducerVersion: "control", NativeOrigin: "control", Fields: telemetry.DCGMExporterFields()})
	reading := sample.Readings[0]
	fmt.Println(err, reading.IsAvailable(), reading.UnavailableReason, reading.Source.RawValue)
	// Output: <nil> false blank 140737488355344
}

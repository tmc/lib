package cudadiag_test

import (
	"bytes"
	"context"
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudadiag"
	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

func ExampleDiagnoseTelemetry_v5() {
	// A valid format and exact value do not supply independent evidence links.
	var stream bytes.Buffer
	writer := nvtelemetry.NewWriter(&stream, nvtelemetry.CollectorConfig{})
	manifest := nvtelemetry.Manifest{SchemaVersion: nvtelemetry.SchemaVersionV5,
		RunID: "example", Devices: []nvtelemetry.DeviceMetadata{{UUID: "GPU-example"}}}
	if err := writer.WriteManifest(&manifest); err != nil {
		fmt.Println(err)
		return
	}
	value := uint64(1<<53 + 1)
	sample := nvtelemetry.Sample{Seq: 1, QueryStartNanos: 1, QueryEndNanos: 2,
		Readings: []nvtelemetry.Reading{{Name: "counter", DeviceUUID: "GPU-example",
			Evidence: nvtelemetry.EvidenceMeasured, Number: &nvtelemetry.Number{Uint: &value},
			Observation: &nvtelemetry.Observation{Kind: "gauge", SourceClass: "synthetic",
				Observer: nvtelemetry.Observer{SamplingCoverage: &nvtelemetry.SamplingCoverage{Status: "unknown"}}}}}}
	if err := writer.WriteSample(&sample); err != nil {
		fmt.Println(err)
		return
	}
	manifest.IsComplete, manifest.TotalSamples = true, 1
	if err := writer.WriteManifest(&manifest); err != nil {
		fmt.Println(err)
		return
	}
	if _, _, _, err := nvtelemetry.ReadStreamStrict(bytes.NewReader(stream.Bytes()), 1<<20); err != nil {
		fmt.Println(err)
		return
	}
	result, err := cudadiag.DiagnoseTelemetry(context.Background(), bytes.NewReader(stream.Bytes()), cudadiag.TelemetryOptions{})
	if err != nil {
		fmt.Println(err)
		return
	}
	finding := result.Observations[0]
	fmt.Println(*finding.Reading.Number.Uint)
	fmt.Println(finding.Finding.Admission.Status, finding.Finding.Admission.Reason)
	// Output:
	// 9007199254740993
	// withheld missing or mismatched workload identity
}

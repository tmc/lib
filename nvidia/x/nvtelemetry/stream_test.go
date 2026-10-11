package nvtelemetry_test

import (
	"bytes"
	"errors"
	"io"
	"testing"

	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

type shortWriter struct{}

func (shortWriter) Write(p []byte) (int, error) {
	if len(p) > 2 {
		return len(p) - 2, nil // short write without error!
	}
	return len(p), nil
}

func TestWriterRejectsShortWrites(t *testing.T) {
	w := nvtelemetry.NewWriter(shortWriter{}, nvtelemetry.CollectorConfig{})
	manifest := &nvtelemetry.Manifest{
		RunID:         "test-run",
		SchemaVersion: nvtelemetry.SchemaVersion,
	}
	err := w.WriteManifest(manifest)
	if !errors.Is(err, io.ErrShortWrite) {
		t.Fatalf("expected ErrShortWrite, got: %v", err)
	}
}

func TestReadStreamDetectsTruncatedRecord(t *testing.T) {
	// A valid manifest followed by a half-written sample
	truncatedData := `{"manifest":{"run_id":"run-1","schema_version":1,"collector_version":"1.0"}}
{"sample":{"seq":1,"query_start_nanos":100,"readings":[{"name":"power_milliwatts",`

	_, _, _, err := nvtelemetry.ReadStream(bytes.NewBufferString(truncatedData), 0)
	if !errors.Is(err, nvtelemetry.ErrTruncatedRecord) {
		t.Fatalf("expected ErrTruncatedRecord, got: %v", err)
	}
}

func TestReadStreamDetectsIdentityMismatch(t *testing.T) {
	streamData := `{"manifest":{"run_id":"run-1","schema_version":1,"devices":[{"uuid":"GPU-001"}]}}
{"sample":{"seq":1,"query_start_nanos":100,"readings":[{"name":"power_milliwatts","unit":"milliwatts","device_uuid":"GPU-ROGUE","value":100}]}}
`
	_, _, _, err := nvtelemetry.ReadStream(bytes.NewBufferString(streamData), 0)
	if !errors.Is(err, nvtelemetry.ErrIdentityMismatch) {
		t.Fatalf("expected ErrIdentityMismatch, got: %v", err)
	}
}

func TestReadStreamDetectsMalformedUnit(t *testing.T) {
	streamData := `{"manifest":{"run_id":"run-1","schema_version":1,"devices":[{"uuid":"GPU-001"}]}}
{"sample":{"seq":1,"query_start_nanos":100,"readings":[{"name":"power_milliwatts","unit":"watts","device_uuid":"GPU-001","value":100}]}}
`
	_, _, _, err := nvtelemetry.ReadStream(bytes.NewBufferString(streamData), 0)
	if !errors.Is(err, nvtelemetry.ErrMalformedStream) {
		t.Fatalf("expected ErrMalformedStream, got: %v", err)
	}
}

func TestReadStreamDetectsOutOfOrderSample(t *testing.T) {
	// Sample seq 2 followed by sample seq 1
	streamData := `{"manifest":{"run_id":"run-1","schema_version":1,"devices":[{"uuid":"GPU-001"}]}}
{"sample":{"seq":2,"query_start_nanos":200,"readings":[{"name":"power_milliwatts","unit":"milliwatts","device_uuid":"GPU-001","value":100}]}}
{"sample":{"seq":1,"query_start_nanos":100,"readings":[{"name":"power_milliwatts","unit":"milliwatts","device_uuid":"GPU-001","value":100}]}}
`
	_, _, _, err := nvtelemetry.ReadStream(bytes.NewBufferString(streamData), 0)
	if !errors.Is(err, nvtelemetry.ErrMalformedStream) {
		t.Fatalf("expected ErrMalformedStream, got: %v", err)
	}
}

func TestReadStreamTrailingManifestUpdatesCompleteness(t *testing.T) {
	streamData := `{"manifest":{"run_id":"run-1","schema_version":1,"devices":[{"uuid":"GPU-001"}],"start_time_monotonic":1000}}
{"sample":{"seq":1,"query_start_nanos":1050,"readings":[{"name":"power_milliwatts","unit":"milliwatts","device_uuid":"GPU-001","value":100}]}}
{"manifest":{"run_id":"run-1","schema_version":1,"devices":[{"uuid":"GPU-001"}],"start_time_monotonic":1000,"end_time_monotonic":2000,"total_samples":1,"lost_records":5,"is_complete":false,"incomplete_reason":"lost_records"}}
`
	m, _, samples, err := nvtelemetry.ReadStream(bytes.NewBufferString(streamData), 0)
	if err != nil {
		t.Fatalf("ReadStream error: %v", err)
	}
	if len(samples) != 1 {
		t.Fatalf("expected 1 sample, got %d", len(samples))
	}
	if m.EndTimeMonotonic != 2000 {
		t.Errorf("EndTimeMonotonic = %d, want 2000", m.EndTimeMonotonic)
	}
	if m.LostRecords != 5 {
		t.Errorf("LostRecords = %d, want 5", m.LostRecords)
	}
	if m.IsComplete {
		t.Errorf("m.IsComplete = true, want false when records are lost")
	}
}

func TestReadStreamRejectsTrailingManifestIdentityMismatch(t *testing.T) {
	// Initial run-1, trailing run-2
	streamData := `{"manifest":{"run_id":"run-1","schema_version":1,"devices":[{"uuid":"GPU-001"}]}}
{"sample":{"seq":1,"query_start_nanos":1050,"readings":[{"name":"power_milliwatts","unit":"milliwatts","device_uuid":"GPU-001","value":100}]}}
{"manifest":{"run_id":"run-2","schema_version":1,"devices":[{"uuid":"GPU-001"}]}}
`
	_, _, _, err := nvtelemetry.ReadStream(bytes.NewBufferString(streamData), 0)
	if !errors.Is(err, nvtelemetry.ErrIdentityMismatch) {
		t.Fatalf("expected ErrIdentityMismatch for run ID change, got: %v", err)
	}
}

func TestReadStreamRejectsTrailingManifestDeviceMismatch(t *testing.T) {
	// Initial GPU-001, trailing GPU-002
	streamData := `{"manifest":{"run_id":"run-1","schema_version":1,"devices":[{"uuid":"GPU-001"}]}}
{"sample":{"seq":1,"query_start_nanos":1050,"readings":[{"name":"power_milliwatts","unit":"milliwatts","device_uuid":"GPU-001","value":100}]}}
{"manifest":{"run_id":"run-1","schema_version":1,"devices":[{"uuid":"GPU-002"}]}}
`
	_, _, _, err := nvtelemetry.ReadStream(bytes.NewBufferString(streamData), 0)
	if !errors.Is(err, nvtelemetry.ErrIdentityMismatch) {
		t.Fatalf("expected ErrIdentityMismatch for device UUID change, got: %v", err)
	}
}

func TestReadStreamClassifiesIncompleteStream(t *testing.T) {
	// Stream ends with sample, no trailing completeness manifest
	streamData := `{"manifest":{"run_id":"run-1","schema_version":1,"devices":[{"uuid":"GPU-001"}]}}
{"sample":{"seq":1,"query_start_nanos":1050,"readings":[{"name":"power_milliwatts","unit":"milliwatts","device_uuid":"GPU-001","value":100}]}}
`
	m, _, _, err := nvtelemetry.ReadStream(bytes.NewBufferString(streamData), 0)
	if err != nil {
		t.Fatalf("unexpected ReadStream error: %v", err)
	}
	if m.IsComplete {
		t.Errorf("m.IsComplete = true, want false for stream missing completeness manifest")
	}
	if m.IncompleteReason != "missing_completion_manifest" {
		t.Errorf("m.IncompleteReason = %s, want missing_completion_manifest", m.IncompleteReason)
	}

	// ReadStreamStrict must return ErrIncompleteStream
	_, _, _, err = nvtelemetry.ReadStreamStrict(bytes.NewBufferString(streamData), 0)
	if !errors.Is(err, nvtelemetry.ErrIncompleteStream) {
		t.Fatalf("expected ErrIncompleteStream from ReadStreamStrict, got: %v", err)
	}
}

func TestReadStreamPreservesProducerIncomplete(t *testing.T) {
	// Trailing manifest carries explicit IsComplete: false from producer
	streamData := `{"manifest":{"run_id":"run-1","schema_version":1,"devices":[{"uuid":"GPU-001"}]}}
{"sample":{"seq":1,"query_start_nanos":1050,"readings":[{"name":"power_milliwatts","unit":"milliwatts","device_uuid":"GPU-001","value":100}]}}
{"manifest":{"run_id":"run-1","schema_version":1,"is_complete":false,"incomplete_reason":"budget_exceeded","devices":[{"uuid":"GPU-001"}]}}
`
	m, _, _, err := nvtelemetry.ReadStream(bytes.NewBufferString(streamData), 0)
	if err != nil {
		t.Fatalf("unexpected ReadStream error: %v", err)
	}
	if m.IsComplete {
		t.Fatalf("m.IsComplete = true, want false (producer marked incomplete)")
	}
	if m.IncompleteReason != "budget_exceeded" {
		t.Fatalf("m.IncompleteReason = %s, want budget_exceeded", m.IncompleteReason)
	}

	_, _, _, err = nvtelemetry.ReadStreamStrict(bytes.NewBufferString(streamData), 0)
	if !errors.Is(err, nvtelemetry.ErrIncompleteStream) {
		t.Fatalf("expected ErrIncompleteStream, got: %v", err)
	}
}

func TestReadStreamHeaderOnlyCompleteNotSuffice(t *testing.T) {
	// Header claims is_complete: true, but stream abruptly ends without trailing manifest
	streamData := `{"manifest":{"run_id":"run-1","schema_version":1,"is_complete":true,"devices":[{"uuid":"GPU-001"}]}}
{"sample":{"seq":1,"query_start_nanos":1050,"readings":[{"name":"power_milliwatts","unit":"milliwatts","device_uuid":"GPU-001","value":100}]}}
`
	m, _, _, err := nvtelemetry.ReadStream(bytes.NewBufferString(streamData), 0)
	if err != nil {
		t.Fatalf("unexpected ReadStream error: %v", err)
	}
	if m.IsComplete {
		t.Fatalf("m.IsComplete = true, want false when trailing manifest is missing")
	}
	if m.IncompleteReason != "missing_completion_manifest" {
		t.Fatalf("m.IncompleteReason = %s, want missing_completion_manifest", m.IncompleteReason)
	}

	_, _, _, err = nvtelemetry.ReadStreamStrict(bytes.NewBufferString(streamData), 0)
	if !errors.Is(err, nvtelemetry.ErrIncompleteStream) {
		t.Fatalf("expected ErrIncompleteStream, got: %v", err)
	}
}

func TestReadStreamRejectsInvalidEnvelopes(t *testing.T) {
	header := `{"manifest":{"run_id":"run","schema_version":1,"devices":[{"uuid":"GPU"}]}}` + "\n"
	complete := `{"manifest":{"run_id":"run","schema_version":1,"devices":[{"uuid":"GPU"}],"is_complete":true}}` + "\n"
	sample := `{"sample":{"seq":0,"readings":[]}}` + "\n"
	tests := []struct {
		name, stream string
		want         error
	}{
		{"record before header", sample + header, nvtelemetry.ErrMalformedStream},
		{"empty envelope", header + "{}\n", nvtelemetry.ErrMalformedStream},
		{"multiple fields", header + `{"sample":{"seq":1},"phase":{"name":"x"}}` + "\n", nvtelemetry.ErrMalformedStream},
		{"post completion sample", header + complete + sample, nvtelemetry.ErrMalformedStream},
		{"zero sequence duplicate", header + sample + sample, nvtelemetry.ErrMalformedStream},
		{"empty device identity", `{"manifest":{"schema_version":1,"devices":[{"uuid":""}]}}` + "\n", nvtelemetry.ErrIdentityMismatch},
		{"duplicate device identity", `{"manifest":{"schema_version":1,"devices":[{"uuid":"GPU"},{"uuid":"GPU"}]}}` + "\n", nvtelemetry.ErrIdentityMismatch},
		{"undeclared device without inventory", `{"manifest":{"schema_version":1}}` + "\n" + `{"sample":{"seq":1,"readings":[{"device_uuid":"GPU"}]}}` + "\n", nvtelemetry.ErrIdentityMismatch},
		{"complete despite loss", header + `{"manifest":{"schema_version":1,"run_id":"run","devices":[{"uuid":"GPU"}],"is_complete":true,"lost_records":1}}` + "\n", nvtelemetry.ErrMalformedStream},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, _, _, err := nvtelemetry.ReadStream(bytes.NewBufferString(tt.stream), 0)
			if !errors.Is(err, tt.want) {
				t.Fatalf("got %v, want %v", err, tt.want)
			}
		})
	}
}

func TestReadStreamCountsWhitespaceTowardBudget(t *testing.T) {
	_, _, _, err := nvtelemetry.ReadStream(bytes.NewBufferString("                        \n"), 8)
	if !errors.Is(err, nvtelemetry.ErrBudgetExceeded) {
		t.Fatalf("got %v, want budget exceeded", err)
	}
}

func TestReadStreamStrict_RejectsV2AdmissionViolations(t *testing.T) {
	headerV2 := `{"manifest":{"run_id":"run-v2","schema_version":2,"clock_monotonic_epoch_id":"epoch-1","devices":[{"uuid":"GPU-001"}]}}` + "\n"
	trailingV2 := `{"manifest":{"run_id":"run-v2","schema_version":2,"clock_monotonic_epoch_id":"epoch-1","total_samples":1,"is_complete":true,"devices":[{"uuid":"GPU-001"}]}}` + "\n"

	tests := []struct {
		name    string
		stream  string
		wantErr error
	}{
		{
			name: "negative_wall_clock_uncertainty",
			stream: headerV2 +
				`{"sample":{"seq":1,"query_start_nanos":1000,"query_end_nanos":2000,"readings":[{"name":"power_milliwatts","unit":"milliwatts","device_uuid":"GPU-001","evidence":"measured","value":100,"has_wall_clock_uncertainty":true,"wall_clock_uncertainty_nanos":-1}]}}` + "\n" +
				trailingV2,
			wantErr: nvtelemetry.ErrMalformedStream,
		},
		{
			name: "negative_clock_calibration_bounds",
			stream: `{"manifest":{"run_id":"run-v2","schema_version":2,"clock_monotonic_epoch_id":"epoch-1","has_clock_calibration":true,"clock_calibration_bounds_nanos":-1,"devices":[{"uuid":"GPU-001"}]}}` + "\n" +
				`{"sample":{"seq":1,"query_start_nanos":1000,"query_end_nanos":2000,"readings":[{"name":"power_milliwatts","unit":"milliwatts","device_uuid":"GPU-001","evidence":"measured","value":100}]}}` + "\n" +
				trailingV2,
			wantErr: nvtelemetry.ErrMalformedStream,
		},
		{
			name: "inverted_reading_query_bounds",
			stream: headerV2 +
				`{"sample":{"seq":1,"query_start_nanos":1000,"query_end_nanos":2000,"readings":[{"name":"power_milliwatts","unit":"milliwatts","device_uuid":"GPU-001","evidence":"measured","value":100,"has_query_bounds":true,"query_start_nanos":20,"query_end_nanos":10}]}}` + "\n" +
				trailingV2,
			wantErr: nvtelemetry.ErrMalformedStream,
		},
		{
			name: "trailing_manifest_epoch_mismatch",
			stream: headerV2 +
				`{"sample":{"seq":1,"query_start_nanos":1000,"query_end_nanos":2000,"readings":[{"name":"power_milliwatts","unit":"milliwatts","device_uuid":"GPU-001","evidence":"measured","value":100}]}}` + "\n" +
				`{"manifest":{"run_id":"run-v2","schema_version":2,"clock_monotonic_epoch_id":"different-epoch","total_samples":1,"is_complete":true,"devices":[{"uuid":"GPU-001"}]}}` + "\n",
			wantErr: nvtelemetry.ErrIdentityMismatch,
		},
		{
			name: "trailing_manifest_schema_downgrade",
			stream: headerV2 +
				`{"sample":{"seq":1,"query_start_nanos":1000,"query_end_nanos":2000,"readings":[{"name":"power_milliwatts","unit":"milliwatts","device_uuid":"GPU-001","evidence":"measured","value":100}]}}` + "\n" +
				`{"manifest":{"run_id":"run-v2","schema_version":1,"clock_monotonic_epoch_id":"epoch-1","total_samples":1,"is_complete":true,"devices":[{"uuid":"GPU-001"}]}}` + "\n",
			wantErr: nvtelemetry.ErrIdentityMismatch,
		},
		{
			name: "out_of_pass_query_bounds",
			stream: headerV2 +
				`{"sample":{"seq":1,"query_start_nanos":1000,"query_end_nanos":2000,"readings":[{"name":"power_milliwatts","unit":"milliwatts","device_uuid":"GPU-001","evidence":"measured","value":100,"has_query_bounds":true,"query_start_nanos":1500,"query_end_nanos":2500}]}}` + "\n" +
				trailingV2,
			wantErr: nvtelemetry.ErrMalformedStream,
		},
		{
			name: "trailing_manifest_provenance_mismatch",
			stream: `{"manifest":{"run_id":"run-v2","schema_version":2,"clock_monotonic_epoch_id":"epoch-1","clock_provenance":"system_kernel","devices":[{"uuid":"GPU-001"}]}}` + "\n" +
				`{"sample":{"seq":1,"query_start_nanos":1000,"query_end_nanos":2000,"readings":[{"name":"power_milliwatts","unit":"milliwatts","device_uuid":"GPU-001","evidence":"measured","value":100}]}}` + "\n" +
				`{"manifest":{"run_id":"run-v2","schema_version":2,"clock_monotonic_epoch_id":"epoch-1","clock_provenance":"custom_ntp","total_samples":1,"is_complete":true,"devices":[{"uuid":"GPU-001"}]}}` + "\n",
			wantErr: nvtelemetry.ErrIdentityMismatch,
		},
		{
			name: "trailing_manifest_binary_digest_mismatch",
			stream: `{"manifest":{"run_id":"run-v2","schema_version":2,"clock_monotonic_epoch_id":"epoch-1","binary_digest":"sha256-original","devices":[{"uuid":"GPU-001"}]}}` + "\n" +
				`{"sample":{"seq":1,"query_start_nanos":1000,"query_end_nanos":2000,"readings":[{"name":"power_milliwatts","unit":"milliwatts","device_uuid":"GPU-001","evidence":"measured","value":100}]}}` + "\n" +
				`{"manifest":{"run_id":"run-v2","schema_version":2,"clock_monotonic_epoch_id":"epoch-1","binary_digest":"sha256-changed","total_samples":1,"is_complete":true,"devices":[{"uuid":"GPU-001"}]}}` + "\n",
			wantErr: nvtelemetry.ErrIdentityMismatch,
		},
		{
			name: "sample_zero_start_reading_negative_start",
			stream: headerV2 +
				`{"sample":{"seq":1,"query_start_nanos":0,"query_end_nanos":10,"readings":[{"name":"power_milliwatts","unit":"milliwatts","device_uuid":"GPU-001","evidence":"measured","value":100,"has_query_bounds":true,"query_start_nanos":-1,"query_end_nanos":5}]}}` + "\n" +
				trailingV2,
			wantErr: nvtelemetry.ErrMalformedStream,
		},
		{
			name: "sample_zero_start_reading_exceeds_end",
			stream: headerV2 +
				`{"sample":{"seq":1,"query_start_nanos":0,"query_end_nanos":10,"readings":[{"name":"power_milliwatts","unit":"milliwatts","device_uuid":"GPU-001","evidence":"measured","value":100,"has_query_bounds":true,"query_start_nanos":1,"query_end_nanos":11}]}}` + "\n" +
				trailingV2,
			wantErr: nvtelemetry.ErrMalformedStream,
		},
		{
			name: "sample_end_zero_inverted",
			stream: headerV2 +
				`{"sample":{"seq":1,"query_start_nanos":10,"query_end_nanos":0,"readings":[{"name":"power_milliwatts","unit":"milliwatts","device_uuid":"GPU-001","evidence":"measured","value":100}]}}` + "\n" +
				trailingV2,
			wantErr: nvtelemetry.ErrMalformedStream,
		},
		{
			name: "sample_inverted_duration",
			stream: headerV2 +
				`{"sample":{"seq":1,"query_start_nanos":10,"query_end_nanos":5,"readings":[{"name":"power_milliwatts","unit":"milliwatts","device_uuid":"GPU-001","evidence":"measured","value":100}]}}` + "\n" +
				trailingV2,
			wantErr: nvtelemetry.ErrMalformedStream,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, _, _, err := nvtelemetry.ReadStreamStrict(bytes.NewBufferString(tt.stream), 0)
			if !errors.Is(err, tt.wantErr) {
				t.Fatalf("got %v, want %v", err, tt.wantErr)
			}
		})
	}
}

func TestReadStreamStrict_ValidZeroPassContainment(t *testing.T) {
	headerV2 := `{"manifest":{"run_id":"run-v2","schema_version":2,"clock_monotonic_epoch_id":"epoch-1","devices":[{"uuid":"GPU-001"}]}}` + "\n"
	trailingV2 := `{"manifest":{"run_id":"run-v2","schema_version":2,"clock_monotonic_epoch_id":"epoch-1","total_samples":1,"is_complete":true,"devices":[{"uuid":"GPU-001"}]}}` + "\n"

	// Valid exact boundaries [0, 10]
	streamExact := headerV2 +
		`{"sample":{"seq":1,"query_start_nanos":0,"query_end_nanos":10,"readings":[{"name":"power_milliwatts","unit":"milliwatts","device_uuid":"GPU-001","evidence":"measured","value":100,"has_query_bounds":true,"query_start_nanos":0,"query_end_nanos":10}]}}` + "\n" +
		trailingV2
	if _, _, _, err := nvtelemetry.ReadStreamStrict(bytes.NewBufferString(streamExact), 0); err != nil {
		t.Fatalf("expected valid stream on exact [0, 10], got: %v", err)
	}

	// Valid inner boundaries [2, 8] within [0, 10]
	streamInner := headerV2 +
		`{"sample":{"seq":1,"query_start_nanos":0,"query_end_nanos":10,"readings":[{"name":"power_milliwatts","unit":"milliwatts","device_uuid":"GPU-001","evidence":"measured","value":100,"has_query_bounds":true,"query_start_nanos":2,"query_end_nanos":8}]}}` + "\n" +
		trailingV2
	if _, _, _, err := nvtelemetry.ReadStreamStrict(bytes.NewBufferString(streamInner), 0); err != nil {
		t.Fatalf("expected valid stream on inner [2, 8], got: %v", err)
	}
}

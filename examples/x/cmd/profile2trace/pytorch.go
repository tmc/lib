package main

import (
	"context"
	"encoding/hex"
	"fmt"
	"io"

	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

// readPyTorch projects only the selected CPU annotation profile. The artifact
// digest namespaces trace-local IDs; it is not an application run or clock epoch.
func readPyTorch(ctx context.Context, r io.Reader, version, digest string, max int) ([]slice, error) {
	if ctx == nil {
		return nil, fmt.Errorf("PyTorch CPU input requires a context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	hash, err := hex.DecodeString(digest)
	if err != nil || len(hash) != 32 || version != "2.10.0" || max < 1 {
		return nil, fmt.Errorf("PyTorch CPU input requires version 2.10.0, SHA256 and a positive slice limit")
	}
	digest = hex.EncodeToString(hash)
	// This is an artifact cohort required by the existing reader, not inferred
	// application identity. The reader owns byte/event/string bounds and parsing.
	source := nvtelemetry.WorkloadSource{Provider: "pytorch", Version: version, RunID: "artifact:" + digest}
	export, annotations, err := nvtelemetry.ReadPyTorchAnnotations(pytorchContextReader{ctx, r}, source, nvtelemetry.WorkloadLimits{})
	if err != nil {
		return nil, fmt.Errorf("read PyTorch CPU input: %w", err)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if export.RawSHA256 != digest {
		return nil, fmt.Errorf("PyTorch CPU input digest mismatch")
	}
	if len(annotations) > max {
		return nil, fmt.Errorf("PyTorch CPU annotation slice limit exceeded")
	}
	rows := make([]slice, 0, len(annotations))
	for i, a := range annotations {
		rows = append(rows, slice{
			name: a.Name, category: "external:pytorch-cpu-user-annotation",
			process: "pytorch:" + digest + "/pid:" + string(a.PID),
			thread:  "tid:" + string(a.TID), start: a.StartNanos, duration: a.DurationNanos,
			args: map[string]any{
				"basis": "external:pytorch-cpu-annotation", "producer_version_declared": version,
				"source_report_sha256": digest, "source_annotation_ordinal": i,
				"source_pid_json": string(a.PID), "source_tid_json": string(a.TID), "source_args_json": string(a.Args),
				"source_start_raw": a.RawStart, "source_duration_raw": a.RawDuration, "source_time_unit": "microseconds",
				"source_clock": "unestablished", "capture_completeness": "unestablished",
				"selection": "complete user_annotation events only", "native_membership": "unavailable",
			},
		})
	}
	return rows, nil
}

type pytorchContextReader struct {
	ctx    context.Context
	reader io.Reader
}

func (r pytorchContextReader) Read(p []byte) (int, error) {
	if err := r.ctx.Err(); err != nil {
		return 0, err
	}
	return r.reader.Read(p)
}

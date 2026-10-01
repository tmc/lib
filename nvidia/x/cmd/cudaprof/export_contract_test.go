package main

import (
	"strings"
	"testing"
)

func TestExportReceiptRequiresWholeRunStatus(t *testing.T) {
	identity := traceExportIdentity{sha256: strings.Repeat("a", 64), path: "capture.pprof"}
	for _, tc := range []struct {
		name   string
		status captureResult
		want   string
	}{
		{"missing", captureResult{}, "unavailable"},
		{"legacy", captureResult{RunCompleteness: runComplete}, "unavailable"},
		{"complete", captureResult{Protocol: captureProtocolVersion, RunCompleteness: runComplete}, "passed"},
		{"partial", captureResult{Protocol: captureProtocolVersion, RunCompleteness: "partial"}, "failed"},
		{"drop", captureResult{Protocol: captureProtocolVersion, RunCompleteness: runComplete, DroppedRecords: 1}, "failed"},
		{"target failed", captureResult{Protocol: captureProtocolVersion, RunCompleteness: runComplete, TargetExitCode: 2}, "failed"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			receipt := exportReceipt("trace export", identity, nil, tc.status)
			if got := receipt.Gates["collector"].Disposition; got != tc.want {
				t.Fatalf("collector=%q, want %q", got, tc.want)
			}
			if tc.want == "unavailable" && receipt.CollectorActivityRecords != nil {
				t.Fatal("unavailable count reported as zero")
			}
		})
	}
}

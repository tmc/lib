package main

import (
	"bytes"
	"encoding/json"
	"os"
	"strings"
	"testing"
)

func TestReportOnSampleProfile(t *testing.T) {
	const samplePath = "/Users/tmc/tmp/ds4-decode-pcsampled.pprof"
	if _, err := os.Stat(samplePath); err != nil {
		t.Skipf("sample profile not present: %v", err)
	}

	t.Run("markdown", func(t *testing.T) {
		var buf bytes.Buffer
		if err := runReport(&buf, samplePath, 5, false); err != nil {
			t.Fatalf("runReport markdown error: %v", err)
		}
		out := buf.String()
		if !strings.Contains(out, "Global Metric Ledger") {
			t.Errorf("output missing Global Metric Ledger")
		}
		if !strings.Contains(out, "ds4_q8_0_matvec_q8_1") {
			t.Errorf("output missing primary kernel ds4_q8_0_matvec_q8_1")
		}
	})

	t.Run("json", func(t *testing.T) {
		var buf bytes.Buffer
		if err := runReport(&buf, samplePath, 5, true); err != nil {
			t.Fatalf("runReport json error: %v", err)
		}
		var parsed map[string]any
		if err := json.Unmarshal(buf.Bytes(), &parsed); err != nil {
			t.Fatalf("invalid json produced: %v", err)
		}
		if _, ok := parsed["kernels"]; !ok {
			t.Errorf("json output missing kernels key")
		}
	})
}

func TestReportMissingFile(t *testing.T) {
	var buf bytes.Buffer
	err := runReport(&buf, "nonexistent-file.pprof", 5, false)
	if err == nil {
		t.Errorf("expected error for nonexistent file, got nil")
	}
}

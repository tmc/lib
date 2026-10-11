package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

func TestCLIDeviceSelectionBeforeNativeAdmission(t *testing.T) {
	uuid := "GPU-12345678-1234-1234-1234-123456789abc"
	for _, input := range []string{",", uuid + "," + uuid, " GPU-0", "GPU-0", "GPU-../../other", uuid + ","} {
		var stdout, stderr bytes.Buffer
		if err := run([]string{"record", "-device-uuids", input}, &stdout, &stderr); err == nil || !strings.Contains(err.Error(), "UUID") {
			t.Fatalf("input %q admitted or reached NVML: %v", input, err)
		}
	}
	if got, err := parseDeviceUUIDs(uuid); err != nil || len(got) != 1 || got[0] != uuid {
		t.Fatalf("valid UUID %v %v", got, err)
	}
	var stdout, stderr bytes.Buffer
	if err := run([]string{"record", "-mock", "-device-uuids", uuid}, &stdout, &stderr); err == nil || !strings.Contains(err.Error(), "mock") {
		t.Fatalf("mock falsely selected devices: %v", err)
	}
}

func TestCLI_Help(t *testing.T) {
	var stdout, stderr bytes.Buffer
	err := run([]string{"help"}, &stdout, &stderr)
	if err != nil {
		t.Fatalf("help command returned error: %v", err)
	}
	if !strings.Contains(stdout.String(), "Usage:") {
		t.Fatalf("expected usage in output, got: %s", stdout.String())
	}
}

func TestCLI_UnknownCommand(t *testing.T) {
	var stdout, stderr bytes.Buffer
	err := run([]string{"nonexistent"}, &stdout, &stderr)
	if err == nil {
		t.Fatal("expected error for unknown command, got nil")
	}
}

func TestCLI_RecordReplaySummaryWorkflow(t *testing.T) {
	tmpDir := t.TempDir()
	outputPath := filepath.Join(tmpDir, "test_telemetry.jsonl")

	// 1. Record with mock sampler for 250ms
	var stdout, stderr bytes.Buffer
	recordArgs := []string{
		"record",
		"-mock",
		"-output=" + outputPath,
		"-interval=50ms",
		"-duration=250ms",
		"-phase=measurement",
	}
	if err := run(recordArgs, &stdout, &stderr); err != nil {
		t.Fatalf("record failed: %v\nstderr: %s", err, stderr.String())
	}
	if !strings.Contains(stderr.String(), "Recording finished:") {
		t.Fatalf("missing completion report: %s", stderr.String())
	}

	if _, err := os.Stat(outputPath); os.IsNotExist(err) {
		t.Fatalf("output file %s was not created", outputPath)
	}

	// 2. Replay with strict validation
	stdout.Reset()
	stderr.Reset()
	replayArgs := []string{
		"replay",
		"-strict",
		"-input=" + outputPath,
	}
	if err := run(replayArgs, &stdout, &stderr); err != nil {
		t.Fatalf("replay failed: %v\nstderr: %s", err, stderr.String())
	}
	if !strings.Contains(stdout.String(), "Stream verified (strict)") {
		t.Fatalf("unexpected replay output: %s", stdout.String())
	}

	// 3. Summary text
	stdout.Reset()
	stderr.Reset()
	summaryArgs := []string{
		"summary",
		"-input=" + outputPath,
		"-window=measurement",
		"-format=text",
	}
	if err := run(summaryArgs, &stdout, &stderr); err != nil {
		t.Fatalf("summary text failed: %v\nstderr: %s", err, stderr.String())
	}
	summaryOut := stdout.String()
	if !strings.Contains(summaryOut, "gpu_utilization_pct") {
		t.Fatalf("summary output missing gpu_utilization_pct: %s", summaryOut)
	}
	if !strings.Contains(summaryOut, "total_energy_millijoules") && !strings.Contains(summaryOut, "Energy Summary:") {
		t.Fatalf("summary output missing energy summary: %s", summaryOut)
	}

	// 4. Summary JSON
	stdout.Reset()
	stderr.Reset()
	summaryJSONArgs := []string{
		"summary",
		"-input=" + outputPath,
		"-window=measurement",
		"-format=json",
	}
	if err := run(summaryJSONArgs, &stdout, &stderr); err != nil {
		t.Fatalf("summary json failed: %v\nstderr: %s", err, stderr.String())
	}

	var jsonResult struct {
		RunID    string                      `json:"run_id"`
		Window   string                      `json:"window"`
		Complete bool                        `json:"complete"`
		Metrics  []nvtelemetry.WindowSummary `json:"metrics"`
		Energy   []nvtelemetry.EnergySummary `json:"energy"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &jsonResult); err != nil {
		t.Fatalf("failed to parse summary json: %v\noutput: %s", err, stdout.String())
	}
	if !jsonResult.Complete {
		t.Fatal("expected jsonResult.Complete to be true")
	}
	if jsonResult.Window != "measurement" {
		t.Fatalf("expected window 'measurement', got %q", jsonResult.Window)
	}
	if len(jsonResult.Metrics) == 0 {
		t.Fatal("expected at least one metric in summary json")
	}
}

func TestCLI_ReplayStrictRejectsIncompleteStream(t *testing.T) {
	tmpDir := t.TempDir()
	incompletePath := filepath.Join(tmpDir, "incomplete.jsonl")

	// Write only initial manifest without trailing completion manifest
	var buf bytes.Buffer
	w := nvtelemetry.NewWriter(&buf, nvtelemetry.CollectorConfig{})
	manifest := nvtelemetry.Manifest{
		RunID:            "test-incomplete",
		SchemaVersion:    nvtelemetry.SchemaVersion,
		CollectorVersion: nvtelemetry.CollectorVersion,
		StartTimeUnix:    time.Now().Unix(),
	}
	if err := w.WriteManifest(&manifest); err != nil {
		t.Fatalf("WriteManifest failed: %v", err)
	}
	if err := os.WriteFile(incompletePath, buf.Bytes(), 0644); err != nil {
		t.Fatalf("write file: %v", err)
	}

	var stdout, stderr bytes.Buffer
	err := run([]string{"replay", "-strict", "-input=" + incompletePath}, &stdout, &stderr)
	if err == nil {
		t.Fatal("expected replay -strict to fail on incomplete stream, got nil")
	}
	if !strings.Contains(err.Error(), "incomplete") {
		t.Fatalf("expected error mentioning incomplete stream, got: %v", err)
	}
}

func TestCLI_RecordDurationZeroFailsOnError(t *testing.T) {
	// Pointing to a directory that cannot be created should cause immediate failure,
	// not hang waiting for duration/signal.
	invalidPath := filepath.Join("/dev/null/impossible/path", "out.jsonl")
	var stdout, stderr bytes.Buffer
	err := run([]string{"record", "-mock", "-output=" + invalidPath, "-duration=0"}, &stdout, &stderr)
	if err == nil {
		t.Fatal("expected record to fail on invalid output path, got nil")
	}
}

func TestCLI_RecordEscalation(t *testing.T) {
	tmpDir := t.TempDir()
	outputPath := filepath.Join(tmpDir, "telemetry.jsonl")
	incidentDir := filepath.Join(tmpDir, "incidents")

	var stdout, stderr bytes.Buffer
	recordArgs := []string{
		"record",
		"-mock",
		"-output=" + outputPath,
		"-interval=20ms",
		"-duration=200ms",
		"-escalate",
		"-trigger-metric=gpu_utilization_pct",
		"-trigger-threshold=50.0",
		"-tenant=test-tenant",
		"-incident-dir=" + incidentDir,
	}

	if err := run(recordArgs, &stdout, &stderr); err != nil {
		t.Fatalf("record with escalation failed: %v\nstderr: %s", err, stderr.String())
	}

	// Verify continuous telemetry stream exists
	if _, err := os.Stat(outputPath); os.IsNotExist(err) {
		t.Fatalf("telemetry stream %s was not created", outputPath)
	}

	// Escalation's resource window must not block periodic sampling.
	raw, err := os.ReadFile(outputPath)
	if err != nil {
		t.Fatal(err)
	}
	_, _, samples, err := nvtelemetry.ReadStreamStrict(bytes.NewReader(raw), 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	if len(samples) < 3 {
		t.Fatalf("escalation blocked continuous sampling: %d samples", len(samples))
	}

	// Verify incident capture bundle was written to incidentDir
	files, err := os.ReadDir(incidentDir)
	if os.IsNotExist(err) || (err == nil && len(files) == 0) {
		// The mock NVML sampler does not invent host resource availability. A
		// short capture can end before the Linux resource window; other hosts
		// explicitly refuse their unsupported sampler.
		if !strings.Contains(stderr.String(), "incident escalation refused or failed") {
			t.Fatalf("missing refusal: %s", stderr.String())
		}
		return
	}
	if err != nil {
		t.Fatalf("read incident dir: %v", err)
	}
	var tarFound bool
	for _, f := range files {
		if strings.HasSuffix(f.Name(), ".tar") {
			tarFound = true
			break
		}
	}
	if !tarFound {
		t.Fatalf("expected .tar incident bundle in %s, got files: %+v", incidentDir, files)
	}
}

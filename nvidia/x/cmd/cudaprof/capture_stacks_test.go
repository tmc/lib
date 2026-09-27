package main

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/x/cudacapture"
)

func TestCaptureRecordAttachesExternalStacks(t *testing.T) {
	scratch := t.TempDir()
	sidecar := "{\"version\":1,\"kind\":\"header\",\"losses\":\"00000000000000000000\"}\n" +
		"{\"version\":1,\"kind\":\"stack\",\"external_kind\":3,\"external_id\":77,\"frames\":[\"main.main\",\"main.cannedTarget\"]}\n"
	if err := os.WriteFile(filepath.Join(scratch, cudacapture.ExternalStacksName), []byte(sidecar), 0600); err != nil {
		t.Fatal(err)
	}
	activities := []cudacapture.ActivityRecord{
		{Subtype: "external_correlation", ExternalKind: uint32(cupti.ExternalCorrelationKindOpenacc), ExternalID: 55, Correlation: 999},
		{Subtype: "external_correlation", ExternalKind: uint32(cupti.ExternalCorrelationKindCustom0), ExternalID: 77, Correlation: 123},
		{Subtype: "kernel", Name: "kernel_canned", Correlation: 123, Start: 100, End: 600, Valid: true},
	}
	path := filepath.Join(t.TempDir(), "capture.cudaprof")
	result := captureResult{Protocol: captureProtocolVersion, Session: "test-session", State: captureStateFinalized,
		Outcome: captureOutcomeObserved, TargetPID: 1234, CaptureSuccess: true, ScratchDir: scratch, Activities: activities}
	if _, err := writeCapture(path, result, captureArtifactBudget{FileBytes: 1 << 20, TotalBytes: 4 << 20}); err != nil {
		t.Fatal(err)
	}
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	capture, err := cudacapture.Read(f)
	if err != nil {
		t.Fatal(err)
	}
	data, ok := capture.Files[cudacapture.ExternalStacksName]
	if !ok || len(data) == 0 {
		t.Fatalf("capture sidecar present=%v bytes=%d", ok, len(data))
	}
	if len(capture.Profile.Sample) != 1 {
		t.Fatalf("samples = %d, want 1", len(capture.Profile.Sample))
	}
	labels := capture.Profile.Sample[0].Label
	if got := labels["cupti.external.stack"]; len(got) != 1 || got[0] != "main.main;main.cannedTarget" {
		t.Fatalf("stack label = %v", got)
	}
	if got := labels["cupti.external.id"]; len(got) != 1 || got[0] != "77" {
		t.Fatalf("external ID label = %v, all labels=%v", got, labels)
	}
}

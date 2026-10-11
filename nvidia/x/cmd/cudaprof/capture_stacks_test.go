package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaprof"
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

func stackActivitySet(process uint64, items ...struct {
	name string
	id   uint64
	corr uint32
}) []cudacapture.ActivityRecord {
	activities := make([]cudacapture.ActivityRecord, 0, 2*len(items))
	for i, item := range items {
		activities = append(activities,
			cudacapture.ActivityRecord{Process: process, Subtype: "external_correlation", ExternalKind: uint32(cupti.ExternalCorrelationKindCustom0), ExternalID: item.id, Correlation: item.corr},
			cudacapture.ActivityRecord{Process: process, Subtype: "kernel", Name: item.name, Correlation: item.corr, Start: uint64(10 + i*10), End: uint64(15 + i*10), Valid: true},
		)
	}
	return activities
}

func readStackCapture(t *testing.T, sidecar []byte, activities []cudacapture.ActivityRecord, budget captureArtifactBudget) (*cudacapture.Capture, []captureOmittedFile) {
	t.Helper()
	scratch := t.TempDir()
	if sidecar != nil {
		if err := os.WriteFile(filepath.Join(scratch, cudacapture.ExternalStacksName), sidecar, 0600); err != nil {
			t.Fatal(err)
		}
	}
	path := filepath.Join(t.TempDir(), "capture.cudaprof")
	result := captureResult{Protocol: captureProtocolVersion, Session: "stack-test", State: captureStateFinalized,
		Outcome: captureOutcomeObserved, TargetPID: 1234, CaptureSuccess: true, ScratchDir: scratch, Activities: activities}
	omitted, err := writeCapture(path, result, budget)
	if err != nil {
		t.Fatal(err)
	}
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	cap, err := cudacapture.Read(f)
	if err != nil {
		t.Fatal(err)
	}
	return cap, omitted
}

func oneExternalActivity() []cudacapture.ActivityRecord {
	return stackActivitySet(1234, struct {
		name string
		id   uint64
		corr uint32
	}{"kernel", 77, 123})
}

func TestCaptureRecordTornStackSidecar(t *testing.T) {
	good := "{\"version\":1,\"kind\":\"header\",\"losses\":\"00000000000000000000\"}\n" +
		"{\"version\":1,\"kind\":\"stack\",\"external_kind\":3,\"external_id\":77,\"frames\":[\"main\",\"tornTarget\"]}\n"
	cap, omitted := readStackCapture(t, []byte(good+"{\"version\":1,\"kind\":"), oneExternalActivity(), captureArtifactBudget{FileBytes: 1 << 20, TotalBytes: 4 << 20})
	if len(omitted) != 0 {
		t.Fatalf("omitted = %+v", omitted)
	}
	if got := cap.Profile.Sample[0].Label["cupti.external.stack"]; len(got) != 1 || got[0] != "main;tornTarget" {
		t.Fatalf("complete stack record was not retained: %v", got)
	}
	if !strings.Contains(strings.Join(cap.Profile.Comments, "\n"), "sidecar damaged") {
		t.Fatalf("profile comments do not report torn sidecar: %v", cap.Profile.Comments)
	}
}

func TestCaptureRecordOversizedStackSidecar(t *testing.T) {
	table := cudaprof.ExternalStackTable{Version: 1, Stacks: []cudaprof.ExternalStackRecord{{ExternalKind: 3, ExternalID: 77, Frames: []string{"main", strings.Repeat("frame-", 100)}}}}
	data, err := json.Marshal(table)
	if err != nil {
		t.Fatal(err)
	}
	cap, omitted := readStackCapture(t, data, oneExternalActivity(), captureArtifactBudget{FileBytes: 128, TotalBytes: 1 << 20})
	if _, ok := cap.Files[cudacapture.ExternalStacksName]; ok {
		t.Fatal("oversized sidecar was attached")
	}
	if !hasStackOmission(omitted) || !strings.Contains(omissionReason(omitted), "exceeds capture file budget") {
		t.Fatalf("oversized sidecar omission = %+v", omitted)
	}
	if got := cap.Profile.Sample[0].Label["cupti.external.stack_status"]; len(got) != 1 || got[0] != "missing" {
		t.Fatalf("oversized sidecar stack status = %v", got)
	}
}

func TestCaptureRecordBadStackSidecarDoesNotLoseCapture(t *testing.T) {
	for _, tc := range []struct {
		name string
		make func(string) error
	}{
		{"directory read error", func(path string) error { return os.Mkdir(path, 0700) }},
		{"undecodable complete record", func(path string) error {
			return os.WriteFile(path, []byte("{\"version\":1,\"kind\":\"unknown\"}\n"), 0600)
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			scratch := t.TempDir()
			if err := tc.make(filepath.Join(scratch, cudacapture.ExternalStacksName)); err != nil {
				t.Fatal(err)
			}
			path := filepath.Join(t.TempDir(), "capture.cudaprof")
			result := captureResult{Protocol: captureProtocolVersion, Session: "bad-stack", State: captureStateFinalized, Outcome: captureOutcomeObserved,
				TargetPID: 1234, CaptureSuccess: true, ScratchDir: scratch, Activities: oneExternalActivity()}
			omitted, err := writeCapture(path, result, captureArtifactBudget{FileBytes: 1 << 20, TotalBytes: 4 << 20})
			if err != nil {
				t.Fatalf("capture was lost: %v", err)
			}
			if !hasStackOmission(omitted) {
				t.Fatalf("sidecar was not listed as omitted: %+v", omitted)
			}
			f, err := os.Open(path)
			if err != nil {
				t.Fatal(err)
			}
			defer f.Close()
			cap, err := cudacapture.Read(f)
			if err != nil {
				t.Fatal(err)
			}
			if _, ok := cap.Files[cudacapture.ExternalStacksName]; ok {
				t.Fatal("invalid sidecar was attached as valid")
			}
		})
	}
}

func TestCaptureRecordStacksScopedByProcess(t *testing.T) {
	pidA, pidB, pidMissing := uint64(101), uint64(202), uint64(303)
	table := cudaprof.ExternalStackTable{Version: 1, Stacks: []cudaprof.ExternalStackRecord{
		{Process: pidA, ExternalKind: 3, ExternalID: 7, Frames: []string{"rootA", "leafA"}},
		{Process: pidB, ExternalKind: 3, ExternalID: 7, Frames: []string{"rootB", "leafB"}},
		{Process: pidA, ExternalKind: 3, ExternalID: 8, Frames: []string{"first"}},
		{Process: pidA, ExternalKind: 3, ExternalID: 8, Frames: []string{"second"}},
	}}
	data, err := json.Marshal(table)
	if err != nil {
		t.Fatal(err)
	}
	activities := append(stackActivitySet(pidA, struct {
		name string
		id   uint64
		corr uint32
	}{"procA", 7, 77}, struct {
		name string
		id   uint64
		corr uint32
	}{"ambiguous", 8, 88}), stackActivitySet(pidB, struct {
		name string
		id   uint64
		corr uint32
	}{"procB", 7, 77})...)
	activities = append(activities, stackActivitySet(pidMissing, struct {
		name string
		id   uint64
		corr uint32
	}{"missing", 9, 99})...)
	cap, _ := readStackCapture(t, data, activities, captureArtifactBudget{FileBytes: 1 << 20, TotalBytes: 4 << 20})
	got := samplesByName(cap)
	assertStack(t, got["procA"], "matched", "rootA;leafA")
	assertStack(t, got["procB"], "matched", "rootB;leafB")
	assertStack(t, got["ambiguous"], "ambiguous", "")
	assertStack(t, got["missing"], "missing", "")
}

func TestCaptureRecordGraphKernelsShareStack(t *testing.T) {
	pid := uint64(1234)
	table := cudaprof.ExternalStackTable{Version: 1, Stacks: []cudaprof.ExternalStackRecord{{Process: pid, ExternalKind: 3, ExternalID: 77, Frames: []string{"graph", "Do"}}}}
	data, _ := json.Marshal(table)
	activities := []cudacapture.ActivityRecord{
		{Process: pid, Subtype: "external_correlation", ExternalKind: 3, ExternalID: 77, Correlation: 123},
		{Process: pid, Subtype: "kernel", Name: "graphA", Correlation: 123, Start: 10, End: 15, Valid: true, GraphID: 1, GraphNodeID: 1},
		{Process: pid, Subtype: "kernel", Name: "graphB", Correlation: 123, Start: 20, End: 25, Valid: true, GraphID: 1, GraphNodeID: 2},
	}
	cap, _ := readStackCapture(t, data, activities, captureArtifactBudget{FileBytes: 1 << 20, TotalBytes: 4 << 20})
	got := samplesByName(cap)
	assertStack(t, got["graphA"], "matched", "graph;Do")
	assertStack(t, got["graphB"], "matched", "graph;Do")
}

func TestCaptureRecordStackUsesLabelledNamespace(t *testing.T) {
	pid := uint64(1234)
	table := cudaprof.ExternalStackTable{Version: 1, Stacks: []cudaprof.ExternalStackRecord{{Process: pid, ExternalKind: uint32(cupti.ExternalCorrelationKindCustom1), ExternalID: 7, Frames: []string{"custom1"}}}}
	data, _ := json.Marshal(table)
	activities := []cudacapture.ActivityRecord{
		{Process: pid, Subtype: "external_correlation", ExternalKind: uint32(cupti.ExternalCorrelationKindCustom0), ExternalID: 99, Correlation: 123},
		{Process: pid, Subtype: "external_correlation", ExternalKind: uint32(cupti.ExternalCorrelationKindCustom1), ExternalID: 7, Correlation: 123},
		{Process: pid, Subtype: "kernel", Name: "namespace", Correlation: 123, Start: 10, End: 15, Valid: true},
	}
	cap, _ := readStackCapture(t, data, activities, captureArtifactBudget{FileBytes: 1 << 20, TotalBytes: 4 << 20})
	sample := samplesByName(cap)["namespace"]
	assertStack(t, sample, "matched", "custom1")
	if got := sample.Label["cupti.external.id"]; len(got) != 1 || got[0] != "7" {
		t.Fatalf("external ID label = %v", got)
	}
}

func TestCaptureRecordReportsExternalStackLosses(t *testing.T) {
	table := cudaprof.ExternalStackTable{Version: 1, Losses: 5, LossesLowerBound: true,
		Stacks: []cudaprof.ExternalStackRecord{{Process: 1234, ExternalKind: 3, ExternalID: 77, Frames: []string{"main", "Do"}}}}
	data, err := json.Marshal(table)
	if err != nil {
		t.Fatal(err)
	}
	cap, _ := readStackCapture(t, data, oneExternalActivity(), captureArtifactBudget{FileBytes: 1 << 20, TotalBytes: 4 << 20})
	if !strings.Contains(strings.Join(cap.Profile.Comments, "\n"), "external stack losses lower bound: 5") {
		t.Fatalf("loss count not reported in capture profile: %v", cap.Profile.Comments)
	}
}

func TestPerProcessSidecarDamageIsIsolated(t *testing.T) {
	dir := t.TempDir()
	base := filepath.Join(dir, cudacapture.ExternalStacksName)
	good := "{\"version\":1,\"kind\":\"header\",\"losses\":\"00000000000000000000\"}\n" +
		"{\"version\":1,\"kind\":\"stack\",\"process\":101,\"external_kind\":3,\"external_id\":77,\"frames\":[\"main\",\"good\"]}\n"
	if err := os.WriteFile(base+".101", []byte(good), 0600); err != nil {
		t.Fatal(err)
	}
	badName := filepath.Base(base + ".202")
	if err := os.WriteFile(base+".202", []byte("{partial"), 0600); err != nil {
		t.Fatal(err)
	}
	got, err := readExternalStackSidecar(base, 101)
	if err != nil {
		t.Fatal(err)
	}
	if !got.damaged || !strings.Contains(strings.Join(got.damageFiles, ","), badName) {
		t.Fatalf("damage report = %+v", got)
	}
	var table cudaprof.ExternalStackTable
	if err := json.Unmarshal(got.data, &table); err != nil {
		t.Fatal(err)
	}
	if len(table.Stacks) != 1 || table.Stacks[0].Process != 101 || table.Stacks[0].Frames[1] != "good" {
		t.Fatalf("healthy sidecar lost: %+v", table.Stacks)
	}
	activities := stackActivitySet(101, struct {
		name string
		id   uint64
		corr uint32
	}{"healthy", 77, 123})
	path := filepath.Join(t.TempDir(), "capture.cudaprof")
	result := captureResult{Protocol: captureProtocolVersion, Session: "damaged-process-sidecar", State: captureStateFinalized, Outcome: captureOutcomeObserved, TargetPID: 101, CaptureSuccess: true, ScratchDir: dir, Activities: activities}
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
	assertStack(t, samplesByName(capture)["healthy"], "matched", "main;good")
	comments := strings.Join(capture.Profile.Comments, "\n")
	if !strings.Contains(comments, badName) {
		t.Fatalf("damaged filename missing from profile comment: %v", capture.Profile.Comments)
	}
}

func TestPerProcessSidecarInvalidCompleteLineIsReported(t *testing.T) {
	dir := t.TempDir()
	base := filepath.Join(dir, cudacapture.ExternalStacksName)
	if err := os.WriteFile(base+".101", []byte("not-json\n"), 0600); err != nil {
		t.Fatal(err)
	}
	got, err := readExternalStackSidecar(base, 101)
	if err != nil {
		t.Fatal(err)
	}
	if !got.damaged || len(got.damageFiles) != 1 || got.damageFiles[0] != filepath.Base(base+".101") {
		t.Fatalf("invalid complete line was not reported: %+v", got)
	}
}

func TestUnreadablePerProcessSidecarIsReported(t *testing.T) {
	dir := t.TempDir()
	base := filepath.Join(dir, cudacapture.ExternalStacksName)
	bad := base + ".303"
	if err := os.WriteFile(bad, []byte("sidecar"), 0600); err != nil {
		t.Fatal(err)
	}
	oldRead := readExternalStackFile
	readExternalStackFile = func(path string, limit int64) ([]byte, error) {
		if path == bad {
			return nil, os.ErrPermission
		}
		return oldRead(path, limit)
	}
	t.Cleanup(func() { readExternalStackFile = oldRead })
	got, err := readExternalStackSidecar(base, 303)
	if err != nil {
		t.Fatal(err)
	}
	if !got.damaged || len(got.damageFiles) != 1 || got.damageFiles[0] != filepath.Base(bad) {
		t.Fatalf("unreadable sidecar was not reported: %+v", got)
	}
}

func TestPerProcessSidecarLossesAreSummed(t *testing.T) {
	dir := t.TempDir()
	base := filepath.Join(dir, cudacapture.ExternalStacksName)
	for pid, losses := range map[int]int{101: 5, 202: 7} {
		line := fmt.Sprintf("{\"version\":1,\"kind\":\"header\",\"losses\":\"%020d\",\"losses_lower_bound\":true}\n", losses)
		if err := os.WriteFile(fmt.Sprintf("%s.%d", base, pid), []byte(line), 0600); err != nil {
			t.Fatal(err)
		}
	}
	got, err := readExternalStackSidecar(base, 101)
	if err != nil {
		t.Fatal(err)
	}
	var table cudaprof.ExternalStackTable
	if err := json.Unmarshal(got.data, &table); err != nil {
		t.Fatal(err)
	}
	if table.Losses != 12 || !table.LossesLowerBound {
		t.Fatalf("losses = %d lower=%v, want 12 lower bound", table.Losses, table.LossesLowerBound)
	}
}

func TestCaptureTargetStacksEndToEnd(t *testing.T) {
	if os.Getenv("CUDAPROF_TEST_END_TO_END_STACKS") == "target" {
		cudaprof.RecordExternalStackKind(cupti.ExternalCorrelationKindCustom0, 71, []string{"main", "targetStackA"})
		cudaprof.RecordExternalStackKind(cupti.ExternalCorrelationKindCustom0, 72, []string{"main", "targetStackB"})
		os.Exit(0)
	}
	result, err := runCapture(context.Background(), captureOptions{command: []string{os.Args[0], "-test.run=^TestCaptureTargetStacksEndToEnd$"}, Env: []string{"CUDAPROF_TEST_END_TO_END_STACKS=target"}, NoInject: true})
	if err != nil {
		t.Fatal(err)
	}
	pid := uint64(result.TargetPID)
	result.Activities = []cudacapture.ActivityRecord{
		{Process: pid, Subtype: "external_correlation", ExternalKind: 3, ExternalID: 71, Correlation: 171},
		{Process: pid, Subtype: "external_correlation", ExternalKind: 3, ExternalID: 72, Correlation: 172},
		{Process: pid, Subtype: "kernel", Name: "targetA", Correlation: 171, Start: 10, End: 15, Valid: true},
		{Process: pid, Subtype: "kernel", Name: "targetB", Correlation: 172, Start: 20, End: 25, Valid: true},
	}
	path := filepath.Join(t.TempDir(), "capture.cudaprof")
	if _, err := writeCapture(path, result, captureArtifactBudget{FileBytes: 1 << 20, TotalBytes: 4 << 20}); err != nil {
		t.Fatal(err)
	}
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	cap, err := cudacapture.Read(f)
	if err != nil {
		t.Fatal(err)
	}
	if len(cap.Files[cudacapture.ExternalStacksName]) == 0 {
		t.Fatal("controller did not attach the target-written sidecar")
	}
	got := samplesByName(cap)
	assertStack(t, got["targetA"], "matched", "main;targetStackA")
	assertStack(t, got["targetB"], "matched", "main;targetStackB")
}

func samplesByName(cap *cudacapture.Capture) map[string]*profile.Sample {
	out := make(map[string]*profile.Sample)
	for _, sample := range cap.Profile.Sample {
		if names := sample.Label["name"]; len(names) == 1 {
			out[names[0]] = sample
		}
	}
	return out
}

func assertStack(t *testing.T, sample *profile.Sample, status, stack string) {
	t.Helper()
	if sample == nil {
		t.Fatalf("missing profile sample for stack status %s", status)
	}
	if got := sample.Label["cupti.external.stack_status"]; len(got) != 1 || got[0] != status {
		t.Fatalf("stack status = %v, want %q (all labels %v)", got, status, sample.Label)
	}
	if stack != "" {
		if got := sample.Label["cupti.external.stack"]; len(got) != 1 || got[0] != stack {
			t.Fatalf("stack = %v, want %q", got, stack)
		}
	}
}

func hasStackOmission(items []captureOmittedFile) bool {
	for _, item := range items {
		if item.Name == cudacapture.ExternalStacksName {
			return true
		}
	}
	return false
}

func omissionReason(items []captureOmittedFile) string {
	for _, item := range items {
		if item.Name == cudacapture.ExternalStacksName {
			return item.Reason
		}
	}
	return ""
}

package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaprof"
)

// This fixture writes synthetic journal-derived values. It exercises the real
// writeCapture consumer; it does not authenticate a native API producer.
func trackerStackCapture(t *testing.T, table cudaprof.ExternalStackTable, activities []cudacapture.ActivityRecord, apis []cudacapture.APIRecord) *cudacapture.Capture {
	t.Helper()
	scratch := t.TempDir()
	if table.Version != 0 {
		data, err := json.Marshal(table)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(scratch, cudacapture.ExternalStacksName), data, 0600); err != nil {
			t.Fatal(err)
		}
	}
	result := captureResult{Protocol: captureProtocolVersion, Session: "tracker-fixture", State: captureStateFinalized, Outcome: captureOutcomeObserved, TargetPID: 101, CaptureSuccess: true, ScratchDir: scratch, Activities: activities, APIRecords: apis}
	path := filepath.Join(t.TempDir(), "capture.cudaprof")
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
	return capture
}

func trackerStackFixture() (cudaprof.ExternalStackTable, []cudacapture.ActivityRecord, []cudacapture.APIRecord) {
	return cudaprof.ExternalStackTable{Version: 1, Stacks: []cudaprof.ExternalStackRecord{{Process: 101, ExternalKind: 3, ExternalID: 7, Frames: []string{"main", "Do"}}}},
		[]cudacapture.ActivityRecord{{Process: 101, Subtype: "external_correlation", Correlation: 77, ExternalKind: 3, ExternalID: 7}, {Process: 101, Subtype: "kernel", Name: "kernel", Correlation: 77, Start: 100, End: 150, Valid: true}},
		[]cudacapture.APIRecord{{Domain: "runtime", ProcessID: 101, ThreadID: 9, Correlation: 77, Start: 1, End: 2}}
}

func TestCaptureStackTrackerRealAPIConsumer(t *testing.T) {
	for _, test := range []struct {
		name                   string
		process                uint32
		wantBasis, wantRuntime string
	}{
		{"matching API", 101, "runtime_api", "matched"},
		{"foreign API", 202, "external_id_sidecar", "missing"},
		{"unknown API", 0, "external_id_sidecar", "missing"},
	} {
		t.Run(test.name, func(t *testing.T) {
			table, activities, apis := trackerStackFixture()
			apis[0].ProcessID = test.process
			cap := trackerStackCapture(t, table, activities, apis)
			sample := samplesByName(cap)["kernel"]
			assertStack(t, sample, "matched", "main;Do")
			if !reflect.DeepEqual(sample.Label["cupti.external.stack_basis"], []string{test.wantBasis}) || !reflect.DeepEqual(sample.Label["cupti.external.runtime_status"], []string{test.wantRuntime}) {
				t.Fatalf("labels=%v", sample.Label)
			}
		})
	}
}

func TestCaptureStackTrackerNoAPIFabrication(t *testing.T) {
	table, activities, _ := trackerStackFixture()
	for _, start := range []uint64{10, 10000} {
		activities[1].Start, activities[1].End = start, start+5
		cap := trackerStackCapture(t, table, activities, nil)
		sample := samplesByName(cap)["kernel"]
		assertStack(t, sample, "matched", "main;Do")
		if got := sample.Label["cupti.external.runtime_status"]; !reflect.DeepEqual(got, []string{"missing"}) {
			t.Fatalf("device time manufactured runtime evidence: %v", sample.Label)
		}
		if got := sample.Label["cupti.external.stack_basis"]; !reflect.DeepEqual(got, []string{"external_id_sidecar"}) {
			t.Fatalf("basis=%v", got)
		}
	}
}

func TestCaptureStackTrackerSelectedKind(t *testing.T) {
	table, activities, apis := trackerStackFixture()
	table.Stacks[0].ExternalKind = 4
	activities[0].ExternalKind = 4
	activities = append(activities, cudacapture.ActivityRecord{Process: 101, Subtype: "external_correlation", Correlation: 77, ExternalKind: 3, ExternalID: 99})
	cap := trackerStackCapture(t, table, activities, apis)
	sample := samplesByName(cap)["kernel"]
	assertStack(t, sample, "matched", "main;Do")
	for key, want := range map[string]string{"cupti.external.kind": "4", "cupti.external.id": "7", "cupti.external.stack_basis": "runtime_api", "cupti.external.runtime_status": "matched"} {
		if !reflect.DeepEqual(sample.Label[key], []string{want}) {
			t.Fatalf("%s=%v want %s", key, sample.Label[key], want)
		}
	}
}

func TestCaptureStackTrackerProcessReuse(t *testing.T) {
	table, activities, apis := trackerStackFixture()
	activities[1].Name = "proc101"
	table.Stacks = append(table.Stacks, cudaprof.ExternalStackRecord{Process: 202, ExternalKind: 3, ExternalID: 7, Frames: []string{"other", "Do"}})
	activities = append(activities, cudacapture.ActivityRecord{Process: 202, Subtype: "external_correlation", Correlation: 77, ExternalKind: 3, ExternalID: 7}, cudacapture.ActivityRecord{Process: 202, Subtype: "kernel", Name: "proc202", Correlation: 77, Start: 200, End: 250, Valid: true})
	apis = append(apis, cudacapture.APIRecord{Domain: "driver", ProcessID: 202, ThreadID: 10, Correlation: 77, Start: 3, End: 4})
	cap := trackerStackCapture(t, table, activities, apis)
	for name, want := range map[string]string{"proc101": "main;Do", "proc202": "other;Do"} {
		sample := samplesByName(cap)[name]
		assertStack(t, sample, "matched", want)
		if !reflect.DeepEqual(sample.Label["cupti.external.stack_basis"], []string{"runtime_api"}) {
			t.Fatalf("process buckets collided: %v", sample.Label)
		}
	}
}

func TestCaptureStackTrackerAmbiguity(t *testing.T) {
	for _, name := range []string{"two APIs", "two threads", "graph siblings", "conflicting stacks"} {
		t.Run(name, func(t *testing.T) {
			table, activities, apis := trackerStackFixture()
			wantStack, wantRuntime := "matched", "ambiguous"
			switch name {
			case "two APIs":
				apis = append(apis, apis[0])
			case "two threads":
				a := apis[0]
				a.ThreadID++
				apis = append(apis, a)
			case "graph siblings":
				a := activities[1]
				a.Name = "sibling"
				a.Start = 200
				a.End = 250
				a.GraphID = 1
				a.GraphNodeID = 2
				activities = append(activities, a)
			case "conflicting stacks":
				r := table.Stacks[0]
				r.Frames = []string{"conflict"}
				table.Stacks = append(table.Stacks, r)
				wantStack = "ambiguous"
			}
			cap := trackerStackCapture(t, table, activities, apis)
			for _, sample := range samplesByName(cap) {
				frames := "main;Do"
				if wantStack != "matched" {
					frames = ""
				}
				assertStack(t, sample, wantStack, frames)
				if !reflect.DeepEqual(sample.Label["cupti.external.runtime_status"], []string{wantRuntime}) || !reflect.DeepEqual(sample.Label["cupti.external.stack_basis"], []string{"external_id_sidecar"}) {
					t.Fatalf("ambiguous runtime claimed complete: %v", sample.Label)
				}
			}
		})
	}
}

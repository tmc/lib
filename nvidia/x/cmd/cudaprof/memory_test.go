package main

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaprof/timeline"
)

func TestMemoryEvidence(t *testing.T) {
	tests := []struct {
		name           string
		records        timeline.Records
		wantTransfer   string
		wantAllocation string
	}{
		{"missing", timeline.Records{}, "unavailable", "unavailable"},
		{"observed", timeline.Records{
			Activities: []timeline.Activity{{Journal: "p1", Subtype: "memcpy", Start: 10, End: 20, Bytes: 4096, CopyKind: 1}},
			Calls:      []timeline.Call{{Domain: "driver", CallbackID: 243}, {Domain: "runtime", CallbackID: 20}},
		}, "observed", "observed"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := summarizeMemory(tt.records, nil)
			if r.TransferEvidence.State != tt.wantTransfer || r.AllocationEvidence.State != tt.wantAllocation {
				t.Fatalf("evidence = %+v", r)
			}
			if r.Residency.State != "unsupported" || r.Migration.State != "unsupported" || r.PageFaults.State != "unsupported" || r.DeviceSupport.State != "unprobed" {
				t.Fatalf("unsupported evidence invented: %+v", r)
			}
			if tt.name == "observed" {
				if r.AllocationRecords != 2 || r.TransferRecords != 1 || r.Transfers[0].Bytes != 4096 || r.Transfers[0].Journal != "p1" {
					t.Fatalf("retained evidence = %+v", r)
				}
			}
			if _, err := json.Marshal(r); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestMemoryCapabilityPreserved(t *testing.T) {
	capability := json.RawMessage(`{"activity":true,"api_records":false,"driver_api":false,"runtime_api":false}`)
	r := summarizeMemory(timeline.Records{}, capability)
	if string(r.Capability) != string(capability) || r.AllocationEvidence.State != "unavailable" {
		t.Fatalf("capability = %+v", r)
	}
	if !strings.Contains(r.AllocationEvidence.Reason, "disabled") {
		t.Fatal("missing disabled collection explanation")
	}
}

func TestMemoryRejectsIncompleteCapture(t *testing.T) {
	path := t.TempDir() + "/incomplete.pprof"
	writeTestCaptureFile(t, path, testCapture(nil, map[string]any{"run_completeness": "partial"}))
	if err := runMemoryCommand([]string{path}); err == nil {
		t.Fatal("accepted incomplete memory evidence")
	}
}

func TestMemoryFlags(t *testing.T) {
	for _, args := range [][]string{{"-max-records=-1", "missing.pprof"}, {"-journal-bytes=-1", "missing.pprof"}} {
		if err := runMemoryCommand(args); err == nil || !strings.Contains(err.Error(), "negative") {
			t.Fatalf("memory %v = %v, want negative-limit refusal", args, err)
		}
	}
	if err := runMemoryCommand([]string{"-help"}); err != nil {
		t.Fatalf("memory help: %v", err)
	}
}

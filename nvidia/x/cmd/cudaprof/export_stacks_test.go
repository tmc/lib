package main

import (
	"bytes"
	"testing"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/x/cudacapture"
	languagestack "github.com/tmc/lib/nvidia/x/cudaprof/stack"
	"github.com/tmc/lib/nvidia/x/cuptiactivity"
)

func TestAttachLanguageStacks(t *testing.T) {
	for _, tt := range []struct {
		name       string
		session    string
		generation uint64
		status     string
		want       bool
	}{
		{"matched", "session", 7, "attached", true},
		{"other session", "other", 7, "attached", false},
		{"reused pid", "session", 8, "attached", false},
		{"failed push", "session", 7, "push_failed", false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			var data bytes.Buffer
			r := languagestack.Record{Version: 1, Process: "producer", PID: 12, Session: tt.session, Generation: tt.generation, Language: "go", Attribution: "submission", ExternalKind: uint32(cupti.ExternalCorrelationKindCustom2), ExternalID: 42, Status: tt.status, Frames: []languagestack.Frame{{Function: "caller", File: "main.go", Line: 10}}}
			if err := languagestack.NewJournal(&data, languagestack.Limits{}).Record(r); err != nil {
				t.Fatal(err)
			}
			cap := &cudacapture.Capture{Files: map[string][]byte{"stacks-go.jsonl": data.Bytes()}}
			records := []traceRecord{{session: "session", process: 12, generation: 7, rec: cuptiactivity.Record{Kind: cupti.ActivityKindConcurrentKernel, CorrelationID: 9}}}
			external := []cuptiactivity.Record{{Kind: cupti.ActivityKindExternalCorrelation, ExternalKind: cupti.ExternalCorrelationKindCustom2, ExternalID: 42, CorrelationID: 9}}
			if err := retainStackCorrelations(records, external); err != nil {
				t.Fatal(err)
			}
			if err := attachLanguageStacks(cap, records); err != nil {
				t.Fatal(err)
			}
			if got := records[0].stack != nil; got != tt.want {
				t.Fatalf("attached = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestStackNamespaceAndAmbiguity(t *testing.T) {
	records := []traceRecord{{session: "session", process: 12, generation: 7, rec: cuptiactivity.Record{Kind: cupti.ActivityKindDriver, CorrelationID: 9}}}
	external := []cuptiactivity.Record{
		{Kind: cupti.ActivityKindExternalCorrelation, ExternalKind: cupti.ExternalCorrelationKindCustom0, ExternalID: 99, CorrelationID: 9},
		{Kind: cupti.ActivityKindExternalCorrelation, ExternalKind: cupti.ExternalCorrelationKindCustom2, ExternalID: 42, CorrelationID: 9},
		{Kind: cupti.ActivityKindExternalCorrelation, ExternalKind: cupti.ExternalCorrelationKindCustom2, ExternalID: 43, CorrelationID: 9},
	}
	if err := retainStackCorrelations(records, external); err != nil {
		t.Fatal(err)
	}
	if got := records[0].external[uint32(cupti.ExternalCorrelationKindCustom0)]; got.Status != "matched" || got.ExternalID != 99 {
		t.Fatalf("framework namespace = %+v", got)
	}
	if got := records[0].external[uint32(cupti.ExternalCorrelationKindCustom2)]; got.Status != "ambiguous" {
		t.Fatalf("stack namespace = %+v", got)
	}
}

package cudacapture_test

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaprof"
)

func TestAuditTrackerReadsActualLossCheckpoint(t *testing.T) {
	cudaprof.ResetExternalStacks()
	t.Cleanup(cudaprof.ResetExternalStacks)
	base := filepath.Join(t.TempDir(), "external-stacks.json")
	t.Setenv(cudaprof.ExternalStacksEnv, base)
	for id := uint64(1); id <= 4096; id++ {
		if !cudaprof.RecordExternalStackKind(3, id, []string{"retained"}) {
			t.Fatalf("producer refused retained stack %d", id)
		}
	}
	if cudaprof.RecordExternalStackKind(3, 4097, []string{"lost"}) || cudaprof.ExternalStackLosses() != 1 {
		t.Fatal("overflow control did not emit one loss")
	}
	file, err := os.Open(fmt.Sprintf("%s.%d", base, os.Getpid()))
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	ct := cudacapture.NewCorrelationTracker()
	if err := ct.ReadExternalStackTable(file); err != nil {
		t.Fatalf("reader rejected actual producer loss checkpoint: %v", err)
	}
	if ct.ExternalStackLosses() != 1 {
		t.Fatalf("losses = %d, want 1", ct.ExternalStackLosses())
	}
	if !ct.ExternalStackLossesLowerBound() {
		t.Fatal("checkpoint qualification lost")
	}
	for id := uint32(1); id <= 4096; id++ {
		if err := ct.RecordRuntime(cudacapture.RuntimeLaunch{CorrelationID: uint64(id)}); err != nil {
			t.Fatal(err)
		}
		for _, row := range []cudacapture.ActivityRecord{
			{Process: uint64(os.Getpid()), Correlation: id, Subtype: "external_correlation", ExternalKind: 3, ExternalID: uint64(id)},
			{Process: uint64(os.Getpid()), Correlation: id, Subtype: "kernel"},
		} {
			if err := ct.RecordActivity(row); err != nil {
				t.Fatal(err)
			}
		}
	}
	matches, ambiguous, _, _ := ct.Join()
	if len(matches) != 4096 || len(ambiguous) != 0 {
		t.Fatalf("matches=%d ambiguous=%d", len(matches), len(ambiguous))
	}
	for _, match := range matches {
		if !match.HasExternalID || match.CallstackMissing || len(match.Callstack) != 1 || match.Callstack[0] != "retained" {
			t.Fatalf("lost retained record: %+v", match)
		}
	}
}

func TestAuditTrackerProcessIsolation(t *testing.T) {
	for _, process := range []uint64{222, 111} {
		t.Run(map[bool]string{true: "matching-process", false: "foreign-process"}[process == 222], func(t *testing.T) {
			ct := cudacapture.NewCorrelationTracker()
			data, err := json.Marshal(cudaprof.ExternalStackTable{Version: 1, Stacks: []cudaprof.ExternalStackRecord{{Process: process, ExternalKind: 3, ExternalID: 9, Frames: []string{"process-owned-stack"}}}})
			if err != nil {
				t.Fatal(err)
			}
			if err := ct.ReadExternalStackTable(bytes.NewReader(data)); err != nil {
				if process != 222 {
					return
				}
				t.Fatal(err)
			}
			if err := ct.RecordRuntime(cudacapture.RuntimeLaunch{CorrelationID: 7, KernelName: "kernel"}); err != nil {
				t.Fatal(err)
			}
			for _, record := range []cudacapture.ActivityRecord{
				{Process: 222, Correlation: 7, Subtype: "external_correlation", ExternalKind: 3, ExternalID: 9},
				{Process: 222, Correlation: 7, Subtype: "kernel", Name: "kernel", Start: 1, End: 2, Valid: true},
			} {
				if err := ct.RecordActivity(record); err != nil {
					t.Fatal(err)
				}
			}
			matched, _, _, _ := ct.Join()
			if len(matched) != 1 {
				t.Fatalf("matched=%+v", matched)
			}
			if process == 222 && len(matched[0].Callstack) == 0 {
				t.Fatal("positive control lost matching stack")
			}
			if process != 222 && len(matched[0].Callstack) != 0 {
				t.Fatalf("foreign process %d stack joined to process 222: %+v", process, matched[0])
			}
		})
	}
}

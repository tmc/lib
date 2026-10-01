package main

import (
	"bytes"
	"io"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/x/cudaprof/timeline"
	"github.com/tmc/lib/nvidia/x/cuptiactivity"
)

func TestComparisonSharesAuxiliaryTrackBudget(t *testing.T) {
	records := timeline.Records{GraphLifecycle: make([]timeline.GraphEvent, 60000)}
	raw := []traceRecord{{rec: cuptiactivity.Record{Kind: cupti.ActivityKindConcurrentKernel, Start: 1, End: 2, Complete: true}}}
	bound := projectionTrackBound(raw, nil, &records, 0, false, true, false, true)
	if err := requireProjectionTracks(bound, defaultExportTracks); err != nil {
		t.Fatalf("individual arm refused: %v", err)
	}
	var receipt alignmentReceipt
	for i := range receipt.Arms {
		receipt.Arms[i] = alignmentArm{records: raw, provenance: &records, showLifecycle: true}
	}
	var output bytes.Buffer
	if err := writeComparisonBundle(&output, receipt); err == nil || !strings.Contains(err.Error(), "combined trace track budget") {
		t.Fatalf("shared auxiliary budget not enforced: %v", err)
	}
	if output.Len() != 0 {
		t.Fatal("archive partly written before track budget refusal")
	}
}

func TestProjectionBudgetsAllowExplicitOverrides(t *testing.T) {
	if err := requireProjectionTracks(120000, 120000); err != nil {
		t.Fatal(err)
	}
	if err := requireProjectionTracks(120000, 100000); err == nil {
		t.Fatal("lowered track budget ignored")
	}
	if err := requireSummaryReferences(2500000, 2500000); err != nil {
		t.Fatal(err)
	}
	if err := requireSummaryReferences(2500000, 2000000); err == nil {
		t.Fatal("lowered reference budget ignored")
	}
	opts, err := parseExportArgs([]string{"-max-tracks", "120000", "-max-summary-references", "2500000", "capture.pprof"})
	if err != nil || opts.maxTracks != 120000 || opts.maxSummaryReferences != 2500000 {
		t.Fatalf("export limits: %+v, %v", opts, err)
	}
	compare, err := parseExportCompareArgs([]string{"-max-tracks=120000", "-max-summary-references=2500000", "-iteration-marker=k", "a", "b"}, io.Discard)
	if err != nil || compare.maxTracks != 120000 || compare.maxSummaryReferences != 2500000 {
		t.Fatalf("compare limits: %+v, %v", compare, err)
	}
	for _, flag := range []string{"-max-tracks=0", "-max-summary-references=-1"} {
		if _, err := parseExportArgs([]string{flag, "capture.pprof"}); err == nil {
			t.Fatalf("accepted %s", flag)
		}
	}
}

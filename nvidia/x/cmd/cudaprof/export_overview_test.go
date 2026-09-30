package main

import (
	"bytes"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaprof/overview"
	"github.com/tmc/lib/nvidia/x/cudaprof/timeline"
)

func TestExportOverviewUnions(t *testing.T) {
	r := overview.Analyze(timeline.Records{Activities: []timeline.Activity{{Journal: "j", Name: "raw", Start: 10, End: 30, Subtype: "kernel"}, {Journal: "j", Name: "other", Start: 20, End: 40, Subtype: "kernel"}}}, "driver", nil)
	var b bytes.Buffer
	if err := writeOverviewTrace(&b, r, 10, 1<<48); err != nil {
		t.Fatal(err)
	}
	for _, text := range []string{"GPU 0 activity union", "summed_kernel_ns", "raw_references", "unattributed"} {
		if !bytes.Contains(b.Bytes(), []byte(text)) {
			t.Fatalf("trace lacks %q", text)
		}
	}
	if bytes.Contains(b.Bytes(), []byte("observed delivered")) {
		t.Fatal("invented delivery")
	}
}

func TestExportSummaryFlags(t *testing.T) {
	opts, err := parseExportArgs([]string{"-summary", "capture.pprof"})
	if err != nil || !opts.summary {
		t.Fatalf("parse flags: %+v %v", opts, err)
	}
	if !strings.HasSuffix(opts.outPath, ".pftrace") {
		t.Fatal(opts.outPath)
	}
}

func TestExportObservedAndInferredEvents(t *testing.T) {
	delivered := int64(40)
	r := overview.Report{Hierarchy: &overview.Node{Kind: "capture", Label: "capture"}, Tokens: []overview.Token{{Index: 2, Journal: "j", ID: "token2", Provenance: "observed", Delivery: &delivered, References: []overview.Reference{{Journal: "j", Kind: "marker", Index: 3}}, Materialization: []overview.Reference{{Journal: "j", Kind: "range", Index: 1}}, MaterializationRanges: []overview.Interval{{Journal: "j", Start: 30, End: 35}}}}, Inferred: []overview.Token{{Index: 0, Journal: "j", InferredBoundary: &overview.Interval{Journal: "j", Start: 10, End: 20}}}}
	var b bytes.Buffer
	if err := writeOverviewTrace(&b, r, 10, 1<<48); err != nil {
		t.Fatal(err)
	}
	for _, v := range []string{"observed delivered", "host materialization scope", "inferred kernel interval", "not delivered/materialized", "marker", "token2"} {
		if !bytes.Contains(b.Bytes(), []byte(v)) {
			t.Fatalf("missing %q", v)
		}
	}
}

package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaprof/iteration"
	"github.com/tmc/lib/nvidia/x/cudaprof/shape"
)

func testComparisonReceipts() (comparisonShapeReceipt, alignmentReceipt) {
	var s comparisonShapeReceipt
	s.Version = 1
	s.Shape = shape.Report{Version: 1, Failures: []string{"application context unavailable"}, BaseUnmatched: []shape.Launch{{Journal: "journal", Activity: 4, RawSymbol: "old_kernel", Missing: []string{"dtype", "token"}}}}
	a := alignmentReceipt{Version: 1, TimeScale: 1}
	for i := range 2 {
		s.Arms[i].SHA256 = []string{"baseSHA", "targetSHA"}[i]
		s.Arms[i].Completeness = "passed"
		s.Arms[i].Stationarity = iteration.Gate{Failures: []string{"stationarity failed"}}
		a.Arms[i] = alignmentArm{SHA256: s.Arms[i].SHA256, Label: []string{"base", "target"}[i], Trace: []string{"base.pftrace", "target.pftrace"}[i], Completeness: "passed", MarkerVerdict: "unchecked", Offset: 100}
	}
	return s, a
}

func TestComparisonSummaryUnqualified(t *testing.T) {
	s, a := testComparisonReceipts()
	r, err := makeComparisonSummary(s, a, nil)
	if err != nil {
		t.Fatal(err)
	}
	var out bytes.Buffer
	if err := comparisonSummaryHTML.Execute(&out, r); err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"No qualified numerical", "1 unmatched", "stationarity failed", "unchecked", "application context unavailable", "No independent campaign"} {
		if !strings.Contains(out.String(), want) {
			t.Errorf("HTML missing %q", want)
		}
	}
	if len(r.Operations) != 0 || r.ShapeQualified {
		t.Fatalf("unqualified numerical comparison: %+v", r)
	}
}

func TestComparisonSummaryQualified(t *testing.T) {
	s, a := testComparisonReceipts()
	id := shape.Identity{Phase: "decode", Operation: "gemv", DType: "bf16", Grid: [3]int{1, 1, 1}, Block: [3]int{32, 1, 1}}
	l := shape.Launch{Identity: &id, Journal: "journal", Activity: 7, APIStatus: "matched", MetadataStatus: "matched", Start: 50, Duration: 10, RawSymbol: "<kernel>"}
	s.Shape = shape.Report{Version: 1, Pass: true, Groups: []shape.Group{{Identity: id, Base: []shape.Launch{l}, Target: []shape.Launch{l}, BaseDurations: shape.Distribution{Count: 1, Min: 10, Median: 10, Max: 10, Sum: 10}, TargetDurations: shape.Distribution{Count: 1, Min: 15, Median: 15, Max: 15, Sum: 15}}}}
	c := &campaignReport{Version: 1, Interval: &campaignInterval{Blocks: 3, Median: 2, Lower: 1, Upper: 3, Assumption: "exchangeability unverified"}, Blocks: []campaignBlock{{Number: 1, Excluded: []string{"crashed"}}}}
	r, err := makeComparisonSummary(s, a, c)
	if err != nil {
		t.Fatal(err)
	}
	if r.Operations[0].SumDelta != 5 || r.Operations[0].References[0][0].AlignedStart != 150 || r.Campaign != c {
		t.Fatalf("summary = %+v", r)
	}
	var out bytes.Buffer
	if err := comparisonSummaryHTML.Execute(&out, r); err != nil {
		t.Fatal(err)
	}
	if strings.Contains(out.String(), "<kernel>") || !strings.Contains(out.String(), "&lt;kernel&gt;") || !strings.Contains(out.String(), "3 qualified independent blocks") || !strings.Contains(out.String(), "crashed") {
		t.Fatalf("HTML = %s", &out)
	}
	s.Shape.Groups[0].Base[0].Missing = []string{"token"}
	if _, err := makeComparisonSummary(s, a, nil); err == nil {
		t.Fatal("accepted unknown matching identity")
	}
}

func TestComparisonSummaryRefusals(t *testing.T) {
	for _, bad := range []string{"identity", "timescale", "empty", "different counts"} {
		t.Run(bad, func(t *testing.T) {
			s, a := testComparisonReceipts()
			switch bad {
			case "identity":
				a.Arms[0].SHA256 = "other"
			case "timescale":
				a.TimeScale = 2
			case "empty":
				s.Shape.Pass = true
			case "different counts":
				s.Shape.Pass = true
				s.Shape.TargetUnmatched = []shape.Launch{{RawSymbol: "extra"}}
			}
			if _, err := makeComparisonSummary(s, a, nil); err == nil {
				t.Fatal("accepted invalid receipt")
			}
		})
	}
}

func TestComparisonSummaryFiles(t *testing.T) {
	s, a := testComparisonReceipts()
	dir := t.TempDir()
	paths := []string{filepath.Join(dir, "shape.json"), filepath.Join(dir, "alignment.json")}
	for i, v := range []any{s, a} {
		data, err := json.Marshal(v)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(paths[i], data, 0600); err != nil {
			t.Fatal(err)
		}
	}
	out := filepath.Join(dir, "summary")
	if err := runComparisonSummaryCommand([]string{"-shape", paths[0], "-alignment", paths[1], "-o", out}, &bytes.Buffer{}); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"summary.json", "summary.html"} {
		if _, err := os.Stat(filepath.Join(out, name)); err != nil {
			t.Fatal(err)
		}
	}
}

package advise

import (
	"regexp"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaprof/iteration"
	"github.com/tmc/lib/nvidia/x/cudaprof/timeline"
)

func iterationReport(t *testing.T, n int, uneven bool) *iteration.Report {
	t.Helper()
	var rec timeline.Records
	for i := range int64(n) {
		at := 100_000 * i
		rec.Activities = append(rec.Activities,
			timeline.Activity{Subtype: "kernel", Name: "Embed", Start: at, End: at + 10_000},
			timeline.Activity{Subtype: "kernel", Name: "gemm", Start: at + 10_000, End: at + 50_000},
		)
		if uneven && i == 2 {
			rec.Activities = append(rec.Activities, timeline.Activity{Subtype: "kernel", Name: "extra", Start: at + 60_000, End: at + 61_000})
		}
	}
	r, err := iteration.Analyze(rec, iteration.Options{Marker: regexp.MustCompile(`^Embed$`), Skip: 1})
	if err != nil {
		t.Fatal(err)
	}
	return r
}

func TestAddIterations(t *testing.T) {
	tests := []struct {
		name   string
		uneven bool
		want   []string // severity: finding prefix, in order
	}{
		{"pass", false, []string{"Read first: existing", "Cost: every iteration: the device is idle 50.0%", "Cost: existing"}},
		{"gate fails", true, []string{"Read first: existing", "Read first: iteration gate failed: steady iterations launched", "Cost: every iteration: the device is idle", "Cost: existing"}},
		{"capture idle", false, []string{"Read first: existing", "Cost: every iteration: the device is idle 50.0%", "Cost: existing", "Context: idle. This counts the whole capture, warm-up included; the per-iteration findings skip the first 1 iterations."}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := &Report{Items: []Item{
				{Severity: SeverityReadFirst, Finding: "existing"},
				{Severity: SeverityCost, Finding: "existing"},
			}}
			if tt.name == "capture idle" {
				r.Items = append([]Item{{Severity: SeverityReadFirst, Finding: "idle.", rule: "capture-idle", severityByKind: true}}, r.Items...)
			}
			r.AddIterations(iterationReport(t, 5, tt.uneven))
			if len(r.Items) != len(tt.want) {
				t.Fatalf("got %d items, want %d: %+v", len(r.Items), len(tt.want), r.Items)
			}
			for i, w := range tt.want {
				got := string(r.Items[i].Severity) + ": " + r.Items[i].Finding
				if !strings.HasPrefix(got, w) {
					t.Errorf("item %d = %q, want prefix %q", i, got, w)
				}
			}
			md := r.MarkdownWithLimit(0)
			if !tt.uneven && !strings.Contains(md, "(0.050 ms per iteration, 50.0% of the iteration)") {
				t.Errorf("markdown lacks the per-iteration magnitude:\n%s", md)
			}
		})
	}
}

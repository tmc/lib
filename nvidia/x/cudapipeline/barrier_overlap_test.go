package cudapipeline_test

import (
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudapipeline"
)

func TestPipelineBarrierRanges(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	for _, tt := range []struct {
		name          string
		first, second int
		want          string
	}{
		{"adjacent", 256, 264, ""},
		{"reverse adjacent", 264, 256, ""},
		{"same offset", 256, 256, "overlaps with stage"},
		{"partial overlap", 256, 260, "overlaps with stage"},
		{"reverse partial overlap", 260, 256, "overlaps with stage"},
		{"negative first", -8, 264, "invalid mbarrier range"},
		{"negative second", 256, -8, "invalid mbarrier range"},
		{"overflow first", maxInt - 7, 264, "invalid mbarrier range"},
		{"overflow second", 256, maxInt - 7, "invalid mbarrier range"},
		{"maximum offset", 256, maxInt, "invalid mbarrier range"},
		{"upper adjacent", maxInt - 23, maxInt - 15, ""},
	} {
		t.Run(tt.name, func(t *testing.T) {
			pipeline, err := cudapipeline.NewPipeline(cudapipeline.PipelineConfig{
				TargetArch: cudapipeline.ArchSM90,
				NumStages:  2,
				Box:        cudapipeline.TMABox{DimX: 16, DimY: 1, ElementSize: 1},
			})
			if err != nil {
				t.Fatal(err)
			}
			pipeline.Stages[0].Barrier.Offset = tt.first
			pipeline.Stages[1].Barrier.Offset = tt.second
			report := pipeline.Validate()
			if tt.want == "" {
				if !report.Valid() {
					t.Fatalf("valid disjoint barriers rejected: %v", report.Errors())
				}
				return
			}
			for _, issue := range report.Errors() {
				if issue.Category == cudapipeline.CatBufferCapacity && strings.Contains(issue.Message, tt.want) {
					return
				}
			}
			t.Fatalf("missing barrier range rejection %q: %v", tt.want, report.Errors())
		})
	}
}

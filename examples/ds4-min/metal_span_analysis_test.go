//go:build darwin && arm64

package main

import "testing"

func TestAnalyzeSpans(t *testing.T) {
	span := func(start, end float64) metalRegion {
		return metalRegion{Start: start, End: end}
	}
	tests := []struct {
		name  string
		spans []metalRegion
		want  spanStats
	}{{
		name:  "empty",
		spans: nil,
		want:  spanStats{},
	}, {
		name:  "single",
		spans: []metalRegion{span(1, 3)},
		want:  spanStats{Count: 1, Sum: 2, Union: 2, Extent: 2},
	}, {
		name:  "disjoint leaves a gap",
		spans: []metalRegion{span(0, 1), span(2, 3)},
		want:  spanStats{Count: 2, Sum: 2, Union: 2, Idle: 1, Extent: 3},
	}, {
		// The case a sum gets wrong: two buffers in flight together.
		name:  "overlapping counts the shared interval once",
		spans: []metalRegion{span(0, 2), span(1, 3)},
		want:  spanStats{Count: 2, Sum: 4, Union: 3, Overlap: 1, Extent: 3},
	}, {
		name:  "nested span adds no union",
		spans: []metalRegion{span(0, 10), span(2, 4)},
		want:  spanStats{Count: 2, Sum: 12, Union: 10, Overlap: 2, Extent: 10},
	}, {
		name:  "unsorted input is handled",
		spans: []metalRegion{span(5, 6), span(0, 1)},
		want:  spanStats{Count: 2, Sum: 2, Union: 2, Idle: 4, Extent: 6},
	}, {
		name:  "touching spans leave no gap",
		spans: []metalRegion{span(0, 1), span(1, 2)},
		want:  spanStats{Count: 2, Sum: 2, Union: 2, Extent: 2},
	}}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := analyzeSpans(tt.spans)
			if got != tt.want {
				t.Errorf("analyzeSpans() = %+v, want %+v", got, tt.want)
			}
		})
	}
}

// A sum exceeds the union exactly when buffers are in flight together, which is
// what makes a sum-derived occupancy an upper bound rather than an estimate.
func TestAnalyzeSpansSumIsUpperBound(t *testing.T) {
	spans := []metalRegion{{Start: 0, End: 2}, {Start: 1, End: 3}}
	st := analyzeSpans(spans)
	if st.Sum <= st.Union {
		t.Fatalf("sum %v should exceed union %v when spans overlap", st.Sum, st.Union)
	}
	if got := st.Sum / st.Extent; got <= 1 {
		t.Errorf("sum-based occupancy = %v, want > 1 to show it can exceed 100%%", got)
	}
	if got := st.Union / st.Extent; got > 1 {
		t.Errorf("union-based occupancy = %v, must never exceed 1", got)
	}
}

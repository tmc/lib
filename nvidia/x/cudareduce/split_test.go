package cudareduce

import (
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudamem"
)

func TestSplit(t *testing.T) {
	for _, rows := range []int{0, 1, 31, 32, 33, 1000, 4096, 1 << 20} {
		for _, n := range []int{1, 255, 256, 257, 768, 1 << 16} {
			splits, rps := Split(rows, n)
			if splits < 1 || rps < 1 {
				t.Fatalf("Split(%d, %d) = %d, %d", rows, n, splits, rps)
			}
			if rows > 0 && (splits*rps < rows || (splits-1)*rps >= rows) {
				t.Errorf("Split(%d, %d) = %d, %d: ranges do not cover the rows exactly", rows, n, splits, rps)
			}
			if rows > 0 && rps < min(minRowsPerSplit, rows) {
				t.Errorf("Split(%d, %d) = %d, %d: ranges shorter than %d", rows, n, splits, rps, minRowsPerSplit)
			}
			if got := ScratchLen(rows, n, 3); got != 3*splits*n {
				t.Errorf("ScratchLen(%d, %d, 3) = %d, want %d", rows, n, got, 3*splits*n)
			}
		}
	}
}

// TestCheck covers argument validation, which runs before any launch.
func TestCheck(t *testing.T) {
	var r Reducer
	var f cudamem.Slice[float32]
	var s cudamem.Slice[float64]
	var seg cudamem.Slice[int32]
	tests := []struct {
		name string
		err  error
		want string
	}{
		{"short out", r.ColumnSum(nil, f, f, 4, 3, s, false), "out has 0 elements, need 3"},
		{"negative", r.ColumnSum(nil, f, f, -1, 3, s, false), "negative"},
		{"groups", r.SegmentColumnSum(nil, f, f, seg, 4, 3, MaxGroups+1, s, false), "groups"},
		{"no groups", r.SegmentColumnSum(nil, f, f, seg, 4, 3, 0, s, false), "groups"},
		{"layernorm", r.LayerNormGrad(nil, f, f, f, f, f, f, 4, 3, s, false), "dw has 0"},
		{"finish k", r.Finish(nil, f, s, 4, 3, 0, false), "sums per column"},
	}
	for _, tt := range tests {
		if tt.err == nil || !strings.Contains(tt.err.Error(), tt.want) {
			t.Errorf("%s: err = %v, want %q", tt.name, tt.err, tt.want)
		}
	}
}

package cudafuse_test

import (
	"testing"

	"github.com/tmc/lib/nvidia/x/cudafuse"
)

func TestClassifyOpFusionAndExplicitBoundaries(t *testing.T) {
	for _, op := range []string{"add", "sub", "mul", "div", "relu", "silu", "exp", "sqrt", "neg", "reshape"} {
		if got := cudafuse.ClassifyOp(op); got != cudafuse.OpFusibleElementwise {
			t.Errorf("ClassifyOp(%q) = %v, want fusible", op, got)
		}
	}
	for _, op := range []string{"transpose", "gather", "cycle", "replicate", "sum", "scan", "to_host"} {
		if got := cudafuse.ClassifyOp(op); got != cudafuse.OpMaterializingBoundary {
			t.Errorf("ClassifyOp(%q) = %v, want materializing boundary", op, got)
		}
	}
	for _, op := range []string{"where", "filter", "keep"} {
		if got := cudafuse.ClassifyOp(op); got != cudafuse.OpConditionallyFusedCompaction {
			t.Errorf("ClassifyOp(%q) = %v, want compaction boundary", op, got)
		}
	}
	for _, op := range []string{"map", "map2", "materialize", "bogus"} {
		if got := cudafuse.ClassifyOp(op); got != cudafuse.OpUnknown {
			t.Errorf("ClassifyOp(%q) = %v, want unknown", op, got)
		}
	}
}

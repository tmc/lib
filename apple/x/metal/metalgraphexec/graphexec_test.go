package metalgraphexec

import (
	"testing"
)

func TestExecutionGraph(t *testing.T) {
	g := New()
	g.AddDispatch("RMSNorm", "RMSNormPipeline", []int{0, 1, 2}, [3]int{1, 1, 1}, [3]int{128, 1, 1})
	g.AddDispatch("LinearGEMV", "GEMVPipeline", []int{2, 3, 4}, [3]int{32, 1, 1}, [3]int{128, 1, 1})

	if g.NodeCount() != 2 {
		t.Errorf("expected 2 nodes, got %d", g.NodeCount())
	}

	if err := g.Validate(); err != nil {
		t.Errorf("Validate failed: %v", err)
	}
}

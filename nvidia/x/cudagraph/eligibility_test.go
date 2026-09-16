package cudagraph_test

import (
	"testing"

	"github.com/tmc/lib/nvidia/x/cudagraph"
	"github.com/tmc/lib/nvidia/x/cudatest"
)

func TestEligibilityClosedGraph(t *testing.T) {
	var g *cudagraph.Graph
	el, err := g.Eligibility()
	if err == nil {
		t.Fatal("expected error on nil graph")
	}
	if el.Captured {
		t.Errorf("expected Captured=false on nil graph")
	}
	if len(el.Reasons) == 0 {
		t.Errorf("expected Reasons populated on nil graph")
	}
}

func TestEligibilityEmptyGraph(t *testing.T) {
	dev := cudatest.Require(t)
	g, err := cudagraph.New(dev)
	if err != nil {
		t.Fatalf("cudagraph.New: %v", err)
	}
	defer g.Destroy()

	el, err := g.Eligibility()
	if err != nil {
		t.Fatalf("g.Eligibility: %v", err)
	}
	if el.Captured {
		t.Errorf("expected Captured=false for empty graph")
	}
	if el.Nodes != 0 {
		t.Errorf("Nodes = %d, want 0", el.Nodes)
	}
	if len(el.Reasons) == 0 {
		t.Errorf("expected Reasons populated for empty graph")
	}
}

func TestEligibilityEmptyNodes(t *testing.T) {
	dev := cudatest.Require(t)
	g, err := cudagraph.New(dev)
	if err != nil {
		t.Fatalf("cudagraph.New: %v", err)
	}
	defer g.Destroy()

	_, err = g.AddEmptyNode()
	if err != nil {
		t.Fatalf("AddEmptyNode: %v", err)
	}

	el, err := g.Eligibility()
	if err != nil {
		t.Fatalf("g.Eligibility: %v", err)
	}
	if !el.Captured {
		t.Errorf("expected Captured=true for graph with valid empty node, reasons: %v", el.Reasons)
	}
	if el.Nodes != 1 {
		t.Errorf("Nodes = %d, want 1", el.Nodes)
	}
	if el.Kernels != 0 {
		t.Errorf("Kernels = %d, want 0", el.Kernels)
	}
}

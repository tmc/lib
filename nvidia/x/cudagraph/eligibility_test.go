package cudagraph_test

import (
	"context"
	"testing"
	"time"

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
	// admission bounds the wait for the GPU lease and the admission checks
	// after it, not driver calls or cleanup.
	admission, cancel := context.WithTimeout(t.Context(), 30*time.Second)
	defer cancel()
	dev, owner := cudatest.RequireOwned(admission, t)
	g, err := cudagraph.New(dev)
	if g != nil {
		owner.Defer(g.Destroy)
	}
	if err != nil {
		t.Fatalf("cudagraph.New: %v", err)
	}

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
	// admission bounds the wait for the GPU lease and the admission checks
	// after it, not driver calls or cleanup.
	admission, cancel := context.WithTimeout(t.Context(), 30*time.Second)
	defer cancel()
	dev, owner := cudatest.RequireOwned(admission, t)
	g, err := cudagraph.New(dev)
	if g != nil {
		owner.Defer(g.Destroy)
	}
	if err != nil {
		t.Fatalf("cudagraph.New: %v", err)
	}

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

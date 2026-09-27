package egraph_test

import (
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/egraph"
)

// TestLazyFusionBoundary_EGraphFusionNodeExtraction verifies egraph rewrite rules
// correctly group lazy operations into a single extracted cudafuse fusion region.
func TestLazyFusionBoundary_EGraphFusionNodeExtraction(t *testing.T) {
	// 1. Verify E-Graph rewrite rule unification
	g := egraph.New()
	a := g.Add(egraph.NewLeaf("a"))
	b := g.Add(egraph.NewLeaf("b"))
	c := g.Add(egraph.NewLeaf("c"))

	addNode := g.Add(egraph.NewNode("add", a, b))
	mulNode := g.Add(egraph.NewNode("mul", addNode, c))
	rootNode := g.Add(egraph.NewNode("sum", mulNode))

	rules := egraph.LazyFusionRewriteRules()
	ruleApplied := false
	for _, r := range rules {
		if r.Name == "fuse_add_mul" {
			merges := r.Apply(g)
			if merges > 0 {
				ruleApplied = true
			}
		}
	}
	if !ruleApplied {
		t.Fatalf("expected fuse_add_mul rewrite rule to match and apply in egraph")
	}

	// 2. Verify maximal contiguous fusion region extraction
	exprA := &egraph.Expr{Op: "a"}
	exprB := &egraph.Expr{Op: "b"}
	exprC := &egraph.Expr{Op: "c"}

	exprAdd := &egraph.Expr{Op: "add", Children: []*egraph.Expr{exprA, exprB}}
	exprMul := &egraph.Expr{Op: "mul", Children: []*egraph.Expr{exprAdd, exprC}}
	exprRoot := &egraph.Expr{Op: "sum", Children: []*egraph.Expr{exprMul}}

	plan, err := egraph.ExtractFusionRegions(exprRoot)
	if err != nil {
		t.Fatalf("ExtractFusionRegions failed: %v", err)
	}

	if plan.RootBoundary != "sum" {
		t.Errorf("plan root boundary = %q, want \"sum\"", plan.RootBoundary)
	}
	if len(plan.Regions) != 1 {
		t.Fatalf("expected exactly 1 unified fusion region, got %d", len(plan.Regions))
	}

	region := plan.Regions[0]
	if len(region.OpNames) != 2 || region.OpNames[0] != "add" || region.OpNames[1] != "mul" {
		t.Errorf("expected region ops [add, mul], got %v", region.OpNames)
	}
	if len(region.Inputs) != 3 {
		t.Errorf("expected 3 external inputs [a, b, c], got %d", len(region.Inputs))
	}
	if region.Boundary != "sum" {
		t.Errorf("expected boundary \"sum\", got %q", region.Boundary)
	}
	if region.Kernel == nil {
		t.Fatalf("expected compiled cudafuse.FusedKernel in extracted region")
	}
	if !strings.Contains(region.Kernel.Source, "v_in_") {
		t.Errorf("expected generated kernel arithmetic in source:\n%s", region.Kernel.Source)
	}

	_ = rootNode
}

// TestEGraphFusionBoundary_FilterIsolation verifies that filter/where compactions
// cleanly isolate upstream fusion regions.
func TestEGraphFusionBoundary_FilterIsolation(t *testing.T) {
	// (sum (add (filter (mul (add a b) c) mask) d))
	a := &egraph.Expr{Op: "a"}
	b := &egraph.Expr{Op: "b"}
	c := &egraph.Expr{Op: "c"}
	mask := &egraph.Expr{Op: "mask"}
	d := &egraph.Expr{Op: "d"}

	upstreamAdd := &egraph.Expr{Op: "add", Children: []*egraph.Expr{a, b}}
	upstreamMul := &egraph.Expr{Op: "mul", Children: []*egraph.Expr{upstreamAdd, c}}
	filter := &egraph.Expr{Op: "filter", Children: []*egraph.Expr{upstreamMul, mask}}
	downstreamAdd := &egraph.Expr{Op: "add", Children: []*egraph.Expr{filter, d}}
	root := &egraph.Expr{Op: "sum", Children: []*egraph.Expr{downstreamAdd}}

	plan, err := egraph.ExtractFusionRegions(root)
	if err != nil {
		t.Fatalf("ExtractFusionRegions failed: %v", err)
	}

	// Should extract 2 distinct fusion regions:
	// Region 1: downstream (add d) bounded by sum
	// Region 2: upstream (add, mul) bounded by filter
	if len(plan.Regions) != 2 {
		t.Fatalf("expected 2 isolated fusion regions across filter boundary, got %d", len(plan.Regions))
	}

	foundUpstream := false
	foundDownstream := false

	for _, reg := range plan.Regions {
		if reg.Boundary == "filter" {
			foundUpstream = true
			if len(reg.OpNames) != 2 {
				t.Errorf("upstream ops=%v, want [add, mul]", reg.OpNames)
			}
		}
		if reg.Boundary == "sum" {
			foundDownstream = true
		}
	}

	if !foundUpstream {
		t.Errorf("missing upstream fusion region bounded by filter")
	}
	if !foundDownstream {
		t.Errorf("missing downstream fusion region bounded by sum")
	}
}

// TestEGraphFusion_MutantControls tests negative controls.
func TestEGraphFusion_MutantControls(t *testing.T) {
	// Nil expression must fail gracefully
	if _, err := egraph.ExtractFusionRegions(nil); err == nil {
		t.Errorf("expected error for nil expression, got nil")
	}
}

package egraph_test

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/egraph"
)

// TestLazyFusionBoundary_EGraphFusionNodeExtraction verifies egraph rewrite rules
// correctly group lazy operations into a single extracted cudafuse fusion region.
func TestLazyFusionBoundary_EGraphFusionNodeExtraction(t *testing.T) {
	g := egraph.New()
	a := g.Add(egraph.NewLeaf("a"))
	b := g.Add(egraph.NewLeaf("b"))
	c := g.Add(egraph.NewLeaf("c"))
	addNode := g.Add(egraph.NewNode("add", a, b))
	mulNode := g.Add(egraph.NewNode("mul", addNode, c))
	root := g.Add(egraph.NewNode("sum", mulNode))
	rules := egraph.LazyFusionRewriteRules()
	_, merges := egraph.Saturate(g, rules, 10)
	if merges == 0 {
		t.Fatal("Saturate applied no lazy fusion rewrite")
	}
	expr, _, err := egraph.ExtractBest(g, root, preferFusionCost{})
	if err != nil {
		t.Fatalf("ExtractBest: %v", err)
	}
	plan, err := egraph.ExtractFusionRegions(expr)
	if err != nil {
		t.Fatalf("ExtractFusionRegions: %v", err)
	}
	if plan.RootBoundary != "sum" {
		t.Errorf("root boundary = %q, want sum", plan.RootBoundary)
	}
	if len(plan.Regions) != 1 {
		t.Fatalf("got %d regions, want exactly one", len(plan.Regions))
	}
	region := plan.Regions[0]
	if region.Boundary != "sum" {
		t.Errorf("boundary = %q, want sum", region.Boundary)
	}
	if region.Root.Op != "mul" || len(region.Root.Children) != 2 || region.Root.Children[0].Op != "add" {
		t.Errorf("extracted fusion does not compute (a+b)*c: %#v", region.Root)
	}
	if region.Kernel == nil {
		t.Fatal("no kernel compiled for extracted e-graph fusion")
	}
	if !strings.Contains(region.Kernel.Source, " + ") || !strings.Contains(region.Kernel.Source, " * ") || strings.Contains(region.Kernel.Source, "in_leaf_") {
		t.Errorf("kernel does not contain extracted add/mul computation:\n%s", region.Kernel.Source)
	}
}

type preferFusionCost struct{}

func (preferFusionCost) NodeCost(n egraph.ENode, childCosts []float64) float64 {
	cost := 1.0
	if strings.HasPrefix(n.Op, "fused_") {
		cost = 0.1
	}
	for _, child := range childCosts {
		cost += child
	}
	return cost
}

func TestEGraphFusionRegion_KernelComputesRegionOps(t *testing.T) {
	x := &egraph.Expr{Op: "x"}
	y := &egraph.Expr{Op: "y"}
	relu := &egraph.Expr{Op: "relu", Children: []*egraph.Expr{x}}
	reshape := &egraph.Expr{Op: "reshape", Children: []*egraph.Expr{relu}}
	add := &egraph.Expr{Op: "add", Children: []*egraph.Expr{reshape, y}}
	root := &egraph.Expr{Op: "sum", Children: []*egraph.Expr{add}}
	plan, err := egraph.ExtractFusionRegions(root)
	if err != nil {
		t.Fatal(err)
	}
	if len(plan.Regions) != 1 {
		t.Fatalf("regions = %d, want 1", len(plan.Regions))
	}
	region := plan.Regions[0]
	if !reflect.DeepEqual(region.OpNames, []string{"relu", "reshape", "add"}) {
		t.Fatalf("OpNames = %v", region.OpNames)
	}
	if region.Kernel == nil {
		t.Fatal("region has no kernel")
	}
	src := region.Kernel.Source
	if !strings.Contains(src, "> 0.0f") || !strings.Contains(src, " + ") || strings.Contains(src, "in_leaf_") {
		t.Fatalf("kernel source does not compute region ops or has dropped subtree inputs:\n%s", src)
	}
}

func TestEGraphFusionRegion_KernelPreservesDistinctOperands(t *testing.T) {
	x := &egraph.Expr{Op: "x"}
	y := &egraph.Expr{Op: "y"}
	z := &egraph.Expr{Op: "z"}
	relu := &egraph.Expr{Op: "relu", Children: []*egraph.Expr{x}}
	add := &egraph.Expr{Op: "add", Children: []*egraph.Expr{relu, y}}
	sub := &egraph.Expr{Op: "sub", Children: []*egraph.Expr{add, z}}
	plan, err := egraph.ExtractFusionRegions(&egraph.Expr{Op: "sum", Children: []*egraph.Expr{sub}})
	if err != nil {
		t.Fatal(err)
	}
	if len(plan.Regions) != 1 || plan.Regions[0].Kernel == nil {
		t.Fatalf("regions = %#v, want one compiled region", plan.Regions)
	}
	region := plan.Regions[0]
	if !reflect.DeepEqual(region.OpNames, []string{"relu", "add", "sub"}) {
		t.Fatalf("OpNames = %v", region.OpNames)
	}
	if len(region.Kernel.Inputs) != 3 || len(region.Kernel.Outputs) != 1 {
		t.Fatalf("kernel inputs/outputs = %v/%v, want three inputs and one output", region.Kernel.Inputs, region.Kernel.Outputs)
	}
	compiler, err := exec.LookPath("clang++")
	if err != nil {
		t.Skip("host emitted-kernel value check requires clang++")
	}
	inputNames := []string{"x", "y", "z"}
	args := make([]string, 0, len(region.Kernel.Inputs)+len(region.Kernel.Outputs)+1)
	for i := range region.Kernel.Inputs {
		args = append(args, inputNames[i])
	}
	args = append(args, "out", "4")
	main := fmt.Sprintf(`
dim blockDim, blockIdx, threadIdx, gridDim;
int main() {
  float x[4] = {-2, 1, 2, -1};
  float y[4] = {4, 6, -1, 3};
  float z[4] = {3, 2, 5, 4};
  float out[4] = {};
  blockDim.x = 4; blockDim.y = 1; blockDim.z = 1;
  gridDim.x = 1; gridDim.y = 1; gridDim.z = 1;
  blockIdx.x = 0; blockIdx.y = 0; blockIdx.z = 0;
  threadIdx.y = 0; threadIdx.z = 0;
  for (threadIdx.x = 0; threadIdx.x < 4; ++threadIdx.x)
    %s(%s);
  const float want[4] = {1, 5, -4, -1};
  for (int i = 0; i < 4; ++i) printf("%%g ", out[i]);
  for (int i = 0; i < 4; ++i) if (out[i] != want[i]) return i + 1;
  return 0;
}
`, region.Kernel.Name, strings.Join(args, ", "))
	dir := t.TempDir()
	header := `#define __global__
struct dim { unsigned int x=1,y=1,z=1; };
extern dim blockDim, blockIdx, threadIdx, gridDim;
`
	if err := os.WriteFile(filepath.Join(dir, "cuda_runtime.h"), []byte(header), 0600); err != nil {
		t.Fatal(err)
	}
	source := "#include <stdio.h>\n" + region.Kernel.Source + "\n" + main
	sourcePath := filepath.Join(dir, "region.cc")
	if err := os.WriteFile(sourcePath, []byte(source), 0600); err != nil {
		t.Fatal(err)
	}
	binary := filepath.Join(dir, "region")
	if out, err := exec.Command(compiler, "-std=c++17", "-O1", "-I", dir, sourcePath, "-o", binary).CombinedOutput(); err != nil {
		t.Fatalf("compile emitted region: %v\n%s", err, out)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if out, err := exec.CommandContext(ctx, binary).CombinedOutput(); err != nil {
		t.Fatalf("emitted region produced wrong values: %v\n%s\ninputs=%v outputs=%v\n%s", err, out, region.Kernel.Inputs, region.Kernel.Outputs, source)
	}
}

func TestExtractedFusionPlanRootBoundary(t *testing.T) {
	for _, tt := range []struct {
		name string
		expr *egraph.Expr
		want string
	}{
		{"root sum", &egraph.Expr{Op: "sum", Children: []*egraph.Expr{{Op: "relu", Children: []*egraph.Expr{{Op: "x"}}}}}, "sum"},
		{"nested sum", &egraph.Expr{Op: "add", Children: []*egraph.Expr{{Op: "sum", Children: []*egraph.Expr{{Op: "x"}}}, {Op: "y"}}}, ""},
	} {
		t.Run(tt.name, func(t *testing.T) {
			plan, err := egraph.ExtractFusionRegions(tt.expr)
			if err != nil {
				t.Fatal(err)
			}
			if plan.RootBoundary != tt.want {
				t.Fatalf("RootBoundary = %q, want %q", plan.RootBoundary, tt.want)
			}
		})
	}
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

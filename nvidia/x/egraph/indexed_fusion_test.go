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

func indexedExpr(op string, shape []int, children ...*egraph.Expr) *egraph.Expr {
	return &egraph.Expr{Op: op, Data: egraph.FusionTensor{Shape: shape}, Children: children}
}

func indexedTranspose(input *egraph.Expr, shape, perm []int) *egraph.Expr {
	return &egraph.Expr{Op: "transpose", Data: egraph.FusionTensor{Shape: shape, Permutation: perm}, Children: []*egraph.Expr{input}}
}

func TestIndexedFusionPreservesValuesAndSharedInputs(t *testing.T) {
	for _, name := range []string{"distinct operands", "shared differently indexed input", "expanded alias"} {
		t.Run(name, func(t *testing.T) {
			x := indexedExpr("x", []int{2, 3})
			transpose := indexedTranspose(x, []int{3, 2}, []int{1, 0})
			var root *egraph.Expr
			want := []int{13, 11, 14, 12, 15, 13}
			inputs := 2
			switch name {
			case "distinct operands":
				root = indexedExpr("add", []int{3, 2}, indexedExpr("neg", []int{3, 2}, transpose), indexedExpr("y", []int{3, 2}))
			case "shared differently indexed input":
				root = indexedExpr("add", []int{6}, indexedExpr("reshape", []int{6}, transpose), indexedExpr("reshape", []int{6}, x))
				want, inputs = []int{-6, -2, -3, 1, 0, 4}, 1
			case "expanded alias":
				root = indexedExpr("fused_neg_add", []int{3, 2}, transpose, indexedExpr("y", []int{3, 2}))
			}
			plan, err := egraph.ExtractFusionRegions(root)
			if err != nil {
				t.Fatal(err)
			}
			if len(plan.Regions) != 1 || len(plan.Regions[0].Inputs) != inputs || plan.RootBoundary != "" {
				t.Fatal("indexed extraction split a region or duplicated shared input", plan)
			}
			region := plan.Regions[0]
			if len(region.Kernel.Inputs) != inputs {
				t.Fatal("kernel input count differs")
			}
			runIndexedEgraphSource(t, region, want)
		})
	}
}

func runIndexedEgraphSource(t *testing.T, region *egraph.ExtractedFusionRegion, want []int) {
	t.Helper()
	compiler, err := exec.LookPath("clang++")
	if err != nil {
		t.Skip("host indexed egraph value check requires clang++")
	}
	dir := t.TempDir()
	header := "#define __global__\nstruct dim {unsigned int x=1,y=1,z=1;};\nextern dim blockDim,blockIdx,threadIdx,gridDim;\n"
	if err := os.WriteFile(filepath.Join(dir, "cuda_runtime.h"), []byte(header), 0600); err != nil {
		t.Fatal(err)
	}
	var main strings.Builder
	fmt.Fprintln(&main, "dim blockDim,blockIdx,threadIdx,gridDim;\nint main(){float x[6]={-3,-2,-1,0,1,2},y[6]={10,11,12,13,14,15},output[8];for(int i=0;i<8;i++)output[i]=-99;")
	var args []string
	for _, expr := range region.Inputs {
		if expr.Op != "x" && expr.Op != "y" {
			t.Fatalf("unknown fixture input %s", expr.Op)
		}
		args = append(args, expr.Op)
	}
	args = append(args, "output+1", "6")
	fmt.Fprintf(&main, "blockDim.x=4;gridDim.x=2;for(blockIdx.x=0;blockIdx.x<2;blockIdx.x++)for(threadIdx.x=0;threadIdx.x<4;threadIdx.x++)%s(%s);\n", region.Kernel.Name, strings.Join(args, ","))
	for i, value := range want {
		fmt.Fprintf(&main, "if(output[%d]!=%d) return 1;\n", i+1, value)
	}
	fmt.Fprintln(&main, "return output[0]!=-99 || output[7]!=-99;}")
	path, binary := filepath.Join(dir, "indexed.cc"), filepath.Join(dir, "indexed")
	if err := os.WriteFile(path, []byte("#include <math.h>\n"+region.Kernel.Source+main.String()), 0600); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if output, err := exec.CommandContext(ctx, compiler, "-std=c++17", "-O1", "-I", dir, path, "-o", binary).CombinedOutput(); err != nil {
		t.Fatalf("compile indexed region: %v\n%s", err, output)
	}
	ctx, cancelRun := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancelRun()
	if output, err := exec.CommandContext(ctx, binary).CombinedOutput(); err != nil {
		t.Fatalf("indexed values/guards: %v\n%s", err, output)
	}
}

func TestIndexedFusionRefusesMissingOrContradictoryMetadata(t *testing.T) {
	for _, tc := range []struct {
		name   string
		mutate func(*egraph.Expr, *egraph.Expr, *egraph.Expr)
	}{
		{"input missing", func(x, _, _ *egraph.Expr) { x.Data = nil }},
		{"transpose missing", func(_, tr, _ *egraph.Expr) { tr.Data = nil }},
		{"unary missing", func(_, _, root *egraph.Expr) { root.Data = nil }},
		{"wrong type", func(_, tr, _ *egraph.Expr) { tr.Data = []int{1, 0} }},
		{"missing permutation", func(_, tr, _ *egraph.Expr) { tr.Data = egraph.FusionTensor{Shape: []int{3, 2}} }},
		{"duplicate permutation", func(_, tr, _ *egraph.Expr) {
			tr.Data = egraph.FusionTensor{Shape: []int{3, 2}, Permutation: []int{0, 0}}
		}},
		{"output mismatch", func(_, tr, _ *egraph.Expr) {
			tr.Data = egraph.FusionTensor{Shape: []int{2, 3}, Permutation: []int{1, 0}}
		}},
		{"negative input after zero", func(x, _, _ *egraph.Expr) { x.Data = egraph.FusionTensor{Shape: []int{0, -1}} }},
		{"unbounded input", func(x, _, _ *egraph.Expr) { x.Data = egraph.FusionTensor{Shape: []int{1 << 30, 3}} }},
		{"unexpected unary permutation", func(_, _, root *egraph.Expr) {
			root.Data = egraph.FusionTensor{Shape: []int{3, 2}, Permutation: []int{1, 0}}
		}},
		{"wrong arity", func(_, tr, _ *egraph.Expr) { tr.Children = nil }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			x := indexedExpr("x", []int{2, 3})
			tr := indexedTranspose(x, []int{3, 2}, []int{1, 0})
			root := indexedExpr("neg", []int{3, 2}, tr)
			tc.mutate(x, tr, root)
			if plan, err := egraph.ExtractFusionRegions(root); err == nil || plan != nil {
				t.Fatal("invalid indexed region admitted", plan, err)
			}
		})
	}
}

func TestIndexedFusionRejectsCyclesAndNilChildren(t *testing.T) {
	for _, cyclic := range []bool{false, true} {
		root := indexedExpr("neg", []int{1}, nil)
		if cyclic {
			root.Children[0] = root
		}
		if plan, err := egraph.ExtractFusionRegions(root); err == nil || plan != nil {
			t.Fatal("malformed AST admitted", err)
		}
	}
}

func TestIndexedFusionScalarsAndEmptyDomains(t *testing.T) {
	for _, shape := range [][]int{nil, {2, 0, 3}, {0, 1 << 30, 1 << 30}} {
		perm := make([]int, len(shape))
		output := make([]int, len(shape))
		for i := range shape {
			perm[i] = len(shape) - 1 - i
			output[i] = shape[perm[i]]
		}
		root := indexedTranspose(indexedExpr("x", shape), output, perm)
		plan, err := egraph.ExtractFusionRegions(root)
		if err != nil {
			t.Fatal(err)
		}
		if len(plan.Regions) != 1 || !reflect.DeepEqual(plan.Regions[0].OpNames, []string{"transpose"}) {
			t.Fatal("empty/scalar transpose not extracted")
		}
	}
}

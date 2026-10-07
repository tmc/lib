package cudalaunch

import (
	"go/ast"
	"go/parser"
	"go/token"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudashared"
)

func TestBoundPlanRequiresFreshOwner(t *testing.T) {
	p, err := cudashared.NewPlan(cudashared.Limits{StaticMax: 64, DynamicMax: 64, TotalMax: 64, Source: "synthetic host plan"})
	if err != nil {
		t.Fatal(err)
	}
	if _, err = p.Add(cudashared.Request{Name: "dynamic", Size: 16, Alignment: 1, Dynamic: true}); err != nil {
		t.Fatal(err)
	}
	l := OwnedKernel(&cuda.Kernel{})
	l.sharedPlan = p
	l.shmem = 16
	for i := 0; i < 2; i++ {
		if err = l.checkSharedPlan(); err == nil || !strings.Contains(err.Error(), "kernel has no device") {
			t.Fatalf("fresh owner refused: %v", err)
		}
	}
}

// No kernel is dispatched by this source wiring control. The behavioral probe
// tests cover fresh query/readback refusal; this pins the public launch callsite.
func TestLaunchCallsFreshPlanCheckBeforeDispatch(t *testing.T) {
	f, err := parser.ParseFile(token.NewFileSet(), "launch.go", nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	var launch, check *ast.FuncDecl
	for _, decl := range f.Decls {
		fn, ok := decl.(*ast.FuncDecl)
		if !ok {
			continue
		}
		if fn.Name.Name == "Launch" {
			launch = fn
		}
		if fn.Name.Name == "checkSharedPlan" {
			check = fn
		}
	}
	if launch == nil || check == nil {
		t.Fatal("missing owned launch path")
	}
	calls := 0
	fresh := 0
	var freshPos, lockPos token.Pos
	ast.Inspect(launch.Body, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}
		s, ok := call.Fun.(*ast.SelectorExpr)
		if !ok {
			return true
		}
		if s.Sel.Name == "checkSharedPlan" {
			calls++
			freshPos = call.Pos()
		}
		if s.Sel.Name == "LockOSThread" {
			lockPos = call.Pos()
		}
		return true
	})
	ast.Inspect(check.Body, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}
		s, ok := call.Fun.(*ast.SelectorExpr)
		if ok && s.Sel.Name == "CheckPlan" {
			id, ok := s.X.(*ast.Ident)
			if ok && id.Name == "cudaheadroom" {
				fresh++
			}
		}
		return true
	})
	if calls != 1 || fresh != 1 || freshPos >= lockPos {
		t.Fatalf("fresh launch wiring calls=%d inspect=%d positions=%d/%d", calls, fresh, freshPos, lockPos)
	}
}

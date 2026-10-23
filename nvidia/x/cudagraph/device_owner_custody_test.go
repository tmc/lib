package cudagraph_test

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"slices"
	"strings"
	"testing"
)

// TestDeviceOwnerCustody pins the graph tests that own their device through
// cudatest.RequireOwned. It reads eligibility_test.go as it is on disk.
func TestDeviceOwnerCustody(t *testing.T) {
	const path = "eligibility_test.go"
	src, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, path, src, 0)
	if err != nil {
		t.Fatal(err)
	}
	text := func(n ast.Node) string {
		return string(src[fset.Position(n.Pos()).Offset:fset.Position(n.End()).Offset])
	}

	funcs := make(map[string]*ast.FuncDecl)
	for _, d := range file.Decls {
		if fn, ok := d.(*ast.FuncDecl); ok {
			funcs[fn.Name.Name] = fn
		}
	}
	if fn := funcs["TestEligibilityClosedGraph"]; fn == nil || slices.ContainsFunc(callNames(fn.Body), isCudatest) {
		t.Error("TestEligibilityClosedGraph is missing or acquires a device")
	}

	for _, name := range []string{"TestEligibilityEmptyGraph", "TestEligibilityEmptyNodes"} {
		fn := funcs[name]
		if fn == nil {
			t.Errorf("%s not found", name)
			continue
		}
		var owned int
		for _, call := range callNames(fn.Body) {
			switch call {
			case "cudatest.RequireOwned":
				owned++
			case "g.Destroy", "dev.Close", "t.Cleanup":
				t.Errorf("%s calls %s; the device owner releases the graph, device and lease", name, call)
			default:
				if isCudatest(call) {
					t.Errorf("%s calls %s", name, call)
				}
			}
		}
		if owned != 1 {
			t.Errorf("%s calls cudatest.RequireOwned %d times, want 1", name, owned)
		}

		// The only defer cancels the admission context.
		ast.Inspect(fn.Body, func(n ast.Node) bool {
			if d, ok := n.(*ast.DeferStmt); ok && text(d) != "defer cancel()" {
				t.Errorf("%s: unexpected %s", name, text(d))
			}
			return true
		})

		// The graph is registered with its owner before its error is checked.
		want := []string{
			"admission, cancel := context.WithTimeout(t.Context(), 30*time.Second)",
			"defer cancel()",
			"dev, owner := cudatest.RequireOwned(admission, t)",
			"g, err := cudagraph.New(dev)",
			"if g != nil {\n\t\towner.Defer(g.Destroy)\n\t}",
			"if err != nil {\n\t\tt.Fatalf(\"cudagraph.New: %v\", err)\n\t}",
		}
		var got []string
		for _, s := range fn.Body.List {
			got = append(got, text(s))
		}
		if len(got) < len(want) || !slices.Equal(got[:len(want)], want) {
			t.Errorf("%s begins with:\n%q\nwant:\n%q", name, got[:min(len(got), len(want))], want)
		}
	}
}

// callNames returns the names of the functions called in n, in source order.
func callNames(n ast.Node) []string {
	var names []string
	ast.Inspect(n, func(n ast.Node) bool {
		if call, ok := n.(*ast.CallExpr); ok {
			if sel, ok := call.Fun.(*ast.SelectorExpr); ok {
				if x, ok := sel.X.(*ast.Ident); ok {
					names = append(names, x.Name+"."+sel.Sel.Name)
				}
			}
		}
		return true
	})
	return names
}

func isCudatest(call string) bool {
	return strings.HasPrefix(call, "cudatest.")
}

package cudahotswap

import (
	"go/ast"
	"go/parser"
	"go/token"
	"testing"
)

// The native readback requires KernelOrder for both the registered executable
// and the original raw executable. Keep that producer prerequisite explicit.
func TestPublicGraphReadbackRetainsKernelOrder(t *testing.T) {
	f, err := parser.ParseFile(token.NewFileSet(), "push_native_test.go", nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	calls, reads := 0, 0
	ast.Inspect(f, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}
		selector, ok := call.Fun.(*ast.SelectorExpr)
		if !ok {
			return true
		}
		if selector.Sel.Name == "KernelOrder" {
			reads++
			return true
		}
		receiver, ok := selector.X.(*ast.Ident)
		if !ok || receiver.Name != "g" {
			return true
		}
		if selector.Sel.Name == "Instantiate" {
			t.Error("graph readback cannot use default unretained executable")
		}
		if selector.Sel.Name != "InstantiateWithOptions" {
			return true
		}
		calls++
		if len(call.Args) != 1 {
			t.Error("missing graph options")
			return true
		}
		opts, ok := call.Args[0].(*ast.CompositeLit)
		if !ok {
			t.Error("unrecognized graph options")
			return true
		}
		retained := false
		for _, field := range opts.Elts {
			kv, ok := field.(*ast.KeyValueExpr)
			if !ok {
				continue
			}
			key, ok := kv.Key.(*ast.Ident)
			if !ok || key.Name != "RetainKernelOrder" {
				continue
			}
			value, ok := kv.Value.(*ast.Ident)
			retained = ok && value.Name == "true"
		}
		if !retained {
			t.Error("graph readback requires retained kernel order")
		}
		return true
	})
	if calls != 1 || reads != 1 {
		t.Fatalf("graph producer wiring calls=%d reads=%d, want 1 each", calls, reads)
	}
}

func TestPublicGraphHandleCapturedBeforeMutation(t *testing.T) {
	f, err := parser.ParseFile(token.NewFileSet(), "push_native_test.go", nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	var capture, bind token.Pos
	lookup := 0
	ast.Inspect(f, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}
		if name, ok := call.Fun.(*ast.Ident); ok && name.Name == "publicGraphHandle" {
			lookup++
			return true
		}
		selector, ok := call.Fun.(*ast.SelectorExpr)
		if !ok {
			return true
		}
		switch selector.Sel.Name {
		case "KernelOrder":
			receiver, ok := selector.X.(*ast.Ident)
			if !ok || receiver.Name != "e" {
				t.Error("kernel-order evidence reused after executable mutation")
			}
			capture = call.Pos()
		case "RegisterGraph":
			bind = call.Pos()
		}
		return true
	})
	if capture == 0 || bind == 0 || capture >= bind || lookup != 1 {
		t.Fatalf("capture=%d bind=%d lookup=%d", capture, bind, lookup)
	}
}

func TestPublicGraphHandleCaptureBothOwners(t *testing.T) {
	f, err := parser.ParseFile(token.NewFileSet(), "push_native_test.go", nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	loops := 0
	ast.Inspect(f, func(n ast.Node) bool {
		loop, ok := n.(*ast.RangeStmt)
		if !ok {
			return true
		}
		register, ok := loop.Value.(*ast.Ident)
		if !ok || register.Name != "register" {
			return true
		}
		values, ok := loop.X.(*ast.CompositeLit)
		if !ok || len(values.Elts) != 2 {
			t.Error("graph owners must cover both registration branches")
			return true
		}
		for i, want := range []string{"true", "false"} {
			value, ok := values.Elts[i].(*ast.Ident)
			if !ok || value.Name != want {
				t.Error("graph owners must include registered and raw executables")
			}
		}
		loops++
		capture, mapping, bind := -1, -1, -1
		for i, statement := range loop.Body.List {
			if assignment, ok := statement.(*ast.AssignStmt); ok && len(assignment.Rhs) == 1 {
				if call, ok := assignment.Rhs[0].(*ast.CallExpr); ok {
					if selector, ok := call.Fun.(*ast.SelectorExpr); ok && selector.Sel.Name == "KernelOrder" {
						receiver, ok := selector.X.(*ast.Ident)
						if ok && receiver.Name == "e" {
							capture = i
						}
					}
				}
				if len(assignment.Lhs) == 1 {
					if index, ok := assignment.Lhs[0].(*ast.IndexExpr); ok {
						table, okTable := index.X.(*ast.Ident)
						key, okKey := index.Index.(*ast.Ident)
						value, okValue := assignment.Rhs[0].(*ast.Ident)
						if okTable && table.Name == "graphHandles" && okKey && key.Name == "e" && okValue && value.Name == "graphExec" {
							mapping = i
						}
					}
				}
			}
			ast.Inspect(statement, func(n ast.Node) bool {
				call, ok := n.(*ast.CallExpr)
				if !ok {
					return true
				}
				selector, ok := call.Fun.(*ast.SelectorExpr)
				if ok && selector.Sel.Name == "RegisterGraph" {
					bind = i
				}
				return true
			})
		}
		if capture < 0 || mapping <= capture || bind <= mapping {
			t.Errorf("both owners require unconditional capture then mapping before bind: %d,%d,%d", capture, mapping, bind)
		}
		return true
	})
	if loops != 1 {
		t.Fatalf("graph owner loops=%d, want 1", loops)
	}
}

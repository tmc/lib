package driver

import (
	"errors"
	"go/ast"
	"go/parser"
	"go/token"
	"strconv"
	"strings"
	"testing"
)

// Generated files can drift even when the generator selects wrappers and
// registrations consistently. Check the shipped pointers and symbol guards.
func TestGeneratedRegistrationCoverage(t *testing.T) {
	declared := make(map[string]bool)
	registered := make(map[string]string)
	cleared := make(map[string]bool)
	guarded := make(map[string]bool)
	for _, filename := range []string{"funcs.gen.go", "result.gen.go", "init.gen.go"} {
		f, err := parser.ParseFile(token.NewFileSet(), filename, nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		for _, d := range f.Decls {
			if g, ok := d.(*ast.GenDecl); ok && g.Tok == token.VAR {
				for _, spec := range g.Specs {
					v := spec.(*ast.ValueSpec)
					if _, ok := v.Type.(*ast.FuncType); !ok {
						continue
					}
					for _, name := range v.Names {
						if strings.HasPrefix(name.Name, "fn") {
							declared[name.Name] = true
						}
					}
				}
			}
			fn, ok := d.(*ast.FuncDecl)
			if !ok {
				continue
			}
			ast.Inspect(fn.Body, func(n ast.Node) bool {
				if fn.Name.Name == "clearFuncs" {
					if a, ok := n.(*ast.AssignStmt); ok && len(a.Lhs) == 1 && len(a.Rhs) == 1 {
						lhs, lok := a.Lhs[0].(*ast.Ident)
						rhs, rok := a.Rhs[0].(*ast.Ident)
						if lok && rok && rhs.Name == "nil" {
							cleared[lhs.Name] = true
						}
					}
				}
				call, ok := n.(*ast.CallExpr)
				if !ok {
					return true
				}
				callee, ok := call.Fun.(*ast.Ident)
				if !ok {
					return true
				}
				if fn.Name.Name == "registerFuncs" && callee.Name == "register" && len(call.Args) == 2 {
					addr := call.Args[0].(*ast.UnaryExpr)
					name := addr.X.(*ast.Ident).Name
					symbol, err := strconv.Unquote(call.Args[1].(*ast.BasicLit).Value)
					if err != nil {
						t.Fatal(err)
					}
					if _, exists := registered[name]; exists {
						t.Errorf("%s registered more than once", name)
					}
					registered[name] = symbol
				}
				if callee.Name == "checkLibSym" && len(call.Args) == 1 {
					if literal, ok := call.Args[0].(*ast.BasicLit); ok {
						symbol, err := strconv.Unquote(literal.Value)
						if err != nil {
							t.Fatal(err)
						}
						guarded[symbol] = true
					}
				}
				return true
			})
		}
	}
	if len(declared) == 0 || len(guarded) == 0 {
		t.Fatal("no generated function pointers or guards found")
	}
	symbols := make(map[string]bool)
	for name, symbol := range registered {
		if !declared[name] {
			t.Errorf("registered pointer %s is not declared", name)
		}
		symbols[symbol] = true
	}
	for name := range declared {
		if _, ok := registered[name]; !ok {
			t.Errorf("declared pointer %s is not registered", name)
		}
		if !cleared[name] {
			t.Errorf("declared pointer %s is not cleared", name)
		}
	}
	for symbol := range guarded {
		if !symbols[symbol] {
			t.Errorf("guarded symbol %s is not registered", symbol)
		}
	}
}

// An older library may omit even a generated, registered optional API. Model
// that state without loading a native library or calling a function pointer.
func TestDeprecatedWrapperMissingSymbol(t *testing.T) {
	loadMu.Lock()
	defer loadMu.Unlock()
	oldReady, oldPresent := ready.Load(), present
	ready.Store(true)
	present = make(map[string]bool)
	defer func() {
		present = oldPresent
		ready.Store(oldReady)
	}()
	err := CtxGetSharedMemConfig(nil)
	var symErr *SymbolError
	if !errors.As(err, &symErr) || symErr.Symbol != "cuCtxGetSharedMemConfig" {
		t.Fatalf("error = %v, want SymbolError for cuCtxGetSharedMemConfig", err)
	}
}

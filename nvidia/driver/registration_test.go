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
	registered := make(map[string][]string)
	chains := make(map[string][]string)
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
					for i, name := range v.Names {
						if i < len(v.Values) {
							if symbols := symbolChainNames(v.Values[i]); symbols != nil {
								chains[name.Name] = symbols
							}
						}
					}
					prefix := "fn"
					switch t := v.Type.(type) {
					case *ast.FuncType:
					case *ast.Ident:
						if t.Name != "uintptr" {
							continue
						}
						prefix = "sym"
					default:
						continue
					}
					for _, name := range v.Names {
						if strings.HasPrefix(name.Name, prefix) {
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
				if a, ok := n.(*ast.AssignStmt); ok && len(a.Lhs) == 1 && len(a.Rhs) == 1 {
					lhs, _ := a.Lhs[0].(*ast.Ident)
					switch rhs := a.Rhs[0].(type) {
					case *ast.Ident:
						if fn.Name.Name == "clearFuncs" && lhs != nil && rhs.Name == "nil" {
							cleared[lhs.Name] = true
						}
					case *ast.BasicLit:
						if fn.Name.Name == "clearFuncs" && lhs != nil && rhs.Value == "0" {
							cleared[lhs.Name] = true
						}
					case *ast.CallExpr:
						callee, ok := rhs.Fun.(*ast.Ident)
						if fn.Name.Name != "registerFuncs" || lhs == nil || !ok {
							break
						}
						var symbols []string
						switch {
						case callee.Name == "registerSym" && len(rhs.Args) == 1:
							symbol, err := strconv.Unquote(rhs.Args[0].(*ast.BasicLit).Value)
							if err != nil {
								t.Fatal(err)
							}
							symbols = []string{symbol}
						case callee.Name == "registerSymChain" && len(rhs.Args) == 1:
							if addr, ok := rhs.Args[0].(*ast.UnaryExpr); ok {
								if name, ok := addr.X.(*ast.Ident); ok {
									symbols = chains[name.Name]
								}
							}
						}
						if symbols != nil {
							if _, exists := registered[lhs.Name]; exists {
								t.Errorf("%s registered more than once", lhs.Name)
							}
							registered[lhs.Name] = symbols
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
					registered[name] = []string{symbol}
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
	for name, registeredSymbols := range registered {
		if !declared[name] {
			t.Errorf("registered pointer %s is not declared", name)
		}
		for _, symbol := range registeredSymbols {
			symbols[symbol] = true
		}
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

func symbolChainNames(expr ast.Expr) []string {
	lit, ok := expr.(*ast.CompositeLit)
	if !ok {
		return nil
	}
	typ, ok := lit.Type.(*ast.Ident)
	if !ok || typ.Name != "symbolChain" {
		return nil
	}
	for _, elt := range lit.Elts {
		field, ok := elt.(*ast.KeyValueExpr)
		if !ok {
			continue
		}
		key, ok := field.Key.(*ast.Ident)
		if !ok || key.Name != "symbols" {
			continue
		}
		list, ok := field.Value.(*ast.CompositeLit)
		if !ok {
			return nil
		}
		var names []string
		for _, item := range list.Elts {
			value, ok := item.(*ast.BasicLit)
			if !ok || value.Kind != token.STRING {
				return nil
			}
			name, err := strconv.Unquote(value.Value)
			if err != nil {
				return nil
			}
			names = append(names, name)
		}
		return names
	}
	return nil
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
	_, err := CtxGetSharedMemConfig()
	var symErr *SymbolError
	if !errors.As(err, &symErr) || symErr.Symbol != "cuCtxGetSharedMemConfig" {
		t.Fatalf("error = %v, want SymbolError for cuCtxGetSharedMemConfig", err)
	}
}

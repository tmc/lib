package cudakernel_test

import (
	"bytes"
	"errors"
	"go/ast"
	"go/format"
	"go/parser"
	"go/token"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudatest"
)

func TestAffineModuleOwner(t *testing.T) {
	loadErr := errors.New("load")
	enterErr := errors.New("enter context")
	unloadErr := errors.New("unload")
	restoreErr := errors.New("restore context")
	for _, tt := range []struct {
		name       string
		module     driver.Module
		loadErr    error
		enterErr   error
		unloadErr  error
		restoreErr error
	}{
		{name: "zero error", loadErr: loadErr},
		{name: "partial error", module: 7, loadErr: loadErr},
		{name: "success", module: 7},
		{name: "unload retry", module: 7, unloadErr: unloadErr},
		{name: "enter retry", module: 7, enterErr: enterErr},
		{name: "restore retry", module: 7, restoreErr: restoreErr},
		{name: "partial and cleanup error", module: 7, loadErr: loadErr, unloadErr: unloadErr},
	} {
		t.Run(tt.name, func(t *testing.T) {
			var owner cudatest.DeviceOwner
			var registered, contexts, unloads, releases int
			owner.Defer(func() error {
				releases++
				return nil
			})
			err := affineLoadModule(func(close func() error) {
				registered++
				owner.Defer(close)
			}, func() (driver.Module, error) {
				return tt.module, tt.loadErr
			}, func(body func() error) error {
				contexts++
				if contexts == 1 && tt.enterErr != nil {
					return tt.enterErr
				}
				if err := body(); err != nil {
					return err
				}
				if contexts == 1 {
					return tt.restoreErr
				}
				return nil
			}, func(mod *driver.Module) error {
				unloads++
				if *mod != tt.module {
					t.Errorf("unload module = %v, want %v", *mod, tt.module)
				}
				if unloads == 1 {
					return tt.unloadErr
				}
				return nil
			})
			if err != tt.loadErr {
				t.Fatalf("load error = %v, want identical %v", err, tt.loadErr)
			}
			wantRegistered := 0
			if tt.module != 0 {
				wantRegistered = 1
			}
			if registered != wantRegistered || contexts != 0 || unloads != 0 || releases != 0 {
				t.Fatalf("before Close: registered/contexts/unloads/releases = %d/%d/%d/%d, want %d/0/0/0", registered, contexts, unloads, releases, wantRegistered)
			}
			wantErr := errors.Join(tt.enterErr, tt.unloadErr, tt.restoreErr)
			err = owner.Close()
			if wantErr == nil {
				if err != nil {
					t.Fatalf("Close: %v", err)
				}
			} else {
				for _, cause := range []error{tt.enterErr, tt.unloadErr, tt.restoreErr} {
					if cause != nil && !errors.Is(err, cause) {
						t.Errorf("Close error = %v, want identity %v", err, cause)
					}
				}
				if releases != 0 {
					t.Fatalf("dependent released after failed cleanup")
				}
			}
			if err := owner.Close(); err != nil {
				t.Fatalf("retry Close: %v", err)
			}
			if err := owner.Close(); err != nil {
				t.Fatalf("repeated Close: %v", err)
			}
			wantContexts, wantUnloads := 1, 1
			if tt.module == 0 {
				wantContexts, wantUnloads = 0, 0
			} else if tt.enterErr != nil {
				wantContexts = 2
			} else if tt.unloadErr != nil {
				wantContexts, wantUnloads = 2, 2
			}
			if contexts != wantContexts || unloads != wantUnloads || releases != 1 {
				t.Errorf("final contexts/unloads/releases = %d/%d/%d, want %d/%d/1", contexts, unloads, releases, wantContexts, wantUnloads)
			}
		})
	}
}

// This binds the CPU-tested helper to the actual gated native caller.
func TestAffineNativeOwnershipSource(t *testing.T) {
	fs := token.NewFileSet()
	f, err := parser.ParseFile(fs, "affine_test.go", nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	var body *ast.BlockStmt
	for _, decl := range f.Decls {
		if fn, ok := decl.(*ast.FuncDecl); ok && fn.Name.Name == "TestNativeAffineArtifact" {
			body = fn.Body
		}
	}
	if body == nil {
		t.Fatal("missing native fixture")
	}
	text := func(n ast.Node) string {
		var b bytes.Buffer
		if err := format.Node(&b, fs, n); err != nil {
			t.Fatal(err)
		}
		return b.String()
	}
	var admission, owned, load token.Pos
	ast.Inspect(body, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}
		switch text(call.Fun) {
		case "affineNativeAdmission":
			admission = call.Pos()
		case "cuda.NewDevice", "dev.Close":
			t.Error("native caller bypasses DeviceOwner")
		}
		return true
	})
	for _, stmt := range body.List {
		switch stmt := stmt.(type) {
		case *ast.AssignStmt:
			if len(stmt.Rhs) != 1 {
				continue
			}
			call, ok := stmt.Rhs[0].(*ast.CallExpr)
			if !ok || text(call.Fun) != "cudatest.RequireOwned" {
				continue
			}
			if len(stmt.Lhs) != 2 || text(stmt.Lhs[0]) != "dev" || text(stmt.Lhs[1]) != "owner" {
				t.Fatal("RequireOwned result is not dev, owner")
			}
			owned = call.Pos()
		case *ast.IfStmt:
			init, ok := stmt.Init.(*ast.AssignStmt)
			if !ok || len(init.Rhs) != 1 {
				continue
			}
			call, ok := init.Rhs[0].(*ast.CallExpr)
			if !ok || text(call.Fun) != "affineLoadModule" {
				continue
			}
			load = call.Pos()
			if text(call) != `affineLoadModule(owner.Defer, func() (driver.Module, error) {
	return art.Load(dev)
}, dev.WithContext, (*driver.Module).Close)` {
				t.Fatalf("module helper is not bound to the actual owner/artifact/context/unload: %s", text(call))
			}
			if text(stmt.Cond) != "err != nil" || len(stmt.Body.List) != 1 || text(stmt.Body.List[0]) != `t.Fatalf("art.Load on device: %v", err)` {
				t.Fatal("load error is not reported after module registration")
			}
		}
	}
	if admission == 0 || owned <= admission || load <= owned {
		t.Fatalf("native admission/owned setup/load order = %v/%v/%v", admission, owned, load)
	}
}

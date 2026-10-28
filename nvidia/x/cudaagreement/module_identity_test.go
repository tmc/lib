package cudaagreement

import (
	"errors"
	"go/ast"
	"go/parser"
	"go/token"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudakernel"
)

func moduleIdentityArtifact(t *testing.T, entry string) *cudakernel.Artifact {
	t.Helper()
	a, err := cudakernel.NewArtifact(cudakernel.KindPTX, []byte("host-only "+entry), entry, cudakernel.Schema{}, "sm_80")
	if err != nil {
		t.Fatal(err)
	}
	return a
}

func TestModuleIdentityCapturedByValue(t *testing.T) {
	first := moduleIdentityArtifact(t, "first")
	second := moduleIdentityArtifact(t, "second")
	firstHandle := *first
	cache := NewModules(nil)
	entries := make(map[*cuda.Module]string)
	loads := 0
	var loading string
	ops := moduleFunctionOps{
		load: func(a cudakernel.Artifact) (driver.Module, error) {
			loads++
			loading = a.EntryPoint()
			return driver.Module(loads), nil
		},
		own: func(driver.Module) (*cuda.Module, error) {
			mod := &cuda.Module{}
			entries[mod] = loading
			return mod, nil
		},
		function: func(mod *cuda.Module, entry string) (*cuda.Kernel, error) {
			if entries[mod] != entry {
				t.Fatalf("module for %q used for %q", entries[mod], entry)
			}
			return nil, nil
		},
	}
	if _, err := cache.function(first, ops); err != nil {
		t.Fatal(err)
	}
	// Replacing a caller wrapper must load a new image, even though the
	// address passed to Function is unchanged.
	*first = *second
	if _, err := cache.function(first, ops); err != nil {
		t.Fatal(err)
	}
	// A different wrapper for an already captured handle must reuse its image.
	copy := firstHandle
	if _, err := cache.function(&copy, ops); err != nil {
		t.Fatal(err)
	}
	// Equal bytes from an independent construction are a distinct identity.
	// A digest-only cache key would incorrectly coalesce these owners.
	sameBytes := moduleIdentityArtifact(t, "first")
	if *sameBytes == firstHandle || sameBytes.Digest() != firstHandle.Digest() {
		t.Fatal("independent identity precondition changed")
	}
	if _, err := cache.function(sameBytes, ops); err != nil {
		t.Fatal(err)
	}
	if loads != 3 || len(cache.loaded) != 3 || len(cache.pending) != 0 {
		t.Fatalf("loads=%d cached=%d pending=%v", loads, len(cache.loaded), cache.pending)
	}
}

func TestModuleIdentityCapturedBeforeLoad(t *testing.T) {
	wrapper := moduleIdentityArtifact(t, "before")
	original := *wrapper
	replacement := moduleIdentityArtifact(t, "after")
	cache := NewModules(nil)
	owner := &cuda.Module{}
	loads := 0
	ops := moduleFunctionOps{
		load: func(a cudakernel.Artifact) (driver.Module, error) {
			loads++
			if a != original {
				t.Fatal("load received replacement identity")
			}
			// Sequential re-entry from the loader is not concurrent mutation.
			*wrapper = *replacement
			return 1, nil
		},
		own: func(driver.Module) (*cuda.Module, error) { return owner, nil },
		function: func(mod *cuda.Module, entry string) (*cuda.Kernel, error) {
			if mod != owner || entry != "before" {
				t.Fatalf("lookup lost captured image: owner=%p entry=%q", mod, entry)
			}
			return nil, nil
		},
	}
	if _, err := cache.function(wrapper, ops); err != nil {
		t.Fatal(err)
	}
	if _, err := cache.function(&original, ops); err != nil {
		t.Fatal(err)
	}
	if loads != 1 || len(cache.loaded) != 1 || len(cache.pending) != 0 {
		t.Fatalf("loads=%d cached=%d pending=%v", loads, len(cache.loaded), cache.pending)
	}
}

func TestModuleIdentityFailureRetainsPendingOwner(t *testing.T) {
	cause := errors.New("module ownership failure")
	for _, stage := range []string{"load", "own"} {
		t.Run(stage, func(t *testing.T) {
			cache := NewModules(nil)
			art := moduleIdentityArtifact(t, "failed")
			ops := moduleFunctionOps{
				load: func(cudakernel.Artifact) (driver.Module, error) {
					if stage == "load" {
						return 7, cause
					}
					return 7, nil
				},
				own: func(driver.Module) (*cuda.Module, error) {
					if stage != "own" {
						t.Fatal("adopted failed load")
					}
					return nil, cause
				},
				function: func(*cuda.Module, string) (*cuda.Kernel, error) {
					t.Fatal("lookup reached after failed ownership")
					return nil, nil
				},
			}
			if _, err := cache.function(art, ops); !errors.Is(err, cause) {
				t.Fatalf("lost primary error: %v", err)
			}
			if len(cache.pending) != 1 || cache.pending[0] != 7 || len(cache.loaded) != 0 {
				t.Fatalf("lost pending owner: %v, cache=%v", cache.pending, cache.loaded)
			}
		})
	}
}

// The runtime controls above use the same private ownership path as Function.
// This guard rejects moving production back to a separate untested path.
func TestModuleIdentityPublicPath(t *testing.T) {
	file, err := parser.ParseFile(token.NewFileSet(), "executor.go", nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	for _, decl := range file.Decls {
		fn, ok := decl.(*ast.FuncDecl)
		if !ok || fn.Name.Name != "Function" || fn.Recv == nil {
			continue
		}
		ret, ok := fn.Body.List[len(fn.Body.List)-1].(*ast.ReturnStmt)
		if !ok || len(ret.Results) != 1 {
			t.Fatal("Function does not return the ownership path")
		}
		call, ok := ret.Results[0].(*ast.CallExpr)
		if !ok || len(call.Args) != 2 {
			t.Fatal("Function does not call the ownership path")
		}
		sel, ok := call.Fun.(*ast.SelectorExpr)
		if !ok || sel.Sel.Name != "function" {
			t.Fatal("Function bypassed the ownership path")
		}
		arg, ok := call.Args[0].(*ast.Ident)
		if !ok || arg.Name != "art" {
			t.Fatal("Function replaced the input wrapper")
		}
		ops, ok := call.Args[1].(*ast.CompositeLit)
		if !ok || len(ops.Elts) != 3 {
			t.Fatal("Function lost its three real operations")
		}
		want := map[string]string{"load": "Load", "own": "OwnModule", "function": "Function"}
		for _, elt := range ops.Elts {
			kv, ok := elt.(*ast.KeyValueExpr)
			if !ok {
				t.Fatal("operation is not keyed")
			}
			key, ok := kv.Key.(*ast.Ident)
			if !ok {
				t.Fatal("operation has no name")
			}
			lit, ok := kv.Value.(*ast.FuncLit)
			if !ok || len(lit.Body.List) != 1 {
				t.Fatal("operation is not a direct binding")
			}
			binding, ok := lit.Body.List[0].(*ast.ReturnStmt)
			if !ok || len(binding.Results) != 1 {
				t.Fatal("operation does not return its real call")
			}
			realCall, ok := binding.Results[0].(*ast.CallExpr)
			if !ok {
				t.Fatal("operation has no real call")
			}
			realSelector, ok := realCall.Fun.(*ast.SelectorExpr)
			if !ok || want[key.Name] == "" || realSelector.Sel.Name != want[key.Name] {
				t.Fatalf("operation %s is not bound to its real implementation", key.Name)
			}
			delete(want, key.Name)
		}
		if len(want) != 0 {
			t.Fatalf("missing operations: %v", want)
		}
		return
	}
	t.Fatal("public Modules.Function not found")
}

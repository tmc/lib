//go:build native && linux

package cudasearch

import (
	"context"
	"crypto/sha256"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

func memoryNativeSelectPTX(compiled *nvrtcx.CompiledModule) ([]byte, string, nvrtcx.OutputKind, error) {
	if compiled == nil {
		return nil, "", nvrtcx.OutputPTX, fmt.Errorf("missing compiled PTX")
	}
	image := []byte(compiled.PTX())
	if len(image) == 0 || len(image) > 8<<20 {
		return nil, "", nvrtcx.OutputPTX, fmt.Errorf("selected PTX unavailable or excessive")
	}
	return image, fmt.Sprintf("%x", sha256.Sum256(image)), nvrtcx.OutputPTX, nil
}

func TestMemoryNativeSelectorCPU(t *testing.T) {
	if os.Getenv("LANE46_SELECTOR_CPU") != "1" {
		t.Skip("CPU NVRTC compiler control opt-in; no GPU")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	compiled, err := nvrtcx.CompileContext(ctx, memoryNativeSource, nvrtcx.WithArch(12, 1), nvrtcx.WithFlag("--fmad=false"), nvrtcx.WithNoCache())
	if err != nil {
		t.Fatal(err)
	}
	image, digest, kind, err := memoryNativeSelectPTX(compiled)
	if err != nil || kind != nvrtcx.OutputPTX || string(image) != compiled.PTX() || digest != memoryNativeDigest(image) {
		t.Fatalf("actual producer PTX selection: %v", err)
	}
	image[0] ^= 1
	if string(image) == compiled.PTX() {
		t.Fatal("selection did not detach bytes")
	}
	if _, _, _, err := memoryNativeSelectPTX(nil); err == nil {
		t.Fatal("missing PTX accepted")
	}
	lto, err := nvrtcx.CompileContext(ctx, memoryNativeSource, nvrtcx.WithArch(12, 1), nvrtcx.WithOutput(nvrtcx.OutputLTOIR), nvrtcx.WithFlag("--fmad=false"), nvrtcx.WithNoCache())
	if err != nil {
		t.Fatal(err)
	}
	if len(lto.LTOIR()) == 0 || lto.PTX() != "" {
		t.Fatal("actual alternate-only compiler output missing")
	}
	if _, _, _, err := memoryNativeSelectPTX(lto); err == nil {
		t.Fatal("non-PTX output substituted")
	}
}

func TestMemoryNativeSelectedLoadSource(t *testing.T) {
	if os.Getenv("LANE46_SELECTOR_CPU") != "1" {
		t.Skip("owned source control opt-in")
	}
	path := os.Getenv("LANE46_SELECTOR_SOURCE")
	if path == "" {
		t.Fatal("exact owned producer source required")
	}
	source, err := parser.ParseFile(token.NewFileSet(), path, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	var loads int
	ast.Inspect(source, func(node ast.Node) bool {
		call, ok := node.(*ast.CallExpr)
		if !ok {
			return true
		}
		sel, ok := call.Fun.(*ast.SelectorExpr)
		if !ok {
			return true
		}
		recv, ok := sel.X.(*ast.Ident)
		if !ok || recv.Name != "compiled" {
			return true
		}
		if sel.Sel.Name == "Load" {
			t.Fatal("ambiguous auto loader restored")
		}
		if sel.Sel.Name == "LoadOutput" {
			loads++
			if len(call.Args) != 2 {
				t.Fatal("selected load arguments")
			}
			id, ok := call.Args[1].(*ast.Ident)
			if !ok || id.Name != "selected" {
				t.Fatal("selected PTX kind bypassed")
			}
		}
		return true
	})
	if loads != 1 {
		t.Fatal("exact selected load missing")
	}
}

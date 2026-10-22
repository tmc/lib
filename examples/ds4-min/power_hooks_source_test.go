package main

import (
	"go/ast"
	"go/parser"
	"go/token"
	"io"
	"os"
	"strings"
	"testing"
)

// These controls check owning scheduling seams on either host. They do not
// execute a driver, command buffer, model, or production clock.
func powerMethodCalls(source, method string) (map[string]bool, error) {
	f, err := parser.ParseFile(token.NewFileSet(), "source.go", source, 0)
	if err != nil {
		return nil, err
	}
	calls := make(map[string]bool)
	for _, decl := range f.Decls {
		fn, ok := decl.(*ast.FuncDecl)
		if !ok || fn.Name.Name != method {
			continue
		}
		ast.Inspect(fn.Body, func(n ast.Node) bool {
			call, ok := n.(*ast.CallExpr)
			if !ok {
				return true
			}
			selector, ok := call.Fun.(*ast.SelectorExpr)
			if ok {
				calls[selector.Sel.Name] = true
			}
			return true
		})
	}
	return calls, nil
}

func TestPowerRuntimeHooks(t *testing.T) {
	for _, tt := range []struct {
		file, method string
		want         []string
	}{
		{"cuda_session_linux.go", "evalPosition", []string{"start", "finishLayer", "finishToken"}},
		{"cuda_session_linux.go", "evalBatchReadout", []string{"start", "finishLayer"}},
		{"cuda_state_adapter_linux.go", "newIndependentState", []string{"newRequest"}},
		{"metal_runtime_darwin.go", "evalPosition", []string{"start", "finishLayer", "finishToken"}},
		{"metal_streaming_prefill_darwin.go", "evalStreamingPrefill", []string{"start", "finishLayer"}},
		{"metal_state_adapter_darwin.go", "newIndependentState", []string{"newRequest"}},
	} {
		source, err := os.ReadFile(tt.file)
		if err != nil {
			t.Fatal(err)
		}
		calls, err := powerMethodCalls(string(source), tt.method)
		if err != nil {
			t.Fatal(err)
		}
		for _, want := range tt.want {
			if !calls[want] {
				t.Errorf("%s.%s missing %s", tt.file, tt.method, want)
			}
		}
	}
	source, err := os.ReadFile("cuda_session_linux.go")
	if err != nil {
		t.Fatal(err)
	}
	needle := "r.power.finishToken(started)"
	if strings.Count(string(source), needle) != 1 {
		t.Fatal("decode pacing mutation seam")
	}
	mutant := strings.Replace(string(source), needle, "", 1)
	calls, err := powerMethodCalls(mutant, "evalPosition")
	if err != nil {
		t.Fatal(err)
	}
	if calls["finishToken"] {
		t.Fatal("disabled decode pacing passed source gate")
	}
}

func TestPowerCLI(t *testing.T) {
	for _, value := range []string{"0", "101", "-1"} {
		_, err := parseConfig([]string{"-model", "fixture.gguf", "-inspect", "-power", value}, strings.NewReader(""), io.Discard)
		if err == nil || !strings.Contains(err.Error(), "power") {
			t.Fatalf("invalid power %s: %v", value, err)
		}
	}
	cfg, err := parseConfig([]string{"-model", "fixture.gguf", "-inspect", "-power", "25"}, strings.NewReader(""), io.Discard)
	if err != nil || cfg.power != 25 {
		t.Fatalf("power flag %d: %v", cfg.power, err)
	}
}

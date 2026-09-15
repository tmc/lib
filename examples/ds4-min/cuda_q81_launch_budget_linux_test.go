//go:build linux

package main

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// The batched prefill front end exists so that one pass over a weight matrix
// serves a whole group of tokens. Every unbatched ds4_q8_0_matvec_q8_1 launch
// left on the prefill path costs a second full read of its weights, and a
// reverted call site is invisible in the outputs: they stay bit-identical and
// only device time moves. These tests read the routing source so that returning
// a call site to the per-token encoder fails here, without a GPU.
//
// The census that fixes the budget (2026-09-15, prefill of 2048 tokens) measured
// 86.00 ds4_q8_0_matvec_q8_1 launches per prefill token. That is exactly two per
// layer over cudaLayers layers: l.sharedGate and l.sharedUp, the only unbatched
// callers the prefill path reached.

// cudaSessionSource is the file that owns prefill routing.
const cudaSessionSource = "cuda_session_linux.go"

// cudaQ81UnbatchedCallers names every function that may call the unbatched Q8_1
// matvec encoder. evalLayerQKV and evalLayerFFNShared serve the single-token
// path; encodeQ80Matvec is the generic dense route.
var cudaQ81UnbatchedCallers = []string{"encodeQ80Matvec", "evalLayerFFNShared", "evalLayerQKV"}

// cudaQ81BatchTensors is the group-amortized weight list, in launch order.
var cudaQ81BatchTensors = []string{"l.attnQA", "l.attnQB", "l.attnKV", "l.sharedGate", "l.sharedUp"}

// cudaSessionFunctions indexes the routing source by function name. Names repeat
// across receivers, so every declaration is kept and the callee queries take the
// union: a duplicate can only raise a launch count, never hide one.
func cudaSessionFunctions(t *testing.T) map[string][]*ast.FuncDecl {
	t.Helper()
	file, err := parser.ParseFile(token.NewFileSet(), cudaSessionSource, nil, 0)
	if err != nil {
		t.Fatalf("parse %s: %v", cudaSessionSource, err)
	}
	functions := make(map[string][]*ast.FuncDecl)
	for _, decl := range file.Decls {
		if fn, ok := decl.(*ast.FuncDecl); ok {
			functions[fn.Name.Name] = append(functions[fn.Name.Name], fn)
		}
	}
	for _, name := range []string{"evalLayerBatch", "evalLayerFFN", "evalLayerFFNShared"} {
		if len(functions[name]) != 1 {
			t.Fatalf("%s: %s has %d declarations, want 1", cudaSessionSource, name, len(functions[name]))
		}
	}
	return functions
}

// cudaCallees returns the names called directly by any of decls, both
// r.method(...) and plain call(...) forms, without duplicates.
func cudaCallees(decls []*ast.FuncDecl) []string {
	var names []string
	for _, fn := range decls {
		ast.Inspect(fn, func(n ast.Node) bool {
			if name, ok := cudaCalledName(n); ok && !slices.Contains(names, name) {
				names = append(names, name)
			}
			return true
		})
	}
	return names
}

// cudaCallCount counts direct calls to name across decls, repeats included.
func cudaCallCount(decls []*ast.FuncDecl, name string) int {
	count := 0
	for _, fn := range decls {
		ast.Inspect(fn, func(n ast.Node) bool {
			if called, ok := cudaCalledName(n); ok && called == name {
				count++
			}
			return true
		})
	}
	return count
}

func cudaCalledName(n ast.Node) (string, bool) {
	call, ok := n.(*ast.CallExpr)
	if !ok {
		return "", false
	}
	switch target := call.Fun.(type) {
	case *ast.SelectorExpr:
		return target.Sel.Name, true
	case *ast.Ident:
		return target.Name, true
	}
	return "", false
}

// cudaReachable returns the names reachable from root within the routing source,
// root excluded.
func cudaReachable(functions map[string][]*ast.FuncDecl, root string) []string {
	var order []string
	seen := map[string]bool{root: true}
	queue := []string{root}
	for len(queue) > 0 {
		name := queue[0]
		queue = queue[1:]
		for _, callee := range cudaCallees(functions[name]) {
			if seen[callee] {
				continue
			}
			seen[callee] = true
			order = append(order, callee)
			queue = append(queue, callee)
		}
	}
	slices.Sort(order)
	return order
}

func cudaCallersOf(functions map[string][]*ast.FuncDecl, callee string) []string {
	var callers []string
	for name, decls := range functions {
		if slices.Contains(cudaCallees(decls), callee) {
			callers = append(callers, name)
		}
	}
	slices.Sort(callers)
	return callers
}

// TestCUDAPrefillQ81LaunchBudget pins the prefill path's per-token count of
// unbatched Q8_1 matvec launches at zero and the group's batched count at one
// launch per shared weight.
func TestCUDAPrefillQ81LaunchBudget(t *testing.T) {
	functions := cudaSessionFunctions(t)

	callers := cudaCallersOf(functions, "encodeQ80MatvecQ81")
	if !slices.Equal(callers, cudaQ81UnbatchedCallers) {
		t.Errorf("unbatched Q8_1 matvec callers %v, want %v", callers, cudaQ81UnbatchedCallers)
	}

	// The prefill front end must reach none of the single-token callers. It
	// reaches encodeQ80Matvec only through compress, for the optional F16-or-Q8_0
	// indexer query, which sits inside the token-ordered compressor region.
	reachable := cudaReachable(functions, "evalLayerBatch")
	for _, name := range []string{"evalLayerQKV", "evalLayerFFNShared", "evalLayerFFN", "evalLayerProjected"} {
		if slices.Contains(reachable, name) {
			t.Errorf("evalLayerBatch reaches %s: prefill has returned to the per-token shared FFN", name)
		}
	}
	for _, name := range []string{"evalLayerAttention", "evalLayerFFNPre", "evalLayerFFNPost"} {
		if !slices.Contains(reachable, name) {
			t.Errorf("evalLayerBatch no longer reaches %s", name)
		}
	}
	if slices.Contains(reachable, "encodeQ80Matvec") {
		if indexer := cudaCallersOf(functions, "encodeQ80Matvec"); !slices.Contains(indexer, "compress") {
			t.Errorf("evalLayerBatch reaches encodeQ80Matvec from %v, want only compress", indexer)
		}
	}

	// Decode keeps the single-token shared projection, two launches per layer.
	if n := cudaCallCount(functions["evalLayerFFN"], "evalLayerFFNShared"); n != 1 {
		t.Errorf("evalLayerFFN calls evalLayerFFNShared %d times, want 1", n)
	}
	if n := cudaCallCount(functions["evalLayerFFNShared"], "encodeQ80MatvecQ81"); n != 2 {
		t.Fatalf("evalLayerFFNShared encodes %d unbatched Q8_1 matvecs, want 2 (shared gate, shared up)", n)
	}

	batched := cudaQ81BatchWeights(t, functions["evalLayerBatch"])
	if !slices.Equal(batched, cudaQ81BatchTensors) {
		t.Fatalf("evalLayerBatch batches %v, want %v", batched, cudaQ81BatchTensors)
	}

	// Count what the prefill path still encodes per token per layer, from the
	// source rather than from a constant, so restoring a call site raises the
	// count instead of leaving the budget vacuously true. encodeQ80Matvec is
	// excluded and audited above.
	unbatchedPerLayer := 0
	for _, name := range append([]string{"evalLayerBatch"}, reachable...) {
		if name == "encodeQ80Matvec" {
			continue
		}
		unbatchedPerLayer += cudaCallCount(functions[name], "encodeQ80MatvecQ81")
	}
	if unbatchedPerLayer != 0 {
		t.Errorf("prefill encodes %d unbatched Q8_1 matvecs per token per layer, want 0", unbatchedPerLayer)
	}

	// The census measured 86.00 unbatched launches per prefill token: the two
	// shared projections over every layer. Batching them at the group width
	// leaves len(cudaQ81BatchTensors) launches per layer per group.
	if eliminated := 2 * cudaLayers; eliminated != 86 {
		t.Errorf("the shared FFN is %d launches per token but the census pinned 86.00; re-measure before trusting this budget", eliminated)
	}
	if per := float64(len(cudaQ81BatchTensors)*cudaLayers) / float64(cudaPrefillBatchWidth); per != 26.875 {
		t.Errorf("batched Q8_1 launches per prefill token %g, want 26.875", per)
	}
}

// cudaQ81BatchWeights returns the weight expression of each direct
// encodeQ80MatvecQ81Batch call in decls, in source order.
func cudaQ81BatchWeights(t *testing.T, decls []*ast.FuncDecl) []string {
	t.Helper()
	var weights []string
	for _, fn := range decls {
		ast.Inspect(fn, func(n ast.Node) bool {
			call, ok := n.(*ast.CallExpr)
			if !ok {
				return true
			}
			target, ok := call.Fun.(*ast.SelectorExpr)
			if !ok || target.Sel.Name != "encodeQ80MatvecQ81Batch" || len(call.Args) == 0 {
				return true
			}
			arg, ok := call.Args[0].(*ast.SelectorExpr)
			if !ok {
				t.Fatalf("encodeQ80MatvecQ81Batch weight argument is not a field selector")
			}
			base, ok := arg.X.(*ast.Ident)
			if !ok {
				t.Fatalf("encodeQ80MatvecQ81Batch weight argument has no simple base")
			}
			weights = append(weights, base.Name+"."+arg.Sel.Name)
			return true
		})
	}
	return weights
}

// TestCUDAQ81UnbatchedEncoderIsConfined keeps the unbatched encoder from
// acquiring a caller in a file the launch budget does not read.
func TestCUDAQ81UnbatchedEncoderIsConfined(t *testing.T) {
	names, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatalf("glob: %v", err)
	}
	want := []string{"cuda_q81_linux.go", cudaSessionSource}
	slices.Sort(want)
	var found []string
	for _, name := range names {
		if strings.HasSuffix(name, "_test.go") {
			continue
		}
		data, err := os.ReadFile(name)
		if err != nil {
			t.Fatalf("read %s: %v", name, err)
		}
		if strings.Contains(string(data), "encodeQ80MatvecQ81(") {
			found = append(found, name)
		}
	}
	slices.Sort(found)
	if !slices.Equal(found, want) {
		t.Errorf("files naming the unbatched Q8_1 matvec encoder %v, want %v", found, want)
	}
}

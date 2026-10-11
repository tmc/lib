package egraph

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaref"
)

func sourceArtifactFixture() Source {
	return Source{CUDA: "// source\n__device__ float eg_c4_add() {\n    return 1;\n}\n", Expression: &Expr{Class: 4, Origin: "rule-add", Op: "add", Children: []*Expr{{Class: 0, Origin: "input", Op: "a", Data: "a"}, {Class: 1, Origin: "input", Op: "constant", Data: float32(1)}}}, Symbols: map[string]Symbol{"eg_c4_add": {Class: 4, Operator: "add", Origin: "rule-add", FirstLine: 2, LastLine: 4}}, Parameters: []Parameter{{Name: "a", Shape: []int{8}}}, CostModel: "AstSizeCost", Cost: 3}
}

func TestSourceSidecarRoundTripAndExpressionOracle(t *testing.T) {
	s := sourceArtifactFixture()
	path := filepath.Join(t.TempDir(), "chain.cu")
	if err := s.WriteFile(path); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(filepath.Join(filepath.Dir(path), "chain.egraph.json"))
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := ReadMetadata(bytes.NewReader(data), s.CUDA)
	if err != nil {
		t.Fatal(err)
	}
	if decoded.Expression.Fingerprint() != s.Expression.Fingerprint() || decoded.Expression.Children[1].Data != float32(1) {
		t.Fatal("sidecar changed expression types")
	}
	for _, line := range []int{2, 3, 4} {
		if c, ok := decoded.ClassAt(line); !ok || c != 4 {
			t.Fatal(line, c, ok)
		}
	}
	for _, line := range []int{-1, 0, 1, 5} {
		if _, ok := decoded.ClassAt(line); ok {
			t.Fatal("outer line attributed", line)
		}
	}
	// Editing the source changes neither expression nor parameter order. Its
	// original line table is explicitly invalidated rather than misattributed.
	edited, err := ReadMetadata(bytes.NewReader(data), "// edit\n"+s.CUDA)
	if err != nil {
		t.Fatal(err)
	}
	if edited.Expression.Fingerprint() != decoded.Expression.Fingerprint() || len(edited.Symbols) != 0 {
		t.Fatal("edited source changed reference or retained stale lines")
	}
	var raw map[string]any
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatal(err)
	}
	raw["Fingerprint"] = "forged"
	bad, _ := json.Marshal(raw)
	if _, err := ReadMetadata(bytes.NewReader(bad), s.CUDA); err == nil {
		t.Fatal("forged expression fingerprint accepted")
	}
}

func TestSourceMetadataRefusalsBeforeWriting(t *testing.T) {
	s := sourceArtifactFixture()
	dir := t.TempDir()
	path := filepath.Join(dir, "bad.cu")
	s.Expression.Data = make(chan int)
	if err := s.WriteFile(path); err == nil {
		t.Fatal("unsupported data encoded")
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatal("invalid source created before validation", err)
	}
	s = sourceArtifactFixture()
	s.Symbols["overlap"] = Symbol{Class: 2, FirstLine: 3, LastLine: 4}
	if err := s.WriteFile(path); err == nil {
		t.Fatal("overlapping source symbols accepted")
	}
	if _, err := ReadMetadata(nil, ""); err == nil {
		t.Fatal("nil reader accepted")
	}
	if _, err := ReadMetadata(bytes.NewReader([]byte("{} {}")), ""); err == nil {
		t.Fatal("trailing sidecar accepted")
	}
}

func ExampleSource_ClassAt() {
	// See TestSourceSidecarRoundTripAndExpressionOracle for the file round-trip.
	s := Source{Symbols: map[string]Symbol{"eg_c7_silu": {Class: 7, FirstLine: 2, LastLine: 5}}}
	class, ok := s.ClassAt(3)
	fmt.Println(class, ok)
	// Output: 7 true
}

func FuzzReadSourceMetadata(f *testing.F) {
	f.Add([]byte(`{"Version":1}`))
	f.Fuzz(func(t *testing.T, data []byte) {
		if len(data) > 32<<10 {
			return
		}
		ReadMetadata(bytes.NewReader(data), "// source\n")
	})
}

func TestReferenceGraphElementwiseBar(t *testing.T) {
	a := &Expr{Class: 1, Origin: "input", Op: "a", Data: "a"}
	b := &Expr{Class: 2, Origin: "input", Op: "b", Data: "b"}
	root := &Expr{Class: 5, Origin: "input", Op: "add", Children: []*Expr{{Class: 3, Origin: "input", Op: "silu", Children: []*Expr{a}}, {Class: 4, Origin: "input", Op: "mul", Children: []*Expr{a, b}}}}
	s := Source{Expression: root, Parameters: []Parameter{{Name: "a", Shape: []int{3}}, {Name: "b", Shape: []int{3}}}}
	input := map[string][]float32{"a": {-2, 0, 3}, "b": {4, 5, -1}}
	graph, err := s.ReferenceGraph()
	if err != nil {
		t.Fatal(err)
	}
	values, err := cudaref.Evaluate(graph, input)
	if err != nil {
		t.Fatal(err)
	}
	for i, a := range input["a"] {
		want := float32(float64(a)/(1+math.Exp(-float64(a)))) + a*input["b"][i]
		if values["result"][i] != want {
			t.Fatal(values, want)
		}
	}
	// Reference construction never reads or interprets the edited CUDA.
	s.CUDA = "extern wrong_source_with_no_expression() {}"
	graph, err = s.ReferenceGraph()
	if err != nil {
		t.Fatal(err)
	}
	again, err := cudaref.Evaluate(graph, input)
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range values["result"] {
		if again["result"][i] != v {
			t.Fatal("source edit changed expression oracle")
		}
	}
}

func TestReferenceGraphNormGemvAndFused(t *testing.T) {
	x := &Expr{Op: "x", Data: "x"}
	w := &Expr{Op: "w", Data: "w"}
	eps := &Expr{Op: "0.25", Data: float32(.25)}
	matrix := &Expr{Op: "matrix", Data: "matrix"}
	normal := &Expr{Op: "rmsnorm", Children: []*Expr{x, w, eps}}
	input := map[string][]float32{"x": {1, 2}, "w": {1, 2}, "matrix": {1, 0, 0, 1, 1, 1}}
	scale := float32(1) / float32(math.Sqrt(2.75))
	u := float32(1) * scale
	v := (float32(2) * scale) * 2
	for _, root := range []*Expr{{Op: "gemv", Children: []*Expr{normal, matrix}}, {Op: "fused_rmsnorm_gemv", Children: []*Expr{x, w, eps, matrix}}} {
		s := Source{Expression: root, Parameters: []Parameter{{Name: "x", Shape: []int{2}}, {Name: "w", Shape: []int{2}}, {Name: "matrix", Shape: []int{3, 2}}}}
		g, err := s.ReferenceGraph()
		if err != nil {
			t.Fatal(err)
		}
		out, err := cudaref.Evaluate(g, input)
		if err != nil {
			t.Fatal(err)
		}
		for i, want := range []float32{u, v, u + v} {
			if out["result"][i] != want {
				t.Fatal(root.Op, out, want)
			}
		}
	}
}

func TestReferenceGraphRefusesUnsupportedAndShapeGuessing(t *testing.T) {
	s := Source{Expression: &Expr{Class: 17, Op: "unimplemented", Children: []*Expr{{Op: "missing"}}}}
	if _, err := s.ReferenceGraph(); !errors.Is(err, ErrEmissionUnsupported) {
		t.Fatal(err)
	}
	s = Source{Expression: &Expr{Op: "a", Data: "a"}}
	if _, err := s.ReferenceGraph(); err == nil {
		t.Fatal("missing authoritative input shape guessed")
	}
	s.Parameters = []Parameter{{Name: "a", Shape: []int{2}}}
	s.Expression.Data = float32(9)
	if _, err := s.ReferenceGraph(); err == nil {
		t.Fatal("constant payload silently converted to named input")
	}
	s.Expression = &Expr{Op: "add", Children: []*Expr{{Op: "a"}, {Op: "b"}}}
	s.Parameters = append(s.Parameters, Parameter{Name: "b", Shape: []int{1}})
	if _, err := s.ReferenceGraph(); err == nil {
		t.Fatal("broadcast shape guessed")
	}
}

func emittedBar(t *testing.T) (*Expr, EmitOptions) {
	t.Helper()
	g := New()
	a := g.Add(NewLeaf("a"))
	b := g.Add(NewLeaf("b"))
	silu := g.Add(NewNode("silu", a))
	mul := g.Add(NewNode("mul", a, b))
	root := g.Add(NewNode("add", silu, mul))
	expr, cost, err := ExtractBest(g, root, AstSizeCost{})
	if err != nil {
		t.Fatal(err)
	}
	return expr, EmitOptions{KernelName: "bar", Parameters: []Parameter{{Name: "b", Shape: []int{3}}, {Name: "a", Shape: []int{3}}}, CostModel: "AstSizeCost", Cost: cost, Stats: ExtractionStats{Nodes: 7}}
}

func TestEmitCUDAElementwiseBarAndStableSource(t *testing.T) {
	expr, opts := emittedBar(t)
	s, err := EmitCUDA(expr, opts)
	if err != nil {
		t.Fatal(err)
	}
	again, err := EmitCUDA(expr, opts)
	if err != nil {
		t.Fatal(err)
	}
	if s.CUDA != again.CUDA {
		t.Fatal("source not byte stable")
	}
	if len(s.Symbols) != 5 || s.Parameters[0].Name != "a" || s.Parameters[1].Name != "b" {
		t.Fatal("shared leaf not deduplicated or parameter order unstable", s.Symbols, s.Parameters)
	}
	for name, symbol := range s.Symbols {
		if strings.Count(s.CUDA, "__device__ auto "+name+"(") != 1 {
			t.Fatal("duplicate/missing symbol", name)
		}
		for _, line := range []int{symbol.FirstLine, symbol.LastLine} {
			if c, ok := s.ClassAt(line); !ok || c != symbol.Class {
				t.Fatal("ClassAt disagrees with symbol", name, line)
			}
		}
	}
	if !strings.Contains(s.CUDA, "auto arg0 = eg_c2_silu(input, params, fwd_ctx)") || !strings.Contains(s.CUDA, "auto arg1 = eg_c3_mul(input, params, fwd_ctx)") {
		t.Fatal("extracted children absent")
	}
	// Original AST/option mutation must not rewrite the returned reference.
	original := s.Expression.Fingerprint()
	expr.Children[0].Op = "wrong"
	opts.Parameters[0].Shape[0] = 99
	if s.Expression.Fingerprint() != original || s.Parameters[1].Shape[0] != 3 {
		t.Fatal("source retains mutable caller metadata")
	}
	if got := fmt.Sprintf("%x", sha256.Sum256([]byte(s.CUDA))); got != "1a6c037c4c1ef5af1e7750e2c0c8691e602f88c15dd210a55c7326bc85ca0882" {
		t.Fatalf("source golden changed: %s", got)
	}
}

func TestEmitCUDARefusesUnsupportedWithoutTemplateFallback(t *testing.T) {
	expr, opts := emittedBar(t)
	expr.Op = "fused_swiglu_gemv"
	if _, err := EmitCUDA(expr, opts); !errors.Is(err, ErrEmissionUnsupported) || !strings.Contains(err.Error(), "fused_swiglu_gemv") || !strings.Contains(err.Error(), "e-class 4") {
		t.Fatal("unsupported emission did not identify node", err)
	}
	expr, opts = emittedBar(t)
	opts.Parameters = nil
	if _, err := EmitCUDA(expr, opts); err == nil {
		t.Fatal("leaf shapes guessed")
	}
	expr, opts = emittedBar(t)
	opts.KernelName = "switch"
	if _, err := EmitCUDA(expr, opts); err == nil {
		t.Fatal("reserved kernel name accepted")
	}
}

func TestEmitCUDAOperatorClosures(t *testing.T) {
	for _, op := range []string{"gelu", "swish", "rmsnorm", "gemv", "fused_rmsnorm_gemv"} {
		t.Run(op, func(t *testing.T) {
			g := New()
			x := g.Add(NewLeaf("x"))
			w := g.Add(NewLeaf("w"))
			eps := g.Add(ENode{Op: "0.25", Data: float32(.25)})
			matrix := g.Add(NewLeaf("matrix"))
			var root Id
			params := []Parameter{{Name: "x", Shape: []int{2}}}
			switch op {
			case "gelu", "swish":
				root = g.Add(NewNode(op, x))
			case "rmsnorm":
				root = g.Add(NewNode(op, x, w, eps))
				params = append(params, Parameter{Name: "w", Shape: []int{2}})
			case "gemv":
				root = g.Add(NewNode(op, x, matrix))
				params = append(params, Parameter{Name: "matrix", Shape: []int{3, 2}})
			case "fused_rmsnorm_gemv":
				root = g.Add(NewNode(op, x, w, eps, matrix))
				params = append(params, Parameter{Name: "w", Shape: []int{2}}, Parameter{Name: "matrix", Shape: []int{3, 2}})
			}
			expr, cost, err := ExtractBest(g, root, AstSizeCost{})
			if err != nil {
				t.Fatal(err)
			}
			source, err := EmitCUDA(expr, EmitOptions{Parameters: params, Cost: cost, CostModel: "AstSizeCost"})
			if err != nil {
				t.Fatal(err)
			}
			if !strings.Contains(source.CUDA, fmt.Sprintf("eg_c%d_%s(eg_input, nullptr, nullptr)", expr.Class, op)) {
				t.Fatal("root function not called")
			}
			if _, err := source.ReferenceGraph(); err != nil {
				t.Fatal(err)
			}
			path := filepath.Join(t.TempDir(), "kernel.cu")
			if err := source.WriteFile(path); err != nil {
				t.Fatal(err)
			}
			sidecar, err := os.Open(strings.TrimSuffix(path, ".cu") + ".egraph.json")
			if err != nil {
				t.Fatal(err)
			}
			restored, err := ReadMetadata(sidecar, source.CUDA)
			closeErr := sidecar.Close()
			if err != nil || closeErr != nil {
				t.Fatal(err, closeErr)
			}
			if restored.Expression.Fingerprint() != expr.Fingerprint() || len(restored.Symbols) != len(source.Symbols) {
				t.Fatal("sidecar closure lost")
			}
		})
	}
}

func ExampleEmitCUDA() {
	g := New()
	x := g.Add(NewLeaf("x"))
	root := g.Add(NewNode("silu", x))
	expr, cost, _ := ExtractBest(g, root, AstSizeCost{})
	source, err := EmitCUDA(expr, EmitOptions{Parameters: []Parameter{{Name: "x", Shape: []int{2}}}, Cost: cost})
	fmt.Println(err, len(source.Symbols))
	// Output: <nil> 2
}

func ExampleSource_ReferenceGraph() {
	source := Source{Expression: &Expr{Op: "add", Children: []*Expr{{Op: "x"}, {Op: "x"}}}, Parameters: []Parameter{{Name: "x", Shape: []int{2}}}}
	graph, _ := source.ReferenceGraph()
	result, _ := cudaref.Evaluate(graph, map[string][]float32{"x": {1, 3}})
	fmt.Println(result["result"])
	// Output: [2 6]
}

type metadataFailReader struct{ err error }

func (r metadataFailReader) Read([]byte) (int, error) { return 0, r.err }

func TestSourceMetadataBoundsAndWriteFailureLifetime(t *testing.T) {
	sentinel := errors.New("reader failed")
	if _, err := ReadMetadata(metadataFailReader{sentinel}, ""); !errors.Is(err, sentinel) {
		t.Fatal("reader error lost", err)
	}
	for _, tc := range []struct {
		name string
		data string
		want string
	}{
		{"bytes", strings.Repeat(" ", (8<<20)+2), "size limit"},
		{"depth", strings.Repeat("[", 257) + strings.Repeat("]", 257), "depth limit"},
		{"tokens", "[" + strings.Repeat("0,", 65536) + "0]", "token limit"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := ReadMetadata(strings.NewReader(tc.data), ""); err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatal("bound not refused", err)
			}
		})
	}
	dir := t.TempDir()
	path := filepath.Join(dir, "kernel.cu")
	if err := os.Mkdir(path, 0700); err != nil {
		t.Fatal(err)
	}
	source := sourceArtifactFixture()
	if err := source.WriteFile(path); err == nil {
		t.Fatal("rename onto directory accepted")
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 1 || entries[0].Name() != "kernel.cu" {
		t.Fatal("temporary artifact leaked", entries)
	}
	if _, err := os.Stat(filepath.Join(dir, "kernel.egraph.json")); !errors.Is(err, os.ErrNotExist) {
		t.Fatal("sidecar emitted after failed source rename", err)
	}
	source.Expression.Children = []*Expr{source.Expression}
	if err := source.WriteFile(filepath.Join(dir, "cycle.cu")); err == nil {
		t.Fatal("cyclic expression accepted")
	}
}

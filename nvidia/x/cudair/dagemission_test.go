package cudair

import (
	"crypto/sha256"
	"fmt"
	"strings"
	"testing"
)

type dagTestNode struct {
	key, symbol                     string
	in, out, ctx, shmem             int
	deps                            []DAGEmitter
	body                            string
	calls                           int
	alter                           func(string) string
	inType, paramsType, computeType string
}

func (n *dagTestNode) DAGKey() string                { return n.key }
func (n *dagTestNode) DAGSymbol() string             { return n.symbol }
func (n *dagTestNode) Dependencies() []DAGEmitter    { return n.deps }
func (n *dagTestNode) InputWidth() int               { return n.in }
func (n *dagTestNode) OutputWidth() int              { return n.out }
func (n *dagTestNode) PaddedOutputWidth() int        { return n.out }
func (n *dagTestNode) FwdCtxBytes() (int, error)     { return n.ctx, nil }
func (n *dagTestNode) FwdCtxAlignedPerElement() bool { return false }
func (n *dagTestNode) ShmemBytes() int               { return n.shmem }
func (n *dagTestNode) InputType() string             { return n.inType }
func (n *dagTestNode) ParamsType() string            { return n.paramsType }
func (n *dagTestNode) ComputeType() string           { return n.computeType }
func (n *dagTestNode) EmitForward(name string) (string, error) {
	return "", fmt.Errorf("DAG must use dependency emission")
}
func (n *dagTestNode) EmitForwardWithDependencies(name string, children []ChildSymbol) (string, error) {
	n.calls++
	body := n.body
	if body == "" {
		if len(children) == 0 {
			body = fmt.Sprintf("tvec<float, %d> out; out[0] = input[0]; return out;", n.out)
		} else {
			for i, c := range children {
				body += fmt.Sprintf("auto v%d = %s(input, params, %s);\n", i, c.Name, c.FwdCtxArg())
			}
			if len(children) == 1 {
				body += "return v0;"
			} else {
				body += fmt.Sprintf("tvec<float, %d> out; out[0] = v0[0] + v1[0]; return out;", n.out)
			}
		}
	}
	s, err := EmitFromBody(n, name, body)
	if err == nil && n.alter != nil {
		s = n.alter(s)
	}
	return s, err
}

func dagBar() (*dagTestNode, *dagTestNode) {
	a := &dagTestNode{key: "a", symbol: "eg_c1_leaf", in: 4, out: 2}
	b := &dagTestNode{key: "b", symbol: "eg_c2_leaf", in: 4, out: 2}
	unary := &dagTestNode{key: "silu", symbol: "eg_c3_silu", in: 4, out: 2, deps: []DAGEmitter{a}}
	mul := &dagTestNode{key: "mul", symbol: "eg_c4_mul", in: 4, out: 2, deps: []DAGEmitter{a, b}}
	return &dagTestNode{key: "add", symbol: "eg_c5_add", in: 4, out: 2, deps: []DAGEmitter{unary, mul}}, a
}

func TestDAGEmissionFanoutNamedSharedLeaf(t *testing.T) {
	root, a := dagBar()
	unit, err := EmitTree(root, root.symbol)
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"eg_c1_leaf", "eg_c3_silu", "eg_c2_leaf", "eg_c4_mul", "eg_c5_add"}
	if fmt.Sprint(unit.Symbols) != fmt.Sprint(want) {
		t.Fatal(unit.Symbols)
	}
	if strings.Count(unit.Source, "__device__ auto eg_c1_leaf(") != 1 || strings.Count(unit.Source, "eg_c1_leaf(input, params, fwd_ctx)") != 2 {
		t.Fatal("shared definition or operand calls dropped", unit.Source)
	}
	if a.calls != 2 {
		t.Fatal("repeated key source was not checked", a.calls)
	}
	if unit.FwdCtxBytes != 0 {
		t.Fatal(unit)
	}
	second, err := EmitTree(root, root.symbol)
	if err != nil || second.Source != unit.Source {
		t.Fatal("nondeterministic emission", err)
	}
	if _, err := EmitTree(root, "renamed"); err == nil {
		t.Fatal("root symbol silently renamed")
	}
}

func TestDAGEmissionRefusesStructuralErrorsBeforeBodies(t *testing.T) {
	for _, tc := range []struct {
		name   string
		change func(*dagTestNode)
	}{
		{"input width", func(r *dagTestNode) { r.deps[1].(*dagTestNode).in = 5 }},
		{"positive context", func(r *dagTestNode) { r.deps[1].(*dagTestNode).ctx = 1 }},
		{"negative context", func(r *dagTestNode) { r.ctx = -1 }},
		{"nil child", func(r *dagTestNode) { r.deps = append(r.deps, nil) }},
		{"typed nil child", func(r *dagTestNode) { var n *dagTestNode; r.deps = append(r.deps, n) }},
		{"cycle", func(r *dagTestNode) { r.deps = append(r.deps, r) }},
		{"symbol collision", func(r *dagTestNode) { r.deps[1].(*dagTestNode).symbol = r.deps[0].DAGSymbol() }},
		{"descriptor conflict", func(r *dagTestNode) { r.deps[1].(*dagTestNode).key = r.deps[0].DAGKey() }},
		{"invalid symbol", func(r *dagTestNode) { r.symbol = "bad-name" }},
		{"keyword symbol", func(r *dagTestNode) { r.symbol = "return" }},
		{"width bound", func(r *dagTestNode) { r.in = dagMaxWidth + 1 }},
		{"shared memory", func(r *dagTestNode) { r.shmem = -1 }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r, a := dagBar()
			tc.change(r)
			if _, err := EmitTree(r, r.symbol); err == nil {
				t.Fatal("invalid DAG admitted")
			}
			if a.calls != 0 {
				t.Fatal("body emitted before structural refusal")
			}
		})
	}
	var nilRoot *dagTestNode
	if _, err := EmitTree(nilRoot, "anything"); err == nil {
		t.Fatal("typed nil root")
	}
}

func TestDAGEmissionBodyContractAndOmission(t *testing.T) {
	for _, body := range []string{
		"return input;",
		"input[0] = 1; return input;",
		"params[0] = 1; return input;",
		"fwd_ctx[0] = 1; return input;",
		"auto v = unknown(input, params, fwd_ctx); return v;",
		"auto v = eg_c3_silu(input, params, fwd_ctx); return v;",
		"auto v = eg_c4_mul(input, params, fwd_ctx); auto w = eg_c3_silu(input, params, fwd_ctx); return v;",
		"auto v = eg_c3_silu(input, params, fwd_ctx); auto w = eg_c4_mul(v, params, fwd_ctx); return v;",
	} {
		r, _ := dagBar()
		r.body = body
		if _, err := EmitTree(r, r.symbol); err == nil {
			t.Fatalf("invalid body admitted: %s", body)
		}
	}
	a := &dagTestNode{key: "a", symbol: "a", in: 4, out: 2}
	b := &dagTestNode{key: "a", symbol: "a", in: 4, out: 2, body: "tvec<float, 2> out; out[0] = input[1]; return out;"}
	r := &dagTestNode{key: "r", symbol: "r", in: 4, out: 2, deps: []DAGEmitter{a, b}}
	if _, err := EmitTree(r, "r"); err == nil {
		t.Fatal("same key with different computation admitted")
	}
}

func TestDAGEmissionRepeatedOperandAndGeneratedSignature(t *testing.T) {
	a := &dagTestNode{key: "a", symbol: "a", in: 4, out: 2}
	r := &dagTestNode{key: "r", symbol: "r", in: 4, out: 2, deps: []DAGEmitter{a, a}}
	u, err := EmitTree(r, "r")
	if err != nil {
		t.Fatal(err)
	}
	if len(u.Symbols) != 2 || strings.Count(u.Source, "__device__ auto a(") != 1 || strings.Count(u.Source, "a(input, params, fwd_ctx)") != 2 {
		t.Fatal("repeated operand deduplicated", u.Source)
	}
	for _, alter := range []func(string) string{
		func(s string) string { return strings.Replace(s, "auto r(", "auto renamed(", 1) },
		func(s string) string {
			return strings.Replace(s, "const tvec<float, 4>& input", "tvec<float, 4>& input", 1)
		},
		func(s string) string { return "#define input writable\n" + s },
	} {
		r.alter = alter
		if _, err := EmitTree(r, "r"); err == nil {
			t.Fatal("changed signature or preprocessor admitted")
		}
	}
}

func TestDAGEmissionPackedInputAndParamsABI(t *testing.T) {
	for _, tc := range []struct {
		name   string
		change func(*dagTestNode)
	}{
		{"child half input", func(r *dagTestNode) { r.deps[1].(*dagTestNode).inType = "__half" }},
		{"parent half input", func(r *dagTestNode) { r.inType = "__half" }},
		{"child params", func(r *dagTestNode) { r.deps[1].(*dagTestNode).paramsType = "__half" }},
		{"parent params", func(r *dagTestNode) { r.paramsType = "__half" }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r, a := dagBar()
			tc.change(r)
			if _, err := EmitTree(r, r.symbol); err == nil || !strings.Contains(err.Error(), "ABI differs") {
				t.Fatal("incompatible ABI admitted", err)
			}
			if a.calls != 0 {
				t.Fatal("body emitted before ABI admission")
			}
		})
	}
}

func TestDAGEmissionReservesMathAndGeneratedHelpers(t *testing.T) {
	for _, symbol := range []string{"expf", "erff", "sqrtf", "rsqrtf", "__expf", "lane_id", "input", "params", "fwd_ctx", "tvec"} {
		n := &dagTestNode{key: "k", symbol: symbol, in: 1, out: 1, body: "return input;"}
		if _, err := EmitTree(n, symbol); err == nil {
			t.Fatal("reserved symbol admitted", symbol)
		}
		if n.calls != 0 {
			t.Fatal("body emitted before symbol admission")
		}
	}
}

func TestDAGEmissionBounds(t *testing.T) {
	r, _ := dagBar()
	r.body = strings.Repeat(" ", dagMaxSource+1)
	if _, err := EmitTree(r, r.symbol); err == nil {
		t.Fatal("oversized source")
	}
	leaf := &dagTestNode{key: "leaf", symbol: "leaf", in: 4, out: 2}
	for i := 0; i < dagMaxNodes; i++ {
		leaf = &dagTestNode{key: fmt.Sprint(i), symbol: fmt.Sprintf("n%d", i), in: 4, out: 2, deps: []DAGEmitter{leaf}}
	}
	if _, err := EmitTree(leaf, leaf.symbol); err == nil {
		t.Fatal("node/depth bound")
	}
}

type sequentialTestNode struct {
	in, out  int
	children []Emitter
}

func (n *sequentialTestNode) InputWidth() int               { return n.in }
func (n *sequentialTestNode) OutputWidth() int              { return n.out }
func (n *sequentialTestNode) PaddedOutputWidth() int        { return n.out }
func (n *sequentialTestNode) FwdCtxBytes() (int, error)     { return 0, nil }
func (n *sequentialTestNode) FwdCtxAlignedPerElement() bool { return false }
func (n *sequentialTestNode) Children() []Emitter           { return n.children }
func (n *sequentialTestNode) EmitForward(name string) (string, error) {
	return EmitFromBody(n, name, fmt.Sprintf("tvec<float, %d> out; out[0] = input[0]; return out;", n.out))
}

func TestDAGEmissionPreservesSequentialGoldenAndWidthRules(t *testing.T) {
	r := &sequentialTestNode{in: 2, out: 1, children: []Emitter{&sequentialTestNode{in: 2, out: 3}, &sequentialTestNode{in: 3, out: 1}}}
	old, err := emitTreeRecursive(r, "seq", map[string]bool{"seq": true})
	if err != nil {
		t.Fatal(err)
	}
	// Baseline e73f372b1's unchanged sequential emitter and context framing.
	const golden = "209fb73725a52827ab260066a86a63d8f0cfe6262064b5b8dc31780750f4e835"
	if fmt.Sprintf("%x", sha256.Sum256([]byte(old.Source))) != golden {
		t.Fatal("sequential golden changed")
	}
	got, err := EmitTree(r, "seq")
	if err != nil || got.Source != old.Source || fmt.Sprint(got.Symbols) != fmt.Sprint(old.Symbols) {
		t.Fatal("sequential source changed", err)
	}
	r.children[1] = &sequentialTestNode{in: 2, out: 1}
	if _, err := EmitTree(r, "seq"); err == nil {
		t.Fatal("sequential adjacent mismatch relaxed")
	}
}

func FuzzDAGEmissionSymbol(f *testing.F) {
	f.Add("leaf")
	f.Add("bad-name")
	f.Fuzz(func(t *testing.T, symbol string) {
		if len(symbol) > 512 {
			return
		}
		n := &dagTestNode{key: "k", symbol: symbol, in: 1, out: 1, body: "return input;"}
		u, err := EmitTree(n, symbol)
		if !dagIdentifier(symbol) {
			if err == nil {
				t.Fatal("invalid identifier admitted")
			}
			return
		}
		if err != nil {
			t.Fatal(err)
		}
		if len(u.Symbols) != 1 || u.Symbols[0] != symbol {
			t.Fatal("symbol changed")
		}
	})
}

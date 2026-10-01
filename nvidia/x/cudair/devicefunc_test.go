package cudair_test

import (
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudair"
)

// mockNode is a plain object that does NOT implement cudair.Emitter.
type mockNode struct {
	name string
}

func (m *mockNode) Name() string { return m.name }

func TestEmitter_UnsupportedNode_RefusesWithNodeName(t *testing.T) {
	node := &mockNode{name: "unimplemented_relu"}
	_, err := cudair.EmitForward(node, "eval")
	if err == nil {
		t.Fatalf("expected error for unimplemented node, got nil")
	}
	if !errors.Is(err, cudair.ErrEmissionUnsupported) {
		t.Fatalf("expected ErrEmissionUnsupported, got: %v", err)
	}
	if !strings.Contains(err.Error(), "unimplemented_relu") {
		t.Fatalf("expected error to contain node name 'unimplemented_relu', got: %v", err)
	}

	// Also test with standard cudair.Node
	irNode := &cudair.Node{
		ID:   42,
		Name: "graph_matmul_node",
		Op:   "MatMul",
	}
	_, err = cudair.EmitForward(irNode, "eval_ir")
	if err == nil {
		t.Fatalf("expected error for *cudair.Node without emitter, got nil")
	}
	if !errors.Is(err, cudair.ErrEmissionUnsupported) {
		t.Fatalf("expected ErrEmissionUnsupported, got: %v", err)
	}
	if !strings.Contains(err.Error(), "graph_matmul_node") {
		t.Fatalf("expected error to name node, got: %v", err)
	}
}

// forwardOnlyEmitter implements Emitter but intentionally does NOT implement BackwardEmitter.
type forwardOnlyEmitter struct {
	name        string
	inWidth     int
	outWidth    int
	paddedWidth int
	fwdBytes    int
	aligned     bool
}

func (f *forwardOnlyEmitter) Name() string                  { return f.name }
func (f *forwardOnlyEmitter) InputWidth() int               { return f.inWidth }
func (f *forwardOnlyEmitter) OutputWidth() int              { return f.outWidth }
func (f *forwardOnlyEmitter) PaddedOutputWidth() int        { return f.paddedWidth }
func (f *forwardOnlyEmitter) FwdCtxBytes() (int, error)     { return f.fwdBytes, nil }
func (f *forwardOnlyEmitter) FwdCtxAlignedPerElement() bool { return f.aligned }
func (f *forwardOnlyEmitter) EmitForward(name string) (string, error) {
	return cudair.EmitFromBody(f, name, "\treturn input;")
}

func TestEmitter_ForwardSupportedBackwardNot_SeparateErrors(t *testing.T) {
	emitter := &forwardOnlyEmitter{
		name:        "forward_only_enc",
		inWidth:     16,
		outWidth:    16,
		paddedWidth: 16,
		fwdBytes:    0,
		aligned:     true,
	}

	// Forward emission must succeed
	src, err := cudair.EmitForward(emitter, "fwd_fun")
	if err != nil {
		t.Fatalf("unexpected forward error: %v", err)
	}
	if !strings.Contains(src, "fwd_fun") {
		t.Fatalf("expected emitted function to contain 'fwd_fun'")
	}

	// Backward emission must fail with ErrBackwardEmissionUnsupported, NOT ErrEmissionUnsupported
	_, err = cudair.EmitBackward(emitter, "bwd_fun", 128)
	if err == nil {
		t.Fatalf("expected error for backward emission, got nil")
	}
	if !errors.Is(err, cudair.ErrBackwardEmissionUnsupported) {
		t.Fatalf("expected ErrBackwardEmissionUnsupported, got: %v", err)
	}
	if errors.Is(err, cudair.ErrEmissionUnsupported) {
		t.Fatalf("should not be ErrEmissionUnsupported")
	}
	if !strings.Contains(err.Error(), "forward_only_enc") {
		t.Fatalf("expected error to name node, got: %v", err)
	}

	// Backward shmem query must also fail with ErrBackwardEmissionUnsupported
	_, err = cudair.BackwardShmemBytes(emitter, 128, cudair.GradientModeOverwrite)
	if err == nil {
		t.Fatalf("expected error for backward shmem query, got nil")
	}
	if !errors.Is(err, cudair.ErrBackwardEmissionUnsupported) {
		t.Fatalf("expected ErrBackwardEmissionUnsupported, got: %v", err)
	}
}

type fullEmitter struct {
	forwardOnlyEmitter
	shmemBytes int
}

func (fe *fullEmitter) EmitBackward(name string, nThreads int) (string, error) {
	return cudair.EmitBackwardFromBody(fe, name, "\t// bwd")
}

func (fe *fullEmitter) BackwardShmemBytes(nThreads int, mode cudair.GradientMode) int {
	return fe.shmemBytes
}

func TestEmitFromBody_SignatureIsGeneratedNotHandwritten(t *testing.T) {
	n := &fullEmitter{
		forwardOnlyEmitter: forwardOnlyEmitter{
			name:        "dense_proj",
			inWidth:     64,
			outWidth:    32,
			paddedWidth: 48, // deliberately different from outWidth (32)
			fwdBytes:    0,
			aligned:     true,
		},
		shmemBytes: 1024,
	}

	fwd, err := cudair.EmitFromBody(n, "dense_proj_eval", "\treturn out;")
	if err != nil {
		t.Fatalf("EmitFromBody failed: %v", err)
	}

	// Expected forward signature:
	// __device__ auto dense_proj_eval(const tvec<float, 64>& input, const float* __restrict__ params, uint8_t* __restrict__ fwd_ctx = nullptr) -> tvec<float, 32>
	expectedSig := "__device__ auto dense_proj_eval(const tvec<float, 64>& input, const float* __restrict__ params, uint8_t* __restrict__ fwd_ctx = nullptr) -> tvec<float, 32>"
	if !strings.Contains(fwd, expectedSig) {
		t.Fatalf("expected forward signature:\n%s\ngot:\n%s", expectedSig, fwd)
	}

	// Mutant check: return type MUST NOT use PaddedOutputWidth (48)
	if strings.Contains(fwd, "tvec<float, 48>") {
		t.Fatalf("emitted return type used padded width (48) instead of unpadded output width (32)")
	}

	bwd, err := cudair.EmitBackwardFromBody(n, "dense_proj_bwd", "\t// backward")
	if err != nil {
		t.Fatalf("EmitBackwardFromBody failed: %v", err)
	}

	// Expected backward signature:
	// __device__ void dense_proj_bwd(const tvec<float, 32>& dL_dy, const float* __restrict__ params, const uint8_t* __restrict__ fwd_ctx, float* __restrict__ dL_dparams = nullptr, tvec<float, 64>* __restrict__ dL_dx = nullptr)
	expectedBwdSig := "__device__ void dense_proj_bwd(const tvec<float, 32>& dL_dy, const float* __restrict__ params, const uint8_t* __restrict__ fwd_ctx, float* __restrict__ dL_dparams = nullptr, tvec<float, 64>* __restrict__ dL_dx = nullptr)"
	if !strings.Contains(bwd, expectedBwdSig) {
		t.Fatalf("expected backward signature:\n%s\ngot:\n%s", expectedBwdSig, bwd)
	}
}

func TestEmitFromBody_FwdCtxLaneOffsetPreamble(t *testing.T) {
	// Case 1: aligned per element with context > 0 -> must emit offset
	nAligned := &fullEmitter{
		forwardOnlyEmitter: forwardOnlyEmitter{
			name:        "layer_with_ctx",
			inWidth:     16,
			outWidth:    16,
			paddedWidth: 16,
			fwdBytes:    256,
			aligned:     true,
		},
	}

	fwd, err := cudair.EmitFromBody(nAligned, "ctx_fwd", "\treturn input;")
	if err != nil {
		t.Fatalf("EmitFromBody: %v", err)
	}
	expectedFwdPreamble := "if (fwd_ctx) fwd_ctx += lane_id() * 256;"
	if !strings.Contains(fwd, expectedFwdPreamble) {
		t.Fatalf("expected forward preamble to contain %q, got:\n%s", expectedFwdPreamble, fwd)
	}

	bwd, err := cudair.EmitBackwardFromBody(nAligned, "ctx_bwd", "\t// bwd")
	if err != nil {
		t.Fatalf("EmitBackwardFromBody: %v", err)
	}
	expectedBwdPreamble := "fwd_ctx += lane_id() * 256;"
	if !strings.Contains(bwd, expectedBwdPreamble) {
		t.Fatalf("expected backward preamble to contain %q, got:\n%s", expectedBwdPreamble, bwd)
	}

	// Case 2: aligned per element but fwdBytes == 0 -> must NOT emit offset
	nZero := &fullEmitter{
		forwardOnlyEmitter: forwardOnlyEmitter{
			name:        "zero_ctx",
			inWidth:     16,
			outWidth:    16,
			paddedWidth: 16,
			fwdBytes:    0,
			aligned:     true,
		},
	}
	fwdZero, err := cudair.EmitFromBody(nZero, "zero_fwd", "\treturn input;")
	if err != nil {
		t.Fatalf("EmitFromBody: %v", err)
	}
	if strings.Contains(fwdZero, "lane_id()") {
		t.Fatalf("expected zero ctx to omit lane_id() preamble, got:\n%s", fwdZero)
	}

	// Case 3: aligned == false -> must NOT emit offset
	nUnaligned := &fullEmitter{
		forwardOnlyEmitter: forwardOnlyEmitter{
			name:        "unaligned_ctx",
			inWidth:     16,
			outWidth:    16,
			paddedWidth: 16,
			fwdBytes:    128,
			aligned:     false,
		},
	}
	fwdUnaligned, err := cudair.EmitFromBody(nUnaligned, "unaligned_fwd", "\treturn input;")
	if err != nil {
		t.Fatalf("EmitFromBody: %v", err)
	}
	if strings.Contains(fwdUnaligned, "lane_id()") {
		t.Fatalf("expected unaligned ctx to omit lane_id() preamble, got:\n%s", fwdUnaligned)
	}
}

func TestFwdCtxBytesMismatch(t *testing.T) {
	node := &fullEmitter{
		forwardOnlyEmitter: forwardOnlyEmitter{
			name:        "undermem_node",
			inWidth:     16,
			outWidth:    16,
			paddedWidth: 16,
			fwdBytes:    64, // declares 64 bytes
			aligned:     true,
		},
	}

	// Writing 64 bytes should pass
	if err := cudair.ValidateFwdContextBounds(node, 64); err != nil {
		t.Fatalf("unexpected error within bounds: %v", err)
	}

	// Mutate / write 128 bytes (exceeding 64 declared) -> must fail with ErrContextBoundsExceeded
	err := cudair.ValidateFwdContextBounds(node, 128)
	if err == nil {
		t.Fatalf("expected error when context bytes exceed declared, got nil")
	}
	if !errors.Is(err, cudair.ErrContextBoundsExceeded) {
		t.Fatalf("expected ErrContextBoundsExceeded, got: %v", err)
	}
	if !strings.Contains(err.Error(), "undermem_node") {
		t.Fatalf("expected error to name node, got: %v", err)
	}
}

// compositeNode implements cudair.CompositeEmitter
type compositeNode struct {
	name     string
	inWidth  int
	outWidth int
	children []cudair.Emitter
}

func (c *compositeNode) Name() string                  { return c.name }
func (c *compositeNode) InputWidth() int               { return c.inWidth }
func (c *compositeNode) OutputWidth() int              { return c.outWidth }
func (c *compositeNode) PaddedOutputWidth() int        { return c.outWidth }
func (c *compositeNode) FwdCtxBytes() (int, error)     { return 0, nil }
func (c *compositeNode) FwdCtxAlignedPerElement() bool { return true }
func (c *compositeNode) Children() []cudair.Emitter    { return c.children }

func (c *compositeNode) EmitForward(name string) (string, error) {
	var calls []string
	for i, ch := range c.children {
		calls = append(calls, fmt.Sprintf("\t// call child %d (%s)", i, cudair.NodeName(ch)))
	}
	calls = append(calls, "\treturn input;")
	return cudair.EmitFromBody(c, name, strings.Join(calls, "\n"))
}

func (c *compositeNode) EmitForwardWithChildren(name string, children []cudair.ChildSymbol) (string, error) {
	var calls []string
	for i, child := range children {
		in := "input"
		if i > 0 {
			in = fmt.Sprintf("h%d", i-1)
		}
		ctx := child.FwdCtxArg()
		if i == len(children)-1 {
			calls = append(calls, fmt.Sprintf("\treturn %s(%s, params, %s);", child.Name, in, ctx))
		} else {
			calls = append(calls, fmt.Sprintf("\tauto h%d = %s(%s, params, %s);", i, child.Name, in, ctx))
		}
	}
	return cudair.EmitFromBody(c, name, strings.Join(calls, "\n"))
}

func TestEmitTree_NestedNamesUnique(t *testing.T) {
	// Create composite node with multiple children having the same name
	c1 := &forwardOnlyEmitter{name: "layer", inWidth: 16, outWidth: 16, paddedWidth: 16}
	c2 := &forwardOnlyEmitter{name: "layer", inWidth: 16, outWidth: 16, paddedWidth: 16}
	c3 := &forwardOnlyEmitter{name: "layer", inWidth: 16, outWidth: 16, paddedWidth: 16}

	root := &compositeNode{
		name:     "mlp",
		inWidth:  16,
		outWidth: 16,
		children: []cudair.Emitter{c1, c2, c3},
	}

	unit, err := cudair.EmitTree(root, "eval_mlp")
	if err != nil {
		t.Fatalf("EmitTree failed: %v", err)
	}

	if len(unit.Symbols) != 4 {
		t.Fatalf("expected 4 symbols (3 children + root), got %d: %v", len(unit.Symbols), unit.Symbols)
	}

	// Verify all symbol names are unique
	seen := make(map[string]bool)
	for _, sym := range unit.Symbols {
		if seen[sym] {
			t.Fatalf("duplicate symbol name detected in Unit: %q", sym)
		}
		seen[sym] = true
	}
}

func TestEmitTree_ParentCallsEmittedSymbols(t *testing.T) {
	child := &forwardOnlyEmitter{name: "enc.hash", inWidth: 16, outWidth: 16, paddedWidth: 16}
	root := &compositeNode{name: "root", inWidth: 16, outWidth: 16, children: []cudair.Emitter{child}}
	unit, err := cudair.EmitTree(root, "root_eval")
	if err != nil {
		t.Fatal(err)
	}
	if len(unit.Symbols) != 2 {
		t.Fatalf("symbols = %v", unit.Symbols)
	}
	childSymbol := unit.Symbols[0]
	if !strings.Contains(unit.Source, "return "+childSymbol+"(input, params, fwd_ctx);") {
		t.Fatalf("parent does not call emitted symbol %q:\n%s", childSymbol, unit.Source)
	}
}

func TestEmitTree_ParentCallsNestedCompositeSymbol(t *testing.T) {
	leaf := &forwardOnlyEmitter{name: "leaf", inWidth: 16, outWidth: 16, paddedWidth: 16}
	middle := &compositeNode{name: "middle", inWidth: 16, outWidth: 16, children: []cudair.Emitter{leaf}}
	root := &compositeNode{name: "root", inWidth: 16, outWidth: 16, children: []cudair.Emitter{middle}}
	unit, err := cudair.EmitTree(root, "root_eval")
	if err != nil {
		t.Fatal(err)
	}
	if len(unit.Symbols) != 3 {
		t.Fatalf("symbols = %v, want leaf, middle composite, and root", unit.Symbols)
	}
	middleSymbol := unit.Symbols[1]
	if !strings.Contains(unit.Source, "return "+middleSymbol+"(input, params, fwd_ctx);") {
		t.Fatalf("root did not call nested composite symbol %q:\n%s", middleSymbol, unit.Source)
	}
}

type contextWriter struct {
	forwardOnlyEmitter
	emit string
}

func (c *contextWriter) EmitForward(name string) (string, error) {
	return cudair.EmitFromBody(c, name, c.emit)
}

func TestEmitTree_FwdCtxRegionsDisjoint(t *testing.T) {
	a := &contextWriter{forwardOnlyEmitter: forwardOnlyEmitter{name: "ctx_a", inWidth: 16, outWidth: 16, paddedWidth: 16, fwdBytes: 16, aligned: true}, emit: "\treturn input;"}
	b := &contextWriter{forwardOnlyEmitter: forwardOnlyEmitter{name: "ctx_b", inWidth: 16, outWidth: 16, paddedWidth: 16, fwdBytes: 16, aligned: true}, emit: "\treturn input;"}
	root := &compositeNode{name: "ctx_root", inWidth: 16, outWidth: 16, children: []cudair.Emitter{a, b}}
	unit, err := cudair.EmitTree(root, "ctx_eval")
	if err != nil {
		t.Fatal(err)
	}
	if unit.FwdCtxBytes != 32 {
		t.Fatalf("FwdCtxBytes = %d, want 32", unit.FwdCtxBytes)
	}
	if !strings.Contains(unit.Source, "fwd_ctx ? fwd_ctx + WARP_SIZE * 16 : nullptr") {
		t.Fatalf("child context regions are not disjoint:\n%s", unit.Source)
	}
}

type contextComposite struct {
	*compositeNode
	ownBytes int
}

func (c *contextComposite) EmitForward(name string) (string, error) {
	return cudair.EmitFromBody(c, name, "return input;")
}
func (c *contextComposite) EmitForwardWithChildren(name string, children []cudair.ChildSymbol) (string, error) {
	var calls []string
	for i, child := range children {
		if i == len(children)-1 {
			calls = append(calls, fmt.Sprintf("return %s(input,params,%s);", child.Name, child.FwdCtxArg()))
		} else {
			calls = append(calls, fmt.Sprintf("auto h%d = %s(input,params,%s);", i, child.Name, child.FwdCtxArg()))
		}
	}
	return cudair.EmitFromBody(c, name, strings.Join(calls, "\n"))
}
func (c *contextComposite) FwdCtxBytes() (int, error) { return c.ownBytes, nil }

func TestEmitTree_ParentAndChildContextRegionsDoNotOverlap(t *testing.T) {
	a := &contextWriter{forwardOnlyEmitter: forwardOnlyEmitter{name: "ctx_a", inWidth: 16, outWidth: 16, paddedWidth: 16, fwdBytes: 16, aligned: true}, emit: "return input;"}
	b := &contextWriter{forwardOnlyEmitter: forwardOnlyEmitter{name: "ctx_b", inWidth: 16, outWidth: 16, paddedWidth: 16, fwdBytes: 16, aligned: true}, emit: "return input;"}
	base := &compositeNode{name: "ctx_parent", inWidth: 16, outWidth: 16, children: []cudair.Emitter{a, b}}
	root := &contextComposite{compositeNode: base, ownBytes: 8}
	unit, err := cudair.EmitTree(root, "ctx_parent_eval")
	if err != nil {
		t.Fatal(err)
	}
	if unit.FwdCtxBytes != 40 {
		t.Fatalf("FwdCtxBytes = %d, want 40", unit.FwdCtxBytes)
	}
	for _, expr := range []string{"WARP_SIZE * 8 - lane_id() * 8", "WARP_SIZE * 24 - lane_id() * 8"} {
		if !strings.Contains(unit.Source, expr) {
			t.Fatalf("missing disjoint context expression %q:\n%s", expr, unit.Source)
		}
	}
}

func TestEmitTree_FwdCtxBytesMismatchFromEmitter(t *testing.T) {
	node := &contextWriter{forwardOnlyEmitter: forwardOnlyEmitter{name: "writing_layer", inWidth: 16, outWidth: 16, paddedWidth: 16, fwdBytes: 0}, emit: "\t((float*)fwd_ctx)[0] = 1;\n\treturn input;"}
	_, err := cudair.EmitTree(node, "eval")
	if !errors.Is(err, cudair.ErrContextBoundsExceeded) {
		t.Fatalf("EmitTree error = %v, want ErrContextBoundsExceeded", err)
	}
	if !strings.Contains(err.Error(), "writing_layer") {
		t.Fatalf("EmitTree error does not name node: %v", err)
	}
}

type backwardContextReader struct{ fullEmitter }

func (b *backwardContextReader) EmitBackward(name string, nThreads int) (string, error) {
	return cudair.EmitBackwardFromBody(&b.fullEmitter, name, "\tfloat saved = ((const float*)fwd_ctx)[0]; (void)saved;")
}

func TestEmitBackwardTree_RejectsUndeclaredContextRead(t *testing.T) {
	node := &backwardContextReader{fullEmitter: fullEmitter{forwardOnlyEmitter: forwardOnlyEmitter{name: "backward_reader", inWidth: 16, outWidth: 16, paddedWidth: 16}}}
	_, err := cudair.EmitBackwardTree(node, "eval_bwd", 32, cudair.GradientModeIgnore)
	if !errors.Is(err, cudair.ErrContextBoundsExceeded) || !strings.Contains(err.Error(), "backward_reader") {
		t.Fatalf("EmitBackwardTree error = %v, want named ErrContextBoundsExceeded", err)
	}
}

func TestEmitTree_ContextAccessorIsBounded(t *testing.T) {
	node := &contextWriter{forwardOnlyEmitter: forwardOnlyEmitter{name: "bounded_writer", inWidth: 16, outWidth: 16, paddedWidth: 16, fwdBytes: 4}, emit: "bool ok = __cudair_ctx.StoreByte(3, 1); return input;"}
	unit, err := cudair.EmitTree(node, "bounded_eval")
	if err != nil {
		t.Fatalf("within-region accessor: %v", err)
	}
	if !strings.Contains(unit.Source, "i >= n") || !strings.Contains(unit.Source, "StoreByte(3, 1)") {
		t.Fatalf("missing bounded accessor:\n%s", unit.Source)
	}
	node.emit = "bool ok = __cudair_ctx.StoreByte(4, 1); return input;"
	_, err = cudair.EmitTree(node, "bounded_eval")
	if !errors.Is(err, cudair.ErrContextBoundsExceeded) || !strings.Contains(err.Error(), "bounded_writer") {
		t.Fatalf("constant out-of-region access must fail at emission: %v", err)
	}
	for _, offset := range []string{"(2 + 2)", "-1", "0x4u", "18446744073709551616ULL"} {
		node.emit = "bool ok = __cudair_ctx.StoreByte(" + offset + ", 1); return input;"
		if _, err := cudair.EmitTree(node, "bounded_eval"); !errors.Is(err, cudair.ErrContextBoundsExceeded) {
			t.Fatalf("offset %s: %v", offset, err)
		}
	}
	node.emit = "int offset = 4; bool ok = __cudair_ctx.StoreByte(offset, 1); return input;"
	unit, err = cudair.EmitTree(node, "bounded_eval")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(unit.Source, "i >= n") {
		t.Fatalf("out-of-region accessor has no runtime bound:\n%s", unit.Source)
	}
}

type gradientWriter struct {
	fullEmitter
	body string
}

func (g *gradientWriter) EmitBackward(name string, nThreads int) (string, error) {
	return cudair.EmitBackwardFromBody(g, name, g.body)
}

func TestEmitBackwardTree_GradientOutputs(t *testing.T) {
	for _, test := range []struct {
		name, body string
		accepted   bool
	}{
		{"unguarded", "(*dL_dx)[0] = 1; dL_dparams[0] = 2;", true},
		{"guarded", "if (dL_dx) { (*dL_dx)[0] = 1; } if (dL_dparams) { dL_dparams[0] += 2; }", true},
		{"params", "params[0] = 1;", false},
		{"upstream", "dL_dy[0] = 1;", false},
		{"output pointer rebind", "dL_dx = nullptr;", false},
	} {
		t.Run(test.name, func(t *testing.T) {
			node := &gradientWriter{fullEmitter: fullEmitter{forwardOnlyEmitter: forwardOnlyEmitter{name: "gradient_writer", inWidth: 16, outWidth: 16, paddedWidth: 16}}, body: test.body}
			_, err := cudair.EmitBackwardTree(node, "gradient_bwd", 32, cudair.GradientModeOverwrite)
			if test.accepted && err != nil {
				t.Fatal(err)
			}
			if !test.accepted && (!errors.Is(err, cudair.ErrContextBoundsExceeded) || !strings.Contains(err.Error(), "gradient_writer")) {
				t.Fatalf("error=%v, want named refusal", err)
			}
		})
	}
}

func TestEmitTree_LocalDeclarationBoundary(t *testing.T) {
	for _, test := range []struct {
		body     string
		accepted bool
	}{
		{"float saved; saved = 1; return input;", true},
		{"half saved; saved = 1; return input;", false},
	} {
		node := &contextWriter{forwardOnlyEmitter: forwardOnlyEmitter{name: "local_type", inWidth: 16, outWidth: 16, paddedWidth: 16}, emit: test.body}
		_, err := cudair.EmitTree(node, "local_eval")
		if test.accepted && err != nil {
			t.Fatal(err)
		}
		if !test.accepted && (err == nil || !strings.Contains(err.Error(), "unsupported declaration half saved") || !strings.Contains(err.Error(), "local_type")) {
			t.Fatalf("error=%v, want named unsupported declaration", err)
		}
	}
}

type boundedBackwardReader struct{ fullEmitter }

func (b *boundedBackwardReader) EmitBackward(name string, nThreads int) (string, error) {
	return cudair.EmitBackwardFromBody(&b.fullEmitter, name, "uint8_t saved; bool ok = __cudair_ctx.LoadByte(3, &saved); (void)ok; return;")
}

func TestEmitBackwardTree_ContextAccessorIsReadOnlyAndBounded(t *testing.T) {
	node := &boundedBackwardReader{fullEmitter: fullEmitter{forwardOnlyEmitter: forwardOnlyEmitter{name: "bounded_reader", inWidth: 16, outWidth: 16, paddedWidth: 16, fwdBytes: 4}}}
	unit, err := cudair.EmitBackwardTree(node, "bounded_bwd", 32, cudair.GradientModeIgnore)
	if err != nil {
		t.Fatalf("within-region backward load: %v", err)
	}
	if !strings.Contains(unit.Source, "LoadByte(3, &saved)") || !strings.Contains(unit.Source, "i >= n") {
		t.Fatalf("missing bounded backward accessor:\n%s", unit.Source)
	}
	// The accessor is generated for the declared region and safely rejects
	// any runtime offset outside that region.
	if strings.Contains(unit.Source, "StoreByte") {
		t.Fatalf("backward accessor unexpectedly permits writes:\n%s", unit.Source)
	}
}

type exactTextEmitter struct {
	*forwardOnlyEmitter
	source string
}

func (e *exactTextEmitter) EmitForward(name string) (string, error) { return e.source, nil }

func TestEmitTree_RefusesCompilerSourceOutsideGeneratedEnvelope(t *testing.T) {
	base := &forwardOnlyEmitter{name: "generated_guard", inWidth: 16, outWidth: 16, paddedWidth: 16}
	good, err := cudair.EmitFromBody(base, "guard_eval", "return input;")
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct{ name, source string }{
		{"prefix directive", "#define C fwd_ctx\n" + good},
		{"token paste", strings.Replace(good, "return input;", "#define CAT(a,b) a##b\nreturn input;", 1)},
		{"line splice", strings.Replace(good, "return input;", "float* fwd_\\\nctx; return input;", 1)},
		{"renamed signature", strings.Replace(good, "uint8_t* __restrict__ fwd_ctx", "uint8_t* __restrict__ c", 1)},
		{"nested context store", strings.Replace(good, "return input;", "if (true) { ((float*)fwd_ctx)[0] = 1; } return input;", 1)},
		{"extra trailing source", good + "\n#define AFTER 1"},
		{"unbalanced body", strings.Replace(good, "return input;", "if (true) { return input;", 1)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			node := &exactTextEmitter{forwardOnlyEmitter: base, source: tc.source}
			_, err := cudair.EmitTree(node, "guard_eval")
			if !errors.Is(err, cudair.ErrContextBoundsExceeded) || !strings.Contains(err.Error(), "generated_guard") {
				t.Fatalf("EmitTree error = %v", err)
			}
		})
	}
}

func TestEmitTree_IgnoresContextNameInCommentAndString(t *testing.T) {
	base := &forwardOnlyEmitter{name: "context_text", inWidth: 16, outWidth: 16, paddedWidth: 16}
	for _, body := range []string{
		"return input; // fwd_ctx is documentation only",
		"if (\"fwd_ctx\"[0]) return input; return input;",
	} {
		node := &contextWriter{forwardOnlyEmitter: *base, emit: body}
		if _, err := cudair.EmitTree(node, "context_text_eval"); err != nil {
			t.Errorf("EmitTree rejected non-code context text %q: %v", body, err)
		}
	}
}

type bodyComposite struct {
	*compositeNode
	body string
}

func (c *bodyComposite) EmitForwardWithChildren(name string, children []cudair.ChildSymbol) (string, error) {
	return cudair.EmitFromBody(c, name, fmt.Sprintf(c.body, children[0].Name))
}

func TestEmitTree_ChildContextMustUseExactArgument(t *testing.T) {
	child := &forwardOnlyEmitter{name: "safe_child", inWidth: 16, outWidth: 16, paddedWidth: 16}
	for _, body := range []string{
		"return %s(input, ((float*)fwd_ctx)[0], fwd_ctx);",
		"return %s(input, params, (fwd_ctx ? fwd_ctx + WARP_SIZE * 7 : nullptr));",
		"return %sx(input, params, fwd_ctx);",
	} {
		base := &compositeNode{name: "bad_parent", inWidth: 16, outWidth: 16, children: []cudair.Emitter{child}}
		root := &bodyComposite{compositeNode: base, body: body}
		_, err := cudair.EmitTree(root, "bad_parent_eval")
		if !errors.Is(err, cudair.ErrContextBoundsExceeded) || !strings.Contains(err.Error(), "bad_parent") {
			t.Errorf("body %q: error = %v", body, err)
		}
	}
}

type badContextComposite struct{ *contextComposite }

func (c *badContextComposite) EmitForwardWithChildren(name string, children []cudair.ChildSymbol) (string, error) {
	body := fmt.Sprintf("fwd_ctx[0] = 1; return %s(input, params, fwd_ctx);", children[0].Name)
	return cudair.EmitFromBody(c, name, body)
}

func TestEmitTree_RejectsCompositeContextWrite(t *testing.T) {
	child := &forwardOnlyEmitter{name: "safe_child", inWidth: 16, outWidth: 16, paddedWidth: 16}
	base := &compositeNode{name: "bad_parent", inWidth: 16, outWidth: 16, children: []cudair.Emitter{child}}
	root := &badContextComposite{contextComposite: &contextComposite{compositeNode: base}}
	_, err := cudair.EmitTree(root, "bad_parent_eval")
	if !errors.Is(err, cudair.ErrContextBoundsExceeded) || !strings.Contains(err.Error(), "bad_parent") {
		t.Fatalf("EmitTree error = %v, want ErrContextBoundsExceeded naming bad_parent", err)
	}
}

func TestEmitTree_RootBoundaryWidthMismatch(t *testing.T) {
	child := &forwardOnlyEmitter{name: "root_child", inWidth: 16, outWidth: 16, paddedWidth: 16}
	root := &compositeNode{name: "root_boundary", inWidth: 32, outWidth: 16, children: []cudair.Emitter{child}}
	_, err := cudair.EmitTree(root, "eval")
	if err == nil || !strings.Contains(err.Error(), "root_boundary") || !strings.Contains(err.Error(), "root_child") || !strings.Contains(err.Error(), "32") || !strings.Contains(err.Error(), "16") {
		t.Fatalf("root mismatch error = %v", err)
	}
}

func TestEmitTree_UsesOutputNotPaddedWidth(t *testing.T) {
	a := &forwardOnlyEmitter{name: "padded", inWidth: 16, outWidth: 15, paddedWidth: 32}
	b := &forwardOnlyEmitter{name: "next", inWidth: 15, outWidth: 8, paddedWidth: 16}
	root := &compositeNode{name: "padded_chain", inWidth: 16, outWidth: 8, children: []cudair.Emitter{a, b}}
	if _, err := cudair.EmitTree(root, "eval"); err != nil {
		t.Fatalf("EmitTree rejected unpadded adjacent widths: %v", err)
	}
}

func TestEmitBackwardTree_UsesBackwardSharedMemory(t *testing.T) {
	node := &fullEmitter{forwardOnlyEmitter: forwardOnlyEmitter{name: "backward_shared", inWidth: 16, outWidth: 16, paddedWidth: 16}, shmemBytes: 64 << 10}
	unit, err := cudair.EmitBackwardTree(node, "backward_eval", 128, cudair.GradientModeAccumulate)
	if err != nil {
		t.Fatal(err)
	}
	if unit.ShmemBytes != 64<<10 {
		t.Fatalf("backward shmem = %d, want %d", unit.ShmemBytes, 64<<10)
	}
}

type backwardDemandNode struct {
	fullEmitter
	children   []cudair.Emitter
	gotThreads []int
	gotModes   []cudair.GradientMode
}

func (n *backwardDemandNode) Children() []cudair.Emitter { return n.children }

func (n *backwardDemandNode) BackwardShmemBytes(nThreads int, mode cudair.GradientMode) int {
	n.gotThreads = append(n.gotThreads, nThreads)
	n.gotModes = append(n.gotModes, mode)
	return n.shmemBytes
}

func TestEmitBackwardTree_QueriesChildDemandWithCallerLaunch(t *testing.T) {
	child := &backwardDemandNode{fullEmitter: fullEmitter{
		forwardOnlyEmitter: forwardOnlyEmitter{name: "backward_child", inWidth: 16, outWidth: 16, paddedWidth: 16},
		shmemBytes:         96 << 10,
	}}
	root := &backwardDemandNode{fullEmitter: fullEmitter{
		forwardOnlyEmitter: forwardOnlyEmitter{name: "backward_root", inWidth: 16, outWidth: 16, paddedWidth: 16},
		shmemBytes:         32 << 10,
	}, children: []cudair.Emitter{child}}
	const threads = 192
	const mode = cudair.GradientModeAccumulate
	unit, err := cudair.EmitBackwardTree(root, "backward_root_eval", threads, mode)
	if err != nil {
		t.Fatal(err)
	}
	if unit.ShmemBytes != child.shmemBytes {
		t.Fatalf("tree shared memory = %d, want child demand %d", unit.ShmemBytes, child.shmemBytes)
	}
	for name, node := range map[string]*backwardDemandNode{"root": root, "child": child} {
		if len(node.gotThreads) != 1 || node.gotThreads[0] != threads || len(node.gotModes) != 1 || node.gotModes[0] != mode {
			t.Errorf("%s demand calls = threads %v, modes %v; want %d, %v", name, node.gotThreads, node.gotModes, threads, mode)
		}
	}
}

type backwardContextWriter struct{ fullEmitter }

func (b *backwardContextWriter) EmitBackward(name string, nThreads int) (string, error) {
	return cudair.EmitBackwardFromBody(&b.fullEmitter, name, "fwd_ctx[0] = 1; return;")
}

func TestEmitBackwardTree_RejectsContextWrite(t *testing.T) {
	node := &backwardContextWriter{fullEmitter: fullEmitter{forwardOnlyEmitter: forwardOnlyEmitter{
		name: "backward_writer", inWidth: 16, outWidth: 16, paddedWidth: 16, fwdBytes: 1,
	}}}
	_, err := cudair.EmitBackwardTree(node, "backward_writer_eval", 32, cudair.GradientModeIgnore)
	if !errors.Is(err, cudair.ErrContextBoundsExceeded) || !strings.Contains(err.Error(), "backward_writer") {
		t.Fatalf("EmitBackwardTree error = %v, want named ErrContextBoundsExceeded", err)
	}
}

func TestEmitTree_WidthMismatch_ReportedInGoNotNVRTC(t *testing.T) {
	// Child 0 outputs 64; Child 1 expects 32 -> Adjacent width mismatch!
	c0 := &forwardOnlyEmitter{name: "encoder_projection", inWidth: 16, outWidth: 64, paddedWidth: 64}
	c1 := &forwardOnlyEmitter{name: "decoder_layer", inWidth: 32, outWidth: 32, paddedWidth: 32}

	root := &compositeNode{
		name:     "mismatched_model",
		inWidth:  16,
		outWidth: 32,
		children: []cudair.Emitter{c0, c1},
	}

	_, err := cudair.EmitTree(root, "eval")
	if err == nil {
		t.Fatalf("expected error from EmitTree on adjacent width mismatch, got nil")
	}

	errMsg := err.Error()
	if !strings.Contains(errMsg, "encoder_projection") || !strings.Contains(errMsg, "decoder_layer") {
		t.Fatalf("expected error to name both nodes, got: %v", errMsg)
	}
	if !strings.Contains(errMsg, "64") || !strings.Contains(errMsg, "32") {
		t.Fatalf("expected error to name both widths (64 and 32), got: %v", errMsg)
	}
}

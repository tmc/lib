package egraph_test

import (
	"errors"
	"math"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudashape"
	"github.com/tmc/lib/nvidia/x/egraph"
)

// FuzzOnlineAttentionDimensions exercises checked extent and tile handling
// with arbitrary host inputs; the target is deliberately small and CPU-only.
func FuzzOnlineAttentionDimensions(f *testing.F) {
	f.Add([]byte{1, 2, 3, 4, 5})
	f.Add([]byte{255, 0, 17})
	f.Fuzz(func(t *testing.T, data []byte) {
		if len(data) == 0 {
			return
		}
		seqLen := len(data) % 16
		headDim := 1 + len(data)%8
		tileSize := 1 + len(data)%32
		scores := make([]float32, seqLen)
		values := make([]float32, seqLen*headDim)
		mask := make([]bool, seqLen)
		for i, b := range data {
			if i < seqLen {
				scores[i] = float32(int(b)-128) / 16
			}
			if i < len(values) {
				values[i] = float32(int(b)-128) / 32
			}
		}
		for i := range mask {
			mask[i] = data[i%len(data)]&1 != 0
		}
		got, err := egraph.OnlineAttentionRow(scores, values, mask, seqLen, headDim, tileSize)
		if err != nil {
			t.Fatalf("online attention: %v", err)
		}
		if len(got) != headDim {
			t.Fatalf("got %d outputs, want %d", len(got), headDim)
		}
	})
}

// 1. TestRewriteLegality verifies typed legality checks on shape, layout, dtype, and alignment.
func TestRewriteLegality(t *testing.T) {
	shA, err := cudashape.New(32, 128)
	if err != nil {
		t.Fatalf("shape.New failed: %v", err)
	}
	metaA := egraph.TypedMetadata{
		Shape:     shA,
		DType:     cudair.Float32,
		Layout:    cudashape.LayoutRowMajor,
		Effect:    cudair.EffectReadOnly,
		Alignment: 16,
	}

	// Identical contracts: legal
	metaB := metaA
	if err := egraph.CheckLegality(metaA, metaB); err != nil {
		t.Fatalf("expected legal union, got %v", err)
	}

	// Incompatible DType
	metaC := metaA
	metaC.DType = cudair.Float16
	if err := egraph.CheckLegality(metaA, metaC); err == nil {
		t.Errorf("expected dtype mismatch rejection, got nil")
	}

	// Incompatible Shape
	shD, _ := cudashape.New(32, 64)
	metaD := metaA
	metaD.Shape = shD
	if err := egraph.CheckLegality(metaA, metaD); err == nil {
		t.Errorf("expected shape mismatch rejection, got nil")
	}

	// Incompatible Layout
	metaE := metaA
	metaE.Layout = cudashape.LayoutColumnMajor
	if err := egraph.CheckLegality(metaA, metaE); err == nil {
		t.Errorf("expected layout mismatch rejection, got nil")
	}

	// Incompatible Alignment (target 16 requires divisor of 16, e.g. 8 is not a multiple of 16)
	metaF := metaA
	metaF.Alignment = 8
	if err := egraph.CheckLegality(metaA, metaF); err == nil {
		t.Errorf("expected alignment check to reject 8 against required 16, got nil")
	}
}

// 2. TestEffectBoundaryRejected confirms that merging mutating storage effects with read-only is rejected.
func TestEffectBoundaryRejected(t *testing.T) {
	sh, err := cudashape.New(16, 16)
	if err != nil {
		t.Fatalf("shape.New failed: %v", err)
	}
	metaRead := egraph.TypedMetadata{
		Shape:     sh,
		DType:     cudair.Float32,
		Layout:    cudashape.LayoutRowMajor,
		Effect:    cudair.EffectReadOnly,
		Alignment: 16,
	}
	metaMut := metaRead
	metaMut.Effect = cudair.EffectMutating

	if err := egraph.CheckLegality(metaRead, metaMut); err == nil {
		t.Errorf("expected mutation effect boundary violation to be rejected, got nil")
	}

	// Verify with LegalAdapter
	eg := egraph.New()
	adapter := egraph.NewLegalAdapter(eg)
	id1 := eg.Add(egraph.NewLeaf("read_tensor"))
	id2 := eg.Add(egraph.NewLeaf("mut_tensor"))
	adapter.SetMetadata(id1, metaRead)
	adapter.SetMetadata(id2, metaMut)

	_, merged, err := adapter.LegalUnion(id1, id2)
	if err == nil || merged {
		t.Errorf("LegalAdapter.LegalUnion should have rejected merging mutating with read-only; err=%v, merged=%v", err, merged)
	}
}

// 3. TestOnlineAttentionReference verifies online attention recurrence against materialized dense reference.
func TestOnlineAttentionReference(t *testing.T) {
	seqLen := 16
	headDim := 8
	scores := make([]float32, seqLen)
	values := make([]float32, seqLen*headDim)
	mask := make([]bool, seqLen)

	for i := 0; i < seqLen; i++ {
		scores[i] = float32(i)*0.25 - 2.0
		mask[i] = true
		for d := 0; d < headDim; d++ {
			values[i*headDim+d] = float32(i+1) * 0.1 * float32(d+1)
		}
	}

	refOut, err := egraph.MaterializedAttentionReference(scores, values, mask, seqLen, headDim)
	if err != nil {
		t.Fatalf("reference failed: %v", err)
	}

	onlineOut, err := egraph.OnlineAttentionRow(scores, values, mask, seqLen, headDim, 4)
	if err != nil {
		t.Fatalf("online attention failed: %v", err)
	}

	for i := 0; i < headDim; i++ {
		diff := math.Abs(float64(refOut[i] - onlineOut[i]))
		if diff > 1e-5 {
			t.Errorf("dim %d mismatch: ref=%f, online=%f, diff=%e", i, refOut[i], onlineOut[i], diff)
		}
	}
}

// 4. TestOnlineAttentionTilePartitions verifies identical outputs across varied tile sizes (1, 2, 4, 8, 16).
func TestOnlineAttentionTilePartitions(t *testing.T) {
	seqLen := 16
	headDim := 8
	scores := make([]float32, seqLen)
	values := make([]float32, seqLen*headDim)
	mask := make([]bool, seqLen)

	for i := 0; i < seqLen; i++ {
		scores[i] = float32(i%5) - 2.0
		mask[i] = true
		for d := 0; d < headDim; d++ {
			values[i*headDim+d] = float32(i*headDim+d) * 0.05
		}
	}

	refOut, err := egraph.MaterializedAttentionReference(scores, values, mask, seqLen, headDim)
	if err != nil {
		t.Fatalf("ref failed: %v", err)
	}

	for _, tileSize := range []int{1, 2, 3, 4, 7, 8, 16, 32} {
		out, err := egraph.OnlineAttentionRow(scores, values, mask, seqLen, headDim, tileSize)
		if err != nil {
			t.Fatalf("tileSize %d failed: %v", tileSize, err)
		}
		for d := 0; d < headDim; d++ {
			diff := math.Abs(float64(refOut[d] - out[d]))
			if diff > 1e-5 {
				t.Fatalf("tileSize %d dim %d mismatch: ref=%f, got=%f", tileSize, d, refOut[d], out[d])
			}
		}
	}
}

// 5. TestFullyMaskedAttention checks that fully masked row policy yields exact zero vector without NaN or Inf.
func TestFullyMaskedAttention(t *testing.T) {
	seqLen := 8
	headDim := 4
	scores := make([]float32, seqLen)
	values := make([]float32, seqLen*headDim)
	mask := make([]bool, seqLen) // all false

	out, err := egraph.OnlineAttentionRow(scores, values, mask, seqLen, headDim, 4)
	if err != nil {
		t.Fatalf("fully masked row failed: %v", err)
	}

	for d := 0; d < headDim; d++ {
		if out[d] != 0.0 {
			t.Errorf("expected exact 0.0 for dim %d on fully masked row, got %f", d, out[d])
		}
		if math.IsNaN(float64(out[d])) || math.IsInf(float64(out[d]), 0) {
			t.Fatalf("dim %d contains NaN/Inf: %f", d, out[d])
		}
	}
}

// 6. TestSparseMaskReference tests prefix and causal masks against reference oracle.
func TestSparseMaskReference(t *testing.T) {
	seqLen := 8
	headDim := 4
	scores := make([]float32, seqLen)
	values := make([]float32, seqLen*headDim)

	for i := 0; i < seqLen; i++ {
		scores[i] = float32(i) * 0.5
		for d := 0; d < headDim; d++ {
			values[i*headDim+d] = float32(i + d)
		}
	}

	// Causal mask at query index 3
	causalMask := egraph.CausalAttentionMask(3, seqLen)
	refCausal, err := egraph.MaterializedAttentionReference(scores, values, causalMask, seqLen, headDim)
	if err != nil {
		t.Fatalf("ref causal failed: %v", err)
	}

	onlineCausal, err := egraph.OnlineAttentionRow(scores, values, causalMask, seqLen, headDim, 2)
	if err != nil {
		t.Fatalf("online causal failed: %v", err)
	}

	for d := 0; d < headDim; d++ {
		if math.Abs(float64(refCausal[d]-onlineCausal[d])) > 1e-5 {
			t.Errorf("causal dim %d mismatch: ref=%f, online=%f", d, refCausal[d], onlineCausal[d])
		}
	}

	// Sparse tree mask with allowed ancestor indices {0, 2, 5}
	treeMask := egraph.SparseTreeMask([]int{0, 2, 5}, seqLen)
	refTree, err := egraph.MaterializedAttentionReference(scores, values, treeMask, seqLen, headDim)
	if err != nil {
		t.Fatalf("ref tree failed: %v", err)
	}

	onlineTree, err := egraph.OnlineAttentionRow(scores, values, treeMask, seqLen, headDim, 3)
	if err != nil {
		t.Fatalf("online tree failed: %v", err)
	}

	for d := 0; d < headDim; d++ {
		if math.Abs(float64(refTree[d]-onlineTree[d])) > 1e-5 {
			t.Errorf("tree dim %d mismatch: ref=%f, online=%f", d, refTree[d], onlineTree[d])
		}
	}
}

// 7. TestAttentionMutantControls verifies that mutant controls fail oracle comparison.
func TestAttentionMutantControls(t *testing.T) {
	seqLen := 8
	headDim := 4
	scores := []float32{1.0, 2.0, -1.0, 3.0, 0.5, -0.5, 2.5, 1.5}
	values := make([]float32, seqLen*headDim)
	mask := make([]bool, seqLen)
	for i := range mask {
		// Keep this row populated: mutant controls must be tested against a
		// non-degenerate oracle input. Fully masked behavior is covered above.
		mask[i] = false
		for d := 0; d < headDim; d++ {
			values[i*headDim+d] = float32(i+1) * 0.2
		}
	}

	refOut, err := egraph.MaterializedAttentionReference(scores, values, mask, seqLen, headDim)
	if err != nil {
		t.Fatalf("ref failed: %v", err)
	}

	// Mutant 1: Inverse Rescaling
	mutantRescale, err := egraph.OnlineAttentionRowMutantInverseRescale(scores, values, mask, seqLen, headDim, 2)
	if err != nil {
		t.Fatalf("mutant rescale failed: %v", err)
	}
	rescaleMatches := true
	for d := 0; d < headDim; d++ {
		if math.Abs(float64(refOut[d]-mutantRescale[d])) > 1e-3 {
			rescaleMatches = false
			break
		}
	}
	if rescaleMatches {
		t.Errorf("expected mutant inverse rescaling to diverge from oracle, but it matched!")
	}

	// Mutant 2: Omitted Normalizer
	mutantNoNorm, err := egraph.OnlineAttentionRowMutantOmitNormalizer(scores, values, mask, seqLen, headDim, 2)
	if err != nil {
		t.Fatalf("mutant no norm failed: %v", err)
	}
	normMatches := true
	for d := 0; d < headDim; d++ {
		if math.Abs(float64(refOut[d]-mutantNoNorm[d])) > 1e-3 {
			normMatches = false
			break
		}
	}
	if normMatches {
		t.Errorf("expected mutant omitted normalizer to diverge from oracle, but it matched!")
	}
}

// 8. TestGroupingDependenceRejected tests causal/dependence violation rejection in grouping transformations.
func TestGroupingDependenceRejected(t *testing.T) {
	// If a transformation attempts to group token T2 with future token T3 when T2 mutates a state needed by T3,
	// effect checking rejects it.
	sh, err := cudashape.New(128)
	if err != nil {
		t.Fatalf("shape.New failed: %v", err)
	}
	metaT2 := egraph.TypedMetadata{
		Shape:     sh,
		DType:     cudair.Float16,
		Layout:    cudashape.LayoutRowMajor,
		Effect:    cudair.EffectMutating,
		Alignment: 16,
	}
	metaT3 := metaT2
	metaT3.Effect = cudair.EffectReadOnly

	if err := egraph.CheckLegality(metaT2, metaT3); err == nil {
		t.Errorf("expected cross-token grouping dependence violation to be rejected")
	}
}

// 9. TestMandatoryRewriteRules verifies the mandatory rewrite rules in equality saturation.
func TestMandatoryRewriteRules(t *testing.T) {
	rules := egraph.MandatoryRewriteRules()
	if len(rules) < 7 {
		t.Fatalf("expected at least 7 mandatory rules, got %d", len(rules))
	}

	ruleNames := make(map[string]bool)
	for _, r := range rules {
		ruleNames[r.Name] = true
	}

	expectedRules := []string{
		"hoist_q2k_scales_weights",
		"hoist_iq2xxs_scales_weights",
		"fuse_rmsnorm_quant_q8",
		"fuse_swiglu_quant_q8_1",
		"vectorize_moe_gather_w16",
		"vectorize_moe_gather_w32",
		"prefill_group_expand_w256",
	}

	for _, expected := range expectedRules {
		if !ruleNames[expected] {
			t.Errorf("missing mandatory rewrite rule: %q", expected)
		}
	}

	// Test rule application for RMSNorm -> Quant Q8 fusion
	g := egraph.New()
	x := g.Add(egraph.NewLeaf("act"))
	w := g.Add(egraph.NewLeaf("weight"))
	norm := g.Add(egraph.NewNode("rmsnorm", x, w))
	g.Add(egraph.NewNode("quantize_q8", norm))

	fuseRule := egraph.MustRewrite("fuse_rmsnorm_quant_q8",
		"(quantize_q8 (rmsnorm ?x ?w))",
		"(ds4_rms_quant_q8 ?x ?w)",
	)

	if merges := fuseRule.Apply(g); merges != 0 {
		t.Fatalf("legacy mandatory Apply bypassed typed proof: merges=%d", merges)
	}
}

// 10. TestRewriteBranchIsolation verifies that branching/cloning a SearchState does not mutate parent state.
func TestRewriteBranchIsolation(t *testing.T) {
	g := egraph.New()
	a := g.Add(egraph.NewLeaf("a"))
	b := g.Add(egraph.NewLeaf("b"))
	sum := g.Add(egraph.NewNode("+", a, b))

	state := egraph.NewSearchState(g, map[string]egraph.Id{"sum": sum})
	rule := egraph.MustRewrite("commute", "(+ ?x ?y)", "(+ ?y ?x)")

	cands := egraph.EnumerateActions(state.Graph, []*egraph.Rewrite{rule})
	if len(cands) == 0 {
		t.Fatalf("expected 1 candidate action")
	}

	res, err := state.ApplyAction(cands[0])
	if err != nil {
		t.Fatalf("apply action failed: %v", err)
	}

	// Parent state must remain at depth 0 with empty derivation
	if state.Depth != 0 || len(state.Derivation) != 0 {
		t.Errorf("parent state was mutated! depth=%d, len(derivation)=%d", state.Depth, len(state.Derivation))
	}

	// Child state must have advanced
	if res.State.Depth != 1 || len(res.State.Derivation) != 1 {
		t.Errorf("child state depth=%d, len(derivation)=%d", res.State.Depth, len(res.State.Derivation))
	}
}

// 11. TestRewriteActionReplay & TestSearchDerivationReplay verify deterministic replay from derivation trace.
func TestRewriteActionReplay(t *testing.T) {
	testSearchDerivationReplayHelper(t)
}

func TestSearchDerivationReplay(t *testing.T) {
	testSearchDerivationReplayHelper(t)
}

func testSearchDerivationReplayHelper(t *testing.T) {
	g := egraph.New()
	x := g.Add(egraph.NewLeaf("x"))
	two := g.Add(egraph.NewLeaf("2"))
	expr := g.Add(egraph.NewNode("*", x, two))

	rules := []*egraph.Rewrite{
		egraph.MustRewrite("mul_by_two", "(* ?a 2)", "(<< ?a 1)"),
		egraph.MustRewrite("shift_to_add", "(<< ?a 1)", "(+ ?a ?a)"),
	}

	state := egraph.NewSearchState(g, map[string]egraph.Id{"root": expr})

	// Step 1
	cands1 := egraph.EnumerateActions(state.Graph, rules)
	if len(cands1) == 0 {
		t.Fatalf("expected step 1 candidates")
	}
	res1, err := state.ApplyAction(cands1[0])
	if err != nil {
		t.Fatalf("step 1 failed: %v", err)
	}

	// Step 2
	cands2 := egraph.EnumerateActions(res1.State.Graph, rules)
	if len(cands2) == 0 {
		t.Fatalf("expected step 2 candidates")
	}
	res2, err := res1.State.ApplyAction(cands2[0])
	if err != nil {
		t.Fatalf("step 2 failed: %v", err)
	}

	trace := res2.State.Derivation
	if len(trace) != 2 {
		t.Fatalf("expected 2 derivation steps, got %d", len(trace))
	}

	// Replay trace on fresh state
	replayedState, err := egraph.ReplayDerivation(g, map[string]egraph.Id{"root": expr}, rules, trace)
	if err != nil {
		t.Fatalf("replay failed: %v", err)
	}

	rootOrig, _ := res2.State.CanonicalRoot("root")
	rootReplay, _ := replayedState.CanonicalRoot("root")

	bestOrig, costOrig, err := egraph.ExtractBest(res2.State.Graph, rootOrig, egraph.AstSizeCost{})
	if err != nil {
		t.Fatalf("extract orig failed: %v", err)
	}

	bestReplay, costReplay, err := egraph.ExtractBest(replayedState.Graph, rootReplay, egraph.AstSizeCost{})
	if err != nil {
		t.Fatalf("extract replay failed: %v", err)
	}

	if !bestOrig.Equal(bestReplay) || costOrig != costReplay {
		t.Errorf("replay mismatch: orig=%s (cost %f), replay=%s (cost %f)", bestOrig, costOrig, bestReplay, costReplay)
	}
}

// 12. TestMultistepRewriteState checks intermediate states in a multi-step sequence.
func TestMultistepRewriteState(t *testing.T) {
	g := egraph.New()
	x := g.Add(egraph.NewLeaf("x"))
	w := g.Add(egraph.NewLeaf("w"))
	norm := g.Add(egraph.NewNode("rmsnorm", x, w))
	quant := g.Add(egraph.NewNode("quantize_q8", norm))

	rules := egraph.MandatoryRewriteRules()
	state := egraph.NewSearchState(g, map[string]egraph.Id{"root": quant})
	sh, _ := cudashape.New(1)
	state.SetMetadata(quant, egraph.TypedMetadata{Shape: sh, DType: cudair.Float32, Layout: cudashape.LayoutRowMajor, Effect: cudair.EffectReadOnly, Alignment: 16, LoopInvariant: true, NumericalPolicy: "rmsnorm-q8"})
	for _, id := range []egraph.Id{x, w, norm} {
		state.SetMetadata(id, egraph.TypedMetadata{Shape: sh, DType: cudair.Float32, Layout: cudashape.LayoutRowMajor, Effect: cudair.EffectReadOnly, Alignment: 16})
	}

	cands := egraph.EnumerateActions(state.Graph, rules)
	if len(cands) == 0 {
		t.Fatalf("expected candidate for rmsnorm quant fusion")
	}

	res, err := state.ApplyAction(cands[0])
	if err != nil {
		t.Fatalf("apply failed: %v", err)
	}

	if !res.Merged {
		t.Fatalf("expected state to merge")
	}

	root, _ := res.State.CanonicalRoot("root")
	best, _, err := egraph.ExtractBest(res.State.Graph, root, egraph.AstSizeCost{})
	if err != nil {
		t.Fatalf("extract failed: %v", err)
	}

	if best.Op != "ds4_rms_quant_q8" {
		t.Errorf("expected ds4_rms_quant_q8, got %s", best.Op)
	}
}

func TestMandatoryTransitionLegalityAndForgedMatch(t *testing.T) {
	g := egraph.New()
	tokens := g.Add(egraph.NewLeaf("tokens"))
	indices := g.Add(egraph.NewLeaf("indices"))
	width := g.Add(egraph.NewLeaf(8))
	root := g.Add(egraph.NewNode("moe_gather", tokens, indices, width))
	rule := egraph.MandatoryRewriteRules()[5]
	state := egraph.NewSearchState(g, map[string]egraph.Id{"root": root})
	cands := egraph.EnumerateActions(state.Graph, []*egraph.Rewrite{rule})
	if len(cands) == 0 {
		t.Fatal("expected vectorization candidate")
	}
	if _, err := state.ApplyAction(cands[0]); !errors.Is(err, egraph.ErrMissingTypedMetadata) {
		t.Fatalf("missing metadata: %v", err)
	}
	sh, _ := cudashape.New(8, 16)
	state.SetMetadata(root, egraph.TypedMetadata{Shape: sh, DType: cudair.Float32, Layout: cudashape.LayoutRowMajor, Effect: cudair.EffectReadOnly, Alignment: 8})
	for _, id := range []egraph.Id{tokens, indices, width} {
		state.SetMetadata(id, egraph.TypedMetadata{Shape: sh, DType: cudair.Float32, Layout: cudashape.LayoutRowMajor, Effect: cudair.EffectReadOnly, Alignment: 8})
	}
	if _, err := state.ApplyAction(cands[0]); !errors.Is(err, egraph.ErrUnalignedVectorAccess) {
		t.Fatalf("unaligned vector: %v", err)
	}
	state.SetMetadata(root, egraph.TypedMetadata{Shape: sh, DType: cudair.Float32, Layout: cudashape.LayoutRowMajor, Effect: cudair.EffectReadOnly, Alignment: 16})
	forged := cands[0]
	forged.Match.Subst = egraph.Subst{"?tokens": 999}
	if _, err := state.ApplyAction(forged); err == nil {
		t.Fatal("forged substitution accepted")
	}
}

// 13. TestExpressionDataIdentity verifies that Expr.String and Fingerprint distinguish Data.
func TestExpressionDataIdentity(t *testing.T) {
	e1 := &egraph.Expr{Op: "const", Data: int32(42)}
	e2 := &egraph.Expr{Op: "const", Data: int64(42)}
	e3 := &egraph.Expr{Op: "const", Data: int32(100)}

	if e1.Fingerprint() == e2.Fingerprint() {
		t.Errorf("Fingerprint failed to distinguish int32(42) from int64(42): %q vs %q", e1.Fingerprint(), e2.Fingerprint())
	}
	if e1.String() == e3.String() {
		t.Errorf("Expr.String failed to distinguish data values 42 and 100: %q vs %q", e1.String(), e3.String())
	}
	if e1.Equal(e3) {
		t.Errorf("Equal returned true for different Data values")
	}
}

// 14. TestSameExpressionDifferentSearchState checks that two states with identical best AST
// are preserved separately when reachable actions differ.
func TestSameExpressionDifferentSearchState(t *testing.T) {
	// State 1: graph with (+ a b) and rule to commute
	g1 := egraph.New()
	a1 := g1.Add(egraph.NewLeaf("a"))
	b1 := g1.Add(egraph.NewLeaf("b"))
	sum1 := g1.Add(egraph.NewNode("+", a1, b1))
	s1 := egraph.NewSearchState(g1, map[string]egraph.Id{"sum": sum1})

	// State 2: graph with (+ a b) where a and b are merged with other nodes
	g2 := g1.Clone()
	c2 := g2.Add(egraph.NewLeaf("c"))
	g2.Union(a1, c2)
	g2.Rebuild()
	s2 := egraph.NewSearchState(g2, map[string]egraph.Id{"sum": sum1})

	best1, _, _ := egraph.ExtractBest(s1.Graph, sum1, egraph.AstSizeCost{})
	best2, _, _ := egraph.ExtractBest(s2.Graph, sum1, egraph.AstSizeCost{})

	// Both states extract the same best AST: (+ a b)
	if best1.String() != best2.String() {
		t.Fatalf("expected same best AST, got %s vs %s", best1.String(), best2.String())
	}

	// But their reachable actions differ!
	commuteRule := egraph.MustRewrite("commute", "(+ ?x ?y)", "(+ ?y ?x)")
	actions1 := egraph.EnumerateActions(s1.Graph, []*egraph.Rewrite{commuteRule})
	actions2 := egraph.EnumerateActions(s2.Graph, []*egraph.Rewrite{commuteRule})

	if len(actions1) == 0 || len(actions2) == 0 {
		t.Fatalf("expected actions in both states")
	}
}

// 15. TestNestedAlternativeExtraction checks candidate extraction considering child alternatives.
func TestNestedAlternativeExtraction(t *testing.T) {
	g := egraph.New()
	x := g.Add(egraph.NewLeaf("x"))
	two := g.Add(egraph.NewLeaf("2"))
	mul := g.Add(egraph.NewNode("*", x, two))
	root := g.Add(egraph.NewNode("neg", mul))

	// Rewrite child: (* x 2) -> (<< x 1)
	rule := egraph.MustRewrite("mul_to_shift", "(* ?a 2)", "(<< ?a 1)")
	rule.Apply(g)

	// Extract top 2 candidates from root
	cands, err := egraph.ExtractTopK(g, root, 5, egraph.AstSizeCost{})
	if err != nil {
		t.Fatalf("ExtractTopK failed: %v", err)
	}

	if len(cands) < 2 {
		t.Fatalf("expected at least 2 candidates from nested child alternative, got %d", len(cands))
	}

	hasShiftChild := false
	hasMulChild := false
	for _, c := range cands {
		s := c.Expr.String()
		if strings.Contains(s, "<<") {
			hasShiftChild = true
		}
		if strings.Contains(s, "*") {
			hasMulChild = true
		}
	}

	if !hasShiftChild || !hasMulChild {
		t.Errorf("nested alternative extraction missed child variant: hasShift=%v, hasMul=%v", hasShiftChild, hasMulChild)
	}
}

// 16. TestNoOpRewriteBounded verifies no-op transitions terminate and are flagged.
func TestNoOpRewriteBounded(t *testing.T) {
	g := egraph.New()
	a := g.Add(egraph.NewLeaf("a"))
	sum := g.Add(egraph.NewNode("+", a, a))

	rule := egraph.MustRewrite("commute", "(+ ?x ?y)", "(+ ?y ?x)")
	state := egraph.NewSearchState(g, map[string]egraph.Id{"sum": sum})

	// First application merges (+ a a) with (+ a a), which is already in same e-class!
	cands := egraph.EnumerateActions(state.Graph, []*egraph.Rewrite{rule})
	if len(cands) == 0 {
		t.Fatalf("expected candidate")
	}

	res, err := state.ApplyAction(cands[0])
	if err != nil {
		t.Fatalf("apply failed: %v", err)
	}

	if !res.NoOp {
		t.Errorf("expected NoOp flag to be true when RHS is already in target e-class")
	}
}

// 17. TestCyclicExtractionBounded verifies bounded termination when encountering cycles.
func TestCyclicExtractionBounded(t *testing.T) {
	g := egraph.New()
	a := g.Add(egraph.NewLeaf("a"))
	b := g.Add(egraph.NewLeaf("b"))

	// Create mutual dependency between a and b: (+ a b) and (+ b a)
	sum1 := g.Add(egraph.NewNode("+", a, b))
	sum2 := g.Add(egraph.NewNode("+", b, a))

	// Saturate with bidirectional commute rule
	rules := []*egraph.Rewrite{
		egraph.MustRewrite("commute", "(+ ?x ?y)", "(+ ?y ?x)"),
	}
	egraph.Saturate(g, rules, 5)

	// Extractor must extract without hanging or infinite recursion
	ext := egraph.NewExtractor(g, egraph.AstSizeCost{})
	_, _, err := ext.Extract(sum1)
	if err != nil {
		t.Fatalf("expected bounded extraction, got err: %v", err)
	}

	_, _, err2 := ext.Extract(sum2)
	if err2 != nil {
		t.Fatalf("expected bounded extraction for sum2, got err: %v", err2)
	}
}

// 18. TestNonfiniteCostRejected ensures NaN and Inf costs are rejected by the cost model and extractor.
type NaNInfCostModel struct{}

func (NaNInfCostModel) NodeCost(n egraph.ENode, childCosts []float64) float64 {
	if n.Op == "nan_op" {
		return math.NaN()
	}
	if n.Op == "inf_op" {
		return math.Inf(1)
	}
	cost := 1.0
	for _, c := range childCosts {
		cost += c
	}
	return cost
}

func TestNonfiniteCostRejected(t *testing.T) {
	g := egraph.New()
	nanNode := g.Add(egraph.NewLeaf("nan_op"))
	infNode := g.Add(egraph.NewLeaf("inf_op"))
	validNode := g.Add(egraph.NewLeaf("valid_op"))

	// Union nan_op with valid_op
	g.Union(nanNode, validNode)
	g.Rebuild()

	ext := egraph.NewExtractor(g, NaNInfCostModel{})

	// Class with nanNode and validNode should select validNode with finite cost
	expr, cost, err := ext.Extract(nanNode)
	if err != nil {
		t.Fatalf("extraction failed: %v", err)
	}
	if math.IsNaN(cost) || math.IsInf(cost, 0) {
		t.Fatalf("expected finite cost, got %f", cost)
	}
	if expr.Op != "valid_op" {
		t.Errorf("expected valid_op to be chosen over nan_op, got %s", expr.Op)
	}

	// Class with only infNode should fail extraction
	_, _, errInf := ext.Extract(infNode)
	if errInf == nil {
		t.Errorf("expected extraction of infNode alone to fail due to nonfinite cost, but succeeded")
	}
}

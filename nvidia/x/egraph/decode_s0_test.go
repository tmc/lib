package egraph_test

import (
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudashape"
	"github.com/tmc/lib/nvidia/x/cudatest"
	"github.com/tmc/lib/nvidia/x/egraph"
)

// TestDecodeFFNGraphIsCUDAAddressable is the S0 decode-only CPU gate.  It
// exercises saturation, extraction, and code generation over the Q8_1-shaped
// shared/routed FFN terminals used by ds4 decode.
func TestDecodeFFNGraphIsCUDAAddressable(t *testing.T) {
	// Exercise the real mandatory hoist proof before testing the selected
	// terminal.  A hand-written equivalent rule would miss this contract.
	g := egraph.New()
	gw := g.Add(egraph.NewLeaf("q2_weights"))
	gs := g.Add(egraph.NewLeaf("q2_scales"))
	gx := g.Add(egraph.NewLeaf("hidden_q8_1"))
	loadID := g.Add(egraph.NewNode("load_unhoisted_q2k", gw, gs))
	hoistRoot := g.Add(egraph.NewNode("q2_k_matvec", loadID, gx))
	state := egraph.NewSearchState(g, map[string]egraph.Id{"root": hoistRoot})
	shape, err := cudashape.New(256)
	if err != nil {
		t.Fatal(err)
	}
	meta := egraph.TypedMetadata{Shape: shape, DType: cudair.Float32, Layout: cudashape.LayoutRowMajor,
		Effect: cudair.EffectReadOnly, Alignment: 16, LoopInvariant: true, NumericalPolicy: "q2k"}
	for _, id := range []egraph.Id{gw, gs, gx, loadID, hoistRoot} {
		state.SetMetadata(id, meta)
	}
	actions := egraph.EnumerateActions(state.Graph, egraph.MandatoryRewriteRules())
	var hoist egraph.ActionCandidate
	found := false
	for _, action := range actions {
		if action.Rule.Name == "hoist_q2k_scales_weights" {
			hoist, found = action, true
			break
		}
	}
	if !found {
		t.Fatal("mandatory Q2_K hoist did not match the decode graph")
	}
	transition, err := state.ApplyAction(hoist)
	if err != nil {
		t.Fatalf("mandatory Q2_K hoist: %v", err)
	}
	canonical, ok := transition.State.CanonicalRoot("root")
	if !ok {
		t.Fatal("missing hoisted root")
	}
	hoisted, _, err := egraph.ExtractBest(transition.State.Graph, canonical, egraph.AstSizeCost{})
	if err != nil {
		t.Fatalf("extract hoisted root: %v", err)
	}
	if _, err := egraph.GenerateCUDA(hoisted, egraph.CodegenOptions{KernelName: "HoistedQ2K", Precision: "q8_1"}); err != nil {
		t.Fatalf("GenerateCUDA(hoisted): %v", err)
	}

	eg := egraph.New()
	x := eg.Add(egraph.NewLeaf("hidden_q8_1"))
	w := eg.Add(egraph.NewLeaf("expert_q2_k"))
	ids := eg.Add(egraph.NewLeaf("expert_ids"))
	load := eg.Add(egraph.NewNode("load_unhoisted_q2k", w, ids))
	root := eg.Add(egraph.NewNode("q2_k_selected_matvec_q8_1", load, x, ids))

	// These names mirror the mandatory ds4 decode rewrites while keeping this
	// S0 test independent of typed metadata required by the production proof.
	rules := []*egraph.Rewrite{
		egraph.MustRewrite("s0_hoist_q2k", "(load_unhoisted_q2k ?w ?ids)", "(hoist_block_q2k ?w ?ids)"),
	}
	egraph.Saturate(eg, rules, 8)
	best, _, err := egraph.ExtractBest(eg, root, egraph.AstSizeCost{})
	if err != nil {
		t.Fatalf("extract: %v", err)
	}
	src, err := egraph.GenerateCUDA(best, egraph.CodegenOptions{
		KernelName: "DecodeFFN",
		Precision:  "q8_1",
	})
	if err != nil {
		t.Fatalf("GenerateCUDA(%s): %v", best, err)
	}
	if !strings.Contains(src, "struct block_q8_1") || !strings.Contains(src, "static_assert(sizeof(block_q8_1) == 36") {
		t.Fatal("generated source does not carry the ds4 Q8_1 ABI")
	}

	for _, op := range []string{
		"rms_quant_q8_1",
		"fused_swiglu_q8_1",
		"q8_0_matvec_q8_1",
		"iq2_xxs_selected_matvec_q8_1",
		"q2_k_selected_matvec_q8_1",
	} {
		t.Run(op, func(t *testing.T) {
			src, err := egraph.GenerateCUDA(&egraph.Expr{Op: op}, egraph.CodegenOptions{
				KernelName: op,
				Precision:  "q8_1",
			})
			if err != nil {
				t.Fatalf("GenerateCUDA: %v", err)
			}
			if op == "rms_quant_q8_1" || op == "fused_swiglu_q8_1" {
				for _, marker := range []string{"block_q8_1* __restrict__ out", ".d =", ".sum ="} {
					if !strings.Contains(src, marker) {
						t.Fatalf("%s does not emit Q8_1 output marker %q", op, marker)
					}
				}
			}
			if strings.Contains(op, "selected") {
				for _, marker := range []string{"const int* __restrict__ ids", "ids[warp]", "W +"} {
					if !strings.Contains(src, marker) {
						t.Fatalf("%s does not emit selected-expert marker %q", op, marker)
					}
				}
			}
			if strings.HasPrefix(op, "q8_0") || strings.Contains(op, "selected_matvec") {
				if !strings.Contains(src, "q8_1_at") {
					t.Fatalf("%s does not load Q8_1 activations", op)
				}
			}
		})
	}
	// Mutation control: replacing a selected lowering with the ordinary matvec
	// must be observable through the selected ABI markers.
	nonSelected, err := egraph.GenerateCUDA(&egraph.Expr{Op: "q2_k_matvec_q8_1"}, egraph.CodegenOptions{
		KernelName: "MutatedSelected",
		Precision:  "q8_1",
	})
	if err != nil {
		t.Fatalf("GenerateCUDA mutation: %v", err)
	}
	if strings.Contains(nonSelected, "const int* __restrict__ ids") || strings.Contains(nonSelected, "ids[warp]") {
		t.Fatal("selected-expert mutation was not detected by the structural gate")
	}
}

func TestDecodeQ8_1GeneratedKernelsCompile(t *testing.T) {
	major, minor := cudatest.RequireNVRTC(t)
	for _, op := range []string{
		"q8_0_matvec_q8_1", "iq2_xxs_matvec_q8_1", "q2_k_matvec_q8_1",
	} {
		t.Run(op, func(t *testing.T) {
			src, err := egraph.GenerateCUDA(&egraph.Expr{Op: op}, egraph.CodegenOptions{
				KernelName: op,
				Precision:  "q8_1",
			})
			if err != nil {
				t.Fatalf("GenerateCUDA: %v", err)
			}
			if _, err := cudatest.CompileSource(src, major, minor); err != nil {
				t.Fatalf("compile: %v", err)
			}
		})
	}
}

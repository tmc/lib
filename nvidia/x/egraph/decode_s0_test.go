package egraph_test

import (
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudatest"
	"github.com/tmc/lib/nvidia/x/egraph"
)

// TestDecodeFFNGraphIsCUDAAddressable is the S0 decode-only CPU gate.  It
// exercises saturation, extraction, and code generation over the Q8_1-shaped
// shared/routed FFN terminals used by ds4 decode.
func TestDecodeFFNGraphIsCUDAAddressable(t *testing.T) {
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
			if strings.HasPrefix(op, "q8_0") || strings.Contains(op, "selected_matvec") {
				if !strings.Contains(src, "q8_1_at") {
					t.Fatalf("%s does not load Q8_1 activations", op)
				}
			}
		})
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

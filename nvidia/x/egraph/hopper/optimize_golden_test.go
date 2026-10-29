package hopper

import (
	"fmt"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/egraph"
)

// TestOptimizeGolden pins the terms Optimize extracts for llama-infer's
// decoder-layer plans, so changes to saturation bookkeeping cannot change
// which kernels a model runs.
func TestOptimizeGolden(t *testing.T) {
	type dims struct {
		name                           string
		hidden, heads, kvHeads, hd, ff int
		bias                           bool
	}
	models := []dims{
		{"tinyllama", 2048, 32, 4, 64, 5632, false},
		{"llama3_8b", 4096, 32, 8, 128, 14336, false},
		{"qwen2_5_7b", 3584, 28, 4, 128, 18944, true},
		{"mistral_nemo", 5120, 32, 8, 128, 14336, false},
	}
	var out strings.Builder
	for _, d := range models {
		for _, m := range []int{1, 64} {
			for _, p := range layerTerms(d.hidden, d.heads, d.kvHeads, d.hd, d.ff, m, d.bias) {
				e, err := Optimize(p.b.G, p.root)
				if err != nil {
					t.Fatalf("%s m=%d %s: %v", d.name, m, p.name, err)
				}
				fmt.Fprintf(&out, "%s m=%d %s\n\t%v\n", d.name, m, p.name, e)
			}
		}
	}
	checkGolden(t, "optimize_llama.txt", out.String())
}

type layerTerm struct {
	name string
	b    *Builder
	root egraph.Id
}

// layerTerms mirrors buildLayerPlans in examples/nvidia/llama-infer.
func layerTerms(h, heads, kvHeads, hd, inter, m int, bias bool) []layerTerm {
	qDim, kvDim := heads*hd, kvHeads*hd
	const eps = 1e-5

	b := NewBuilder()
	x := b.Input("x", F32, m, h)
	norm := b.Cast(BF16, b.RMSNorm(x, b.Input("attn_norm", F32, h), h, eps))
	qkv := b.MatMul(norm, b.Input("w_qkv", BF16, qDim+2*kvDim, h))
	if bias {
		qkv = b.Map(MustLambda("(add $0 $1)"), qkv, b.Input("b_qkv", F32, qDim+2*kvDim))
	}
	attnIn := layerTerm{"attn_in", b, qkv}

	b = NewBuilder()
	ao := b.Input("attn", BF16, m, qDim)
	x = b.Input("x", F32, m, h)
	hid := b.Map(MustLambda("(add $0 $1)"), b.MatMul(ao, b.Input("w_o", BF16, h, qDim)), x)
	norm = b.Cast(BF16, b.RMSNorm(hid, b.Input("ffn_norm", F32, h), h, eps))
	act := b.SwiGLU(b.MatMul(norm, b.Input("w_gu", BF16, 2*inter, h)))
	post := b.Map(MustLambda("(add $0 $1)"), b.MatMul(act, b.Input("w_down", BF16, h, inter)), hid)
	return []layerTerm{attnIn, {"post", b, post}}
}

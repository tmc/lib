package main

import (
	"testing"

	"github.com/tmc/lib/nvidia/x/egraph/hopper"
)

// llama3_8B is Meta-Llama-3-8B's shape.
var llama3_8B = ModelConfig{
	HiddenSize:        4096,
	IntermediateSize:  14336,
	NumHiddenLayers:   32,
	NumAttentionHeads: 32,
	NumKeyValueHeads:  8,
	VocabSize:         128256,
	MaxSeqLen:         2048,
	RmsNormEps:        1e-5,
	RopeTheta:         500000,
}

func TestBuildEgraphPlans(t *testing.T) {
	p, err := buildEgraphPlans(&llama3_8B, false, egraphOptions{Attn: true, PrefillRows: -1})
	if err != nil {
		t.Fatal(err)
	}
	count := func(p *hopper.Plan) (skinny, tile, mma, row int) {
		for _, k := range p.Kernels {
			switch {
			case k.GEMM != nil:
				skinny++
			case k.MMA != nil:
				mma++
			case k.Tile != nil:
				tile++
				if k.Tile.WGMMA {
					t.Errorf("%s: wgmma is not available on sm_121", k.Tile.Name)
				}
			case k.Row != nil:
				row++
			default:
				t.Errorf("%s: unexpected kernel %+v", p.Name, k)
			}
		}
		return
	}
	tests := []struct {
		plan                   *hopper.Plan
		skinny, tile, mma, row int
	}{
		// rmsnorm row, qkv gemm.
		{p.attnIn, 1, 0, 0, 1},
		{p.pf.attnIn, 0, 1, 0, 1},
		{p.pf.narrowIn, 0, 0, 1, 1},
		// cast row, o_proj+x gemm, rmsnorm row, gate_up gemm, swiglu row,
		// down_proj+h gemm. Prefill attention writes BF16: no cast.
		{p.post, 3, 0, 0, 3},
		{p.pf.post, 0, 3, 0, 2},
		{p.pf.narrowPost, 0, 0, 3, 2},
		{p.head, 1, 0, 0, 1},
	}
	for _, tt := range tests {
		skinny, tile, mma, row := count(tt.plan)
		if skinny != tt.skinny || tile != tt.tile || mma != tt.mma || row != tt.row {
			t.Errorf("%s: %d skinny gemms, %d tile gemms, %d skinny mmas, %d row kernels; want %d, %d, %d, %d",
				tt.plan.Name, skinny, tile, mma, row, tt.skinny, tt.tile, tt.mma, tt.row)
		}
	}
	if p.pf.narrow != 32 {
		t.Errorf("narrow prefill chunks = %d rows, want 32", p.pf.narrow)
	}
	for _, pl := range []*hopper.Plan{p.pf.attnIn, p.pf.post} {
		for _, k := range pl.Kernels {
			if k.Tile != nil && k.Tile.SharedBytes() > 99*1024 {
				t.Errorf("%s: %d bytes of shared memory, more than sm_121's 99 KiB", k.Tile.Name, k.Tile.SharedBytes())
			}
		}
	}
	if p.pf.rows != llama3_8B.MaxSeqLen {
		t.Errorf("prefill chunk = %d rows, want the cache length %d", p.pf.rows, llama3_8B.MaxSeqLen)
	}
	if p.pf.qtok != 15 {
		t.Errorf("prefill attention groups = %d rows, want 15 (G=4, D=128 in 48 KiB of static shared memory)", p.pf.qtok)
	}
}

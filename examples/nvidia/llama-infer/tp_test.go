package main

import (
	"testing"

	"github.com/tmc/lib/nvidia/x/egraph/hopper"
	"github.com/tmc/lib/nvidia/x/egraph/tensorparallel"
)

// TestTPPlansBindable checks that Llama-3-8B's sharded decode plans use
// only the kernels tpDecoder.bindPlan launches: skinny gemms without a
// split-K workspace and row kernels, for every TP degree and fusion.
// TP=8 shards because SwiGLU runs in the gate·up gemm's epilogue: as a
// row kernel over 14336/8 = 1792 columns it would need a width the
// generator's row kernels do not support (above 1024, a multiple of 512).
func TestTPPlansBindable(t *testing.T) {
	l := llama8B()
	for _, tp := range []int{1, 2, 4, 8} {
		for _, fusion := range []tensorparallel.Fusion{tensorparallel.FuseNone, tensorparallel.FuseReduceAdd} {
			p, err := tensorparallel.Shard(l, tensorparallel.Options{TP: tp, Fusion: fusion})
			if err != nil {
				t.Fatal(err)
			}
			for _, rp := range p.Ranks {
				for _, pl := range []*hopper.Plan{rp.AttnIn, rp.OProj, rp.FFN} {
					for _, k := range pl.Kernels {
						switch {
						case k.GEMM != nil:
							if k.GEMM.WorkspaceBytes() > 0 {
								t.Errorf("tp %d %s: %s needs a split-k workspace", tp, pl.Name, k.GEMM.Name)
							}
						case k.Row != nil:
						default:
							t.Errorf("tp %d %s: kernel %+v is neither a skinny gemm nor a row kernel", tp, pl.Name, k)
						}
					}
				}
			}
		}
	}
}

func llama8B() tensorparallel.Layer {
	return tensorparallel.Layer{Hidden: 4096, Heads: 32, KVHeads: 8, HeadDim: 128, Intermediate: 14336, Eps: 1e-5,
		QKVOut: hopper.BF16, AttnDType: hopper.F32}
}

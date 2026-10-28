package tensorparallel_test

import (
	"fmt"
	"log"

	"github.com/tmc/lib/nvidia/x/egraph/tensorparallel"
)

func ExampleShard() {
	// Llama-3-8B's decoder layer, one decode token, over 4 ranks.
	l := tensorparallel.Layer{Hidden: 4096, Heads: 32, KVHeads: 8, HeadDim: 128, Intermediate: 14336, Eps: 1e-5}
	p, err := tensorparallel.Shard(l, tensorparallel.Options{TP: 4})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("per rank:", p.Local.Heads, "heads,", p.Local.KVHeads, "kv heads,", p.Local.Intermediate, "intermediate")
	for _, rp := range p.Ranks[:2] {
		fmt.Println("rank", rp.Rank, "q heads", rp.QHeads, rp.OProj.Name, rp.FFN.Name)
	}
	for _, c := range p.Collectives() {
		fmt.Println("all-reduce", c.Name, c.Count, "f32")
	}
	// Output:
	// per rank: 8 heads, 2 kv heads, 3584 intermediate
	// rank 0 q heads {0 8} tp_o_proj tp_ffn
	// rank 1 q heads {8 16} tp_peer_o_proj tp_peer_ffn
	// all-reduce attn 4096 f32
	// all-reduce ffn 4096 f32
}

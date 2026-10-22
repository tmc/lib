package main

import (
	"github.com/tmc/lib/gguf"
	"testing"
)

func TestProMetadata(t *testing.T) {
	g := &gguf.File{Metadata: flashMetadata()}
	p := proConfig()
	for k, n := range map[string]int{"block_count": 61, "embedding_length": 7168, "attention.head_count": 128, "attention.q_lora_rank": 1536, "attention.output_group_count": 16, "expert_count": 384, "expert_feed_forward_length": 3072, "attention.indexer.top_k": 1024} {
		g.Metadata["deepseek4."+k] = gguf.Value{TypeID: gguf.TypeUint32, Value: uint32(n)}
	}
	g.Metadata["deepseek4.expert_weights_scale"] = gguf.Value{TypeID: gguf.TypeFloat32, Value: float32(2.5)}
	r := make([]any, p.Layers)
	cl := make([]any, p.Layers)
	for i := range r {
		r[i] = uint32(p.CompressRatio[i])
		cl[i] = float32(10)
	}
	g.Metadata["deepseek4.attention.compress_ratios"] = gguf.Value{TypeID: gguf.TypeArray, Value: r}
	g.Metadata["deepseek4.swiglu_clamp_exp"] = gguf.Value{TypeID: gguf.TypeArray, Value: cl}
	c, err := ConfigFor(g)
	if err != nil {
		t.Fatal(err)
	}
	if c.LayerKind(0) != LayerCoarse || c.LayerKind(1) != LayerCoarse || c.LayerKind(2) != LayerFine {
		t.Fatal("Pro schedule interpreted as Flash")
	}
	if err := c.Validate(); err == nil {
		t.Fatal("Pro admitted to Flash backend")
	}
	if _, err := ReadConfig(g); err == nil {
		t.Fatal("Flash reader admitted Pro")
	}
	g.Metadata["deepseek4.attention.indexer.top_k"] = gguf.Value{Value: uint32(1023)}
	if _, err := ConfigFor(g); err == nil {
		t.Fatal("top-k mutation accepted")
	}
}

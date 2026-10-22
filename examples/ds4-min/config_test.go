package main

import (
	"math"
	"strings"
	"testing"

	"github.com/tmc/lib/gguf"
)

func flashMetadata() map[string]gguf.Value {
	m := map[string]gguf.Value{"general.architecture": {TypeID: gguf.TypeString, Value: "deepseek4"}, "deepseek4.expert_weights_norm": {TypeID: gguf.TypeBool, Value: true}}
	put := func(k string, n int) { m[k] = gguf.Value{TypeID: gguf.TypeUint32, Value: uint32(n)} }
	for _, x := range []struct {
		k string
		n int
	}{
		{"deepseek4.block_count", 43}, {"deepseek4.embedding_length", 4096}, {"deepseek4.vocab_size", 129280},
		{"deepseek4.attention.head_count", 64}, {"deepseek4.attention.head_count_kv", 1}, {"deepseek4.attention.key_length", 512}, {"deepseek4.attention.value_length", 512}, {"deepseek4.rope.dimension_count", 64},
		{"deepseek4.attention.q_lora_rank", 1024}, {"deepseek4.attention.output_lora_rank", 1024}, {"deepseek4.attention.output_group_count", 8}, {"deepseek4.expert_count", 256}, {"deepseek4.expert_used_count", 6}, {"deepseek4.expert_shared_count", 1}, {"deepseek4.expert_feed_forward_length", 2048}, {"deepseek4.hash_layer_count", 3},
		{"deepseek4.attention.sliding_window", 128}, {"deepseek4.attention.indexer.head_count", 64}, {"deepseek4.attention.indexer.key_length", 128}, {"deepseek4.attention.indexer.top_k", 512}, {"deepseek4.hyper_connection.count", 4}, {"deepseek4.hyper_connection.sinkhorn_iterations", 20},
	} {
		put(x.k, x.n)
	}
	for _, x := range []struct {
		k string
		n float32
	}{{"deepseek4.rope.freq_base", 10000}, {"deepseek4.rope.scaling.factor", 16}, {"deepseek4.rope.scaling.yarn_beta_fast", 32}, {"deepseek4.rope.scaling.yarn_beta_slow", 1}, {"deepseek4.attention.compress_rope_freq_base", 160000}, {"deepseek4.expert_weights_scale", 1.5}, {"deepseek4.attention.layer_norm_rms_epsilon", 1e-6}, {"deepseek4.hyper_connection.epsilon", 1e-6}} {
		m[x.k] = gguf.Value{TypeID: gguf.TypeFloat32, Value: x.n}
	}
	m["deepseek4.attention.compress_ratios"] = gguf.Value{TypeID: gguf.TypeArray, Value: func() []any {
		a := make([]any, 43)
		for i := range a {
			a[i] = uint32(0)
			if i >= 2 {
				a[i] = uint32(128)
				if i%2 == 0 {
					a[i] = uint32(4)
				}
			}
		}
		return a
	}()}
	m["deepseek4.swiglu_clamp_exp"] = gguf.Value{TypeID: gguf.TypeArray, Value: func() []any {
		a := make([]any, 43)
		for i := range a {
			a[i] = float32(10)
		}
		return a
	}()}
	return m
}

func TestReadConfig(t *testing.T) {
	g := &gguf.File{Metadata: flashMetadata()}
	c, err := ReadConfig(g)
	if err != nil {
		t.Fatal(err)
	}
	if c.Layers != 43 || c.LayerKind(0) != LayerRaw || c.LayerKind(2) != LayerFine || c.LayerKind(3) != LayerCoarse || !c.HashRouted(2) || c.HashRouted(3) {
		t.Fatalf("unexpected config: %+v", c)
	}
	for _, key := range []string{"deepseek4.block_count", "deepseek4.attention.compress_ratios", "deepseek4.swiglu_clamp_exp"} {
		bad := &gguf.File{Metadata: flashMetadata()}
		v := bad.Metadata[key]
		if key == "deepseek4.block_count" {
			v.Value = uint32(42)
		} else {
			v.Value = "bad"
		}
		bad.Metadata[key] = v
		if _, err := ReadConfig(bad); err == nil || !strings.Contains(err.Error(), key) {
			t.Errorf("mutation %q error = %v", key, err)
		}
	}
}

func TestReadConfigActiveLayerPrefix(t *testing.T) {
	for _, key := range []string{"deepseek4.attention.compress_ratios", "deepseek4.swiglu_clamp_exp"} {
		t.Run(key, func(t *testing.T) {
			for _, tc := range []struct {
				name   string
				mutate func([]any) []any
				fail   bool
			}{
				{"suffix", func(a []any) []any { return append(a, "inactive suffix") }, false},
				{"short", func(a []any) []any { return a[:42] }, true},
				{"bad active", func(a []any) []any { a[42] = "bad"; return a }, true},
			} {
				t.Run(tc.name, func(t *testing.T) {
					g := &gguf.File{Metadata: flashMetadata()}
					v := g.Metadata[key]
					v.Value = tc.mutate(v.Value.([]any))
					g.Metadata[key] = v
					c, err := ReadConfig(g)
					if (err != nil) != tc.fail {
						t.Fatalf("error=%v want failure=%v", err, tc.fail)
					}
					if err == nil && (len(c.CompressRatio) != 43 || len(c.SwiGLUClamp) != 43) {
						t.Fatal("inactive metadata entered active shape")
					}
				})
			}
		})
	}
}

func TestReadConfigRejectsNonfiniteClamp(t *testing.T) {
	for _, n := range []float64{math.NaN(), math.Inf(1), math.Inf(-1)} {
		g := &gguf.File{Metadata: flashMetadata()}
		v := g.Metadata["deepseek4.swiglu_clamp_exp"]
		a := v.Value.([]any)
		a[42] = n
		if _, err := ReadConfig(g); err == nil {
			t.Fatal("nonfinite active clamp accepted")
		}
	}
}

func TestSupportsRejectsUnsupportedExpertQuant(t *testing.T) {
	c, err := ReadConfig(&gguf.File{Metadata: flashMetadata()})
	if err != nil {
		t.Fatal(err)
	}
	g := &gguf.File{Tensors: []gguf.Tensor{{Name: "blk.0.ffn_gate_exps.weight", TypeID: 16}}}
	if err := Supports(c, g); err != nil {
		t.Fatal(err)
	}
	g.Tensors[0].TypeID = 12
	if err := Supports(c, g); err == nil || !strings.Contains(err.Error(), "ffn_gate_exps") {
		t.Fatalf("Q4 expert accepted: %v", err)
	}
}

func TestSupportsRejectsUnsupportedNonExpertQuant(t *testing.T) {
	c, err := ReadConfig(&gguf.File{Metadata: flashMetadata()})
	if err != nil {
		t.Fatal(err)
	}
	g := &gguf.File{Tensors: []gguf.Tensor{{Name: "blk.0.hc_ffn_fn.weight", TypeID: typeQ4K}}}
	if err := Supports(c, g); err == nil || !strings.Contains(err.Error(), "hc_ffn_fn.weight") {
		t.Fatalf("unsupported non-expert tensor accepted: %v", err)
	}
}

func TestConfigValidateRejectsUnknownLayerRoute(t *testing.T) {
	c, err := ReadConfig(&gguf.File{Metadata: flashMetadata()})
	if err != nil {
		t.Fatal(err)
	}
	if got := c.LayerKind(43); got != LayerInvalid {
		t.Fatalf("out-of-range LayerKind = %v, want LayerInvalid", got)
	}
	c.CompressRatio[3] = 7
	if err := c.Validate(); err == nil || !strings.Contains(err.Error(), "layer 3") {
		t.Fatalf("unknown compression route error = %v", err)
	}
	if err := Supports(c, &gguf.File{}); err == nil || !strings.Contains(err.Error(), "layer 3") {
		t.Fatalf("Supports accepted unknown compression route: %v", err)
	}
}

func TestConfigValidateRejectsNonFiniteClamp(t *testing.T) {
	c, err := ReadConfig(&gguf.File{Metadata: flashMetadata()})
	if err != nil {
		t.Fatal(err)
	}
	c.SwiGLUClamp[2] = float32(math.NaN())
	if err := c.Validate(); err == nil || !strings.Contains(err.Error(), "SwiGLU clamp") {
		t.Fatalf("non-finite clamp error = %v", err)
	}
}

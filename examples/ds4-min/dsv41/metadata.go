package dsv41

import (
	"fmt"
	"math"

	"github.com/tmc/lib/examples/ds4-min/dsv41engram"
	"github.com/tmc/lib/gguf"
)

var flashIntegers = map[string]uint32{
	"vocab_size": 129280, "hidden_size": 5120, "moe_intermediate_size": 2304, "num_hidden_layers": 40,
	"num_attention_heads": 64, "num_key_value_heads": 1, "head_dim": 512, "qk_rope_head_dim": 64,
	"q_lora_rank": 1280, "o_lora_rank": 1024, "o_groups": 8, "n_routed_experts": 384, "n_shared_experts": 1,
	"num_experts_per_tok": 6, "max_position_embeddings": 1048576, "sliding_window": 128,
	"index_n_heads": 32, "index_head_dim": 128, "index_topk": 512, "candidate_source_layer_id": 20,
	"candidate_topk_blocks": 2048, "candidate_block_size": 8, "hc_mult": 4, "hc_sinkhorn_iters": 20,
	"rope_theta": 10000, "compress_rope_theta": 160000, "engram.compressed_vocab_size": 99092,
}

var flashFloats = map[string]float32{
	"rms_norm_eps": 1e-20, "hc_eps": 1e-6, "swiglu_limit": 10, "routed_scaling_factor": 1.5,
	"rope_scaling.factor": 16, "rope_scaling.beta_fast": 32, "rope_scaling.beta_slow": 1,
	"rope_scaling.original_max_position_embeddings": 65536,
}

// ValidateMetadata checks the actual deepseek41 Flash contract, including the
// layer-owner schedule and complete Engram hash metadata. It returns a layout
// that copies its token map; later metadata edits cannot change request hashes.
// This does not validate tensor extents or establish model/device parity.
func ValidateMetadata(g *gguf.File) (*dsv41engram.Layout, error) {
	if g == nil {
		return nil, fmt.Errorf("missing V4.1 metadata")
	}
	if g.Metadata["general.architecture"].TypeID != gguf.TypeString || g.Metadata["general.architecture"].Value != "deepseek41" {
		return nil, fmt.Errorf("expected deepseek41 architecture")
	}
	for key, want := range flashIntegers {
		v := g.Metadata["deepseek41."+key]
		got, ok := v.Value.(uint32)
		if v.TypeID != gguf.TypeUint32 || !ok || got != want {
			return nil, fmt.Errorf("invalid deepseek41.%s", key)
		}
	}
	for key, want := range flashFloats {
		got, ok := metadataFloat(g.Metadata["deepseek41."+key])
		if !ok || math.IsNaN(float64(got)) || float32(math.Abs(float64(got-want))) > max(float32(math.Abs(float64(want)))*1e-5, 1e-30) {
			return nil, fmt.Errorf("invalid deepseek41.%s", key)
		}
	}
	for key, want := range map[string]string{"scoring_func": "sqrtsoftplus", "hidden_act": "silu", "topk_method": "noaux_tc", "engram.encoding": "e4m3_e8m0_32_row264"} {
		if g.Metadata["deepseek41."+key].TypeID != gguf.TypeString || g.Metadata["deepseek41."+key].Value != want {
			return nil, fmt.Errorf("invalid deepseek41.%s", key)
		}
	}
	if g.Metadata["deepseek41.norm_topk_prob"].TypeID != gguf.TypeBool || g.Metadata["deepseek41.norm_topk_prob"].Value != true {
		return nil, fmt.Errorf("invalid norm_topk_prob")
	}
	arrays := map[string][]uint32{"kv_source_layer_ids": {2, 8, 14, 20}, "index_source_layer_ids": {2, 8, 14, 20, 24, 28, 32, 36}, "engram.layer_ids": {1, 14}, "engram.rows": {384006168, 384016682}}
	ratios := make([]uint32, 40)
	for i := range ratios {
		l, _ := Schedule(i)
		ratios[i] = uint32(l.Ratio)
	}
	arrays["compress_ratios"] = ratios
	for key, want := range arrays {
		values, err := metadataU32(g, "deepseek41."+key, len(want))
		if err != nil {
			return nil, err
		}
		for i := range want {
			if values[i] != want[i] {
				return nil, fmt.Errorf("invalid deepseek41.%s index %d", key, i)
			}
		}
	}
	tokenMap, err := metadataU32(g, "deepseek41.engram.token_map", 129280)
	if err != nil {
		return nil, err
	}
	primes, err := metadataU32(g, "deepseek41.engram.primes", 48)
	if err != nil {
		return nil, err
	}
	values, ok := g.Metadata["deepseek41.engram.multipliers"].Value.([]any)
	if g.Metadata["deepseek41.engram.multipliers"].TypeID != gguf.TypeArray || !ok || len(values) != 8 {
		return nil, fmt.Errorf("invalid engram multipliers")
	}
	padValue := g.Metadata["deepseek41.engram.pad_id"]
	pad, ok := padValue.Value.(uint32)
	if padValue.TypeID != gguf.TypeUint32 || !ok || pad >= 99092 {
		return nil, fmt.Errorf("invalid engram pad id")
	}
	s := dsv41engram.LayoutSpec{TokenMap: tokenMap, CompressedVocabSize: 99092, PadID: uint32(pad), Rows: [2]uint32{384006168, 384016682}}
	for i, value := range values {
		n, ok := value.(uint64)
		if !ok {
			return nil, fmt.Errorf("invalid engram multiplier type")
		}
		s.Multipliers[i/4][i%4] = n
	}
	for i, p := range primes {
		s.Primes[i/24][i%24] = p
	}
	return dsv41engram.NewLayout(s)
}

func metadataFloat(v gguf.Value) (float32, bool) {
	switch n := v.Value.(type) {
	case float32:
		return n, v.TypeID == gguf.TypeFloat32
	case float64:
		return float32(n), v.TypeID == gguf.TypeFloat64
	case uint32:
		return float32(n), v.TypeID == gguf.TypeUint32
	case int32:
		return float32(n), v.TypeID == gguf.TypeInt32
	}
	return 0, false
}

func metadataU32(g *gguf.File, key string, n int) ([]uint32, error) {
	v := g.Metadata[key]
	values, ok := v.Value.([]any)
	if v.TypeID != gguf.TypeArray || !ok || len(values) != n {
		return nil, fmt.Errorf("invalid %s array", key)
	}
	out := make([]uint32, n)
	for i, v := range values {
		u, ok := v.(uint32)
		if !ok {
			return nil, fmt.Errorf("invalid %s index %d type", key, i)
		}
		out[i] = u
	}
	return out, nil
}

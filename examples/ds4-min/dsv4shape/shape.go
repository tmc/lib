package dsv4shape

import "fmt"

// Variant identifies a DeepSeek V4 model layout.
type Variant uint8

const (
	Flash Variant = iota
	Pro
)

// Config is a known V4 shape. All fields participate in shape selection.
// Values match ds4.c at 11e8cad; V4.1 has a separate graph contract.
type Config struct {
	Variant                                                   Variant
	Layers, Embed, Vocab, Heads, HeadsKV                      int
	HeadDim, ValueDim, RopeDim, QLoRA, OutLoRA, OutGroups     int
	Experts, ExpertsUsed, ExpertsShared, ExpertFF, HashLayers int
	SlidingWindow, IndexerHeads, IndexerHeadDim, IndexerTopK  int
	HC, SinkhornIters                                         int
	ExpertWeightScale                                         float32
}

// For returns the known configuration for v, or an error for an unknown variant.
func For(v Variant) (Config, error) {
	c := Config{Variant: v, Layers: 43, Embed: 4096, Vocab: 129280, Heads: 64,
		HeadsKV: 1, HeadDim: 512, ValueDim: 512, RopeDim: 64, QLoRA: 1024,
		OutLoRA: 1024, OutGroups: 8, Experts: 256, ExpertsUsed: 6,
		ExpertsShared: 1, ExpertFF: 2048, HashLayers: 3, SlidingWindow: 128,
		IndexerHeads: 64, IndexerHeadDim: 128, IndexerTopK: 512, HC: 4,
		SinkhornIters: 20, ExpertWeightScale: 1.5}
	switch v {
	case Flash:
	case Pro:
		c.Layers, c.Embed, c.Heads = 61, 7168, 128
		c.QLoRA, c.OutGroups, c.Experts = 1536, 16, 384
		c.ExpertFF, c.IndexerTopK, c.ExpertWeightScale = 3072, 1024, 2.5
	default:
		return Config{}, fmt.Errorf("shape: unknown variant %d", v)
	}
	return c, nil
}

// Validate refuses mutated or partial known shapes.
func (c Config) Validate() error {
	want, err := For(c.Variant)
	if err != nil {
		return err
	}
	if c != want {
		return fmt.Errorf("shape: dimensions do not match variant %d", c.Variant)
	}
	return nil
}

// Layer describes the state and weights bound for one layer.
type Layer struct {
	CompressRatio       int
	Indexer, HashRouter bool
}

// Layer returns the family-dependent layer schedule.
func (c Config) Layer(i int) (Layer, error) {
	if err := c.Validate(); err != nil {
		return Layer{}, err
	}
	if i < 0 || i >= c.Layers {
		return Layer{}, fmt.Errorf("shape: layer %d outside [0,%d)", i, c.Layers)
	}
	ratio := 128
	if i < 2 {
		if c.Variant == Flash {
			ratio = 0
		}
	} else if i%2 == 0 {
		ratio = 4
	}
	return Layer{CompressRatio: ratio, Indexer: ratio == 4, HashRouter: i < c.HashLayers}, nil
}

func (c Config) fields() map[string]int {
	return map[string]int{
		"block_count": c.Layers, "embedding_length": c.Embed, "vocab_size": c.Vocab,
		"attention.head_count": c.Heads, "attention.head_count_kv": c.HeadsKV,
		"attention.key_length": c.HeadDim, "attention.value_length": c.ValueDim,
		"rope.dimension_count": c.RopeDim, "attention.q_lora_rank": c.QLoRA,
		"attention.output_lora_rank": c.OutLoRA, "attention.output_group_count": c.OutGroups,
		"expert_count": c.Experts, "expert_used_count": c.ExpertsUsed,
		"expert_shared_count": c.ExpertsShared, "expert_feed_forward_length": c.ExpertFF,
		"hash_layer_count": c.HashLayers, "attention.sliding_window": c.SlidingWindow,
		"attention.indexer.head_count": c.IndexerHeads, "attention.indexer.key_length": c.IndexerHeadDim,
		"attention.indexer.top_k": c.IndexerTopK, "hyper_connection.count": c.HC,
		"hyper_connection.sinkhorn_iterations": c.SinkhornIters,
	}
}

// Select matches integer GGUF metadata against the complete known shape tables.
// Keys include the deepseek4 prefix. The caller must separately validate the
// architecture, floating-point metadata, arrays and tensor layouts.
func Select(md map[string]uint64) (Config, error) {
	for _, v := range []Variant{Flash, Pro} {
		c, _ := For(v)
		match := true
		for k, n := range c.fields() {
			got, ok := md["deepseek4."+k]
			if !ok || got != uint64(n) {
				match = false
				break
			}
		}
		if match {
			return c, nil
		}
	}
	return Config{}, fmt.Errorf("shape: unsupported or incomplete deepseek4 metadata")
}

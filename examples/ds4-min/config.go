package main

import (
	"fmt"
	"math"
	"strings"

	"github.com/tmc/lib/gguf"
)

// LayerKind describes the attention state used by a transformer layer.
type LayerKind uint8

const (
	LayerRaw LayerKind = iota
	LayerFine
	LayerCoarse
	// LayerInvalid identifies a configuration whose compression ratio is not
	// implemented by this backend. Callers must not silently treat an
	// unknown ratio as the coarse path.
	LayerInvalid
)

// Config is the model configuration used by ds4-min.  The values are read
// from GGUF metadata; the Flash shape is the only shape implemented by the
// current kernels.
type Config struct {
	Variant                                                           string
	Layers, Embed, Vocab                                              int
	Heads, HeadsKV                                                    int
	HeadDim, ValueDim                                                 int
	RopeDim                                                           int
	QLoRA, OutLoRA, OutGroups                                         int
	Experts, ExpertsUsed, ExpertsShared, ExpertFF, HashLayers         int
	SlidingWindow                                                     int
	IndexerHeads, IndexerHeadDim, IndexerTopK                         int
	HC, SinkhornIters                                                 int
	CompressRatio                                                     []int
	SwiGLUClamp                                                       []float32
	RMSEps, HCEps, ExpertWeightScale                                  float32
	RopeBase, RopeScale, YarnBetaFast, YarnBetaSlow, CompressRopeBase float32
	RopeOrigContext                                                   int
	ExpertWeightsNorm                                                 bool
}

// Flash is the supported DeepSeek V4 Flash shape.
var Flash = Config{
	Variant: "flash", Layers: 43, Embed: 4096, Vocab: 129280,
	Heads: 64, HeadsKV: 1, HeadDim: 512, ValueDim: 512, RopeDim: 64,
	QLoRA: 1024, OutLoRA: 1024, OutGroups: 8,
	Experts: 256, ExpertsUsed: 6, ExpertsShared: 1, ExpertFF: 2048, HashLayers: 3,
	SlidingWindow: 128, IndexerHeads: 64, IndexerHeadDim: 128, IndexerTopK: 512,
	HC: 4, SinkhornIters: 20, RMSEps: 1e-6, HCEps: 1e-6,
	RopeBase: 10000, RopeScale: 16, YarnBetaFast: 32, YarnBetaSlow: 1,
	CompressRopeBase: 160000, RopeOrigContext: 65536,
	ExpertWeightScale: 1.5, ExpertWeightsNorm: true,
}

func init() {
	Flash.CompressRatio = make([]int, Flash.Layers)
	Flash.SwiGLUClamp = make([]float32, Flash.Layers)
	for i := range Flash.CompressRatio {
		if i >= 2 {
			Flash.CompressRatio[i] = 128
			if i%2 == 0 {
				Flash.CompressRatio[i] = 4
			}
		}
		Flash.SwiGLUClamp[i] = 10
	}
}

func (c *Config) LayerKind(i int) LayerKind {
	if c == nil || i < 0 || i >= len(c.CompressRatio) {
		return LayerInvalid
	}
	if c.CompressRatio[i] == 0 {
		return LayerRaw
	}
	switch c.CompressRatio[i] {
	case 4:
		return LayerFine
	case 128:
		return LayerCoarse
	default:
		return LayerInvalid
	}
}

func (c *Config) HashRouted(i int) bool { return i >= 0 && i < c.HashLayers }

// Validate checks the in-memory configuration before a backend consumes it.
// ReadConfig already validates GGUF metadata, but callers can construct or
// mutate Config values directly. Backends must reject those values rather
// than interpreting an unknown layer ratio as a supported route.
func (c *Config) Validate() error {
	if c == nil {
		return fmt.Errorf("config: missing configuration")
	}
	if c.Variant != Flash.Variant {
		return fmt.Errorf("config: unsupported variant %q", c.Variant)
	}
	if c.Layers != Flash.Layers {
		return fmt.Errorf("config: layers %d, want %d", c.Layers, Flash.Layers)
	}
	if len(c.CompressRatio) != c.Layers {
		return fmt.Errorf("config: compression ratios %d, want %d", len(c.CompressRatio), c.Layers)
	}
	if len(c.SwiGLUClamp) != c.Layers {
		return fmt.Errorf("config: SwiGLU clamps %d, want %d", len(c.SwiGLUClamp), c.Layers)
	}
	for i, ratio := range c.CompressRatio {
		want := Flash.CompressRatio[i]
		if ratio != want {
			return fmt.Errorf("config: layer %d compression ratio %d, want %d", i, ratio, want)
		}
		if c.LayerKind(i) == LayerInvalid {
			return fmt.Errorf("config: layer %d has unsupported compression ratio %d", i, ratio)
		}
	}
	for i, clamp := range c.SwiGLUClamp {
		if math.IsNaN(float64(clamp)) || math.IsInf(float64(clamp), 0) || clamp != Flash.SwiGLUClamp[i] {
			return fmt.Errorf("config: layer %d SwiGLU clamp %g, want %g", i, clamp, Flash.SwiGLUClamp[i])
		}
	}
	if c.Embed <= 0 || c.Vocab <= 0 || c.Heads <= 0 || c.HeadsKV <= 0 || c.HeadDim <= 0 || c.ValueDim <= 0 {
		return fmt.Errorf("config: non-positive model dimensions")
	}
	if c.Experts <= 0 || c.ExpertsUsed <= 0 || c.ExpertFF <= 0 || c.HashLayers < 0 || c.HashLayers > c.Layers {
		return fmt.Errorf("config: invalid expert or hash-layer counts")
	}
	for name, value := range map[string]float32{
		"RMS epsilon": c.RMSEps, "HC epsilon": c.HCEps,
		"expert weight scale": c.ExpertWeightScale, "rope base": c.RopeBase,
		"rope scale": c.RopeScale, "compress rope base": c.CompressRopeBase,
	} {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) || value <= 0 {
			return fmt.Errorf("config: invalid %s %g", name, value)
		}
	}
	return nil
}

func metadataValue(g *gguf.File, key string) (gguf.Value, error) {
	if g == nil {
		return gguf.Value{}, fmt.Errorf("metadata %q: missing GGUF", key)
	}
	v, ok := g.Metadata[key]
	if !ok {
		return gguf.Value{}, fmt.Errorf("metadata %q: missing", key)
	}
	return v, nil
}

func metadataInt(g *gguf.File, key string, optional bool, def int) (int, error) {
	v, err := metadataValue(g, key)
	if err != nil {
		if optional && stringsHasMissing(err) {
			return def, nil
		}
		return 0, err
	}
	maxInt := uint64(^uint(0) >> 1)
	if n, ok := v.Uint(); ok && n <= maxInt {
		return int(n), nil
	}
	if n, ok := v.Int(); ok && n >= 0 && uint64(n) <= maxInt {
		return int(n), nil
	}
	return 0, fmt.Errorf("metadata %q: want non-negative integer", key)
}

func metadataFloat(g *gguf.File, key string, optional bool, def float32) (float32, error) {
	v, err := metadataValue(g, key)
	if err != nil {
		if optional && stringsHasMissing(err) {
			return def, nil
		}
		return 0, err
	}
	var f float64
	switch n := v.Value.(type) {
	case float32:
		f = float64(n)
	case float64:
		f = n
	default:
		return 0, fmt.Errorf("metadata %q: want float", key)
	}
	if math.IsNaN(f) || math.IsInf(f, 0) || float32(f) == 0 && f != 0 {
		return 0, fmt.Errorf("metadata %q: invalid float", key)
	}
	return float32(f), nil
}

func metadataBool(g *gguf.File, key string) (bool, error) {
	v, err := metadataValue(g, key)
	if err != nil {
		return false, err
	}
	b, ok := v.Value.(bool)
	if !ok {
		return false, fmt.Errorf("metadata %q: want bool", key)
	}
	return b, nil
}

func metadataArray(g *gguf.File, key string) ([]any, error) {
	v, err := metadataValue(g, key)
	if err != nil {
		return nil, err
	}
	if v.TypeID != gguf.TypeArray {
		return nil, fmt.Errorf("metadata %q: want array", key)
	}
	a, ok := v.Value.([]any)
	if !ok {
		return nil, fmt.Errorf("metadata %q: want array", key)
	}
	return a, nil
}

func stringsHasMissing(err error) bool {
	return err != nil && strings.Contains(err.Error(), ": missing")
}

// ReadConfig validates the Flash contract supported by current backends.
func ReadConfig(g *gguf.File) (*Config, error) {
	return readConfig(g, Flash)
}

func readConfig(g *gguf.File, shape Config) (*Config, error) {
	c := shape
	c.CompressRatio = append([]int(nil), shape.CompressRatio...)
	c.SwiGLUClamp = append([]float32(nil), shape.SwiGLUClamp...)
	if g == nil {
		return nil, fmt.Errorf("read config: missing GGUF")
	}
	if v, ok := g.Metadata["general.architecture"]; ok {
		if s, ok := v.Value.(string); !ok || s != "deepseek4" {
			return nil, fmt.Errorf("metadata %q: want deepseek4", "general.architecture")
		}
	}
	if v, ok := g.Metadata["deepseek4.checkpoint_variant"]; ok {
		if s, ok := v.Value.(string); !ok || s != "" {
			return nil, fmt.Errorf("metadata %q: unsupported checkpoint variant", "deepseek4.checkpoint_variant")
		}
	}
	ints := []struct {
		key  string
		dst  *int
		want int
	}{
		{"deepseek4.block_count", &c.Layers, shape.Layers}, {"deepseek4.embedding_length", &c.Embed, shape.Embed}, {"deepseek4.vocab_size", &c.Vocab, shape.Vocab},
		{"deepseek4.attention.head_count", &c.Heads, shape.Heads}, {"deepseek4.attention.head_count_kv", &c.HeadsKV, shape.HeadsKV}, {"deepseek4.attention.key_length", &c.HeadDim, shape.HeadDim}, {"deepseek4.attention.value_length", &c.ValueDim, shape.ValueDim},
		{"deepseek4.rope.dimension_count", &c.RopeDim, shape.RopeDim}, {"deepseek4.attention.q_lora_rank", &c.QLoRA, shape.QLoRA}, {"deepseek4.attention.output_lora_rank", &c.OutLoRA, shape.OutLoRA}, {"deepseek4.attention.output_group_count", &c.OutGroups, shape.OutGroups},
		{"deepseek4.expert_count", &c.Experts, shape.Experts}, {"deepseek4.expert_used_count", &c.ExpertsUsed, shape.ExpertsUsed}, {"deepseek4.expert_shared_count", &c.ExpertsShared, shape.ExpertsShared}, {"deepseek4.expert_feed_forward_length", &c.ExpertFF, shape.ExpertFF}, {"deepseek4.hash_layer_count", &c.HashLayers, shape.HashLayers},
		{"deepseek4.attention.sliding_window", &c.SlidingWindow, shape.SlidingWindow}, {"deepseek4.attention.indexer.head_count", &c.IndexerHeads, shape.IndexerHeads}, {"deepseek4.attention.indexer.key_length", &c.IndexerHeadDim, shape.IndexerHeadDim}, {"deepseek4.attention.indexer.top_k", &c.IndexerTopK, shape.IndexerTopK},
		{"deepseek4.hyper_connection.count", &c.HC, shape.HC}, {"deepseek4.hyper_connection.sinkhorn_iterations", &c.SinkhornIters, shape.SinkhornIters},
	}
	for _, item := range ints {
		got, err := metadataInt(g, item.key, false, 0)
		if err != nil {
			return nil, err
		}
		if got != item.want {
			return nil, fmt.Errorf("metadata %q: got %d, want %d", item.key, got, item.want)
		}
		*item.dst = got
	}
	for _, item := range []struct {
		key      string
		dst      *float32
		want     float32
		optional bool
	}{{"deepseek4.rope.freq_base", &c.RopeBase, shape.RopeBase, false}, {"deepseek4.rope.scaling.factor", &c.RopeScale, shape.RopeScale, true}, {"deepseek4.rope.scaling.yarn_beta_fast", &c.YarnBetaFast, shape.YarnBetaFast, true}, {"deepseek4.rope.scaling.yarn_beta_slow", &c.YarnBetaSlow, shape.YarnBetaSlow, true}, {"deepseek4.attention.compress_rope_freq_base", &c.CompressRopeBase, shape.CompressRopeBase, false}, {"deepseek4.expert_weights_scale", &c.ExpertWeightScale, shape.ExpertWeightScale, false}, {"deepseek4.attention.layer_norm_rms_epsilon", &c.RMSEps, shape.RMSEps, false}, {"deepseek4.hyper_connection.epsilon", &c.HCEps, shape.HCEps, false}} {
		got, err := metadataFloat(g, item.key, item.optional, item.want)
		if err != nil {
			return nil, err
		}
		if math.Abs(float64(got-item.want)) > 1e-6*math.Max(1, math.Abs(float64(item.want))) {
			return nil, fmt.Errorf("metadata %q: got %g, want %g", item.key, got, item.want)
		}
		*item.dst = got
	}
	if c.RopeOrigContext, _ = metadataInt(g, "deepseek4.rope.scaling.original_context_length", true, shape.RopeOrigContext); c.RopeOrigContext != shape.RopeOrigContext {
		return nil, fmt.Errorf("metadata %q: got %d, want %d", "deepseek4.rope.scaling.original_context_length", c.RopeOrigContext, shape.RopeOrigContext)
	}
	if c.ExpertWeightsNorm, _ = metadataBool(g, "deepseek4.expert_weights_norm"); !c.ExpertWeightsNorm {
		return nil, fmt.Errorf("metadata %q: got false, want true", "deepseek4.expert_weights_norm")
	}
	for _, item := range []struct {
		key  string
		want []int
	}{{"deepseek4.attention.compress_ratios", shape.CompressRatio}} {
		a, err := metadataArray(g, item.key)
		if err != nil {
			return nil, err
		}
		if len(a) < len(item.want) {
			return nil, fmt.Errorf("metadata %q: got %d entries, want at least %d", item.key, len(a), len(item.want))
		}
		// Upstream validates only the active-layer prefix. Some model files
		// include additional entries; they are not part of this runtime shape.
		for i, v := range a[:len(item.want)] {
			n, ok := gguf.Value{Value: v}.Int()
			if !ok || n < 0 || int(n) != item.want[i] {
				return nil, fmt.Errorf("metadata %q layer %d: got %v, want %d", item.key, i, v, item.want[i])
			}
		}
	}
	a, err := metadataArray(g, "deepseek4.swiglu_clamp_exp")
	if err != nil {
		return nil, err
	}
	if len(a) < c.Layers {
		return nil, fmt.Errorf("metadata %q: got %d entries, want at least %d", "deepseek4.swiglu_clamp_exp", len(a), c.Layers)
	}
	for i, v := range a[:c.Layers] {
		var f float64
		switch n := v.(type) {
		case float32:
			f = float64(n)
		case float64:
			f = n
		default:
			return nil, fmt.Errorf("metadata %q layer %d: want float", "deepseek4.swiglu_clamp_exp", i)
		}
		if math.IsNaN(f) || math.IsInf(f, 0) || math.Abs(f-10) > 1e-5 {
			return nil, fmt.Errorf("metadata %q layer %d: got %g, want 10", "deepseek4.swiglu_clamp_exp", i, f)
		}
	}
	if n, err := metadataInt(g, "deepseek4.expert_group_count", true, 0); err != nil || n != 0 {
		return nil, fmt.Errorf("metadata %q: want 0", "deepseek4.expert_group_count")
	}
	if n, err := metadataInt(g, "deepseek4.expert_group_used_count", true, 0); err != nil || n != 0 {
		return nil, fmt.Errorf("metadata %q: want 0", "deepseek4.expert_group_used_count")
	}
	if err := c.ValidateShape(); err != nil {
		return nil, fmt.Errorf("read config: %w", err)
	}
	return &c, nil
}

// Supports checks the dimensions and tensor types implemented by ds4-min.
func Supports(c *Config, g *gguf.File) error {
	if err := c.Validate(); err != nil {
		return fmt.Errorf("supports: %w", err)
	}
	if c.Variant != Flash.Variant || c.Layers != Flash.Layers || c.Embed != Flash.Embed || c.Vocab != Flash.Vocab ||
		c.Heads != Flash.Heads || c.HeadsKV != Flash.HeadsKV || c.HeadDim != Flash.HeadDim || c.ValueDim != Flash.ValueDim || c.RopeDim != Flash.RopeDim ||
		c.QLoRA != Flash.QLoRA || c.OutLoRA != Flash.OutLoRA || c.OutGroups != Flash.OutGroups || c.Experts != Flash.Experts ||
		c.ExpertsUsed != Flash.ExpertsUsed || c.ExpertsShared != Flash.ExpertsShared || c.ExpertFF != Flash.ExpertFF || c.HashLayers != Flash.HashLayers ||
		c.SlidingWindow != Flash.SlidingWindow || c.IndexerHeads != Flash.IndexerHeads || c.IndexerHeadDim != Flash.IndexerHeadDim || c.IndexerTopK != Flash.IndexerTopK ||
		c.HC != Flash.HC || c.SinkhornIters != Flash.SinkhornIters || c.RopeOrigContext != Flash.RopeOrigContext ||
		c.RMSEps != Flash.RMSEps || c.HCEps != Flash.HCEps || c.ExpertWeightScale != Flash.ExpertWeightScale || c.RopeBase != Flash.RopeBase ||
		c.RopeScale != Flash.RopeScale || c.YarnBetaFast != Flash.YarnBetaFast || c.YarnBetaSlow != Flash.YarnBetaSlow || c.CompressRopeBase != Flash.CompressRopeBase || !c.ExpertWeightsNorm ||
		!sameInts(c.CompressRatio, Flash.CompressRatio) || !sameFloats(c.SwiGLUClamp, Flash.SwiGLUClamp) {
		return fmt.Errorf("supports: unsupported model shape")
	}
	if g == nil {
		return fmt.Errorf("supports: missing GGUF")
	}
	// Apply the shared non-expert type contract at admission time as well as
	// in LayoutOf. Otherwise a malformed dense or normalization tensor could
	// pass the legacy expert-only checks and fail later during binding.
	if err := validateTensorTypes(g); err != nil {
		return fmt.Errorf("supports: %w", err)
	}
	for _, t := range g.Tensors {
		if strings.Contains(t.Name, ".ffn_gate_exps") || strings.Contains(t.Name, ".ffn_up_exps") {
			if t.TypeID != 16 {
				return fmt.Errorf("supports: tensor %q: unsupported type %d", t.Name, t.TypeID)
			}
		}
		if strings.Contains(t.Name, ".ffn_down_exps") && t.TypeID != 10 {
			return fmt.Errorf("supports: tensor %q: unsupported type %d", t.Name, t.TypeID)
		}
	}
	return nil
}

func sameInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func sameFloats(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

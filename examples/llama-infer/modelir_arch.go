package main

// The architecture record from modelir, behind LLAMA_INFER_MODELIR=1.
//
// With the switch set, LoadModelConfig also resolves the model directory
// through internal/llamamodel and refuses to load when the two records
// disagree on any fact the engine reads. The default path is unchanged: the
// switch adds a check, not a source. Phase 1 of the modelir adoption plan
// (docs/LLAMA_INFER_MODELIR_ADOPTION.md) ends when ModelConfig is gone and
// the engine reads llamamodel.Arch; this is its first half, and the check is
// what makes the second half safe to take.
//
// UNQUALIFIED against a downloaded checkpoint: TestModelirLoadMatchesLegacy
// runs on the Llama 3.2 1B configuration fixture under
// internal/llamamodel/testdata, never on a model directory of the user's.

import (
	"errors"
	"fmt"
	"os"
	"slices"
	"strings"

	"github.com/tmc/lib/examples/llama-infer/internal/llamamodel"
	"github.com/tmc/modelir"
)

func modelirEnabled() bool { return os.Getenv("LLAMA_INFER_MODELIR") == "1" }

// modelConfigFromArch projects a resolved architecture onto ModelConfig,
// which the engine still consumes. It refuses what ModelConfig cannot say:
// layers of different shapes, a norm other than RMSNorm, an MLP other than
// SwiGLU with SiLU, since the engine has exactly those and the JSON loader
// never checked. fuse_egraph is a loader flag, not architecture, and stays
// false.
func modelConfigFromArch(a *llamamodel.Arch) (*ModelConfig, error) {
	if a == nil || len(a.Layers) == 0 {
		return nil, errors.New("architecture has no layers")
	}
	first := a.Layers[0]
	for _, l := range a.Layers[1:] {
		if l.Heads != first.Heads || l.KVHeads != first.KVHeads || l.HeadDim != first.HeadDim || l.Type != first.Type || l.SlidingWindow != first.SlidingWindow {
			return nil, fmt.Errorf("layer %d differs from layer 0; ModelConfig has one layer shape", l.Index)
		}
	}
	if first.Type != modelir.LayerTypeDense {
		return nil, fmt.Errorf("layer type %q; the engine has dense attention only", first.Type)
	}
	if a.Norm != modelir.NormRMSNorm {
		return nil, fmt.Errorf("norm %q; the engine has RMSNorm only", a.Norm)
	}
	if a.MLP != modelir.MLPSwiGLU || a.Activation != modelir.MLPActivationSiLU {
		return nil, fmt.Errorf("MLP %s with %s; the engine has SwiGLU with SiLU only", a.MLP, a.Activation)
	}
	cfg := &ModelConfig{
		ModelType:         a.Family,
		HiddenSize:        a.Hidden,
		IntermediateSize:  a.Intermediate,
		NumHiddenLayers:   len(a.Layers),
		NumAttentionHeads: first.Heads,
		NumKeyValueHeads:  first.KVHeads,
		ExplicitHeadDim:   first.HeadDim,
		VocabSize:         a.Vocab,
		RmsNormEps:        float32(a.NormEps),
		RopeTheta:         float32(a.RoPE.Theta),
		MaxSeqLen:         a.MaxPositions,
		TieWordEmbeddings: a.TieEmbed,
		EOSTokenIDs:       slices.Clone(a.EOS),
	}
	if len(cfg.EOSTokenIDs) > 0 {
		cfg.EOSTokenID = cfg.EOSTokenIDs[0]
	}
	if cfg.MaxSeqLen == 0 {
		cfg.MaxSeqLen = 2048 // the JSON loader's default for a missing max_position_embeddings
	}
	return cfg, nil
}

// modelConfigDiff names the facts the engine reads on which a and b differ,
// by their config.json keys. head_dim is compared through HeadDim, so an
// explicit value and the derived one agree; the raw JSON and the nested
// text_config are loader intermediates and are not compared.
func modelConfigDiff(a, b *ModelConfig) []string {
	var diff []string
	add := func(key string, same bool) {
		if !same {
			diff = append(diff, key)
		}
	}
	add("model_type", a.ModelType == b.ModelType)
	add("hidden_size", a.HiddenSize == b.HiddenSize)
	add("intermediate_size", a.IntermediateSize == b.IntermediateSize)
	add("num_hidden_layers", a.NumHiddenLayers == b.NumHiddenLayers)
	add("num_attention_heads", a.NumAttentionHeads == b.NumAttentionHeads)
	add("num_key_value_heads", a.NumKeyValueHeads == b.NumKeyValueHeads)
	add("head_dim", a.HeadDim() == b.HeadDim())
	add("vocab_size", a.VocabSize == b.VocabSize)
	add("rms_norm_eps", a.RmsNormEps == b.RmsNormEps)
	add("rope_theta", a.RopeTheta == b.RopeTheta)
	add("max_position_embeddings", a.MaxSeqLen == b.MaxSeqLen)
	add("tie_word_embeddings", a.TieWordEmbeddings == b.TieWordEmbeddings)
	add("fuse_egraph", a.FuseEGraph == b.FuseEGraph)
	add("eos_token_id", a.EOSTokenID == b.EOSTokenID && slices.Equal(a.EOSTokenIDs, b.EOSTokenIDs))
	return diff
}

// checkModelConfigAgainstModelir is what LLAMA_INFER_MODELIR=1 adds to
// LoadModelConfig: the directory resolved through modelir must say what the
// JSON loader read, fact for fact, or the load fails naming the facts.
func checkModelConfigAgainstModelir(dir string, legacy *ModelConfig) error {
	arch, _, err := llamamodel.Load(dir)
	if err != nil {
		return fmt.Errorf("modelir: %w", err)
	}
	got, err := modelConfigFromArch(arch)
	if err != nil {
		return fmt.Errorf("modelir: %w", err)
	}
	if diff := modelConfigDiff(got, legacy); len(diff) > 0 {
		return fmt.Errorf("modelir disagrees with config.json on %s", strings.Join(diff, ", "))
	}
	return nil
}

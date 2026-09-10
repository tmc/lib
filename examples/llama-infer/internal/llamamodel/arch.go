package llamamodel

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"

	"github.com/tmc/modelir"
	"github.com/tmc/modelir/config"
	"github.com/tmc/modelir/encoding/sexpr"
)

// Arch is a model's architecture resolved to concrete facts. Family is the
// configuration's model_type, kept for reports; nothing here switches on it.
type Arch struct {
	Family       string
	Vocab        int
	Hidden       int
	Intermediate int
	Layers       []Layer
	Norm         modelir.NormKind
	NormEps      float64
	MLP          modelir.MLPKind
	Activation   modelir.MLPActivation
	RoPE         RoPE
	TieEmbed     bool
	// MaxPositions is max_position_embeddings, 0 when the configuration has
	// none. It is a training fact; the runtime chooses its own context bound.
	MaxPositions int
	KV           KV
	BOS          int32
	EOS          []int32
}

// Layer is one decoder block.
type Layer struct {
	Index         int
	Type          modelir.LayerType
	Heads         int
	KVHeads       int
	HeadDim       int
	SlidingWindow int
}

// RoPE holds the rotary embedding facts modelir resolves.
type RoPE struct {
	Theta       float64
	Traditional bool
}

// KV is the module set's KV-cache declaration. The zero value means the
// module set declared none.
type KV struct {
	Kind       string
	BlockSize  int
	DType      modelir.DType
	WindowSize int
}

// Load resolves the model at path: a directory holding config.json, with
// generation_config.json and tokenizer_config.json read when present, or a
// .sexpr module set. The returned module set is the record Load resolved;
// encode it with modelir/encoding/sexpr to keep it.
func Load(path string) (*Arch, *modelir.ModuleSet, error) {
	info, err := os.Stat(path)
	if err != nil {
		return nil, nil, err
	}
	var ms *modelir.ModuleSet
	if info.IsDir() {
		ms, err = loadDirectory(path)
	} else {
		var data []byte
		data, err = os.ReadFile(path)
		if err == nil {
			ms, err = sexpr.Unmarshal(data)
			if err != nil {
				err = fmt.Errorf("parse %s: %w", path, err)
			}
		}
	}
	if err != nil {
		return nil, nil, err
	}
	arch, err := Resolve(ms)
	if err != nil {
		return nil, nil, fmt.Errorf("resolve %s: %w", path, err)
	}
	return arch, ms, nil
}

// loadDirectory runs modelir's config frontend on dir and adds the facts it
// leaves out; see the package comment.
func loadDirectory(dir string) (*modelir.ModuleSet, error) {
	_, ms, err := config.ParseDirectory(dir)
	if err != nil {
		return nil, fmt.Errorf("modelir config: %w", err)
	}
	data, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, err
	}
	var side struct {
		TieWordEmbeddings     bool            `json:"tie_word_embeddings"`
		MaxPositionEmbeddings int             `json:"max_position_embeddings"`
		BOSTokenID            json.RawMessage `json:"bos_token_id"`
		EOSTokenID            json.RawMessage `json:"eos_token_id"`
	}
	if err := json.Unmarshal(data, &side); err != nil {
		return nil, fmt.Errorf("parse config.json: %w", err)
	}
	if len(ms.Configs) != 1 {
		return nil, fmt.Errorf("modelir config: %d configurations, want 1", len(ms.Configs))
	}
	fields := &ms.Configs[0].Fields
	*fields = append(*fields, modelir.ConfigField{Name: "tie_word_embeddings", Default: strconv.FormatBool(side.TieWordEmbeddings)})
	if side.MaxPositionEmbeddings > 0 {
		*fields = append(*fields, modelir.ConfigField{Name: "max_position_embeddings", Default: strconv.Itoa(side.MaxPositionEmbeddings)})
	}
	if ms.Arch == nil {
		ms.Arch = &modelir.ArchSpec{}
	}
	// The frontend declares a KV cache from nothing in config.json (kind and
	// family alone), and modelir's validator refuses that declaration as
	// incomplete. Dropping it is the fact: the configuration declares none.
	ms.Arch.KVCache = nil
	if ms.Arch.Generation == nil {
		ms.Arch.Generation = &modelir.GenerationSpec{}
	}
	gen := ms.Arch.Generation
	if gen.BOSID == 0 && len(side.BOSTokenID) > 0 {
		var id int64
		if err := json.Unmarshal(side.BOSTokenID, &id); err != nil {
			return nil, fmt.Errorf("parse config.json: bos_token_id: %w", err)
		}
		gen.BOSID = id
	}
	if len(gen.EOSIDs) == 0 && len(side.EOSTokenID) > 0 {
		var single int64
		if err := json.Unmarshal(side.EOSTokenID, &single); err == nil {
			gen.EOSIDs = []int64{single}
		} else if err := json.Unmarshal(side.EOSTokenID, &gen.EOSIDs); err != nil {
			return nil, fmt.Errorf("parse config.json: eos_token_id: %w", err)
		}
	}
	return ms, nil
}

// Resolve turns a module set into an Arch. It refuses what modelir's
// validator refuses, a field reference the configuration cannot satisfy, a
// dimension the configuration leaves at zero, and a norm kind this package
// does not know. modelir's findings are returned as modelir.Diagnostics.
func Resolve(ms *modelir.ModuleSet) (*Arch, error) {
	if ms == nil {
		return nil, errors.New("llamamodel: nil module set")
	}
	if ds := modelir.ValidateModuleSet(ms); len(ds) > 0 {
		return nil, modelir.Diagnostics(ds)
	}
	r, ds := modelir.Resolve(ms, modelir.ConfigSpec{})
	if len(ds) > 0 {
		return nil, ds
	}
	for _, d := range []struct {
		name string
		v    int
	}{
		{"num_hidden_layers", r.NumLayers},
		{"hidden_size", r.HiddenDim},
		{"intermediate_size", r.Intermediate},
		{"vocab_size", r.VocabSize},
		{"num_attention_heads", r.NumHeads},
		{"num_key_value_heads", r.NumKVHeads},
		{"head_dim", r.HeadDim},
	} {
		if d.v <= 0 {
			return nil, fmt.Errorf("llamamodel: %s unresolved", d.name)
		}
	}
	if r.NormEps <= 0 || math.IsNaN(r.NormEps) || math.IsInf(r.NormEps, 0) {
		return nil, fmt.Errorf("llamamodel: norm epsilon %v unresolved", r.NormEps)
	}
	if len(r.Layers) != r.NumLayers {
		return nil, fmt.Errorf("llamamodel: %d layers resolved for %d declared", len(r.Layers), r.NumLayers)
	}
	norm := modelir.NormKind(r.NormKind)
	switch norm {
	case modelir.NormRMSNorm, modelir.NormLayerNorm:
	default:
		return nil, fmt.Errorf("llamamodel: norm kind %q unknown", r.NormKind)
	}

	a := &Arch{
		Family:       r.Family,
		Vocab:        r.VocabSize,
		Hidden:       r.HiddenDim,
		Intermediate: r.Intermediate,
		Layers:       make([]Layer, len(r.Layers)),
		Norm:         norm,
		NormEps:      r.NormEps,
		MLP:          r.MLPKind,
		Activation:   r.Activation,
		RoPE:         RoPE{Theta: r.RoPE.Theta, Traditional: r.RoPE.Traditional},
	}
	for i, l := range r.Layers {
		a.Layers[i] = Layer{Index: l.Index, Type: l.Type, Heads: l.NumHeads, KVHeads: l.NumKVHeads, HeadDim: l.HeadDim, SlidingWindow: l.SlidingWindow}
	}
	cfg := resolvedConfig(ms)
	if v, ok := configField(cfg, "tie_word_embeddings"); ok {
		tie, err := strconv.ParseBool(v)
		if err != nil {
			return nil, fmt.Errorf("llamamodel: tie_word_embeddings %q: %w", v, err)
		}
		a.TieEmbed = tie
	}
	if v, ok := configField(cfg, "max_position_embeddings"); ok {
		n, err := strconv.Atoi(v)
		if err != nil || n < 0 {
			return nil, fmt.Errorf("llamamodel: max_position_embeddings %q is not a count", v)
		}
		a.MaxPositions = n
	}
	if r.KVCache != nil {
		a.KV = KV{Kind: r.KVCache.Kind, BlockSize: r.KVCache.BlockSize, DType: r.KVCache.DType, WindowSize: r.KVCache.WindowSize}
	}
	if g := r.Generation; g != nil {
		id, err := tokenID(g.BOSID)
		if err != nil {
			return nil, fmt.Errorf("llamamodel: bos_id: %w", err)
		}
		a.BOS = id
		for _, e := range g.EOSIDs {
			id, err := tokenID(e)
			if err != nil {
				return nil, fmt.Errorf("llamamodel: eos_id: %w", err)
			}
			a.EOS = append(a.EOS, id)
		}
	}
	return a, nil
}

func tokenID(v int64) (int32, error) {
	if v < 0 || v > math.MaxInt32 {
		return 0, fmt.Errorf("token id %d is out of range", v)
	}
	return int32(v), nil
}

// resolvedConfig is the configuration modelir.Resolve used: the entry
// module's, or the only one. Nil when neither applies.
func resolvedConfig(ms *modelir.ModuleSet) *modelir.ConfigSpec {
	if ms.EntryModule != "" {
		for i := range ms.Modules {
			m := &ms.Modules[i]
			if m.Name != ms.EntryModule || m.ConfigRef == "" {
				continue
			}
			for j := range ms.Configs {
				if ms.Configs[j].Name == m.ConfigRef {
					return &ms.Configs[j]
				}
			}
			return nil
		}
	}
	if len(ms.Configs) == 1 {
		return &ms.Configs[0]
	}
	return nil
}

func configField(cfg *modelir.ConfigSpec, name string) (string, bool) {
	if cfg == nil {
		return "", false
	}
	for _, f := range cfg.Fields {
		if f.Name == name {
			return f.Default, true
		}
	}
	return "", false
}

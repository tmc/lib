package main

import (
	"fmt"
	"strings"

	"github.com/tmc/lib/gguf"
)

// ExpertFormat names the routed-expert encoding accepted by a ds4-min
// backend.  It describes the weights, not the activation format selected by
// a particular kernel.
type ExpertFormat uint8

const (
	ExpertsIQ2Q2K ExpertFormat = iota + 1
	ExpertsQ4K
	ExpertsMXFP4
)

// ExpertFormatForTypes validates one layer's routed expert triplet and
// returns the format a backend would have to implement. Gate and up must
// match; down may differ only for the IQ2_XXS/Q2_K pairing. Keeping this
// decision separate from GGUF traversal lets loaders and mutation tests share
// the same fail-closed registry.
func ExpertFormatForTypes(gate, up, down uint32) (ExpertFormat, error) {
	if gate != up {
		return 0, fmt.Errorf("gate/up types %d/%d differ", gate, up)
	}
	switch {
	case gate == typeIQ2XXS && down == typeQ2K:
		return ExpertsIQ2Q2K, nil
	case gate == typeQ4K && down == typeQ4K:
		return ExpertsQ4K, nil
	case gate == typeMXFP4 && down == typeMXFP4:
		return ExpertsMXFP4, nil
	default:
		return 0, fmt.Errorf("unsupported expert types gate/up/down=%d/%d/%d", gate, up, down)
	}
}

func (f ExpertFormat) String() string {
	switch f {
	case ExpertsIQ2Q2K:
		return "IQ2_XXS/Q2_K"
	case ExpertsQ4K:
		return "Q4_K/Q4_K"
	case ExpertsMXFP4:
		return "MXFP4/MXFP4"
	default:
		return "unknown"
	}
}

// Profile identifies a checkpoint whose manifest is known to ds4-min.  The
// file digest is checked by validatePinnedModel; MatchProfile checks the
// parsed-header part because a gguf.File does not retain the file reader.
type Profile struct {
	Name, File string
	Bytes      uint64
	SHA256     string
	Metadata   string
	Tensors    string
	Quant      string
	Tokenizer  string
}

var supportedProfiles = []Profile{{
	Name: "ds4f-q2", File: "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf",
	Bytes: pinnedModelBytes, SHA256: pinnedModelSHA256,
	Metadata: pinnedMetadataHash, Tensors: pinnedTensorHash,
	Quant: pinnedQuantHash, Tokenizer: pinnedTokenizerHash,
}, {
	Name: "ds4f-q2-q4", File: "DeepSeek-V4-Flash-Layers37-42Q4KExperts-OtherExpertLayersIQ2XXSGateUp-Q2KDown-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-fixed-0731.gguf",
	Bytes:     97591747456,
	SHA256:    "659e22fbd01c9e13ea37a57c8d9c41e0a8819dffa3473d3c5286ee44b2d3398f",
	Metadata:  "0312e0fd129be0b3abd8eaf18e4a06655c84f22af3b14aaa98778c3c21071e5a",
	Tensors:   "f7b33970a49c4438da85212ea2204c51271d0ab04d3dea5364db1a68d2e6ac9e",
	Quant:     "c269bb524561883bb8c88fed7ffa09159c32aa43356d425eb2a6fea4be0bda76",
	Tokenizer: "d58f8cbcd40bfa9319e0954e01ad619c179567cf001258423713d3d604998eec",
}}

// Profiles returns the profiles known to this build.  Variants without
// registered manifest digests are intentionally absent: accepting them by
// filename or tensor type would turn an unqualified experiment into a load.
func Profiles() []Profile {
	return append([]Profile(nil), supportedProfiles...)
}

// MatchProfile matches the size and all parsed manifest digests. Empty
// digests never match. The complete file SHA256 is checked separately while
// reading the model with validatePinnedModel.
func MatchProfile(g *gguf.File) (*Profile, error) {
	if g == nil {
		return nil, fmt.Errorf("nil GGUF file")
	}
	d, err := profileHashes(g)
	if err != nil {
		return nil, fmt.Errorf("profile manifest: %w", err)
	}
	for i := range supportedProfiles {
		p := &supportedProfiles[i]
		if p.Bytes == 0 || p.Metadata == "" || p.Tensors == "" || p.Quant == "" || p.Tokenizer == "" {
			continue
		}
		if g.Size == p.Bytes && d.metadata == p.Metadata && d.tensors == p.Tensors && d.quant == p.Quant && d.tokenizer == p.Tokenizer {
			return p, nil
		}
	}
	return nil, fmt.Errorf("GGUF does not match a registered ds4-min profile")
}

const (
	typeQ2K    uint32 = 10
	typeQ4K    uint32 = 12
	typeIQ2XXS uint32 = 16
	typeMXFP4  uint32 = 39
)

// LayoutOf validates the routed-expert tensor pairs and returns their format
// for each of the 43 DSV4 layers. It is deliberately stricter than the
// current kernels: a format is reported only when all three tensors are
// present and the pair is one that an eventual backend can name. No backend
// dispatch is implied by this function.
func LayoutOf(g *gguf.File) ([43]ExpertFormat, error) {
	var layout [43]ExpertFormat
	if g == nil {
		return layout, fmt.Errorf("nil GGUF file")
	}
	if err := validateTensorTypes(g); err != nil {
		return layout, err
	}
	for layer := range layout {
		prefix := fmt.Sprintf("blk.%d.", layer)
		var typ [3]uint32
		seen := [3]bool{}
		for _, t := range g.Tensors {
			idx := -1
			switch {
			case t.Name == prefix+"ffn_gate_exps.weight":
				idx = 0
			case t.Name == prefix+"ffn_up_exps.weight":
				idx = 1
			case t.Name == prefix+"ffn_down_exps.weight":
				idx = 2
			default:
				continue
			}
			if seen[idx] {
				return layout, fmt.Errorf("layer %d: duplicate expert tensor %q", layer, t.Name)
			}
			seen[idx], typ[idx] = true, t.TypeID
		}
		for i, ok := range seen {
			if !ok {
				return layout, fmt.Errorf("layer %d: missing %s expert tensor", layer, []string{"gate", "up", "down"}[i])
			}
		}
		format, err := ExpertFormatForTypes(typ[0], typ[1], typ[2])
		if err != nil {
			return layout, fmt.Errorf("layer %d: %w", layer, err)
		}
		for _, t := range g.Tensors {
			if t.Name == prefix+"ffn_gate_exps.weight" || t.Name == prefix+"ffn_up_exps.weight" {
				if !expertShape(t.Dimensions, 4096, 2048, 256) {
					return layout, fmt.Errorf("layer %d: %s: dimensions %v, want [4096 2048 256]", layer, t.Name, t.Dimensions)
				}
			}
			if t.Name == prefix+"ffn_down_exps.weight" {
				if !expertShape(t.Dimensions, 2048, 4096, 256) {
					return layout, fmt.Errorf("layer %d: %s: dimensions %v, want [2048 4096 256]", layer, t.Name, t.Dimensions)
				}
			}
		}
		layout[layer] = format
	}
	return layout, nil
}

// validateTensorTypes checks the non-expert tensor type contract from the
// quant-variant specification.  It validates tensors that are present without
// requiring a complete model, which keeps LayoutOf useful for header-only
// fixtures while still rejecting a malformed profile before backend dispatch.
func validateTensorTypes(g *gguf.File) error {
	for _, t := range g.Tensors {
		allowed, checked := tensorTypeSet(t.Name)
		if !checked {
			continue
		}
		if !allowed[t.TypeID] {
			return fmt.Errorf("tensor %q: unsupported type %d", t.Name, t.TypeID)
		}
	}
	return nil
}

func tensorTypeSet(name string) (map[uint32]bool, bool) {
	const (
		f32 = 0
		f16 = 1
		q40 = 2
		q80 = 8
		q4k = 12
		i32 = 26
	)
	switch {
	case name == "token_embd.weight" || name == "output_hc_fn.weight" ||
		strings.HasSuffix(name, ".hc_attn_fn.weight") || strings.HasSuffix(name, ".hc_ffn_fn.weight") ||
		strings.Contains(name, "_compressor_ape.weight") || strings.Contains(name, "_compressor_kv.weight") ||
		strings.Contains(name, "_compressor_gate.weight") || name == "indexer.proj.weight" ||
		strings.HasSuffix(name, ".ffn_gate_inp.weight"):
		return map[uint32]bool{f16: true}, true
	case name == "output_norm.weight" || name == "output_hc_base.weight" || name == "output_hc_scale.weight" || name == "attn_sinks.weight" ||
		strings.HasSuffix(name, ".attn_kv_a_norm.weight") || strings.HasSuffix(name, ".attn_q_a_norm.weight") ||
		strings.Contains(name, "_compressor_norm.weight") ||
		strings.HasSuffix(name, ".ffn_norm.weight") || strings.HasSuffix(name, ".hc_attn_base.weight") ||
		strings.HasSuffix(name, ".hc_attn_scale.weight") || strings.HasSuffix(name, ".hc_ffn_base.weight") ||
		strings.HasSuffix(name, ".hc_ffn_scale.weight") || strings.HasSuffix(name, ".exp_probs_b.bias"):
		return map[uint32]bool{f32: true}, true
	case strings.HasSuffix(name, ".ffn_gate_tid2eid.weight"):
		return map[uint32]bool{i32: true}, true
	case strings.HasSuffix(name, ".indexer.attn_q_b.weight"):
		return map[uint32]bool{f16: true, q80: true}, true
	case name == "output.weight" || strings.HasSuffix(name, ".attn_q_a.weight") ||
		strings.HasSuffix(name, ".attn_q_b.weight") || strings.HasSuffix(name, ".attn_kv.weight") ||
		strings.HasSuffix(name, ".attn_output_a.weight") || strings.HasSuffix(name, ".attn_output_b.weight") ||
		strings.HasSuffix(name, ".ffn_gate_shexp.weight") || strings.HasSuffix(name, ".ffn_up_shexp.weight") ||
		strings.HasSuffix(name, ".ffn_down_shexp.weight"):
		return map[uint32]bool{q40: true, q80: true, q4k: true}, true
	default:
		return nil, false
	}
}

func expertShape(got []uint64, want ...uint64) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range want {
		if got[i] != want[i] {
			return false
		}
	}
	return true
}

// HasExpertTensors is useful to callers that want to distinguish a malformed
// partial GGUF from a non-DSV4 file before asking for LayoutOf.
func HasExpertTensors(g *gguf.File) bool {
	if g == nil {
		return false
	}
	for _, t := range g.Tensors {
		if strings.Contains(t.Name, ".ffn_gate_exps.weight") || strings.Contains(t.Name, ".ffn_down_exps.weight") {
			return true
		}
	}
	return false
}

package main

import (
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/x/egraph/hopper"
)

// FullLayerMegakernelConfig configures a 1-kernel-per-layer or
// 1-kernel-per-N-layers monolithic persistent megakernel for llama-infer,
// fusing RMSNorm + QKV + QKNormRoPEKVAppend + PagedAttention + O-proj + FFN
// across one or more decoder layers.
type FullLayerMegakernelConfig struct {
	// Name is the megakernel identifier (defaults to "llama_full_layer_mega").
	Name string

	// Model optionally supplies LLaMA model dimensions when explicit dimension
	// fields below are zero.
	Model *ModelConfig

	// StartLayer is the first decoder layer index when LayerIndices is empty.
	StartLayer int

	// NumLayers is the number of consecutive decoder layers to fuse into a
	// single monolithic megakernel (defaults to Model.NumHiddenLayers or 1).
	NumLayers int

	// LayerIndices optionally specifies explicit layer indices to fuse.
	LayerIndices []int

	BatchM     int
	HiddenDim  int
	IntermDim  int
	NumQHeads  int
	NumKVHeads int
	HeadDim    int
	PageSize   int
	MaxPages   int
	RoPETheta  float32
	RMSEps     float32
	Arch       hopper.Arch
	Target     string
}

// LayerSpecs resolves cfg into the per-layer [hopper.FullDecoderLayerSpec]
// slice for [hopper.BuildFullLayerMegakernel].
func (cfg FullLayerMegakernelConfig) LayerSpecs() ([]hopper.FullDecoderLayerSpec, error) {
	if cfg.StartLayer < 0 {
		return nil, fmt.Errorf("llama full-layer megakernel: negative StartLayer %d", cfg.StartLayer)
	}
	if cfg.NumLayers < 0 {
		return nil, fmt.Errorf("llama full-layer megakernel: negative NumLayers %d", cfg.NumLayers)
	}

	hidden := cfg.HiddenDim
	interm := cfg.IntermDim
	hq := cfg.NumQHeads
	hkv := cfg.NumKVHeads
	hd := cfg.HeadDim
	theta := cfg.RoPETheta
	eps := cfg.RMSEps
	pageSize := cfg.PageSize
	if pageSize == 0 {
		pageSize = egraphPageSize
	}
	maxPages := cfg.MaxPages
	numLayers := cfg.NumLayers

	if m := cfg.Model; m != nil {
		if hidden == 0 {
			hidden = m.HiddenSize
		}
		if interm == 0 {
			interm = m.IntermediateSize
		}
		if hq == 0 {
			hq = m.NumAttentionHeads
		}
		if hkv == 0 {
			hkv = m.NumKeyValueHeads
		}
		if hd == 0 {
			hd = m.HeadDim()
		}
		if theta <= 0 {
			theta = m.RopeTheta
		}
		if eps <= 0 {
			eps = m.RmsNormEps
		}
		if maxPages == 0 && m.MaxSeqLen > 0 && pageSize > 0 {
			maxPages = (m.MaxSeqLen + pageSize - 1) / pageSize
		}
		if numLayers == 0 && len(cfg.LayerIndices) == 0 && m.NumHiddenLayers > 0 {
			numLayers = m.NumHiddenLayers - cfg.StartLayer
		}
	}

	if hd == 0 && hidden > 0 && hq > 0 && hidden%hq == 0 {
		hd = hidden / hq
	}
	if maxPages == 0 {
		maxPages = 4
	}
	batchM := cfg.BatchM
	if batchM == 0 {
		batchM = 1
	}
	arch := cfg.Arch
	if arch == "" && cfg.Target != "" {
		arch = hopper.Arch(strings.TrimSpace(cfg.Target))
	}
	if arch == "" {
		arch = hopper.TargetSM121A
	}

	var indices []int
	if len(cfg.LayerIndices) > 0 {
		indices = append(indices, cfg.LayerIndices...)
	} else {
		if numLayers == 0 {
			numLayers = 1
		}
		indices = make([]int, numLayers)
		for i := range numLayers {
			indices[i] = cfg.StartLayer + i
		}
	}
	if len(indices) == 0 {
		return nil, fmt.Errorf("llama full-layer megakernel: zero layers")
	}
	if cfg.Model != nil && cfg.Model.NumHiddenLayers > 0 {
		for _, idx := range indices {
			if idx < 0 || idx >= cfg.Model.NumHiddenLayers {
				return nil, fmt.Errorf("llama full-layer megakernel: layer index %d out of range [0, %d)",
					idx, cfg.Model.NumHiddenLayers)
			}
		}
	}

	specs := make([]hopper.FullDecoderLayerSpec, len(indices))
	for i, idx := range indices {
		specs[i] = hopper.FullDecoderLayerSpec{
			LayerIndex: idx,
			BatchM:     batchM,
			HiddenDim:  hidden,
			IntermDim:  interm,
			NumQHeads:  hq,
			NumKVHeads: hkv,
			HeadDim:    hd,
			PageSize:   pageSize,
			MaxPages:   maxPages,
			RoPETheta:  theta,
			RMSEps:     eps,
			Arch:       arch,
		}
	}
	return specs, nil
}

// BuildLlamaFullLayerMegakernel builds a monolithic persistent megakernel
// spanning all 9 stages per decoder layer (attn_norm, qkv_proj,
// QKNormRoPEKVAppend, PagedAttention, o_proj, ffn_norm, gate_proj, up_proj,
// down_proj) across one or more LLaMA decoder layers.
func BuildLlamaFullLayerMegakernel(cfg FullLayerMegakernelConfig) (*hopper.Plan, *hopper.Megakernel, error) {
	specs, err := cfg.LayerSpecs()
	if err != nil {
		return nil, nil, err
	}
	name := strings.TrimSpace(cfg.Name)
	if name == "" {
		if len(specs) == 1 {
			name = fmt.Sprintf("llama_full_l%d_mega", specs[0].LayerIndex)
		} else {
			name = fmt.Sprintf("llama_full_l%d_l%d_mega", specs[0].LayerIndex, specs[len(specs)-1].LayerIndex)
		}
	}
	return hopper.BuildFullLayerMegakernel(name, specs)
}

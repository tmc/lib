package main

import (
	"errors"
	"fmt"

	"github.com/tmc/lib/nvidia/x/egraph/hopper"
)

// buildCrossLayerDecodeMegakernel builds the persistent cross-layer decode
// megakernel fusing layerL's post-attention MLP + residual RMSNorm plan with
// layerNext's pre-attention QKV projection plan across the inter-layer
// boundary.
//
// In chained decode mode (chain = true), post[layerL] normalizes its output
// residual into the BF16 bridge activation "xn", and attnIn[layerNext] reads
// "xn" to compute the next layer's QKV projection. Fusing both plans into a
// single persistent [hopper.Megakernel] synchronizes the inter-layer boundary
// with epoch-tagged device ready counters ([hopper.ReadyPrelude]) rather than a
// host stream launch boundary.
func buildCrossLayerDecodeMegakernel(cfg *ModelConfig, qkvBias bool, layerL, layerNext int, m int, target string) (*hopper.Plan, hopper.Megakernel, error) {
	if cfg == nil {
		return nil, hopper.Megakernel{}, errors.New("cross-layer decode megakernel: nil model config")
	}
	if layerL < 0 || layerNext < 0 || layerL == layerNext {
		return nil, hopper.Megakernel{}, fmt.Errorf("cross-layer decode megakernel: invalid layer pair (%d, %d)", layerL, layerNext)
	}
	if cfg.NumHiddenLayers > 0 && (layerL >= cfg.NumHiddenLayers || layerNext >= cfg.NumHiddenLayers) {
		return nil, hopper.Megakernel{}, fmt.Errorf("cross-layer decode megakernel: layer pair (%d, %d) out of range [0, %d)",
			layerL, layerNext, cfg.NumHiddenLayers)
	}
	if m <= 0 {
		return nil, hopper.Megakernel{}, fmt.Errorf("cross-layer decode megakernel: row count m=%d must be positive", m)
	}

	prefixL := fmt.Sprintf("l%d", layerL)
	prefixNext := fmt.Sprintf("l%d", layerNext)

	_, postL, err := buildLayerPlans(cfg, qkvBias, false, prefixL, m, hopper.F32, target, true)
	if err != nil {
		return nil, hopper.Megakernel{}, fmt.Errorf("cross-layer decode megakernel layer %d post: %w", layerL, err)
	}
	attnInNext, _, err := buildLayerPlans(cfg, qkvBias, false, prefixNext, m, hopper.F32, target, true)
	if err != nil {
		return nil, hopper.Megakernel{}, fmt.Errorf("cross-layer decode megakernel layer %d attnIn: %w", layerNext, err)
	}

	glue := hopper.CrossLayerGlue{
		Name:               fmt.Sprintf("llama_cross_l%d_l%d", layerL, layerNext),
		LayerLPrefix:       prefixL + "_",
		LayerNextPrefix:    prefixNext + "_",
		BridgeInput:        "xn",
		RetainBridgeOutput: true,
	}
	return hopper.BuildCrossLayerMegakernel(postL, attnInNext, glue)
}

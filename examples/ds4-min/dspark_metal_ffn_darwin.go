//go:build darwin && arm64

package main

import (
	"fmt"

	"github.com/tmc/lib/apple/x/metal/metalmem"
)

type metalDSparkFFNRows struct {
	state, flat, normalized         metalmem.Slice[float32]
	logits, probs, weights          metalmem.Slice[float32]
	ids                             metalmem.Slice[int32]
	gate, up, mid, experts          metalmem.Slice[float32]
	sharedGate, sharedUp, sharedMid metalmem.Slice[float32]
	sum, output                     metalmem.Slice[float32]
}

// EncodeStageFFN admits the source-backed IQ2_XXS/Q2_K support profile only.
// Other admitted metadata formats need their own native operator gate, not
// CPU decoding or substitution of target experts.
func (d *metalDSparkDense) EncodeStageFFN(b *metalBatch, stage, drafts int, s metalDSparkFFNRows) error {
	if d == nil || drafts < 1 || drafts > 16 || s.state.Len() != drafts*16384 || s.flat.Len() != s.state.Len() || s.normalized.Len() != drafts*8192 || s.logits.Len() != drafts*256 || s.probs.Len() != s.logits.Len() || s.ids.Len() != drafts*6 || s.weights.Len() != s.ids.Len() || s.sum.Len() != drafts*4096 || s.output.Len() != s.state.Len() || s.experts.Len() != drafts*7*4096 {
		return fmt.Errorf("metal dspark FFN: invalid owned row extent")
	}
	name := func(suffix string) string { return fmt.Sprintf("mtp.%d.%s", stage, suffix) }
	gate, err := d.weights.tensor(name("ffn_gate_exps.weight"))
	if err != nil {
		return err
	}
	up, err := d.weights.tensor(name("ffn_up_exps.weight"))
	if err != nil {
		return err
	}
	down, err := d.weights.tensor(name("ffn_down_exps.weight"))
	if err != nil {
		return err
	}
	if gate.typeID != 16 || up.typeID != 16 || down.typeID != 10 || len(gate.dimensions) != 3 || len(up.dimensions) != 3 || len(down.dimensions) != 3 || gate.dimensions[0] != 4096 || gate.dimensions[2] != 256 || up.dimensions[0] != gate.dimensions[0] || up.dimensions[1] != gate.dimensions[1] || up.dimensions[2] != 256 || down.dimensions[0] != gate.dimensions[1] || down.dimensions[1] != 4096 || down.dimensions[2] != 256 {
		return fmt.Errorf("metal dspark FFN: native expert format unavailable")
	}
	width := int(gate.dimensions[1])
	if width < 1 || width > 1<<20 || s.gate.Len() != drafts*6*width || s.up.Len() != s.gate.Len() || s.mid.Len() != s.gate.Len() || s.sharedGate.Len() != drafts*width || s.sharedUp.Len() != s.sharedGate.Len() || s.sharedMid.Len() != s.sharedGate.Len() {
		return fmt.Errorf("metal dspark FFN: invalid intermediate extent")
	}
	bias, err := d.weights.tensor(name("exp_probs_b.bias"))
	if err != nil {
		return err
	}
	if err := d.EncodeHCPre(b, stage, true, s.state, s.flat, s.normalized, drafts); err != nil {
		return err
	}
	norm, err := s.normalized.Slice(drafts*4096, drafts*8192)
	if err != nil {
		return err
	}
	if err := d.Encode(b, name("ffn_gate_inp.weight"), norm, s.logits, drafts); err != nil {
		return err
	}
	if err := b.EncodeRouterBiasedTokens(s.logits, bias, s.probs, s.ids, s.weights, drafts); err != nil {
		return err
	}
	for row := 0; row < drafts; row++ {
		input, err := tokenRow(norm, row, 4096)
		if err != nil {
			return err
		}
		ids, err := tokenRow(s.ids, row, 6)
		if err != nil {
			return err
		}
		weights, err := tokenRow(s.weights, row, 6)
		if err != nil {
			return err
		}
		g, err := tokenRow(s.gate, row, 6*width)
		if err != nil {
			return err
		}
		u, err := tokenRow(s.up, row, 6*width)
		if err != nil {
			return err
		}
		mid, err := tokenRow(s.mid, row, 6*width)
		if err != nil {
			return err
		}
		out, err := s.experts.Slice(row*7*4096, (row*7+6)*4096)
		if err != nil {
			return err
		}
		if err := b.EncodeMatvecIQ2Selected(gate, up, ids, input, g, u); err != nil {
			return err
		}
		if err := b.EncodeSwiGLUWeightedSlots(g, u, weights, mid); err != nil {
			return err
		}
		if err := b.EncodeMatvecQ2KSelected(down, ids, mid, out); err != nil {
			return err
		}
	}
	if err := d.Encode(b, name("ffn_gate_shexp.weight"), norm, s.sharedGate, drafts); err != nil {
		return err
	}
	if err := d.Encode(b, name("ffn_up_shexp.weight"), norm, s.sharedUp, drafts); err != nil {
		return err
	}
	if err := b.EncodeSwiGLU(s.sharedGate, s.sharedUp, s.sharedMid); err != nil {
		return err
	}
	for row := 0; row < drafts; row++ {
		input, err := tokenRow(s.sharedMid, row, width)
		if err != nil {
			return err
		}
		out, err := s.experts.Slice((row*7+6)*4096, (row*7+7)*4096)
		if err != nil {
			return err
		}
		if err := d.Encode(b, name("ffn_down_shexp.weight"), input, out, 1); err != nil {
			return err
		}
	}
	if err := b.EncodeSumVectorsTokens(s.experts, s.sum, 7, drafts); err != nil {
		return err
	}
	return d.EncodeHCPost(b, stage, true, s.state, s.sum, s.output, drafts)
}

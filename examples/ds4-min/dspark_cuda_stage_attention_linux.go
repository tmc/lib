//go:build linux

package main

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudamem"
)

// encodeAttentionOutput consumes native noncausal attention values and
// emits draft-only HC rows. The preserved feature row stays with the caller.
// Every workspace is private to this request and remains live through stream
// completion. A failed enqueue invalidates the caller's partial proposal.
func (w *cudaDSparkWeights) encodeAttentionOutput(r *cudaRuntime, stage, position int, input, heads, low, projection, output cudamem.Slice[float32]) error {
	if err := dsparkRequestOpen(r); err != nil {
		return err
	}
	if w == nil || r == nil || r.cudaFoundation == nil || w.model == nil || w.closed || r.device != w.device {
		return fmt.Errorf("cuda dspark attention output: missing owner or request")
	}
	rows := w.model.block
	if stage < 0 || stage >= w.model.stages || position < 0 || input.Len() != (rows+1)*Flash.HC*Flash.Embed || heads.Len() != rows*Flash.Heads*Flash.HeadDim || low.Len() != Flash.OutGroups*Flash.OutLoRA || projection.Len() != Flash.Embed || output.Len() != rows*Flash.HC*Flash.Embed {
		return fmt.Errorf("cuda dspark attention output: invalid workspace")
	}
	name := func(suffix string) string { return fmt.Sprintf("mtp.%d.%s", stage, suffix) }
	var weights [3]cudaModelTensor
	for i, suffix := range []string{"hc_attn_fn.weight", "hc_attn_scale.weight", "hc_attn_base.weight"} {
		var err error
		weights[i], err = w.tensor(name(suffix))
		if err != nil {
			return err
		}
	}
	if err := w.encodeRope(r, heads, Flash.Heads, rows, position, true); err != nil {
		return err
	}
	groupWidth := Flash.Heads * Flash.HeadDim / Flash.OutGroups
	for row := 0; row < rows; row++ {
		values := heads.Slice(row*Flash.Heads*Flash.HeadDim, (row+1)*Flash.Heads*Flash.HeadDim)
		for group := 0; group < Flash.OutGroups; group++ {
			if err := w.encodeDenseRange(r, name("attn_output_a.weight"), group*Flash.OutLoRA, Flash.OutLoRA, values.Slice(group*groupWidth, (group+1)*groupWidth), low.Slice(group*Flash.OutLoRA, (group+1)*Flash.OutLoRA)); err != nil {
				return err
			}
		}
		if err := w.encodeDense(r, name("attn_output_b.weight"), low, projection); err != nil {
			return err
		}
		width := Flash.HC * Flash.Embed
		if err := r.cudaFoundation.encodeHCPost(input.Slice((row+1)*width, (row+2)*width), projection, weights[0], weights[1], weights[2], output.Slice(row*width, (row+1)*width)); err != nil {
			return err
		}
	}
	return nil
}

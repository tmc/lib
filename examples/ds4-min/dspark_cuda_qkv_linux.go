//go:build linux

package main

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudamem"
)

// encodeStageQKV consumes feature HC followed by all draft HC rows. Every
// operand and temporary belongs to the consuming request; the target cache
// and target HC are never scratch. Subsequent stages preserve feature HC.
// The owner must remain open through completion of the request stream.
func (w *cudaDSparkWeights) encodeStageQKV(r *cudaRuntime, stage, position int, input, hcWork, low, queries, kv cudamem.Slice[float32]) error {
	if err := dsparkRequestOpen(r); err != nil {
		return err
	}
	if w == nil || r == nil || r.cudaFoundation == nil || w.model == nil || w.closed || r.device != w.device {
		return fmt.Errorf("cuda dspark QKV: missing owner or request")
	}
	rows := w.model.block + 1
	if stage < 0 || stage >= w.model.stages || position < 0 || position > 2147483647-rows || input.Len() != rows*Flash.HC*Flash.Embed || hcWork.Len() != rows*2*Flash.Embed || low.Len() != Flash.QLoRA || queries.Len() != w.model.block*Flash.Heads*Flash.HeadDim || kv.Len() != rows*Flash.HeadDim {
		return fmt.Errorf("cuda dspark QKV: invalid stage or workspace shape")
	}
	name := func(suffix string) string { return fmt.Sprintf("mtp.%d.%s", stage, suffix) }
	var weights [6]cudaModelTensor
	for i, suffix := range []string{"hc_attn_fn.weight", "hc_attn_scale.weight", "hc_attn_base.weight", "attn_norm.weight", "attn_kv_a_norm.weight", "attn_q_a_norm.weight"} {
		var err error
		weights[i], err = w.tensor(name(suffix))
		if err != nil {
			return err
		}
	}
	for row := 0; row < rows; row++ {
		hc := input.Slice(row*Flash.HC*Flash.Embed, (row+1)*Flash.HC*Flash.Embed)
		work := hcWork.Slice(row*2*Flash.Embed, (row+1)*2*Flash.Embed)
		if err := r.cudaFoundation.encodeHCPre(hc, weights[0], weights[1], weights[2], weights[3], work); err != nil {
			return fmt.Errorf("cuda dspark stage %d HC row %d: %w", stage, row, err)
		}
		normalized := work.Slice(Flash.Embed, 2*Flash.Embed)
		key := kv.Slice(row*Flash.HeadDim, (row+1)*Flash.HeadDim)
		if err := w.encodeDense(r, name("attn_kv.weight"), normalized, key); err != nil {
			return err
		}
		if err := r.cudaFoundation.encodeRMS(key, weights[4], key, 1, true); err != nil {
			return err
		}
		if row == 0 {
			continue // selected feature supplies KV, never a query
		}
		if err := w.encodeDense(r, name("attn_q_a.weight"), normalized, low); err != nil {
			return err
		}
		if err := r.cudaFoundation.encodeRMS(low, weights[5], low, 1, true); err != nil {
			return err
		}
		query := queries.Slice((row-1)*Flash.Heads*Flash.HeadDim, row*Flash.Heads*Flash.HeadDim)
		if err := w.encodeDense(r, name("attn_q_b.weight"), low, query); err != nil {
			return err
		}
		if err := r.cudaFoundation.encodeRMS(query, cudaModelTensor{}, query, Flash.Heads, false); err != nil {
			return err
		}
	}
	// Feature and draft0 both represent position, so rotating the whole KV
	// allocation as consecutive positions would shift every draft key by one.
	if err := w.encodeRope(r, kv.Slice(0, Flash.HeadDim), 1, 1, position, false); err != nil {
		return err
	}
	if err := w.encodeRope(r, kv.Slice(Flash.HeadDim, kv.Len()), 1, w.model.block, position, false); err != nil {
		return err
	}
	return w.encodeRope(r, queries, Flash.Heads, w.model.block, position, false)
}

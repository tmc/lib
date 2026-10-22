//go:build darwin && arm64

package main

import (
	"fmt"

	"github.com/tmc/lib/apple/x/metal/metalmem"
)

type metalDSparkAttentionRows struct {
	qkv                                 metalDSparkQKVRows
	raw, heads, low, projection, output metalmem.Slice[float32]
}

// EncodeStageAttention appends native support attention, never target Eval.
// The caller must own a rollback image of raw until the batch completes;
// admission alone does not publish an accepted target prefix.
func (d *metalDSparkDense) EncodeStageAttention(b *metalBatch, a *metalDSparkAttention, stage, position, drafts, capacity, appendStart, visibleStart, visible int, s metalDSparkAttentionRows) error {
	if drafts < 1 || drafts > 16 || capacity < drafts+1 || capacity > 145 || appendStart < 0 || appendStart >= capacity || visibleStart < 0 || visibleStart >= capacity || visible < drafts+1 || visible > capacity || (appendStart-visibleStart+capacity)%capacity+drafts+1 != visible || s.raw.Len() != capacity*512 || s.heads.Len() != drafts*32768 || s.low.Len() != drafts*Flash.OutGroups*Flash.OutLoRA || s.projection.Len() != drafts*4096 || s.output.Len() != drafts*16384 {
		return fmt.Errorf("metal dspark stage attention: invalid cache or output extent")
	}
	if err := d.EncodeQKV(b, a, stage, position, drafts, s.qkv); err != nil {
		return err
	}
	if err := a.EncodeStore(b, s.qkv.kvNorm, s.raw, capacity, appendStart, drafts+1); err != nil {
		return err
	}
	if err := a.EncodeAttention(b, s.qkv.queryNorm, s.raw, s.heads, stage, capacity, visibleStart, visible, drafts); err != nil {
		return err
	}
	if err := a.EncodeRoPE(b, s.heads, 64, drafts, position, true); err != nil {
		return err
	}
	// Head groups are contiguous inside each row, not contiguous across
	// draft rows. Select each activation and matching physical weight rows.
	for row := 0; row < drafts; row++ {
		for group := 0; group < Flash.OutGroups; group++ {
			input, err := s.heads.Slice(row*32768+group*4096, row*32768+(group+1)*4096)
			if err != nil {
				return err
			}
			weightStart, start, end, err := metalDSparkOutputRow(drafts, row, group, s.low.Len())
			if err != nil {
				return err
			}
			output, err := s.low.Slice(start, end)
			if err != nil {
				return err
			}
			if err := d.EncodeRows(b, fmt.Sprintf("mtp.%d.attn_output_a.weight", stage), weightStart, Flash.OutLoRA, input, output, 1); err != nil {
				return err
			}
		}
	}
	if err := d.Encode(b, fmt.Sprintf("mtp.%d.attn_output_b.weight", stage), s.low, s.projection, drafts); err != nil {
		return err
	}
	return d.EncodeHCDraftPost(b, stage, s.qkv.state, s.projection, s.output, drafts)
}

func metalDSparkOutputRow(drafts, row, group, extent int) (weightStart, start, end int, err error) {
	width := Flash.OutGroups * Flash.OutLoRA
	if drafts < 1 || drafts > 16 || row < 0 || row >= drafts || group < 0 || group >= Flash.OutGroups || extent != drafts*width {
		return 0, 0, 0, fmt.Errorf("metal dspark output projection: invalid row geometry")
	}
	weightStart = group * Flash.OutLoRA
	start = row*width + weightStart
	return weightStart, start, start + Flash.OutLoRA, nil
}

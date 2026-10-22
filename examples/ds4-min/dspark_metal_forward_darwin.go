//go:build darwin && arm64

package main

import (
	"fmt"

	libmetal "github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/metalmem"
)

type metalDSparkForwardRows struct {
	features, embeddings, main, mainNorm    metalmem.Slice[float32]
	attention                               []metalDSparkAttentionRows
	ffn                                     []metalDSparkFFNRows
	headFlat, headMix, headWeighted, hidden metalmem.Slice[float32]
}

// EncodeForward appends the actual support transformer. The fixed feature
// row is retained while each stage replaces only draft residuals. Private
// caches must be snapshotted by the caller before submission; this encoder
// does not publish target State or accepted-prefix history.
func (d *metalDSparkDense) EncodeForward(b *metalBatch, a *metalDSparkAttention, position, capacity, appendStart, visibleStart, visible int, s metalDSparkForwardRows) error {
	if d == nil || d.closed || d.weights == nil || d.weights.closed || d.weights.model == nil || a == nil || a.closed || a.weights != d.weights || d.weights.model.block < 1 || d.weights.model.block > 16 || d.weights.model.stages < 1 || len(s.attention) != d.weights.model.stages || len(s.ffn) != len(s.attention) {
		return fmt.Errorf("metal dspark forward: invalid stage owners")
	}
	if err := a.ready(b); err != nil {
		return fmt.Errorf("metal dspark forward: stage batch: %w", err)
	}
	drafts := d.weights.model.block
	state := s.attention[0].qkv.state
	for stage := range s.attention {
		if !metalDSparkSameSlice(state, s.attention[stage].qkv.state) || !metalDSparkSameSlice(s.attention[stage].output, s.ffn[stage].state) {
			return fmt.Errorf("metal dspark forward: stage scratch does not share exact residual")
		}
	}
	if err := d.EncodeInput(b, a, s.features, s.embeddings, s.main, s.mainNorm, state, drafts); err != nil {
		return err
	}
	for stage := range s.attention {
		if err := d.EncodeStageAttention(b, a, stage, position, drafts, capacity, appendStart, visibleStart, visible, s.attention[stage]); err != nil {
			return fmt.Errorf("metal dspark stage %d attention: %w", stage, err)
		}
		if err := d.EncodeStageFFN(b, stage, drafts, s.ffn[stage]); err != nil {
			return fmt.Errorf("metal dspark stage %d FFN: %w", stage, err)
		}
		if stage+1 < len(s.attention) {
			if err := a.ready(b); err != nil {
				return err
			}
			enc := b.encoder()
			enc.SetComputePipelineState(a.replace)
			enc.SetBufferWithOffsetAtIndex(s.ffn[stage].output.Buffer(), s.ffn[stage].output.Offset(), 0)
			enc.SetBufferWithOffsetAtIndex(state.Buffer(), state.Offset(), 1)
			libmetal.SetScalar(enc, uint32(drafts), 2)
			enc.DispatchThreadsThreadsPerThreadgroup(libmetal.Threads(drafts*16384), libmetal.Threads(256))
			b.encoded++
			a.batches = retainMetalDSparkBatch(a.batches, b)
		}
	}
	return d.EncodeFinalHidden(b, a, s.ffn[len(s.ffn)-1].output, s.headFlat, s.headMix, s.headWeighted, s.hidden, drafts)
}

func metalDSparkSameSlice(a, b metalmem.Slice[float32]) bool {
	return a.Len() == b.Len() && a.Offset() == b.Offset() && a.Buffer() != nil && b.Buffer() != nil && a.Buffer().GetID() == b.Buffer().GetID()
}

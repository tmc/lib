//go:build darwin && arm64

package main

import (
	"fmt"

	libmetal "github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/metalmem"
)

// EncodeInput consumes a device row copied from the actual selected-layer
// journal and actual base-model embeddings, not target predictions.
func (d *metalDSparkDense) EncodeInput(b *metalBatch, a *metalDSparkAttention, features, embeddings, projection, normalized, state metalmem.Slice[float32], drafts int) error {
	if d == nil || a == nil || a.weights != d.weights || drafts != d.weights.model.block || drafts < 1 || drafts > 16 || features.Len() != len(d.weights.model.targets)*4096 || embeddings.Len() != drafts*4096 || projection.Len() != 4096 || normalized.Len() != 4096 || state.Len() != (drafts+1)*16384 {
		return fmt.Errorf("metal dspark input: invalid actual feature or embedding extent")
	}
	if err := d.Encode(b, "mtp.0.main_proj.weight", features, projection, 1); err != nil {
		return err
	}
	norm, err := d.weights.tensor("mtp.0.main_norm.weight")
	if err != nil {
		return err
	}
	if err := b.EncodeRMS(projection, norm, normalized, 1, true); err != nil {
		return err
	}
	if err := a.ready(b); err != nil {
		return err
	}
	enc := b.encoder()
	enc.SetComputePipelineState(a.expand)
	for i, v := range []metalmem.Slice[float32]{normalized, embeddings, state} {
		enc.SetBufferWithOffsetAtIndex(v.Buffer(), v.Offset(), uint(i))
	}
	libmetal.SetScalar(enc, uint32(drafts), 3)
	enc.DispatchThreadsThreadsPerThreadgroup(libmetal.Threads(state.Len()), libmetal.Threads(256))
	b.encoded++
	a.batches = retainMetalDSparkBatch(a.batches, b)
	return nil
}

func (d *metalDSparkDense) EncodeFinalHidden(b *metalBatch, a *metalDSparkAttention, state, flat, mix, weighted, hidden metalmem.Slice[float32], drafts int) error {
	if d == nil || a == nil || a.weights != d.weights || drafts < 1 || drafts > 16 || state.Len() != drafts*16384 || flat.Len() != state.Len() || mix.Len() != drafts*4 || weighted.Len() != drafts*4096 || hidden.Len() != weighted.Len() {
		return fmt.Errorf("metal dspark final hidden: invalid owned extent")
	}
	name := func(suffix string) string { return fmt.Sprintf("mtp.%d.%s", d.weights.model.stages-1, suffix) }
	norm, err := d.weights.tensor(name("norm.weight"))
	if err != nil {
		return err
	}
	if err := b.EncodeRMS(state, norm, flat, drafts, false); err != nil {
		return err
	}
	if err := d.Encode(b, name("hc_head_fn.weight"), flat, mix, drafts); err != nil {
		return err
	}
	scale, err := d.weights.tensor(name("hc_head_scale.weight"))
	if err != nil {
		return err
	}
	base, err := d.weights.tensor(name("hc_head_base.weight"))
	if err != nil {
		return err
	}
	if scale.typeID != 0 || base.typeID != 0 || scale.data.Len() != 4 || base.data.Len() != 16 {
		return fmt.Errorf("metal dspark final hidden: invalid HC weights")
	}
	enc := b.encoder()
	enc.SetComputePipelineState(a.finalHC)
	enc.SetBufferWithOffsetAtIndex(state.Buffer(), state.Offset(), 0)
	enc.SetBufferWithOffsetAtIndex(mix.Buffer(), mix.Offset(), 1)
	enc.SetBufferWithOffsetAtIndex(scale.data.Buffer(), scale.data.Offset(), 2)
	enc.SetBufferWithOffsetAtIndex(base.data.Buffer(), base.data.Offset(), 3)
	enc.SetBufferWithOffsetAtIndex(weighted.Buffer(), weighted.Offset(), 4)
	libmetal.SetScalar(enc, uint32(drafts), 5)
	enc.DispatchThreadsThreadsPerThreadgroup(libmetal.Threads(weighted.Len()), libmetal.Threads(256))
	b.encoded++
	a.batches = retainMetalDSparkBatch(a.batches, b)
	return b.EncodeRMS(weighted, norm, hidden, drafts, true)
}

//go:build darwin && arm64

package main

import (
	"fmt"

	applemetal "github.com/tmc/apple/metal"
	libmetal "github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/metalmem"
)

// EncodeHCPre runs actual support learned HC matrices on the device. The
// pinned split/collapse/RMS kernels are reused, but F32 support HC matrices
// are not forced through the target's F16-only matrix admission.
func (d *metalDSparkDense) EncodeHCPre(b *metalBatch, stage int, ffn bool, state, flat, out metalmem.Slice[float32], count int) error {
	if d == nil || d.weights == nil || b == nil || b.ops == nil || b.ops.dev != d.weights.device || b.ops.hc == nil || count < 1 || count > 17 || state.Len() != count*16384 || flat.Len() != state.Len() || out.Len() != count*8192 {
		return fmt.Errorf("metal dspark HC: invalid row extent")
	}
	fn, scale, base, norm, err := d.hcWeights(stage, ffn)
	if err != nil {
		return err
	}
	h := b.ops.hc
	if h.pending != nil && h.pending != b {
		return fmt.Errorf("metal dspark HC: foreign scratch producer")
	}
	if err := b.EncodeRMS(state, norm, flat, count, false); err != nil {
		return err
	}
	if err := h.reserve(count); err != nil {
		return err
	}
	mix, err := h.mix.Slice(0, count*24)
	if err != nil {
		return err
	}
	prefix := "hc_attn"
	if ffn {
		prefix = "hc_ffn"
	}
	if err := d.Encode(b, fmt.Sprintf("mtp.%d.%s_fn.weight", stage, prefix), flat, mix, count); err != nil {
		return err
	}
	enc := b.encoder()
	enc.SetComputePipelineState(h.kernels.Pre)
	enc.SetBufferWithOffsetAtIndex(mix.Buffer(), mix.Offset(), 0)
	enc.SetBufferWithOffsetAtIndex(scale.data.Buffer(), scale.data.Offset(), 1)
	enc.SetBufferWithOffsetAtIndex(base.data.Buffer(), base.data.Offset(), 2)
	enc.SetBufferWithOffsetAtIndex(state.Buffer(), state.Offset(), 3)
	enc.SetBufferWithOffsetAtIndex(h.split.Buffer(), h.split.Offset(), 4)
	enc.SetBufferWithOffsetAtIndex(out.Buffer(), out.Offset(), 5)
	enc.SetBufferWithOffsetAtIndex(norm.data.Buffer(), norm.data.Offset(), 6)
	enc.SetBufferWithOffsetAtIndex(out.Buffer(), out.Offset()+uint(count)*4096*4, 7)
	libmetal.SetScalar(enc, uint32(count), 8)
	enc.SetThreadgroupMemoryLengthAtIndex((4096+4+32)*4, 0)
	enc.DispatchThreadgroupsThreadsPerThreadgroup(libmetal.Threads(count), libmetal.Threads(1024))
	h.key = metalHCIdentity(state, fn, scale, base, count)
	h.pending = b
	b.encoded++
	return nil
}

func (d *metalDSparkDense) EncodeHCPost(b *metalBatch, stage int, ffn bool, state, block, out metalmem.Slice[float32], count int) error {
	return d.encodeHCPostRows(b, stage, ffn, state, block, out, count, 0, count)
}

// EncodeHCDraftPost consumes only the draft rows of a pre containing one
// captured feature followed by drafts. Identity still covers the complete
// pre input; the feature is not a fabricated target state update.
func (d *metalDSparkDense) EncodeHCDraftPost(b *metalBatch, stage int, state, block, out metalmem.Slice[float32], drafts int) error {
	return d.encodeHCPostRows(b, stage, false, state, block, out, drafts+1, 1, drafts)
}

func (d *metalDSparkDense) encodeHCPostRows(b *metalBatch, stage int, ffn bool, state, block, out metalmem.Slice[float32], count, start, rows int) error {
	if d == nil || b == nil || b.ops == nil || b.ops.hc == nil || count < 1 || count > 17 || start < 0 || rows < 1 || start+rows > count || state.Len() != count*16384 || block.Len() != rows*4096 || out.Len() != rows*16384 {
		return fmt.Errorf("metal dspark HC post: invalid row extent")
	}
	fn, scale, base, _, err := d.hcWeights(stage, ffn)
	if err != nil {
		return err
	}
	h := b.ops.hc
	if h.pending != b || h.key != metalHCIdentity(state, fn, scale, base, count) {
		return fmt.Errorf("metal dspark HC post: matching same-buffer pre required")
	}
	if err := b.ready(); err != nil {
		return err
	}
	selectedState, err := state.Slice(start*16384, (start+rows)*16384)
	if err != nil {
		return err
	}
	selectedSplit, err := h.split.Slice(start*24, (start+rows)*24)
	if err != nil {
		return err
	}
	enc := b.encoder()
	enc.SetComputePipelineState(h.kernels.Post)
	for i, v := range []metalmem.Slice[float32]{selectedState, block, selectedSplit, out} {
		enc.SetBufferWithOffsetAtIndex(v.Buffer(), v.Offset(), uint(i))
	}
	enc.DispatchThreadsThreadsPerThreadgroup(applemetal.MTLSize{Width: 4096, Height: uint(rows), Depth: 1}, applemetal.MTLSize{Width: 256, Height: 1, Depth: 1})
	h.pending = nil
	b.encoded++
	return nil
}

func (d *metalDSparkDense) hcWeights(stage int, ffn bool) (fn, scale, base, norm boundMetalTensor, err error) {
	if d == nil || d.closed || d.weights == nil || d.weights.model == nil || stage < 0 || stage >= d.weights.model.stages {
		err = fmt.Errorf("metal dspark HC: invalid stage")
		return
	}
	prefix, normName := "hc_attn", "attn_norm.weight"
	if ffn {
		prefix, normName = "hc_ffn", "ffn_norm.weight"
	}
	for _, field := range []struct {
		name string
		dst  *boundMetalTensor
	}{{prefix + "_fn.weight", &fn}, {prefix + "_scale.weight", &scale}, {prefix + "_base.weight", &base}, {normName, &norm}} {
		*field.dst, err = d.weights.tensor(fmt.Sprintf("mtp.%d.%s", stage, field.name))
		if err != nil {
			return
		}
	}
	if len(fn.dimensions) != 2 || fn.dimensions[0] != 16384 || fn.dimensions[1] != 24 || (fn.typeID != 0 && fn.typeID != 1) || scale.typeID != 0 || base.typeID != 0 || norm.typeID != 0 || scale.data.Len() != 12 || base.data.Len() != 96 || norm.data.Len() != 16384 {
		err = fmt.Errorf("metal dspark HC: unsupported tensor layout")
	}
	return
}

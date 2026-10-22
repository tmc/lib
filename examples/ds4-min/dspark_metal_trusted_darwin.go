//go:build darwin && arm64

package main

import (
	"fmt"

	libmetal "github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/metalmem"
)

// EncodeTrusted reconstructs private support KV from actual journal rows.
// Pinned C's direct-main KV path is ROCm-only (ds4.c:33807); Metal must run
// learned support HC/attention norm before wkv. The CUDA private shortcut
// therefore cannot be transferred to this backend.
func (d *metalDSparkDense) EncodeTrusted(b *metalBatch, a *metalDSparkAttention, f *metalDSparkFeatures, position, capacity int, s metalDSparkForwardRows) (start, count int, err error) {
	if d == nil || a == nil || f == nil || f.closed || a.weights != d.weights || f.weights != d.weights || f.request.runtime.dsparkFeatures != f || f.batch != nil || position < f.journal.frontier-1 || position > f.journal.frontier || capacity < d.weights.model.block+1 || capacity > 145 || len(s.attention) != d.weights.model.stages {
		return 0, 0, fmt.Errorf("metal dspark trusted KV: foreign or unpublished journal")
	}
	if err := f.journal.validate(f.request.runtime.pos); err != nil {
		return 0, 0, err
	}
	start = max(0, position-Flash.SlidingWindow)
	count = position - start
	if count+d.weights.model.block+1 > capacity {
		return 0, 0, fmt.Errorf("metal dspark trusted KV: visible history exceeds bounded cache")
	}
	norm, err := d.weights.tensor("mtp.0.main_norm.weight")
	if err != nil {
		return 0, 0, err
	}
	for row := start; row < position; row++ {
		if _, err := f.journal.row(row, f.epoch); err != nil {
			return 0, 0, err
		}
		width := len(f.targets) * 4096
		feature, err := f.data.Slice(row%len(f.journal.rows)*width, (row%len(f.journal.rows)+1)*width)
		if err != nil {
			return 0, 0, err
		}
		if err := d.Encode(b, "mtp.0.main_proj.weight", feature, s.main, 1); err != nil {
			return 0, 0, err
		}
		if err := b.EncodeRMS(s.main, norm, s.mainNorm, 1, true); err != nil {
			return 0, 0, err
		}
		for stage, frames := range s.attention {
			state, err := frames.qkv.state.Slice(0, 16384)
			if err != nil {
				return 0, 0, err
			}
			flat, err := frames.qkv.flat.Slice(0, 16384)
			if err != nil {
				return 0, 0, err
			}
			pre, err := frames.qkv.normalized.Slice(0, 8192)
			if err != nil {
				return 0, 0, err
			}
			kv, err := frames.qkv.kv.Slice(0, 512)
			if err != nil {
				return 0, 0, err
			}
			kvNorm, err := frames.qkv.kvNorm.Slice(0, 512)
			if err != nil {
				return 0, 0, err
			}
			if frames.raw.Len() != capacity*512 {
				return 0, 0, fmt.Errorf("metal dspark trusted KV: cache extent mismatch")
			}
			if err := a.ready(b); err != nil {
				return 0, 0, err
			}
			enc := b.encoder()
			enc.SetComputePipelineState(a.expand)
			for i, v := range []metalmem.Slice[float32]{s.mainNorm, s.mainNorm, state} {
				enc.SetBufferWithOffsetAtIndex(v.Buffer(), v.Offset(), uint(i))
			}
			libmetal.SetScalar(enc, uint32(0), 3)
			enc.DispatchThreadsThreadsPerThreadgroup(libmetal.Threads(16384), libmetal.Threads(256))
			b.encoded++
			a.batches = retainMetalDSparkBatch(a.batches, b)
			if err := d.EncodeHCPre(b, stage, false, state, flat, pre, 1); err != nil {
				return 0, 0, err
			}
			input, err := pre.Slice(4096, 8192)
			if err != nil {
				return 0, 0, err
			}
			if err := d.Encode(b, fmt.Sprintf("mtp.%d.attn_kv.weight", stage), input, kv, 1); err != nil {
				return 0, 0, err
			}
			weight, err := d.weights.tensor(fmt.Sprintf("mtp.%d.attn_kv_a_norm.weight", stage))
			if err != nil {
				return 0, 0, err
			}
			if err := b.EncodeRMS(kv, weight, kvNorm, 1, true); err != nil {
				return 0, 0, err
			}
			if err := a.EncodeRoPE(b, kvNorm, 1, 1, row, false); err != nil {
				return 0, 0, err
			}
			if err := a.EncodeStore(b, kvNorm, frames.raw, capacity, row%capacity, 1); err != nil {
				return 0, 0, err
			}
		}
	}
	return start, count, nil
}

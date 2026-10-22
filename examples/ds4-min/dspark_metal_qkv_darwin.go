//go:build darwin && arm64

package main

import (
	"fmt"

	"github.com/tmc/lib/apple/x/metal/metalmem"
)

// Scratch belongs to the support request, never the target State. The first
// HC row is the captured feature; only subsequent rows produce queries.
type metalDSparkQKVRows struct {
	state, flat, normalized metalmem.Slice[float32]
	kv, kvNorm              metalmem.Slice[float32]
	qLow, qLowNorm          metalmem.Slice[float32]
	query, queryNorm        metalmem.Slice[float32]
}

func (d *metalDSparkDense) EncodeQKV(b *metalBatch, a *metalDSparkAttention, stage, position, drafts int, s metalDSparkQKVRows) error {
	if d == nil || a == nil || a.weights != d.weights || drafts < 1 || drafts > 16 || position < 0 || position > int(^uint32(0))-drafts || s.state.Len() != (drafts+1)*16384 || s.flat.Len() != s.state.Len() || s.normalized.Len() != (drafts+1)*8192 || s.kv.Len() != (drafts+1)*512 || s.kvNorm.Len() != s.kv.Len() || s.qLow.Len() != drafts*1024 || s.qLowNorm.Len() != s.qLow.Len() || s.query.Len() != drafts*32768 || s.queryNorm.Len() != s.query.Len() {
		return fmt.Errorf("metal dspark QKV: invalid owned row extent")
	}
	name := func(suffix string) string { return fmt.Sprintf("mtp.%d.%s", stage, suffix) }
	if err := d.EncodeHCPre(b, stage, false, s.state, s.flat, s.normalized, drafts+1); err != nil {
		return err
	}
	norm, err := s.normalized.Slice((drafts+1)*4096, (drafts+1)*8192)
	if err != nil {
		return err
	}
	if err := d.Encode(b, name("attn_kv.weight"), norm, s.kv, drafts+1); err != nil {
		return err
	}
	kvWeight, err := d.weights.tensor(name("attn_kv_a_norm.weight"))
	if err != nil {
		return err
	}
	if err := b.EncodeRMS(s.kv, kvWeight, s.kvNorm, drafts+1, true); err != nil {
		return err
	}
	featureKV, err := s.kvNorm.Slice(0, 512)
	if err != nil {
		return err
	}
	draftKV, err := s.kvNorm.Slice(512, s.kvNorm.Len())
	if err != nil {
		return err
	}
	// Feature and first draft share the seed position. Advancing all rows
	// together would rotate every draft one token ahead.
	if err := a.EncodeRoPE(b, featureKV, 1, 1, position, false); err != nil {
		return err
	}
	if err := a.EncodeRoPE(b, draftKV, 1, drafts, position, false); err != nil {
		return err
	}
	draftNorm, err := norm.Slice(4096, norm.Len())
	if err != nil {
		return err
	}
	if err := d.Encode(b, name("attn_q_a.weight"), draftNorm, s.qLow, drafts); err != nil {
		return err
	}
	qWeight, err := d.weights.tensor(name("attn_q_a_norm.weight"))
	if err != nil {
		return err
	}
	if err := b.EncodeRMS(s.qLow, qWeight, s.qLowNorm, drafts, true); err != nil {
		return err
	}
	if err := d.Encode(b, name("attn_q_b.weight"), s.qLowNorm, s.query, drafts); err != nil {
		return err
	}
	if err := b.EncodeRMS(s.query, kvWeight, s.queryNorm, drafts*64, false); err != nil {
		return err
	}
	return a.EncodeRoPE(b, s.queryNorm, 64, drafts, position, false)
}

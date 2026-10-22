//go:build darwin && arm64

package main

import (
	"context"
	"fmt"

	applemetal "github.com/tmc/apple/metal"
	libmetal "github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/metalmem"
)

type metalDSparkNative struct {
	target             *metalRequestState
	weights            *metalDSparkWeights
	dense              *metalDSparkDense
	attention          *metalDSparkAttention
	ops                *metalPinnedOps
	arena              *metalmem.Arena
	rows               metalDSparkForwardRows
	logits, confidence metalmem.Slice[float32]
	confidenceLogits   metalmem.Slice[float32]
	ids                metalmem.Slice[int32]
	result             metalmem.Slice[uint32]
	batch              *metalBatch
	closed, failed     bool
}

func newMetalDSparkNative(target *metalRequestState, w *metalDSparkWeights) (_ *metalDSparkNative, err error) {
	if target == nil || target.closed || target.runtime == nil || target.runtime.dsparkFeatures == nil || target.runtime.dsparkFeatures.support != nil || w == nil || w.closed || w.model == nil || w.device != target.runtime.dev || target.runtime.dsparkFeatures.weights != w {
		return nil, fmt.Errorf("metal dspark native: bound captured target and support required")
	}
	s := &metalDSparkNative{target: target, weights: w}
	defer func() {
		if err != nil {
			err = finishBackendCleanup(err, s)
		}
	}()
	s.ops, err = newMetalPinnedOps(w.device)
	if err != nil {
		return nil, err
	}
	s.dense, err = newMetalDSparkDense(w.device, w)
	if err != nil {
		return nil, err
	}
	s.attention, err = newMetalDSparkAttention(w.device, w)
	if err != nil {
		return nil, err
	}
	s.arena = metalmem.NewArena(w.device)
	take := func(name string, n int) metalmem.Slice[float32] {
		return metalmem.AllocIn[float32](s.arena, "dspark-"+name, n)
	}
	block := w.model.block
	state := take("stage-input", (block+1)*16384)
	s.rows.main = take("main", 4096)
	s.rows.mainNorm = take("main-norm", 4096)
	s.rows.embeddings = take("embeddings", block*4096)
	for stage := 0; stage < w.model.stages; stage++ {
		t, ok := w.model.tensor(fmt.Sprintf("mtp.%d.ffn_gate_exps.weight", stage))
		if !ok || len(t.Dimensions) != 3 || t.Dimensions[1] > 1<<20 {
			return nil, fmt.Errorf("metal dspark native: missing stage expert geometry")
		}
		width := int(t.Dimensions[1])
		prefix := fmt.Sprintf("stage-%d-", stage)
		q := metalDSparkQKVRows{state: state, flat: take(prefix+"flat", state.Len()), normalized: take(prefix+"norm", (block+1)*8192), kv: take(prefix+"kv", (block+1)*512), kvNorm: take(prefix+"kv-norm", (block+1)*512), qLow: take(prefix+"q-low", block*1024), qLowNorm: take(prefix+"q-low-norm", block*1024), query: take(prefix+"query", block*32768), queryNorm: take(prefix+"query-norm", block*32768)}
		a := metalDSparkAttentionRows{qkv: q, raw: take(prefix+"raw", (Flash.SlidingWindow+block+1)*512), heads: take(prefix+"heads", block*32768), low: take(prefix+"low", block*Flash.OutGroups*Flash.OutLoRA), projection: take(prefix+"projection", block*4096), output: take(prefix+"after-attn", block*16384)}
		f := metalDSparkFFNRows{state: a.output, flat: take(prefix+"ffn-flat", a.output.Len()), normalized: take(prefix+"ffn-norm", block*8192), logits: take(prefix+"router", block*256), probs: take(prefix+"probs", block*256), weights: take(prefix+"expert-weights", block*6), ids: metalmem.AllocIn[int32](s.arena, prefix+"expert-ids", block*6), gate: take(prefix+"gate", block*6*width), up: take(prefix+"up", block*6*width), mid: take(prefix+"mid", block*6*width), experts: take(prefix+"experts", block*7*4096), sharedGate: take(prefix+"shared-gate", block*width), sharedUp: take(prefix+"shared-up", block*width), sharedMid: take(prefix+"shared-mid", block*width), sum: take(prefix+"sum", block*4096), output: take(prefix+"output", block*16384)}
		s.rows.attention = append(s.rows.attention, a)
		s.rows.ffn = append(s.rows.ffn, f)
	}
	s.rows.headFlat = take("head-flat", block*16384)
	s.rows.headMix = take("head-mix", block*4)
	s.rows.headWeighted = take("head-weighted", block*4096)
	s.rows.hidden = take("hidden", block*4096)
	s.logits = take("logits", block*Flash.Vocab)
	s.confidence = take("confidence", block)
	s.confidenceLogits = take("confidence-logits", block)
	s.ids = metalmem.AllocIn[int32](s.arena, "dspark-proposals", block)
	s.result = metalmem.AllocIn[uint32](s.arena, "dspark-result", 2)
	if err := s.arena.Err(); err != nil {
		return nil, err
	}
	target.runtime.dsparkFeatures.support = s
	return s, nil
}

// proposal changes only request-private support scratch. Every call rebuilds
// trusted KV from actual target captures before speculative rows overwrite it.
func (s *metalDSparkNative) proposal(seed, count int, threshold float32, incoming bool) ([]int, error) {
	if s == nil || s.closed || s.failed || s.target.closed || count < 1 || count > s.weights.model.block {
		return nil, fmt.Errorf("metal dspark native: unavailable proposal owner")
	}
	f, r := s.target.runtime.dsparkFeatures, s.target.runtime
	if f == nil || f.closed || f.journal.frontier != r.pos || f.batch != nil || r.pos < 1 {
		return nil, fmt.Errorf("metal dspark native: unpublished feature frontier")
	}
	row, err := f.journal.row(r.pos-1, f.epoch)
	if err != nil {
		return nil, err
	}
	if !incoming && seed != row.token {
		return nil, fmt.Errorf("metal dspark native: seed differs from feature token")
	}
	if seed < 0 || seed >= Flash.Vocab {
		return nil, fmt.Errorf("metal dspark native: invalid seed")
	}
	position := r.pos - 1
	if incoming {
		position++
	}
	b, err := newMetalBatch(s.ops)
	if err != nil {
		return nil, err
	}
	s.batch = b
	defer func() {
		if s.batch != nil {
			s.failed = true
		}
	}()
	capacity := Flash.SlidingWindow + s.weights.model.block + 1
	start, trusted, err := s.dense.EncodeTrusted(b, s.attention, f, position, capacity, s.rows)
	if err != nil {
		return nil, err
	}
	width := len(f.targets) * 4096
	slot := (r.pos - 1) % len(f.journal.rows)
	s.rows.features, err = f.data.Slice(slot*width, (slot+1)*width)
	if err != nil {
		return nil, err
	}
	for row := 0; row < s.weights.model.block; row++ {
		token := s.weights.model.noise
		if row == 0 {
			token = seed
		}
		out, err := tokenRow(s.rows.embeddings, row, 4096)
		if err != nil {
			return nil, err
		}
		if r.output.embedding.typeID != 1 || r.output.embedding.dimensions[0] != 4096 {
			return nil, fmt.Errorf("metal dspark native: unsupported base embedding")
		}
		enc := b.encoder()
		enc.SetComputePipelineState(s.ops.kernels.GetRowF16)
		enc.SetBufferWithOffsetAtIndex(r.output.embedding.data.Buffer(), r.output.embedding.data.Offset(), 0)
		enc.SetBufferWithOffsetAtIndex(out.Buffer(), out.Offset(), 1)
		libmetal.SetScalar(enc, uint32(token), 2)
		libmetal.SetScalar(enc, uint32(4096), 3)
		enc.DispatchThreadsThreadsPerThreadgroup(libmetal.Threads(4096), libmetal.Threads(256))
		b.encoded++
	}
	if err := s.dense.EncodeForward(b, s.attention, position, capacity, position%capacity, start%capacity, trusted+s.weights.model.block+1, s.rows); err != nil {
		return nil, err
	}
	hidden, err := s.rows.hidden.Slice(0, count*4096)
	if err != nil {
		return nil, err
	}
	logits, err := s.logits.Slice(0, count*Flash.Vocab)
	if err != nil {
		return nil, err
	}
	ids, err := s.ids.Slice(0, count)
	if err != nil {
		return nil, err
	}
	confidence, err := s.confidence.Slice(0, count)
	if err != nil {
		return nil, err
	}
	confidenceLogits, err := s.confidenceLogits.Slice(0, count)
	if err != nil {
		return nil, err
	}
	if err := s.dense.EncodeProposalHead(b, s.attention, r.output.weight, hidden, logits, ids, confidence, confidenceLogits, s.result, count, seed, threshold); err != nil {
		return nil, err
	}
	if err := b.Wait(context.Background()); err != nil {
		return nil, err
	}
	if b.buffer.Status() != applemetal.MTLCommandBufferStatusCompleted {
		return nil, fmt.Errorf("metal dspark native: command did not complete successfully")
	}
	proposal, err := metalDSparkProposalResult(ids.HostSlice(), confidence.HostSlice(), s.result.HostSlice(), count, threshold)
	if err != nil {
		return nil, err
	}
	s.batch = nil
	return proposal, nil
}

func (s *metalDSparkNative) Close() error {
	if s == nil || s.closed {
		return nil
	}
	if s.batch != nil {
		if !s.batch.done {
			s.batch.endEncoding()
			s.ops.hc.complete(s.batch)
			s.batch.done = true
		}
		if s.batch.buffer.Status() != applemetal.MTLCommandBufferStatusNotEnqueued && !metalGroupedTerminal(s.batch) {
			return fmt.Errorf("metal dspark native: live command retains support resources")
		}
	}
	// Unsubmitted encoders cannot be submitted after this owner closes.
	if s.dense != nil {
		s.dense.batches = nil
		if err := s.dense.Close(); err != nil {
			return err
		}
		s.dense = nil
	}
	if s.attention != nil {
		s.attention.batches = nil
		if err := s.attention.Close(); err != nil {
			return err
		}
		s.attention = nil
	}
	if s.arena != nil {
		s.arena.Free()
		s.arena = nil
	}
	if s.ops != nil {
		s.ops.Close()
		s.ops = nil
	}
	s.closed = true
	if s.target != nil && s.target.runtime != nil && s.target.runtime.dsparkFeatures != nil && s.target.runtime.dsparkFeatures.support == s {
		s.target.runtime.dsparkFeatures.support = nil
	}
	return nil
}

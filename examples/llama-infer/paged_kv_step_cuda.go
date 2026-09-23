//go:build !darwin && !tpu

package main

// Binding the paged KV pool to a forward call.
//
// A pagedSequence is one sequence's hold on blocks in the pool. A pagedStep is
// the attention of one forward call: the engine binds it to a scratchpad with
// the rows' sequences and positions before the call, and the forward body
// calls attend once per layer in place of the dense RoPE, StoreKV and
// attention launches. Everything here is UNQUALIFIED with the rest of the
// paged path; see paged_kv_cuda.go.

import (
	"context"
	"errors"
	"fmt"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// pagedSequence is one sequence's block table in the pool. The zero value is
// not usable; runtimes hand them out with newSequence.
type pagedSequence struct {
	rt *pagedKVRuntime
	id int64

	// reserved is how many positions from zero have a block.
	reserved int
}

// newSequence registers a sequence holding no blocks yet.
func (r *pagedKVRuntime) newSequence() (*pagedSequence, error) {
	r.nextSeq++
	id := r.nextSeq
	if err := r.cache.AllocSequence(id, 0); err != nil {
		return nil, fmt.Errorf("paged kv: sequence %d: %w", id, err)
	}
	return &pagedSequence{rt: r, id: id}, nil
}

// newSequenceWithPrefix registers a sequence whose first tokens positions are
// already held in prefix, blocks another holder (the prefix cache) has, so the
// sequence writes from tokens on. tokens must be exactly the blocks' capacity.
func (r *pagedKVRuntime) newSequenceWithPrefix(prefix []int, tokens int) (*pagedSequence, error) {
	if tokens != len(prefix)*r.cfg.BlockSize {
		return nil, fmt.Errorf("paged kv: %d prefix blocks of %d tokens do not cover exactly %d tokens", len(prefix), r.cfg.BlockSize, tokens)
	}
	if len(prefix) == 0 {
		return r.newSequence()
	}
	r.nextSeq++
	id := r.nextSeq
	if err := r.cache.AllocSequenceWithPrefix(id, prefix, tokens); err != nil {
		return nil, fmt.Errorf("paged kv: sequence %d: %w", id, err)
	}
	return &pagedSequence{rt: r, id: id, reserved: tokens}, nil
}

// blockBytes is the device memory one block holds across every layer.
func (r *pagedKVRuntime) blockBytes() int64 {
	return int64(r.layers) * 2 * int64(r.cfg.BlockSize) * int64(r.kvHeads) * int64(r.headDim) * 4
}

// table is a copy of the sequence's block table, in position order.
func (s *pagedSequence) table() []int {
	if s == nil || s.rt == nil {
		return nil
	}
	return s.rt.cache.BlockTable(s.id)
}

// reserve makes sure every position up to and including pos has a block.
func (s *pagedSequence) reserve(pos int) error {
	if s == nil || s.rt == nil {
		return errors.New("paged kv: freed sequence")
	}
	if pos < 0 || pos >= s.rt.maxSeqLen {
		return fmt.Errorf("paged kv: position %d outside the %d-token context window", pos, s.rt.maxSeqLen)
	}
	for s.reserved <= pos {
		if err := s.rt.cache.AppendToken(s.id); err != nil {
			return fmt.Errorf("paged kv: sequence %d at position %d: %w", s.id, s.reserved, err)
		}
		s.reserved++
	}
	return nil
}

// reset gives the sequence's blocks back so it can be written again from
// position zero. Nothing is zeroed: attention reads only positions at or
// before the current one, all of which are rewritten first.
func (s *pagedSequence) reset() error {
	if s == nil || s.rt == nil {
		return errors.New("paged kv: freed sequence")
	}
	s.rt.cache.FreeSequence(s.id)
	s.reserved = 0
	if err := s.rt.cache.AllocSequence(s.id, 0); err != nil {
		return fmt.Errorf("paged kv: sequence %d: %w", s.id, err)
	}
	return nil
}

// free returns the sequence's blocks to the pool. It is safe on a nil or
// already freed sequence.
func (s *pagedSequence) free() {
	if s == nil || s.rt == nil {
		return
	}
	s.rt.cache.FreeSequence(s.id)
	s.rt = nil
}

// pagedStep is the paged attention of one forward call. It lives on the
// scratchpad the call uses, is bound before every call, and errs on a call it
// was not freshly bound for, so stale side tables are an error and never a
// silently wrong result.
type pagedStep struct {
	rt   *pagedKVRuntime
	rows int

	// armed is set by bind and cleared by the first layer's attend.
	armed bool
	// ran records that a step has launched since the last status check.
	ran bool
}

// bind stages the step for rows whose sequences and positions are given: it
// checks the status word of the previous step, reserves blocks up to each
// position, lays out the side tables and uploads them on stream.
func (s *pagedStep) bind(stream *cuda.Stream, seqs []*pagedSequence, positions []int) error {
	if s == nil || s.rt == nil {
		return errors.New("paged kv: step has no runtime")
	}
	if len(seqs) == 0 || len(seqs) != len(positions) {
		return fmt.Errorf("paged kv: binding %d sequences to %d positions", len(seqs), len(positions))
	}
	if len(seqs) > s.rt.maxSeqs {
		return fmt.Errorf("paged kv: %d rows exceeds the %d the pool was sized for", len(seqs), s.rt.maxSeqs)
	}
	if s.ran {
		// The previous step's kernels report a bad block table through the
		// status word, not a fault. Both forward paths read their result back
		// before returning, so the stream is normally idle here and this
		// synchronize costs nothing.
		if err := stream.Synchronize(context.Background()); err != nil {
			return fmt.Errorf("paged kv: previous step: %w", err)
		}
		if err := s.rt.CheckStatus(); err != nil {
			return fmt.Errorf("paged kv: previous step: %w", err)
		}
		s.ran = false
	}
	tables := make([][]int, len(seqs))
	for i, seq := range seqs {
		if seq == nil || seq.rt == nil {
			return fmt.Errorf("paged kv: row %d has no live sequence", i)
		}
		if seq.rt != s.rt {
			return fmt.Errorf("paged kv: row %d belongs to another pool", i)
		}
		if err := seq.reserve(positions[i]); err != nil {
			return err
		}
		tables[i] = s.rt.cache.BlockTable(seq.id)
	}
	t, err := buildPagedKVTables(tables, positions, s.rt.cfg.BlockSize, s.rt.cfg.MaxBlocksPerSeq)
	if err != nil {
		return err
	}
	if err := s.rt.UploadBatch(stream, t.seqIndex, t.positions, t.seqLens, t.blockTable); err != nil {
		return err
	}
	s.rows = len(seqs)
	s.armed = true
	return nil
}

// attend runs one layer's RoPE, KV store and attention for every bound row. q,
// k, v and out hold rows contiguous rows each, in the scratchpad's layout.
func (s *pagedStep) attend(stream *cuda.Stream, counter *cudalaunch.Counter, layer int, q, k, v, out cudamem.Slice[float32], rows int, ropeTheta float32) error {
	if s == nil || s.rt == nil {
		return errors.New("paged kv: step has no runtime")
	}
	if rows != s.rows {
		return fmt.Errorf("paged kv: step bound for %d rows, forward has %d", s.rows, rows)
	}
	if layer == 0 {
		if !s.armed {
			return errors.New("paged kv: forward call without a fresh bind")
		}
		s.armed = false
		s.ran = true
	}
	qRoped := s.rt.qRoped.Slice(0, rows*s.rt.qHeads*s.rt.headDim)
	if err := s.rt.StoreKV(stream, counter, layer, q, k, v, qRoped, rows, rows, ropeTheta); err != nil {
		return fmt.Errorf("paged kv: layer %d store: %w", layer, err)
	}
	if err := s.rt.Decode(stream, counter, layer, qRoped, out, rows); err != nil {
		return fmt.Errorf("paged kv: layer %d attention: %w", layer, err)
	}
	return nil
}

// pagedSequenceFor returns the sequence a forward call over skv writes: skv's
// own when it has one, the engine's single-stream sequence otherwise.
func (eng *ModelEngine) pagedSequenceFor(skv *SequenceKVCache) *pagedSequence {
	if skv != nil && skv.paged != nil {
		return skv.paged
	}
	return eng.pagedMain
}

// bindPagedSingle stages the engine scratchpad for one token of skv's sequence
// at pos. It does nothing on a dense engine, so callers need not test the
// mode.
func (eng *ModelEngine) bindPagedSingle(skv *SequenceKVCache, pos int) error {
	if eng.paged == nil {
		return nil
	}
	if eng.scratch.paged == nil {
		eng.scratch.paged = &pagedStep{rt: eng.paged}
	}
	return eng.scratch.paged.bind(eng.stream, []*pagedSequence{eng.pagedSequenceFor(skv)}, []int{pos})
}

// bindPagedBatch stages sp for one row per cache at positions. It does nothing
// on a dense engine.
func (eng *ModelEngine) bindPagedBatch(sp *GPUBatchScratchpad, caches []*SequenceKVCache, positions []int) error {
	if eng.paged == nil {
		return nil
	}
	if sp == nil {
		return errors.New("paged kv: nil batch scratchpad")
	}
	seqs := make([]*pagedSequence, len(caches))
	for i, c := range caches {
		if c == nil || c.paged == nil {
			return fmt.Errorf("paged kv: batch row %d has no paged sequence", i)
		}
		seqs[i] = c.paged
	}
	if sp.paged == nil {
		sp.paged = &pagedStep{rt: eng.paged}
	}
	return sp.paged.bind(eng.stream, seqs, positions)
}

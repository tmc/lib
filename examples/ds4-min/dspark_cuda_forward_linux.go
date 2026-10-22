//go:build linux

package main

import (
	"fmt"
	"math"

	"github.com/tmc/lib/nvidia/x/cudamem"
)

// cudaDSparkForward is actual request-private native support scratch. It is
// not an admitted scheduler: emitted values and the FFN MMQ route require
// independent physical qualification before a production caller may use it.
type cudaDSparkForward struct {
	weights                               *cudaDSparkWeights
	target                                *cudaRequestState
	arena                                 *cudamem.Arena
	ffn                                   *cudaDSparkFFN
	input, hcWork, qLow, query, kv, heads cudamem.Slice[float32]
	outLow, projection, afterAttn, output cudamem.Slice[float32]
	main, trustedKV                       cudamem.Slice[float32]
	cache                                 []cudamem.Slice[float32]
	head                                  cudaDSparkHeadWork
	capacity                              int
	closed, failed                        bool
}

func newCUDADSparkForward(target *cudaRequestState, weights *cudaDSparkWeights) (_ *cudaDSparkForward, err error) {
	if target == nil || target.closed || target.runtime == nil || target.runtime.cudaFoundation == nil || weights == nil || weights.closed || weights.model == nil || weights.device != target.runtime.device || weights.model.block != 5 {
		return nil, fmt.Errorf("cuda dspark forward: request and supported resident model required")
	}
	if err := dsparkRequestOpen(target.runtime); err != nil {
		return nil, err
	}
	s := &cudaDSparkForward{weights: weights, target: target, capacity: Flash.SlidingWindow + weights.model.block + 1}
	defer func() {
		if err != nil {
			err = finishBackendCleanup(err, s)
		}
	}()
	s.arena = cudamem.NewArena(weights.device)
	take := func(name string, n int) cudamem.Slice[float32] {
		return cudamem.AllocIn[float32](s.arena, "dspark-"+name, n)
	}
	block := weights.model.block
	hcWidth := Flash.HC * Flash.Embed
	s.input = take("input-hc", (block+1)*hcWidth)
	s.hcWork = take("hc-work", (block+1)*2*Flash.Embed)
	s.qLow = take("q-low", Flash.QLoRA)
	s.query = take("query", block*Flash.Heads*Flash.HeadDim)
	s.kv = take("current-kv", (block+1)*Flash.HeadDim)
	s.heads = take("attention-values", s.query.Len())
	s.outLow = take("attention-low", Flash.OutGroups*Flash.OutLoRA)
	s.projection = take("attention-projection", Flash.Embed)
	s.afterAttn = take("after-attention-hc", block*hcWidth)
	s.output = take("after-ffn-hc", block*hcWidth)
	s.main, s.trustedKV = take("trusted-main", Flash.Embed), take("trusted-kv", Flash.HeadDim)
	for stage := 0; stage < weights.model.stages; stage++ {
		s.cache = append(s.cache, take(fmt.Sprintf("stage-%d-kv", stage), s.capacity*Flash.HeadDim))
	}
	s.head.mixed = take("head-mixed", 2*Flash.Embed)
	s.head.hidden = take("head-hidden", block*Flash.Embed)
	s.head.logits = take("head-logits", block*Flash.Vocab)
	s.head.markov = take("markov-state", weights.model.rank)
	s.head.bias = take("markov-bias", Flash.Vocab)
	s.head.features = take("confidence-input", Flash.Embed+weights.model.rank)
	s.head.confidence = take("confidence", block)
	s.head.proposal = cudamem.AllocIn[int32](s.arena, "dspark-proposal", block)
	s.head.length = cudamem.AllocIn[uint32](s.arena, "dspark-proposal-length", 1)
	s.head.status = cudamem.AllocIn[uint32](s.arena, "dspark-proposal-status", 1)
	if err := s.arena.Err(); err != nil {
		return nil, err
	}
	s.ffn, err = newCUDADSparkFFN(weights.device)
	if err != nil {
		return nil, err
	}
	return s, nil
}

// encodeTrusted rebuilds only real target feature rows. Temporary support
// predictions never enter the journal. This deliberately favors correctness
// over incremental-cache speed until the lifecycle/value gates qualify it.
func (s *cudaDSparkForward) encodeTrusted(position int) (start, count int, err error) {
	f, r, w := s.target.dspark, s.target.runtime, s.weights
	if f == nil || f.history == nil || f.history.journal == nil || !f.bound() || !f.published || f.position != position || f.modelSHA != w.model.sha256 {
		return 0, 0, fmt.Errorf("cuda dspark forward: missing trusted feature history")
	}
	if err := f.history.journal.validate(position + 1); err != nil {
		return 0, 0, err
	}
	start = max(0, position-Flash.SlidingWindow)
	count = position - start
	norm, err := w.tensor("mtp.0.main_norm.weight")
	if err != nil {
		return 0, 0, err
	}
	for row := start; row < position; row++ {
		features, err := f.history.row(row)
		if err != nil {
			return 0, 0, err
		}
		if err := w.encodeDense(r, "mtp.0.main_proj.weight", features, s.main); err != nil {
			return 0, 0, err
		}
		if err := r.cudaFoundation.encodeRMS(s.main, norm, s.main, 1, true); err != nil {
			return 0, 0, err
		}
		for stage := range s.cache {
			name := fmt.Sprintf("mtp.%d.", stage)
			if err := w.encodeDense(r, name+"attn_kv.weight", s.main, s.trustedKV); err != nil {
				return 0, 0, err
			}
			kvNorm, err := w.tensor(name + "attn_kv_a_norm.weight")
			if err != nil {
				return 0, 0, err
			}
			if err := r.cudaFoundation.encodeRMS(s.trustedKV, kvNorm, s.trustedKV, 1, true); err != nil {
				return 0, 0, err
			}
			if err := w.encodeRope(r, s.trustedKV, 1, 1, row, false); err != nil {
				return 0, 0, err
			}
			if err := w.encodeStoreRaw(r, s.trustedKV, s.cache[stage], s.capacity, row%s.capacity, 1); err != nil {
				return 0, 0, err
			}
		}
	}
	return start, count, nil
}

// proposal executes support weights, then reads only emitted proposal IDs,
// confidence, and status. It does not run target Eval or a CPU forward.
func (s *cudaDSparkForward) proposal(seed, count int, threshold float32) (tokens []int, confidence []float32, err error) {
	if s == nil || s.closed || s.failed || s.target == nil || s.target.closed || s.target.runtime == nil || s.target.dspark == nil || s.weights == nil || s.weights.closed || s.weights.model == nil || count < 1 || count > s.weights.model.block || seed < 0 || seed >= Flash.Vocab || threshold < 0 || threshold > 1 || math.IsNaN(float64(threshold)) {
		return nil, nil, fmt.Errorf("cuda dspark forward: unavailable proposal owner")
	}
	if err := dsparkRequestOpen(s.target.runtime); err != nil {
		return nil, nil, err
	}
	defer func() {
		if err != nil {
			s.failed = true
		}
	}()
	f, r, w := s.target.dspark, s.target.runtime, s.weights
	position := f.position
	start, trusted, err := s.encodeTrusted(position)
	if err != nil {
		return nil, nil, err
	}
	if err := w.encodeInput(f, seed, s.input); err != nil {
		return nil, nil, err
	}
	for stage := range s.cache {
		if err := w.encodeStageQKV(r, stage, position, s.input, s.hcWork, s.qLow, s.query, s.kv); err != nil {
			return nil, nil, err
		}
		if err := w.encodeStoreRaw(r, s.kv, s.cache[stage], s.capacity, position%s.capacity, w.model.block+1); err != nil {
			return nil, nil, err
		}
		if err := w.encodeNoncausal(r, stage, s.query, s.cache[stage], s.heads, s.capacity, start%s.capacity, trusted+w.model.block+1, w.model.block); err != nil {
			return nil, nil, err
		}
		if err := w.encodeAttentionOutput(r, stage, position, s.input, s.heads, s.outLow, s.projection, s.afterAttn); err != nil {
			return nil, nil, err
		}
		if err := s.ffn.encode(w, r, stage, s.afterAttn, s.output); err != nil {
			return nil, nil, err
		}
		if stage+1 < len(s.cache) {
			if err := s.input.Slice(Flash.HC*Flash.Embed, s.input.Len()).CopyDeviceToDeviceAsync(s.output, r.stream()); err != nil {
				return nil, nil, err
			}
		}
	}
	head := s.head
	head.hidden = head.hidden.Slice(0, count*Flash.Embed)
	head.logits = head.logits.Slice(0, count*Flash.Vocab)
	head.confidence = head.confidence.Slice(0, count)
	head.proposal = head.proposal.Slice(0, count)
	if err := w.encodeHead(r, s.output, seed, count, threshold, &head); err != nil {
		return nil, nil, err
	}
	if err := syncStream(r.stream()); err != nil {
		return nil, nil, err
	}
	var status, length [1]uint32
	if err := s.head.status.CopyToHost(status[:]); err != nil {
		return nil, nil, err
	}
	if err := s.head.length.CopyToHost(length[:]); err != nil {
		return nil, nil, err
	}
	if status[0] != 0 || length[0] > uint32(count) {
		return nil, nil, fmt.Errorf("cuda dspark forward: invalid emitted proposal status %d length %d", status[0], length[0])
	}
	ids := make([]int32, count)
	confidence = make([]float32, count)
	if err := head.proposal.CopyToHost(ids); err != nil {
		return nil, nil, err
	}
	if err := head.confidence.CopyToHost(confidence); err != nil {
		return nil, nil, err
	}
	for _, id := range ids[:length[0]] {
		if id < 0 || id >= int32(Flash.Vocab) {
			return nil, nil, fmt.Errorf("cuda dspark forward: emitted token outside vocabulary")
		}
		tokens = append(tokens, int(id))
	}
	return tokens, confidence[:length[0]], nil
}

func (s *cudaDSparkForward) close() error {
	if s == nil {
		return nil
	}
	s.closed = true
	if s.target != nil && s.target.runtime != nil {
		if err := syncStream(s.target.runtime.stream()); err != nil {
			return err
		}
	}
	if s.ffn != nil {
		if err := s.ffn.close(); err != nil {
			return err
		}
		s.ffn = nil
	}
	if s.arena != nil {
		if err := s.arena.Free(); err != nil {
			return err
		}
		s.arena = nil
	}
	return nil
}

// Close retries cleanup retained by a failed constructor. A failed release
// leaves the forward terminal for computation while preserving its resources.
func (s *cudaDSparkForward) Close() error { return s.close() }

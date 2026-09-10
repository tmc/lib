//go:build linux && !tpu

package main

import (
	"context"
	"errors"
	"fmt"
	"sync"

	"github.com/tmc/lib/examples/internal/continuousbatch"
	"github.com/tmc/lib/examples/internal/inferbench"
	"github.com/tmc/lib/nvidia/x/cudakv"
)

// cudaServingBackend connects the continuous scheduler to the complete CUDA
// forward pass. Each request owns its KV cache. The engine scratch buffers and
// stream are shared, so device work is serialized until the CUDA model path has
// a true shared-weight batched forward pass.
type cudaServingBackend struct {
	engine          *ModelEngine
	cfg             Config
	batch           *GPUBatchScratchpad
	mu              sync.Mutex
	last            cudaBatchLaunches
	live            map[*SequenceKVCache]uint64
	observedOutputs *observedOutputCollector
	next            uint64
	closed          bool
	physical        *physicalWaveLedger

	// prefixes reuses finished sequences' blocks for prompts that begin with
	// their tokens. It is set only when the engine is paged and
	// LLAMA_INFER_PAGED_PREFIX=1 (prefix_cache_paged.go, UNQUALIFIED).
	prefixes *pagedPrefixCache
}

// resetBenchmarkPrefixes drops what the prefix cache retained, so no sample
// reuses blocks an earlier one computed. Without the cache nothing is
// retained and the reset is exact with no work.
func (b *cudaServingBackend) resetBenchmarkPrefixes() error {
	if b.prefixes == nil {
		return nil
	}
	b.engine.generateMu.Lock()
	defer b.engine.generateMu.Unlock()
	return b.prefixes.reset()
}

func (b *cudaServingBackend) prefixCacheMetrics() (PrefixCacheMetrics, bool) {
	if b == nil || b.prefixes == nil {
		return PrefixCacheMetrics{}, false
	}
	return b.prefixes.metrics(), true
}
func (b *cudaServingBackend) enableBenchmarkLedger() { b.physical = new(physicalWaveLedger) }
func (b *cudaServingBackend) benchmarkLedgerSnapshot() physicalLedgerSnapshot {
	return b.physical.Snapshot()
}
func (b *cudaServingBackend) benchmarkLedgerSince(before physicalLedgerSnapshot) ([]physicalWave, error) {
	return b.physical.Since(before)
}
func (b *cudaServingBackend) recordPhysical(kind string, requests []string, dispatches uint64) error {
	if b.physical == nil {
		return nil
	}
	return b.physical.Record(kind, requests, dispatches, nil, nil)
}
func (b *cudaServingBackend) installBenchmarkObserver() {
	b.observedOutputs = newObservedOutputCollector()
}
func (b *cudaServingBackend) benchmarkOutputs(ids []string) ([]inferbench.ObservedOutput, error) {
	if b.observedOutputs == nil {
		return nil, errors.New("inferbench: observed output collector unavailable")
	}
	return b.observedOutputs.outputs(ids)
}

type cudaSequenceState struct {
	cache      *SequenceKVCache
	generation uint64

	// prefix is the sequence's standing in the prefix cache: the node it
	// joined on and every token it has written, for the leave that stores
	// them. Nil without the cache.
	prefix *cudaPrefixHold
}

type cudaPrefixHold struct {
	node   *cudakv.RadixNode
	tokens []int32
}

func newPlatformServingBackend(engine *ModelEngine, cfg Config, capacity int) (servingBackend, error) {
	if engine == nil || engine.dev == nil || engine.weights == nil || engine.scratch == nil || engine.stream == nil {
		return nil, fmt.Errorf("%w: incomplete CUDA engine", errServingBackendUnavailable)
	}
	if capacity <= 0 {
		return nil, errors.New("cuda serving backend: capacity must be positive")
	}
	if engine.tokenizer == nil {
		return nil, fmt.Errorf("%w: tokenizer unavailable", errServingBackendUnavailable)
	}
	var prefixes *pagedPrefixCache
	if pagedPrefixEnabled() {
		if engine.paged == nil {
			return nil, errors.New("cuda serving backend: LLAMA_INFER_PAGED_PREFIX=1 needs an engine compiled with LLAMA_INFER_PAGED_KV=1")
		}
		var err error
		prefixes, err = newPagedPrefixCache(engine.paged.cache, engine.paged.cfg.BlockSize, defaultPagedPrefixNodes, engine.paged.blockBytes())
		if err != nil {
			return nil, fmt.Errorf("cuda serving backend: %w", err)
		}
	}
	batch, err := newGPUBatchScratchpad(engine.dev, engine.memory, &engine.mCfg, capacity)
	if err != nil {
		return nil, fmt.Errorf("cuda serving backend: allocate batch scratch: %w", err)
	}
	return &cudaServingBackend{engine: engine, cfg: cudaServingConfig(engine, cfg), batch: batch, live: make(map[*SequenceKVCache]uint64), prefixes: prefixes}, nil
}

// joinSequence allocates the cache a prompt is prefilled into and reports the
// position prefill starts at. Without the prefix cache that is a fresh cache
// and position zero; with it, the prompt's stored prefix is reused and only
// the tail is computed. The caller holds generateMu.
func (b *cudaServingBackend) joinSequence(tokens []int32) (*SequenceKVCache, *cudaPrefixHold, int, error) {
	if b.prefixes == nil {
		cache, err := b.engine.NewSequenceKV(b.cfg)
		return cache, nil, 0, err
	}
	blocks, matched, node, err := b.prefixes.join(tokens)
	if err != nil {
		return nil, nil, 0, err
	}
	// The tail needs its blocks now and the first decode step may need one
	// more; make room before the pool is asked, since a full pool is an
	// error there rather than an eviction.
	blockSize := b.engine.paged.cfg.BlockSize
	need := (len(tokens)+blockSize-1)/blockSize - len(blocks) + 1
	if _, err := b.prefixes.ensureFree(need); err != nil {
		b.prefixes.abandon(node)
		return nil, nil, 0, err
	}
	seq, err := b.engine.paged.newSequenceWithPrefix(blocks, matched)
	if err != nil {
		b.prefixes.abandon(node)
		return nil, nil, 0, err
	}
	return &SequenceKVCache{paged: seq}, &cudaPrefixHold{node: node, tokens: append([]int32(nil), tokens...)}, matched, nil
}

// discardSequence frees a cache whose prefill did not complete. Nothing of it
// is stored: the prefix cache only learns from sequences that finished.
func (b *cudaServingBackend) discardSequence(cache *SequenceKVCache, hold *cudaPrefixHold) {
	if hold != nil && b.prefixes != nil {
		b.prefixes.abandon(hold.node)
	}
	cache.Free()
}

// cudaServingConfig takes model semantics from the compiled engine while
// retaining caller-owned runtime options such as BOS handling and endpoints.
func cudaServingConfig(engine *ModelEngine, cfg Config) Config {
	model := engine.mCfg
	cfg.Dim = model.HiddenSize
	cfg.HiddenDim = model.IntermediateSize
	cfg.NumLayers = model.NumHiddenLayers
	cfg.NumHeads = model.NumAttentionHeads
	cfg.NumKVHeads = model.NumKeyValueHeads
	cfg.HeadDim = model.HeadDim()
	cfg.VocabSize = model.VocabSize
	cfg.MaxSeqLen = model.MaxSeqLen
	cfg.Eps = model.RmsNormEps
	cfg.RopeTheta = model.RopeTheta
	cfg.EOSTokenID = model.EOSTokenID
	cfg.EOSTokenIDs = append([]int32(nil), model.EOSTokenIDs...)
	return cfg
}

func cudaContextErr(ctx context.Context, operation string) error {
	if ctx == nil {
		return fmt.Errorf("%s: nil context", operation)
	}
	return ctx.Err()
}

func (b *cudaServingBackend) Prefill(ctx context.Context, state *continuousbatch.State) (continuousbatch.Token, error) {
	results := b.PrefillBatch([]continuousbatch.PrefillRequest{{Context: ctx, State: state}})
	if len(results) != 1 {
		return continuousbatch.Token{}, errors.New("cuda prefill: invalid batch result")
	}
	return results[0].Token, results[0].Err
}

// PrefillBatch preserves request isolation but does not claim a physical CUDA
// batch: requests execute one at a time over shared model scratch storage.
func (b *cudaServingBackend) PrefillBatch(requests []continuousbatch.PrefillRequest) []continuousbatch.Result {
	results := make([]continuousbatch.Result, len(requests))
	for i, request := range requests {
		results[i] = b.prefill(request.Context, request.State)
	}
	return results
}

func (b *cudaServingBackend) prefill(ctx context.Context, state *continuousbatch.State) continuousbatch.Result {
	if err := cudaContextErr(ctx, "cuda prefill"); err != nil {
		return continuousbatch.Result{Err: err}
	}
	if state == nil {
		return continuousbatch.Result{Err: errors.New("cuda prefill: nil state")}
	}
	if err := validateCUDASampler(state.Sampler); err != nil {
		return continuousbatch.Result{Err: err}
	}
	tokens, err := prefillTokens(b.cfg, b.engine.tokenizer, state)
	if err != nil {
		return continuousbatch.Result{Err: err}
	}
	if len(tokens) == 0 {
		return continuousbatch.Result{Err: errors.New("cuda prefill: empty prompt")}
	}
	if len(tokens) >= b.engine.mCfg.MaxSeqLen {
		return continuousbatch.Result{Err: errors.New("cuda prefill: prompt exceeds cache capacity")}
	}
	for _, token := range tokens {
		if token < 0 || int(token) >= b.engine.mCfg.VocabSize {
			return continuousbatch.Result{Err: errors.New("cuda prefill: invalid prompt token")}
		}
	}
	b.engine.generateMu.Lock()
	b.mu.Lock()
	closed := b.closed
	b.mu.Unlock()
	if closed {
		b.engine.generateMu.Unlock()
		return continuousbatch.Result{Err: errors.New("cuda prefill: backend closed")}
	}
	if err = cudaContextErr(ctx, "cuda prefill"); err != nil {
		b.engine.generateMu.Unlock()
		return continuousbatch.Result{Err: fmt.Errorf("cuda prefill: %w", err)}
	}
	cache, hold, start, err := b.joinSequence(tokens)
	if err != nil {
		b.engine.generateMu.Unlock()
		return continuousbatch.Result{Err: fmt.Errorf("cuda prefill: %w", err)}
	}
	next, dispatches, err := gpuBatchedPrefillPassKVMeasured(b.engine, cache, tokens[start:], start)
	b.engine.generateMu.Unlock()
	if err != nil {
		b.discardSequence(cache, hold)
		return continuousbatch.Result{Err: err}
	}
	text, err := b.engine.tokenizer.Decode([]int32{next})
	if err != nil {
		b.discardSequence(cache, hold)
		return continuousbatch.Result{Err: fmt.Errorf("cuda prefill: decode token: %w", err)}
	}
	b.mu.Lock()
	b.next++
	generation := b.next
	b.live[cache] = generation
	b.mu.Unlock()
	state.KV = cudaSequenceState{cache: cache, generation: generation, prefix: hold}
	if err := b.recordPhysical("cuda-prefill", []string{fmt.Sprintf("%s@0/%d", state.ID, generation)}, dispatches); err != nil {
		b.Release(state)
		return continuousbatch.Result{Err: err}
	}
	state.Position = len(tokens)
	return continuousbatch.Result{Token: continuousbatch.Token{ID: next, Text: text}, Done: b.cfg.IsEOS(next)}
}

func (b *cudaServingBackend) DecodeBatch(ctx context.Context, states []*continuousbatch.State) []continuousbatch.Result {
	results := make([]continuousbatch.Result, len(states))
	if err := cudaContextErr(ctx, "cuda decode"); err != nil {
		return batchError(results, err)
	}
	if b == nil {
		return batchError(results, errors.New("cuda decode: backend unavailable"))
	}
	if b.engine != nil {
		b.engine.generateMu.Lock()
		defer b.engine.generateMu.Unlock()
	}
	b.mu.Lock()
	closed := b.closed
	b.mu.Unlock()
	if closed {
		return batchError(results, errors.New("cuda decode: backend closed"))
	}
	var valid []int
	var caches []*SequenceKVCache
	var sequences []cudaSequenceState
	var tokens []int32
	var positions []int
	for i, state := range states {
		if err := cudaContextErr(ctx, "cuda decode"); err != nil {
			results[i].Err = err
			continue
		}
		if state == nil {
			results[i].Err = errors.New("cuda decode: nil state")
			continue
		}
		if err := validateCUDASampler(state.Sampler); err != nil {
			results[i].Err = err
			continue
		}
		sequence, ok := state.KV.(cudaSequenceState)
		if !ok || !sequence.cache.usable() {
			results[i].Err = errors.New("cuda decode: invalid cache state")
			continue
		}
		b.mu.Lock()
		generation := b.live[sequence.cache]
		b.mu.Unlock()
		if generation == 0 || generation != sequence.generation {
			results[i].Err = errors.New("cuda decode: stale cache state")
			continue
		}
		valid = append(valid, i)
		caches = append(caches, sequence.cache)
		sequences = append(sequences, sequence)
		tokens = append(tokens, state.Current)
		positions = append(positions, state.Position)
	}
	if len(valid) == 0 {
		return results
	}
	if b.engine == nil {
		return batchError(results, errors.New("cuda decode: backend unavailable"))
	}
	if b.prefixes != nil {
		// Every row may cross a block boundary this step. Make room before
		// the step binds, where the pool is asked and a full pool is an error.
		if _, err := b.prefixes.ensureFree(len(valid)); err != nil {
			for _, i := range valid {
				results[i].Err = fmt.Errorf("cuda decode: %w", err)
			}
			return results
		}
	}
	next, stats, err := b.decodeCUDA(ctx, states, valid, sequences, caches, tokens, positions)
	b.mu.Lock()
	b.last = stats
	b.mu.Unlock()
	if err != nil {
		for _, i := range valid {
			results[i].Err = fmt.Errorf("cuda decode: %w", err)
		}
		return results
	}
	if b.observedOutputs != nil && !b.engine.benchmarkSerial {
		logits := make([]float32, len(valid)*b.cfg.VocabSize)
		if err := b.batch.Logits.Slice(0, len(logits)).CopyToHost(logits); err != nil {
			return batchError(results, fmt.Errorf("cuda decode: observe logits: %w", err))
		}
		observedStates := make([]*continuousbatch.State, len(valid))
		for row, i := range valid {
			observedStates[row] = states[i]
		}
		if err := b.observedOutputs.observeCUDALogits(observedStates, sequences, logits, b.cfg.VocabSize); err != nil {
			return batchError(results, err)
		}
	}
	for row, i := range valid {
		if sequences[row].prefix != nil {
			// The step wrote tokens[row] at positions[row]; the leave that
			// stores the sequence needs every written token in order.
			sequences[row].prefix.tokens = append(sequences[row].prefix.tokens, tokens[row])
		}
		text, decodeErr := b.engine.tokenizer.Decode([]int32{next[row]})
		results[i] = continuousbatch.Result{Token: continuousbatch.Token{ID: next[row], Text: text}, Done: b.cfg.IsEOS(next[row]), Err: decodeErr}
	}
	return results
}

func (b *cudaServingBackend) decodeCUDA(ctx context.Context, states []*continuousbatch.State, valid []int, sequences []cudaSequenceState, caches []*SequenceKVCache, tokens []int32, positions []int) ([]int32, cudaBatchLaunches, error) {
	widths := cudaBatchWidths(len(valid), b.engine.benchmarkSerial)
	if len(widths) == 1 {
		if err := b.engine.bindKVBatch(b.batch, caches, positions); err != nil {
			return nil, cudaBatchLaunches{}, err
		}
		next, stats, err := GPUForwardBatchKV(ctx, b.engine, b.batch, caches, tokens, positions)
		if err != nil {
			return nil, stats, err
		}
		members := make([]string, len(valid))
		for row, i := range valid {
			members[row] = fmt.Sprintf("%s@%d/%d", states[i].ID, row, sequences[row].generation)
		}
		return next, stats, b.recordPhysical("cuda-decode", members, stats.Dispatches)
	}
	next := make([]int32, len(valid))
	var total cudaBatchLaunches
	for row, i := range valid {
		if err := b.engine.bindKVBatch(b.batch, caches[row:row+1], positions[row:row+1]); err != nil {
			return nil, total, err
		}
		one, stats, err := GPUForwardBatchKV(ctx, b.engine, b.batch, caches[row:row+1], tokens[row:row+1], positions[row:row+1])
		if err != nil {
			return nil, total, err
		}
		next[row] = one[0]
		total.Rows += stats.Rows
		total.Dispatches += stats.Dispatches
		total.WeightLaunches += stats.WeightLaunches
		total.IndependentWeightLaunches += stats.IndependentWeightLaunches
		total.MatrixLaunches += stats.MatrixLaunches
		if b.observedOutputs != nil {
			logits := make([]float32, b.cfg.VocabSize)
			if err := b.batch.Logits.Slice(0, len(logits)).CopyToHost(logits); err != nil {
				return nil, total, fmt.Errorf("cuda decode: observe logits: %w", err)
			}
			if err := b.observedOutputs.observeCUDALogits([]*continuousbatch.State{states[i]}, sequences[row:row+1], logits, b.cfg.VocabSize); err != nil {
				return nil, total, err
			}
		}
		member := fmt.Sprintf("%s@0/%d", states[i].ID, sequences[row].generation)
		if err := b.recordPhysical("cuda-decode", []string{member}, stats.Dispatches); err != nil {
			return nil, total, err
		}
	}
	return next, total, nil
}

func cudaBatchWidths(n int, serial bool) []int {
	if n <= 0 {
		return nil
	}
	if !serial {
		return []int{n}
	}
	widths := make([]int, n)
	for i := range widths {
		widths[i] = 1
	}
	return widths
}

func (b *cudaServingBackend) lastBatchLaunches() cudaBatchLaunches {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.last
}

func (b *cudaServingBackend) Release(state *continuousbatch.State) {
	if state == nil {
		return
	}
	sequence, ok := state.KV.(cudaSequenceState)
	if !ok {
		return
	}
	if b.engine != nil {
		b.engine.generateMu.Lock()
		defer b.engine.generateMu.Unlock()
	}
	b.mu.Lock()
	if b.live[sequence.cache] != sequence.generation {
		b.mu.Unlock()
		state.KV = nil
		return
	}
	delete(b.live, sequence.cache)
	if b.observedOutputs != nil {
		b.observedOutputs.observeCUDAKV(state, sequence, b.cfg)
	}
	b.mu.Unlock()
	if sequence.prefix != nil && b.prefixes != nil && sequence.cache.paged != nil {
		// A bookkeeping failure poisons the cache, which reports it on the
		// next join; there is nothing to return from here.
		_ = b.prefixes.leave(sequence.prefix.node, sequence.prefix.tokens, sequence.cache.paged.table())
	}
	sequence.cache.Free()
	state.KV = nil
}

func (b *cudaServingBackend) Close() {
	if b == nil || b.engine == nil {
		return
	}
	b.engine.generateMu.Lock()
	defer b.engine.generateMu.Unlock()
	b.mu.Lock()
	if b.closed {
		b.mu.Unlock()
		return
	}
	b.closed = true
	batch := b.batch
	b.batch = nil
	b.mu.Unlock()
	batch.Free()
}

func validateCUDASampler(sampler any) error {
	if sampler == nil {
		return nil
	}
	cfg, ok := sampler.(SamplerConfig)
	if !ok {
		return errors.New("cuda serving backend: unsupported sampler state")
	}
	if cfg.Method != "" && cfg.Method != "argmax" {
		return fmt.Errorf("cuda serving backend: sampler %q is unsupported", cfg.Method)
	}
	if cfg.Temperature != 0 {
		return errors.New("cuda serving backend: temperature is unsupported")
	}
	return nil
}

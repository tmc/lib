//go:build linux

package main

import (
	"context"
	"errors"
	"fmt"
	"sync"

	"github.com/tmc/lib/examples/internal/continuousbatch"
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
}

type cudaSequenceState struct {
	cache      *SequenceKVCache
	generation uint64
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
	batch, err := newGPUBatchScratchpad(engine.dev, engine.memory, &engine.mCfg, capacity)
	if err != nil {
		return nil, fmt.Errorf("cuda serving backend: allocate batch scratch: %w", err)
	}
	return &cudaServingBackend{engine: engine, cfg: cudaServingConfig(engine, cfg), batch: batch, live: make(map[*SequenceKVCache]uint64)}, nil
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
	cache, err := b.engine.NewSequenceKV(b.cfg)
	if err != nil {
		return continuousbatch.Result{Err: fmt.Errorf("cuda prefill: %w", err)}
	}
	b.engine.generateMu.Lock()
	b.mu.Lock()
	closed := b.closed
	b.mu.Unlock()
	if closed {
		b.engine.generateMu.Unlock()
		cache.Free()
		return continuousbatch.Result{Err: errors.New("cuda prefill: backend closed")}
	}
	if err = cudaContextErr(ctx, "cuda prefill"); err != nil {
		b.engine.generateMu.Unlock()
		cache.Free()
		return continuousbatch.Result{Err: fmt.Errorf("cuda prefill: %w", err)}
	}
	next, err := GPUPrefillPassKV(b.engine, b.cfg, cache, tokens)
	b.engine.generateMu.Unlock()
	if err != nil {
		cache.Free()
		return continuousbatch.Result{Err: err}
	}
	text, err := b.engine.tokenizer.Decode([]int32{next})
	if err != nil {
		cache.Free()
		return continuousbatch.Result{Err: fmt.Errorf("cuda prefill: decode token: %w", err)}
	}
	b.mu.Lock()
	b.next++
	generation := b.next
	b.live[cache] = generation
	b.mu.Unlock()
	state.KV = cudaSequenceState{cache: cache, generation: generation}
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
		if !ok || sequence.cache == nil || sequence.cache.Cache == nil {
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
	next, stats, err := GPUForwardBatchKV(ctx, b.engine, b.batch, caches, tokens, positions)
	b.mu.Lock()
	b.last = stats
	b.mu.Unlock()
	if err != nil {
		for _, i := range valid {
			results[i].Err = fmt.Errorf("cuda decode: %w", err)
		}
		return results
	}
	if b.observedOutputs != nil {
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
		text, decodeErr := b.engine.tokenizer.Decode([]int32{next[row]})
		results[i] = continuousbatch.Result{Token: continuousbatch.Token{ID: next[row], Text: text}, Done: b.cfg.IsEOS(next[row]), Err: decodeErr}
	}
	return results
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

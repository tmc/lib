//go:build linux

package main

import (
	"context"
	"errors"
	"fmt"

	"github.com/tmc/lib/examples/internal/continuousbatch"
)

// cudaServingBackend connects the continuous scheduler to the complete CUDA
// forward pass. Each request owns its KV cache. The engine scratch buffers and
// stream are shared, so device work is serialized until the CUDA model path has
// a true shared-weight batched forward pass.
type cudaServingBackend struct {
	engine *ModelEngine
	cfg    Config
}

type cudaSequenceState struct {
	cache *SequenceKVCache
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
	return &cudaServingBackend{engine: engine, cfg: cfg}, nil
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
	if state == nil {
		return continuousbatch.Result{Err: errors.New("cuda prefill: nil state")}
	}
	if err := ctx.Err(); err != nil {
		return continuousbatch.Result{Err: err}
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
	if err = ctx.Err(); err != nil {
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
	state.KV = cudaSequenceState{cache: cache}
	state.Position = len(tokens)
	return continuousbatch.Result{Token: continuousbatch.Token{ID: next, Text: text}, Done: b.cfg.IsEOS(next)}
}

func (b *cudaServingBackend) DecodeBatch(ctx context.Context, states []*continuousbatch.State) []continuousbatch.Result {
	results := make([]continuousbatch.Result, len(states))
	for i, state := range states {
		if err := ctx.Err(); err != nil {
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
		b.engine.generateMu.Lock()
		next, err := GPUForwardPassKV(b.engine, b.cfg, sequence.cache, state.Current, state.Position)
		b.engine.generateMu.Unlock()
		if err != nil {
			results[i].Err = fmt.Errorf("cuda decode: %w", err)
			continue
		}
		text, err := b.engine.tokenizer.Decode([]int32{next})
		results[i] = continuousbatch.Result{Token: continuousbatch.Token{ID: next, Text: text}, Done: b.cfg.IsEOS(next), Err: err}
	}
	return results
}

func (b *cudaServingBackend) Release(state *continuousbatch.State) {
	if state == nil {
		return
	}
	sequence, ok := state.KV.(cudaSequenceState)
	if !ok {
		return
	}
	sequence.cache.Free()
	state.KV = nil
}

func (*cudaServingBackend) Close() {}

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

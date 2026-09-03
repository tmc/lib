package main

import (
	"context"
	"errors"
	"fmt"
	"sync"

	"github.com/tmc/lib/examples/internal/continuousbatch"
)

type scheduledEngineAdapter struct {
	engine                              *ModelEngine
	cfg                                 Config
	cache                               *BatchedAttentionCache
	tail                                *BatchedModelTail
	prefillMu, rowsMu, metricsMu, gpuMu sync.Mutex
	freeRows                            []int
	batches                             []fullBatchObservation
}

type scheduledSequenceState struct{ row int }
type fullBatchObservation struct{ Batch, Dispatches int }

func newScheduledEngineAdapter(engine *ModelEngine, cfg Config, capacity int) (*scheduledEngineAdapter, error) {
	cache, err := engine.NewBatchedAttentionCache(cfg, capacity)
	if err != nil {
		return nil, err
	}
	tail, err := engine.NewBatchedModelTail(cfg, capacity)
	if err != nil {
		cache.Free()
		return nil, err
	}
	a := &scheduledEngineAdapter{engine: engine, cfg: cfg, cache: cache, tail: tail, freeRows: make([]int, capacity)}
	for i := range a.freeRows {
		a.freeRows[i] = capacity - i - 1
	}
	return a, nil
}

func (a *scheduledEngineAdapter) Close() { a.cache.Free(); a.tail.Free() }

func (a *scheduledEngineAdapter) Release(state *continuousbatch.State) {
	sequence, ok := state.KV.(scheduledSequenceState)
	if !ok {
		return
	}
	a.rowsMu.Lock()
	a.freeRows = append(a.freeRows, sequence.row)
	a.rowsMu.Unlock()
	state.KV = nil
}

func (a *scheduledEngineAdapter) Prefill(ctx context.Context, state *continuousbatch.State) (continuousbatch.Token, error) {
	if err := ctx.Err(); err != nil {
		return continuousbatch.Token{}, err
	}
	a.rowsMu.Lock()
	if len(a.freeRows) == 0 {
		a.rowsMu.Unlock()
		return continuousbatch.Token{}, errors.New("prefill: no cache row available")
	}
	row := a.freeRows[len(a.freeRows)-1]
	a.freeRows = a.freeRows[:len(a.freeRows)-1]
	a.rowsMu.Unlock()
	releaseRow := true
	defer func() {
		if releaseRow {
			a.rowsMu.Lock()
			a.freeRows = append(a.freeRows, row)
			a.rowsMu.Unlock()
		}
	}()
	a.prefillMu.Lock()
	defer a.prefillMu.Unlock()
	a.gpuMu.Lock()
	defer a.gpuMu.Unlock()
	if a.engine == nil || a.engine.tokenizer == nil {
		return continuousbatch.Token{}, errors.New("prefill: nil engine or tokenizer")
	}
	tokens, err := a.engine.tokenizer.Encode(state.Prompt)
	if err != nil {
		return continuousbatch.Token{}, fmt.Errorf("prefill tokenize: %w", err)
	}
	if len(tokens) == 0 {
		tokens = []int32{1}
	}
	if a.cfg.BOSTokenID != 0 {
		tokens = append([]int32{a.cfg.BOSTokenID}, tokens...)
	}
	if len(tokens) >= a.cfg.MaxSeqLen {
		return continuousbatch.Token{}, errors.New("prefill: prompt exceeds cache capacity")
	}
	sequence, err := a.engine.NewSequenceKV(a.cfg)
	if err != nil {
		return continuousbatch.Token{}, fmt.Errorf("prefill allocate KV: %w", err)
	}
	next, err := GPUPrefillPassKV(a.engine, a.cfg, sequence, tokens)
	if err == nil {
		err = a.cache.ImportSequence(row, len(tokens), sequence)
	}
	sequence.Free()
	if err != nil {
		return continuousbatch.Token{}, fmt.Errorf("prefill GPU state: %w", err)
	}
	text, err := a.engine.tokenizer.Decode([]int32{next})
	if err != nil {
		return continuousbatch.Token{}, fmt.Errorf("prefill decode token: %w", err)
	}
	state.KV, state.Position = scheduledSequenceState{row: row}, len(tokens)
	releaseRow = false
	return continuousbatch.Token{ID: next, Text: text}, nil
}

func (a *scheduledEngineAdapter) DecodeBatch(ctx context.Context, states []*continuousbatch.State) []continuousbatch.Result {
	results := make([]continuousbatch.Result, len(states))
	if err := ctx.Err(); err != nil {
		return batchError(results, err)
	}
	tokens := make([]int32, len(states))
	positions := make([]int, len(states))
	rows := make([]int, len(states))
	samplers := make([]SamplerConfig, len(states))
	for i, state := range states {
		sequence, ok := state.KV.(scheduledSequenceState)
		if !ok {
			return batchError(results, errors.New("decode: invalid cache state"))
		}
		sampler, ok := state.Sampler.(SamplerConfig)
		if state.Sampler != nil && !ok {
			return batchError(results, errors.New("decode: unsupported sampler state"))
		}
		tokens[i], positions[i], rows[i], samplers[i] = state.Current, state.Position, sequence.row, sampler
	}
	a.gpuMu.Lock()
	defer a.gpuMu.Unlock()
	chain, err := RunBatchedLayerChain(a.engine, a.cfg, a.cache, tokens, positions, rows)
	if err != nil {
		return batchError(results, err)
	}
	tail, err := RunBatchedModelTailFromChain(a.engine, a.cfg, a.tail, chain, samplers)
	if err != nil {
		return batchError(results, err)
	}
	a.metricsMu.Lock()
	a.batches = append(a.batches, fullBatchObservation{len(states), chain.Dispatches + tail.Dispatches})
	a.metricsMu.Unlock()
	for i, token := range tail.Tokens {
		text, err := a.engine.tokenizer.Decode([]int32{token})
		results[i] = continuousbatch.Result{Token: continuousbatch.Token{ID: token, Text: text}, Done: a.cfg.IsEOS(token), Err: err}
	}
	return results
}

func batchError(results []continuousbatch.Result, err error) []continuousbatch.Result {
	for i := range results {
		results[i].Err = err
	}
	return results
}
func (a *scheduledEngineAdapter) Observations() []fullBatchObservation {
	a.metricsMu.Lock()
	defer a.metricsMu.Unlock()
	return append([]fullBatchObservation(nil), a.batches...)
}

var _ continuousbatch.Backend = (*scheduledEngineAdapter)(nil)

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
	leases                              map[int]uint64
	nextLease                           uint64
	batches                             []fullBatchObservation
	prefillDispatches                   uint64
}

type scheduledSequenceState struct {
	row   int
	lease uint64
}
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
	a := &scheduledEngineAdapter{engine: engine, cfg: cfg, cache: cache, tail: tail, freeRows: make([]int, capacity), leases: make(map[int]uint64)}
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
	if a.leases[sequence.row] == sequence.lease {
		delete(a.leases, sequence.row)
		a.freeRows = append(a.freeRows, sequence.row)
	}
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
	a.nextLease++
	lease := a.nextLease
	a.leases[row] = lease
	a.rowsMu.Unlock()
	releaseRow := true
	defer func() {
		if releaseRow {
			a.rowsMu.Lock()
			if a.leases[row] == lease {
				delete(a.leases, row)
				a.freeRows = append(a.freeRows, row)
			}
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
	tokens, err := prefillTokens(a.cfg, a.engine.tokenizer, state)
	if err != nil {
		return continuousbatch.Token{}, err
	}
	if len(tokens) == 0 {
		tokens = []int32{1}
	}
	if len(tokens) >= a.cfg.MaxSeqLen {
		return continuousbatch.Token{}, errors.New("prefill: prompt exceeds cache capacity")
	}
	sequence, err := a.engine.NewSequenceKV(a.cfg)
	if err != nil {
		return continuousbatch.Token{}, fmt.Errorf("prefill allocate KV: %w", err)
	}
	start, next := a.engine.restoreScheduledPrefix(tokens, sequence, a.cfg)
	prefillDispatches := 0
	if start < len(tokens) {
		next, err = GPUBatchedPrefillPassKV(a.engine, a.cfg, sequence, tokens[start:], start)
		prefillDispatches = benchmarkPrefillDispatches(a.engine)
	}
	if err == nil {
		err = a.cache.ImportSequence(row, len(tokens), sequence)
	}
	if err == nil {
		a.engine.storeScheduledPrefix(tokens, sequence, next, a.cfg)
	}
	sequence.Free()
	if err != nil {
		return continuousbatch.Token{}, fmt.Errorf("prefill GPU state: %w", err)
	}
	a.metricsMu.Lock()
	a.prefillDispatches += uint64(prefillDispatches)
	a.metricsMu.Unlock()
	text, err := a.engine.tokenizer.Decode([]int32{next})
	if err != nil {
		return continuousbatch.Token{}, fmt.Errorf("prefill decode token: %w", err)
	}
	state.KV, state.Position = scheduledSequenceState{row: row, lease: lease}, len(tokens)
	releaseRow = false
	return continuousbatch.Token{ID: next, Text: text}, nil
}

func prefillTokens(cfg Config, tokenizer Tokenizer, state *continuousbatch.State) ([]int32, error) {
	if len(state.PromptTokenIDs) != 0 {
		return append([]int32(nil), state.PromptTokenIDs...), nil
	}
	tokens, err := tokenizer.Encode(state.Prompt)
	if err != nil {
		return nil, fmt.Errorf("prefill tokenize: %w", err)
	}
	if cfg.BOSTokenID != 0 {
		tokens = append([]int32{cfg.BOSTokenID}, tokens...)
	}
	return tokens, nil
}

// DecodeBatch observes cancellation before dispatch. Once a Metal command buffer
// is committed, cancellation suppresses its result and releases its rows after
// completion; it does not preempt GPU execution.
func (a *scheduledEngineAdapter) DecodeBatch(ctx context.Context, states []*continuousbatch.State) []continuousbatch.Result {
	results := make([]continuousbatch.Result, len(states))
	if err := ctx.Err(); err != nil {
		return batchError(results, err)
	}
	tokens := make([]int32, len(states))
	positions := make([]int, len(states))
	rows := make([]int, len(states))
	samplers := make([]SamplerConfig, len(states))
	a.rowsMu.Lock()
	defer a.rowsMu.Unlock()
	for i, state := range states {
		sequence, ok := state.KV.(scheduledSequenceState)
		if !ok {
			return batchError(results, errors.New("decode: invalid cache state"))
		}
		if a.leases[sequence.row] != sequence.lease {
			return batchError(results, errors.New("decode: stale cache row lease"))
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

func (a *scheduledEngineAdapter) Dispatches() uint64 {
	a.metricsMu.Lock()
	defer a.metricsMu.Unlock()
	total := a.prefillDispatches
	for _, batch := range a.batches {
		total += uint64(batch.Dispatches)
	}
	return total
}

var _ continuousbatch.Backend = (*scheduledEngineAdapter)(nil)

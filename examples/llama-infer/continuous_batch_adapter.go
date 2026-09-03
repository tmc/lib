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
	prefillBatches                      []fullBatchObservation
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
	results := a.PrefillBatch([]continuousbatch.PrefillRequest{{Context: ctx, State: state}})
	if len(results) != 1 {
		return continuousbatch.Token{}, errors.New("prefill: invalid batch result")
	}
	return results[0].Token, results[0].Err
}

type scheduledPrefill struct {
	input     continuousbatch.PrefillRequest
	index     int
	tokens    []int32
	row       int
	lease     uint64
	start     int
	next      int32
	committed bool
}

// PrefillBatch executes all valid uncanceled prompt suffixes in one segmented
// layer chain. Row leases become visible to the scheduler only after the GPU
// wave and model tail have completed successfully.
func (a *scheduledEngineAdapter) PrefillBatch(inputs []continuousbatch.PrefillRequest) []continuousbatch.Result {
	results := make([]continuousbatch.Result, len(inputs))
	if len(inputs) == 0 {
		return results
	}
	a.prefillMu.Lock()
	defer a.prefillMu.Unlock()
	a.gpuMu.Lock()
	defer a.gpuMu.Unlock()
	if a.engine == nil || a.engine.tokenizer == nil {
		return batchError(results, errors.New("prefill: nil engine or tokenizer"))
	}

	work := make([]*scheduledPrefill, 0, len(inputs))
	for i, input := range inputs {
		if input.State == nil {
			results[i].Err = errors.New("prefill: nil state")
			continue
		}
		if err := input.Context.Err(); err != nil {
			results[i].Err = err
			continue
		}
		tokens, err := prefillTokens(a.cfg, a.engine.tokenizer, input.State)
		if err != nil {
			results[i].Err = err
			continue
		}
		if len(tokens) == 0 {
			tokens = []int32{1}
		}
		if len(tokens) >= a.cfg.MaxSeqLen {
			results[i].Err = errors.New("prefill: prompt exceeds cache capacity")
			continue
		}
		valid := true
		for _, token := range tokens {
			if token < 0 || int(token) >= a.cfg.VocabSize {
				results[i].Err = errors.New("prefill: invalid prompt token")
				valid = false
				break
			}
		}
		if !valid {
			continue
		}
		if _, ok := input.State.Sampler.(SamplerConfig); input.State.Sampler != nil && !ok {
			results[i].Err = errors.New("prefill: unsupported sampler state")
			continue
		}
		work = append(work, &scheduledPrefill{input: input, index: i, tokens: tokens})
	}
	if len(work) == 0 {
		return results
	}

	a.rowsMu.Lock()
	if len(a.freeRows) < len(work) {
		a.rowsMu.Unlock()
		return prefillBatchError(results, work, errors.New("prefill: no cache row available"))
	}
	for _, item := range work {
		item.row = a.freeRows[len(a.freeRows)-1]
		a.freeRows = a.freeRows[:len(a.freeRows)-1]
		a.nextLease++
		item.lease = a.nextLease
		a.leases[item.row] = item.lease
	}
	a.rowsMu.Unlock()
	defer func() {
		for _, item := range work {
			if !item.committed {
				a.releasePrefillRow(item)
			}
		}
	}()

	for _, item := range work {
		item.start, item.next = a.engine.restoreScheduledPrefixRow(item.tokens, a.cache, item.row)
	}
	pending := append([]*scheduledPrefill(nil), work...)
	for len(pending) > 0 {
		segments := make([]PrefillSegment, 0, len(pending))
		wave := pending[:0]
		for _, item := range pending {
			if err := item.input.Context.Err(); err != nil {
				results[item.index].Err = err
				a.releasePrefillRow(item)
				continue
			}
			if item.start == len(item.tokens) {
				continue
			}
			requestID := fmt.Sprintf("%s#%d", item.input.State.ID, item.index)
			segments = append(segments, PrefillSegment{RequestID: requestID, Row: item.row, Tokens: item.tokens[item.start:], AbsoluteStart: item.start, RestoredPrefix: item.start})
			wave = append(wave, item)
		}
		pending = wave
		if len(pending) == 0 {
			break
		}
		var packer PrefillPacker
		packed, err := packer.Pack(context.Background(), segments, a.cfg.MaxSeqLen*len(segments), a.cache.rowCapacity, a.cfg.VocabSize, a.cfg.MaxSeqLen)
		if err != nil {
			return prefillBatchError(results, pending, err)
		}
		waveCtx, cancel := context.WithCancel(context.Background())
		stops := make([]func() bool, len(pending))
		for i, item := range pending {
			stops[i] = context.AfterFunc(item.input.Context, cancel)
		}
		chain, err := RunSegmentedLayerChain(waveCtx, a.engine, a.cfg, a.cache, packed)
		for _, stop := range stops {
			stop()
		}
		cancel()
		if errors.Is(err, context.Canceled) {
			next := pending[:0]
			for _, item := range pending {
				if requestErr := item.input.Context.Err(); requestErr != nil {
					results[item.index].Err = requestErr
					a.releasePrefillRow(item)
					continue
				}
				next = append(next, item)
			}
			if len(next) == len(pending) {
				return prefillBatchError(results, pending, err)
			}
			pending = next
			continue
		}
		if err != nil {
			return prefillBatchError(results, pending, fmt.Errorf("prefill GPU state: %w", err))
		}
		samplers := make([]SamplerConfig, len(pending))
		for i, item := range pending {
			samplers[i], _ = item.input.State.Sampler.(SamplerConfig)
		}
		tail, err := RunSegmentedModelTailFromChain(a.engine, a.cfg, a.tail, chain, samplers)
		if err != nil {
			return prefillBatchError(results, pending, fmt.Errorf("prefill model tail: %w", err))
		}
		for i, item := range pending {
			item.next = tail.Tokens[i]
		}
		a.metricsMu.Lock()
		a.prefillDispatches += uint64(chain.Dispatches + tail.Dispatches)
		a.prefillBatches = append(a.prefillBatches, fullBatchObservation{Batch: len(pending), Dispatches: chain.Dispatches + tail.Dispatches})
		a.metricsMu.Unlock()
		break
	}

	for _, item := range work {
		if results[item.index].Err != nil {
			continue
		}
		text, err := a.engine.tokenizer.Decode([]int32{item.next})
		if err != nil {
			results[item.index].Err = fmt.Errorf("prefill decode token: %w", err)
			continue
		}
		item.input.State.KV = scheduledSequenceState{row: item.row, lease: item.lease}
		item.input.State.Position = len(item.tokens)
		item.committed = true
		a.engine.storeScheduledPrefixRow(item.tokens, a.cache, item.row, item.next)
		results[item.index].Token = continuousbatch.Token{ID: item.next, Text: text}
	}
	return results
}

func (a *scheduledEngineAdapter) releasePrefillRow(item *scheduledPrefill) {
	a.rowsMu.Lock()
	if a.leases[item.row] == item.lease {
		delete(a.leases, item.row)
		a.freeRows = append(a.freeRows, item.row)
	}
	a.rowsMu.Unlock()
}

func prefillBatchError(results []continuousbatch.Result, work []*scheduledPrefill, err error) []continuousbatch.Result {
	// A committed command buffer is one fault domain. Without encoder-level
	// evidence, a GPU execution failure cannot be attributed to one request.
	for _, item := range work {
		if results[item.index].Err == nil {
			results[item.index].Err = err
		}
	}
	return results
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
	// Decode rows share command buffers, so execution errors fail the wave.
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

func (a *scheduledEngineAdapter) PrefillObservations() []fullBatchObservation {
	a.metricsMu.Lock()
	defer a.metricsMu.Unlock()
	return append([]fullBatchObservation(nil), a.prefillBatches...)
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
var _ continuousbatch.BatchPrefiller = (*scheduledEngineAdapter)(nil)

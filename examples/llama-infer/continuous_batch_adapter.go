package main

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"sync"

	"github.com/tmc/lib/examples/internal/continuousbatch"
)

type scheduledEngineAdapter struct {
	engine            *ModelEngine
	cfg               Config
	cache             *BatchedAttentionCache
	tail              *BatchedModelTail
	capacity          int
	prefillMu         sync.Mutex
	gpuMu             sync.Mutex // Acquired before rowsMu when an operation needs both.
	rowsMu            sync.Mutex
	metricsMu         sync.Mutex
	freeRows          []int
	leases            map[int]uint64
	nextLease         uint64
	batches           []fullBatchObservation
	prefillBatches    []fullBatchObservation
	prefillDispatches uint64
	prefixes          *productionPrefixRuntime
	prefillObserver   func(*continuousbatch.State, *productionPrefixLease, []float32)
	observedOutputs   *observedOutputCollector
	physical          *physicalWaveLedger
}

type scheduledSequenceState struct {
	row    int
	lease  uint64
	prefix *productionPrefixLease
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
	prefixes, err := newProductionPrefixRuntime(engine, cfg, capacity)
	if err != nil {
		cache.Free()
		tail.Free()
		return nil, err
	}
	a := &scheduledEngineAdapter{engine: engine, cfg: cfg, cache: cache, tail: tail, capacity: capacity, prefixes: prefixes, freeRows: make([]int, capacity), leases: make(map[int]uint64)}
	for i := range a.freeRows {
		a.freeRows[i] = capacity - i - 1
	}
	return a, nil
}

func (a *scheduledEngineAdapter) Close() {
	a.cache.Free()
	a.tail.Free()
	_ = a.prefixes.Close()
}

// resetPrefixes replaces the serving prefix runtime between benchmark
// samples. Callers must have drained every request submitted to the adapter.
func (a *scheduledEngineAdapter) resetPrefixes() error {
	a.prefillMu.Lock()
	defer a.prefillMu.Unlock()
	a.gpuMu.Lock()
	defer a.gpuMu.Unlock()

	next, err := newProductionPrefixRuntime(a.engine, a.cfg, a.capacity)
	if err != nil {
		return fmt.Errorf("reset prefixes: create runtime: %w", err)
	}
	if err := a.prefixes.Close(); err != nil {
		_ = next.Close()
		return fmt.Errorf("reset prefixes: close runtime: %w", err)
	}
	a.prefixes = next
	return nil
}

func (a *scheduledEngineAdapter) Release(state *continuousbatch.State) {
	sequence, ok := state.KV.(scheduledSequenceState)
	if !ok {
		return
	}
	if a.observedOutputs != nil {
		a.observedOutputs.observeKV(state, a.cache, a.cfg)
		read := uint64(2 * a.cfg.NumLayers * state.Position * a.cfg.NumKVHeads * a.cfg.HeadDim * 4)
		if err := a.recordPhysical("observed-output-kv", []string{fmt.Sprintf("%s@%d/%d", state.ID, sequence.row, sequence.lease)}, 0, map[string]uint64{"observed_kv_host": read}, nil); err != nil {
			a.observedOutputs.fail(err)
		}
	}
	a.rowsMu.Lock()
	if a.leases[sequence.row] == sequence.lease {
		delete(a.leases, sequence.row)
		a.freeRows = append(a.freeRows, sequence.row)
		if sequence.prefix != nil {
			sequence.prefix.Release()
		}
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
	prefix    *productionPrefixLease
	logits    []float32
}

// PrefillBatch executes valid uncanceled prompts in shared GPU waves. Cache
// misses use a segmented layer chain; hits read leased prefix blocks directly.
// Row leases become visible only after GPU work and the model tail complete.
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
				if item.prefix != nil {
					item.prefix.Release()
					item.prefix = nil
				}
				a.releasePrefillRow(item)
			}
		}
	}()

	for _, item := range work {
		lease, ok, err := a.prefixes.Acquire(item.tokens)
		if err != nil {
			results[item.index].Err = fmt.Errorf("prefill: acquire Metal prefix: %w", err)
			continue
		}
		if ok {
			item.prefix = lease
			item.start = lease.Tokens()
			if item.start == len(item.tokens) {
				item.next = lease.Current()
			}
		}
	}
	leased := make([]*scheduledPrefill, 0, len(work))
	pending := make([]*scheduledPrefill, 0, len(work))
	for _, item := range work {
		if item.prefix != nil {
			leased = append(leased, item)
		} else {
			pending = append(pending, item)
		}
	}
	if len(leased) != 0 {
		a.prefillWithLeasedBlocks(results, leased)
	}
	for len(pending) > 0 {
		segments := make([]PrefillSegment, 0, len(pending))
		wave := pending[:0]
		for _, item := range pending {
			if results[item.index].Err != nil {
				a.releasePrefillRow(item)
				continue
			}
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
		packed, err := packer.Pack(context.Background(), segments, a.cfg.MaxSeqLen*len(segments), a.capacity, a.cfg.VocabSize, a.cfg.MaxSeqLen)
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
		members := prefillRequestIDs(pending)
		if err := a.recordPhysical("prefill-chain", members, uint64(chain.Dispatches), nil, nil); err != nil {
			return prefillBatchError(results, pending, err)
		}
		samplers := make([]SamplerConfig, len(pending))
		for i, item := range pending {
			samplers[i], _ = item.input.State.Sampler.(SamplerConfig)
		}
		tail, err := RunSegmentedModelTailFromChain(a.engine, a.cfg, a.tail, chain, samplers)
		if err != nil {
			return prefillBatchError(results, pending, fmt.Errorf("prefill model tail: %w", err))
		}
		if err := a.recordPhysical("prefill-tail", members, uint64(tail.Dispatches), nil, nil); err != nil {
			return prefillBatchError(results, pending, err)
		}
		var diagnostic []float32
		if a.prefillObserver != nil {
			full, err := a.tail.Run(a.engine, a.cfg, chain.Hidden, samplers)
			if err != nil {
				return prefillBatchError(results, pending, fmt.Errorf("prefill diagnostic tail: %w", err))
			}
			if !slices.Equal(full.Tokens, tail.Tokens) {
				return prefillBatchError(results, pending, errors.New("prefill diagnostic tail token mismatch"))
			}
			diagnostic = full.Logits
		}
		for i, item := range pending {
			item.next = tail.Tokens[i]
			if diagnostic != nil {
				item.logits = append(item.logits[:0], diagnostic[i*a.cfg.VocabSize:(i+1)*a.cfg.VocabSize]...)
			}
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
		if item.prefix == nil {
			beforeTransfer := a.prefixes.Stats()
			if err := a.prefixes.Put(item.input.Context, item.tokens, item.next, a.cache, item.row); err != nil {
				results[item.index].Err = fmt.Errorf("prefill: publish Metal prefix: %w", err)
				continue
			}
			if err := a.recordPrefixTransfer(prefillRequestID(item), beforeTransfer); err != nil {
				results[item.index].Err = err
				continue
			}
			lease, err := a.prefixes.AcquirePublished(item.tokens)
			if err != nil || lease == nil {
				results[item.index].Err = fmt.Errorf("prefill: acquire published Metal prefix: %w", err)
				continue
			}
			item.prefix = lease
		} else if item.prefix.Tokens() < len(item.tokens) {
			beforeTransfer := a.prefixes.Stats()
			if err := a.prefixes.Extend(item.input.Context, item.prefix, item.tokens, item.next, a.cache, item.row); err != nil {
				results[item.index].Err = fmt.Errorf("prefill: extend Metal prefix: %w", err)
				continue
			}
			if err := a.recordPrefixTransfer(prefillRequestID(item), beforeTransfer); err != nil {
				results[item.index].Err = err
				continue
			}
			lease, err := a.prefixes.AcquirePublished(item.tokens)
			if err != nil || lease == nil {
				results[item.index].Err = fmt.Errorf("prefill: acquire extended Metal prefix: %w", err)
				continue
			}
			item.prefix.Release()
			item.prefix = lease
		}
		item.input.State.KV = scheduledSequenceState{row: item.row, lease: item.lease, prefix: item.prefix}
		item.input.State.Position = len(item.tokens)
		item.committed = true
		results[item.index].Token = continuousbatch.Token{ID: item.next, Text: text}
		if a.prefillObserver != nil {
			a.prefillObserver(item.input.State, item.prefix, item.logits)
		}
	}
	return results
}

func (a *scheduledEngineAdapter) prefillWithLeasedBlocks(results []continuousbatch.Result, work []*scheduledPrefill) {
	for {
		active := make([]*scheduledPrefill, 0, len(work))
		for _, item := range work {
			if results[item.index].Err != nil || item.start >= len(item.tokens) {
				continue
			}
			if err := item.input.Context.Err(); err != nil {
				results[item.index].Err = err
				continue
			}
			active = append(active, item)
		}
		if len(active) == 0 {
			return
		}
		tokens := make([]int32, len(active))
		positions := make([]int, len(active))
		rows := make([]int, len(active))
		leases := make([]*productionPrefixLease, len(active))
		samplers := make([]SamplerConfig, len(active))
		waveCtx, cancel := context.WithCancel(context.Background())
		stops := make([]func() bool, len(active))
		for i, item := range active {
			tokens[i], positions[i], rows[i], leases[i] = item.tokens[item.start], item.start, item.row, item.prefix
			samplers[i], _ = item.input.State.Sampler.(SamplerConfig)
			stops[i] = context.AfterFunc(item.input.Context, cancel)
		}
		chain, err := a.prefixes.Run(waveCtx, a.engine, a.cfg, a.cache, leases, tokens, positions, rows)
		for _, stop := range stops {
			stop()
		}
		cancel()
		if errors.Is(err, context.Canceled) {
			canceled := 0
			for _, item := range active {
				if requestErr := item.input.Context.Err(); requestErr != nil {
					results[item.index].Err = requestErr
					canceled++
				}
			}
			if canceled != 0 {
				continue
			}
		}
		if err != nil {
			for _, item := range active {
				results[item.index].Err = fmt.Errorf("prefill leased Metal state: %w", err)
			}
			return
		}
		members := prefillRequestIDs(active)
		if err := a.recordPhysical("prefill-prefix-chain", members, uint64(chain.Dispatches), nil, nil); err != nil {
			for _, item := range active {
				results[item.index].Err = err
			}
			return
		}
		tail, err := RunBatchedModelTailFromChain(a.engine, a.cfg, a.tail, chain, samplers)
		if err != nil {
			for _, item := range active {
				results[item.index].Err = fmt.Errorf("prefill model tail: %w", err)
			}
			return
		}
		if err := a.recordPhysical("prefill-prefix-tail", members, uint64(tail.Dispatches), nil, nil); err != nil {
			for _, item := range active {
				results[item.index].Err = err
			}
			return
		}
		var diagnostic []float32
		if a.prefillObserver != nil {
			full, err := a.tail.Run(a.engine, a.cfg, chain.Hidden, samplers)
			if err != nil {
				for _, item := range active {
					results[item.index].Err = fmt.Errorf("prefill diagnostic tail: %w", err)
				}
				return
			}
			if !slices.Equal(full.Tokens, tail.Tokens) {
				for _, item := range active {
					results[item.index].Err = errors.New("prefill diagnostic tail token mismatch")
				}
				return
			}
			diagnostic = full.Logits
		}
		for i, item := range active {
			item.start++
			item.next = tail.Tokens[i]
			if diagnostic != nil {
				item.logits = append(item.logits[:0], diagnostic[i*a.cfg.VocabSize:(i+1)*a.cfg.VocabSize]...)
			}
		}
		a.metricsMu.Lock()
		a.prefillDispatches += uint64(chain.Dispatches + tail.Dispatches)
		a.prefillBatches = append(a.prefillBatches, fullBatchObservation{Batch: len(active), Dispatches: chain.Dispatches + tail.Dispatches})
		a.metricsMu.Unlock()
	}
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
	a.lockDecode()
	defer a.unlockDecode()
	tokens := make([]int32, len(states))
	positions := make([]int, len(states))
	rows := make([]int, len(states))
	samplers := make([]SamplerConfig, len(states))
	prefixes := make([]*productionPrefixLease, len(states))
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
		tokens[i], positions[i], rows[i], samplers[i], prefixes[i] = state.Current, state.Position, sequence.row, sampler, sequence.prefix
	}
	chain, tail, err := a.prefixes.Decode(ctx, a.engine, a.cfg, a.cache, prefixes, a.tail, samplers, tokens, positions, rows)
	if err != nil {
		return batchError(results, err)
	}
	members := stateRequestIDs(states)
	if err := a.recordPhysical("decode-chain-tail", members, uint64(chain.Dispatches+tail.Dispatches), nil, nil); err != nil {
		return batchError(results, err)
	}
	if a.observedOutputs != nil {
		full, err := a.tail.Run(a.engine, a.cfg, chain.Hidden, samplers)
		if err != nil {
			return batchError(results, fmt.Errorf("decode observed tail: %w", err))
		}
		if !slices.Equal(full.Tokens, tail.Tokens) {
			return batchError(results, errors.New("decode observed tail token mismatch"))
		}
		if err := a.observedOutputs.observeLogits(states, full.Logits, a.cfg.VocabSize); err != nil {
			return batchError(results, err)
		}
		read := map[string]uint64{"observed_logits_host": uint64(len(full.Logits) * 4)}
		if err := a.recordPhysical("decode-observed-tail", members, uint64(full.Dispatches), read, nil); err != nil {
			return batchError(results, err)
		}
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

func (a *scheduledEngineAdapter) lockDecode() {
	a.gpuMu.Lock()
	a.rowsMu.Lock()
}

func (a *scheduledEngineAdapter) unlockDecode() {
	a.rowsMu.Unlock()
	a.gpuMu.Unlock()
}

func batchError(results []continuousbatch.Result, err error) []continuousbatch.Result {
	// Decode rows share command buffers, so execution errors fail the wave.
	for i := range results {
		results[i].Err = err
	}
	return results
}

func prefillRequestIDs(work []*scheduledPrefill) []string {
	ids := make([]string, len(work))
	for i, item := range work {
		ids[i] = prefillRequestID(item)
	}
	return ids
}

func prefillRequestID(item *scheduledPrefill) string {
	return fmt.Sprintf("%s@%d/%d", item.input.State.ID, item.row, item.lease)
}

func stateRequestIDs(states []*continuousbatch.State) []string {
	ids := make([]string, len(states))
	for i, state := range states {
		sequence, _ := state.KV.(scheduledSequenceState)
		ids[i] = fmt.Sprintf("%s@%d/%d", state.ID, sequence.row, sequence.lease)
	}
	return ids
}

func (a *scheduledEngineAdapter) recordPrefixTransfer(request string, before productionPrefixStats) error {
	after := a.prefixes.Stats()
	if after.DeviceReadBytes < before.DeviceReadBytes || after.DeviceWriteBytes < before.DeviceWriteBytes {
		return errors.New("physical ledger: prefix transfer counter reset")
	}
	read := after.DeviceReadBytes - before.DeviceReadBytes
	write := after.DeviceWriteBytes - before.DeviceWriteBytes
	if read == 0 && write == 0 {
		return nil
	}
	return a.recordPhysical("prefix-publish", []string{request}, 0, map[string]uint64{"metal_row_cache": read}, map[string]uint64{"metal_prefix_cache": write})
}

func (a *scheduledEngineAdapter) enablePhysicalLedger() {
	a.physical = new(physicalWaveLedger)
}

func (a *scheduledEngineAdapter) recordPhysical(kind string, requests []string, dispatches uint64, read, write map[string]uint64) error {
	if a.physical == nil {
		return nil
	}
	return a.physical.Record(kind, requests, dispatches, read, write)
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

//go:build linux

package main

import (
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"sync"

	"github.com/tmc/lib/examples/internal/continuousbatch"
	"github.com/tmc/lib/examples/internal/inferbench"
)

type observedOutputCollector struct {
	mu      sync.Mutex
	records map[string]inferbench.ObservedOutput
	members map[string]cudaObservedMembership
	err     error
}

type cudaObservedMembership struct {
	cache      *SequenceKVCache
	generation uint64
	position   int
}

func newObservedOutputCollector() *observedOutputCollector {
	return &observedOutputCollector{records: make(map[string]inferbench.ObservedOutput), members: make(map[string]cudaObservedMembership)}
}

func (c *observedOutputCollector) observeLogits(states []*continuousbatch.State, logits []float32, vocab int) error {
	if len(logits) != len(states)*vocab {
		return errors.New("observed output: inconsistent logits shape")
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	for row, state := range states {
		if state == nil || state.ID == "" {
			return errors.New("observed output: invalid request")
		}
		record := c.records[state.ID]
		record.FullLogitsSHA256 = hashFloat32(logits[row*vocab : (row+1)*vocab])
		record.FullLogitsElements = vocab
		c.records[state.ID] = record
	}
	return nil
}

func (c *observedOutputCollector) observeCUDALogits(states []*continuousbatch.State, sequences []cudaSequenceState, logits []float32, vocab int) error {
	if len(states) != len(sequences) {
		return errors.New("observed output: inconsistent CUDA membership")
	}
	for _, sequence := range sequences {
		if sequence.cache == nil || sequence.generation == 0 {
			return errors.New("observed output: invalid CUDA membership")
		}
	}
	if err := c.observeLogits(states, logits, vocab); err != nil {
		return err
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	for row, state := range states {
		sequence := sequences[row]
		c.members[state.ID] = cudaObservedMembership{cache: sequence.cache, generation: sequence.generation, position: state.Position}
	}
	return nil
}

// observeKV is used only by the Metal scheduled adapter. CUDA calls
// observeCUDAKV with its per-sequence cache instead.
func (c *observedOutputCollector) observeKV(*continuousbatch.State, *BatchedAttentionCache, Config) {
	c.fail(errors.New("observed output: invalid CUDA cache type"))
}

func (c *observedOutputCollector) observeCUDAKV(state *continuousbatch.State, sequence cudaSequenceState, cfg Config) {
	if state == nil || state.ID == "" || sequence.cache == nil || sequence.cache.Cache == nil {
		c.fail(errors.New("observed output: invalid cache state"))
		return
	}
	c.mu.Lock()
	membership, ok := c.members[state.ID]
	c.mu.Unlock()
	if !ok || membership.cache != sequence.cache || membership.generation != sequence.generation || state.Position != membership.position+1 {
		c.fail(errors.New("observed output: stale or mismatched CUDA membership"))
		return
	}
	tokens := state.Position
	kvDim := cfg.NumKVHeads * cfg.HeadDim
	if tokens <= 0 || tokens > cfg.MaxSeqLen || cfg.NumLayers <= 0 || kvDim <= 0 {
		c.fail(errors.New("observed output: invalid populated KV shape"))
		return
	}
	n := tokens * kvDim
	layers := make([]inferbench.ObservedKVLayer, cfg.NumLayers)
	for layer := range cfg.NumLayers {
		device, err := sequence.cache.Cache.Layer(layer)
		if err != nil {
			c.fail(fmt.Errorf("observed output: layer %d: %w", layer, err))
			return
		}
		key := make([]float32, n)
		value := make([]float32, n)
		if err := device.K.Slice(0, n).CopyToHost(key); err != nil {
			c.fail(fmt.Errorf("observed output: read layer %d key: %w", layer, err))
			return
		}
		if err := device.V.Slice(0, n).CopyToHost(value); err != nil {
			c.fail(fmt.Errorf("observed output: read layer %d value: %w", layer, err))
			return
		}
		layers[layer] = inferbench.ObservedKVLayer{Layer: layer, KeySHA256: hashFloat32(key), ValueSHA256: hashFloat32(value), ElementsEach: n}
	}
	kvHash, err := inferbench.HashJSON(layers)
	if err != nil {
		c.fail(fmt.Errorf("observed output: hash KV layers: %w", err))
		return
	}
	c.mu.Lock()
	record := c.records[state.ID]
	record.PopulatedKVSHA256 = kvHash
	record.PopulatedKVElements = 2 * cfg.NumLayers * n
	record.PopulatedTokens = tokens
	record.Layers = layers
	c.records[state.ID] = record
	c.mu.Unlock()
}

func (c *observedOutputCollector) outputs(ids []string) ([]inferbench.ObservedOutput, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.err != nil {
		return nil, c.err
	}
	out := make([]inferbench.ObservedOutput, len(ids))
	for i, id := range ids {
		record, ok := c.records[id]
		if !ok || record.FullLogitsSHA256 == "" || record.PopulatedKVSHA256 == "" {
			return nil, fmt.Errorf("observed output: incomplete request %q", id)
		}
		record.Request = i
		out[i] = record
		delete(c.records, id)
		delete(c.members, id)
	}
	return out, nil
}

func (c *observedOutputCollector) fail(err error) {
	c.mu.Lock()
	if c.err == nil {
		c.err = err
	}
	c.mu.Unlock()
}

func hashFloat32(values []float32) string {
	h := sha256.New()
	var scratch [4]byte
	for _, value := range values {
		binary.LittleEndian.PutUint32(scratch[:], math.Float32bits(value))
		h.Write(scratch[:])
	}
	return fmt.Sprintf("%x", h.Sum(nil))
}

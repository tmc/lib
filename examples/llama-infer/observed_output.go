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
	err     error
}

func newObservedOutputCollector() *observedOutputCollector {
	return &observedOutputCollector{records: make(map[string]inferbench.ObservedOutput)}
}

func (c *observedOutputCollector) observeLogits(states []*continuousbatch.State, logits []float32, vocab int) error {
	if len(logits) != len(states)*vocab {
		return errors.New("observed output: inconsistent logits shape")
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	for i, state := range states {
		record := c.records[state.ID]
		record.FullLogitsSHA256 = hashFloat32(logits[i*vocab : (i+1)*vocab])
		record.FullLogitsElements = vocab
		c.records[state.ID] = record
	}
	return nil
}

func (c *observedOutputCollector) observeKV(state *continuousbatch.State, cache *BatchedAttentionCache, cfg Config) {
	sequence, ok := state.KV.(scheduledSequenceState)
	if !ok {
		c.fail(errors.New("observed output: invalid cache state"))
		return
	}
	tokens := state.Position
	if tokens <= 0 || tokens > cfg.MaxSeqLen || len(cache.K) != cfg.NumLayers || len(cache.V) != cfg.NumLayers {
		c.fail(errors.New("observed output: invalid populated KV shape"))
		return
	}
	k := make([][]float32, cfg.NumLayers)
	v := make([][]float32, cfg.NumLayers)
	n := tokens * cache.kvDim
	for layer := range cfg.NumLayers {
		k[layer] = make([]float32, n)
		v[layer] = make([]float32, n)
	}
	prefixTokens := 0
	if sequence.prefix != nil {
		prefixTokens = min(sequence.prefix.Tokens(), tokens)
		if err := sequence.prefix.Restore(k, v, prefixTokens); err != nil {
			c.fail(fmt.Errorf("observed output: restore prefix: %w", err))
			return
		}
	}
	cache.mu.Lock()
	if cache.freed {
		cache.mu.Unlock()
		c.fail(errors.New("observed output: cache is freed"))
		return
	}
	offset := sequence.row * cache.maxSeqLen * cache.kvDim
	lo := prefixTokens * cache.kvDim
	for layer := range cfg.NumLayers {
		copy(k[layer][lo:], cache.K[layer].HostSlice()[offset+lo:offset+n])
		copy(v[layer][lo:], cache.V[layer].HostSlice()[offset+lo:offset+n])
	}
	cache.mu.Unlock()

	layers := make([]inferbench.ObservedKVLayer, cfg.NumLayers)
	for layer := range cfg.NumLayers {
		layers[layer] = inferbench.ObservedKVLayer{Layer: layer, KeySHA256: hashFloat32(k[layer]), ValueSHA256: hashFloat32(v[layer]), ElementsEach: n}
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

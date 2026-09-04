//go:build linux

package main

import (
	"errors"
	"sync"

	"github.com/tmc/lib/examples/internal/continuousbatch"
	"github.com/tmc/lib/examples/internal/inferbench"
)

// observedOutputCollector keeps the Linux build fail-closed until CUDA serving
// can export complete per-request logits and populated KV receipts.
type observedOutputCollector struct {
	mu  sync.Mutex
	err error
}

func newObservedOutputCollector() *observedOutputCollector {
	return &observedOutputCollector{err: errors.New("observed output: CUDA full-logit and KV receipts are unavailable")}
}

func (c *observedOutputCollector) observeLogits([]*continuousbatch.State, []float32, int) error {
	c.fail(errors.New("observed output: CUDA full-logit receipts are unavailable"))
	return c.err
}

func (c *observedOutputCollector) observeKV(*continuousbatch.State, *BatchedAttentionCache, Config) {
	c.fail(errors.New("observed output: CUDA populated KV receipts are unavailable"))
}

func (c *observedOutputCollector) outputs([]string) ([]inferbench.ObservedOutput, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return nil, c.err
}

func (c *observedOutputCollector) fail(err error) {
	c.mu.Lock()
	if c.err == nil {
		c.err = err
	}
	c.mu.Unlock()
}

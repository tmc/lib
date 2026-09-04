//go:build darwin

package main

import (
	"context"

	"github.com/tmc/lib/examples/internal/metalprefix"
)

type runLeasedPrefixFunc func(context.Context, *ModelEngine, Config, *BatchedAttentionCache, *metalBlockWorkspace, []*metalprefix.Lease, []int32, []int, []int) (BatchedLayerChainResult, error)
type runLeasedDecodeFunc func(context.Context, *ModelEngine, Config, *BatchedAttentionCache, *metalBlockWorkspace, []*metalprefix.Lease, *BatchedModelTail, []SamplerConfig, []int32, []int, []int) (BatchedLayerChainResult, BatchedModelTailResult, error)

type productionPrefixLease struct {
	lease *metalprefix.Lease
}

func (l *productionPrefixLease) Tokens() int    { return l.lease.Tokens() }
func (l *productionPrefixLease) Current() int32 { return l.lease.Current() }
func (l *productionPrefixLease) Release()       { l.lease.Release() }
func (l *productionPrefixLease) Restore(k, v [][]float32, tokens int) error {
	return l.lease.Restore(k, v, tokens)
}

type productionPrefixRuntime struct {
	cache     *metalprefix.Cache
	workspace *metalBlockWorkspace
	runLeased runLeasedPrefixFunc
	decode    runLeasedDecodeFunc
}

type productionPrefixStats struct {
	Entries, Blocks                   int
	Bytes                             int64
	Hits, Misses                      int64
	SavedTokens                       int64
	WorkspaceBytes                    int64
	DescriptorAddress, RowAddress     uint64
	DeviceReadBytes, DeviceWriteBytes uint64
}

func (*productionPrefixRuntime) Available() bool { return true }

func newProductionPrefixRuntime(engine *ModelEngine, cfg Config, capacity int) (*productionPrefixRuntime, error) {
	cache, err := metalprefix.New(metalprefix.Config{Device: engine.dev, MaxBytes: 64 << 20, MaxEntries: 8, BlockTokens: 16, Layers: cfg.NumLayers, KVHeads: cfg.NumKVHeads, HeadDim: cfg.HeadDim, Tracker: engine.memory})
	if err != nil {
		return nil, err
	}
	workspace, err := newMetalBlockWorkspace(engine, cfg, capacity)
	if err != nil {
		_ = cache.Close()
		return nil, err
	}
	return &productionPrefixRuntime{cache: cache, workspace: workspace, runLeased: RunLeasedBatchedLayerChain, decode: RunLeasedBatchedDecode}, nil
}

func (p *productionPrefixRuntime) Decode(ctx context.Context, engine *ModelEngine, cfg Config, cache *BatchedAttentionCache, leases []*productionPrefixLease, tail *BatchedModelTail, samplers []SamplerConfig, tokens []int32, positions, rows []int) (BatchedLayerChainResult, BatchedModelTailResult, error) {
	raw := make([]*metalprefix.Lease, len(leases))
	for i, lease := range leases {
		if lease != nil {
			raw[i] = lease.lease
		}
	}
	return p.decode(ctx, engine, cfg, cache, p.workspace, raw, tail, samplers, tokens, positions, rows)
}

func (p *productionPrefixRuntime) Acquire(tokens []int32) (*productionPrefixLease, bool, error) {
	lease, ok, err := p.cache.Acquire(tokens)
	if err != nil || !ok {
		return nil, ok, err
	}
	return &productionPrefixLease{lease: lease}, true, nil
}

func (p *productionPrefixRuntime) AcquirePublished(tokens []int32) (*productionPrefixLease, error) {
	lease, err := p.cache.AcquireExact(tokens)
	if err != nil {
		return nil, err
	}
	return &productionPrefixLease{lease: lease}, nil
}

func (p *productionPrefixRuntime) Put(ctx context.Context, tokens []int32, current int32, cache *BatchedAttentionCache, row int) error {
	offset := row * cache.maxSeqLen * cache.kvDim
	return p.cache.PutBuffers(ctx, tokens, current, cache.K, cache.V, offset)
}

func (p *productionPrefixRuntime) Extend(ctx context.Context, base *productionPrefixLease, tokens []int32, current int32, cache *BatchedAttentionCache, row int) error {
	if base == nil {
		return p.Put(ctx, tokens, current, cache, row)
	}
	offset := row * cache.maxSeqLen * cache.kvDim
	return p.cache.ExtendBuffers(ctx, base.lease, tokens, current, cache.K, cache.V, offset)
}

func (p *productionPrefixRuntime) Run(ctx context.Context, engine *ModelEngine, cfg Config, cache *BatchedAttentionCache, leases []*productionPrefixLease, tokens []int32, positions, rows []int) (BatchedLayerChainResult, error) {
	raw := make([]*metalprefix.Lease, len(leases))
	for i, lease := range leases {
		if lease != nil {
			raw[i] = lease.lease
		}
	}
	return p.runLeased(ctx, engine, cfg, cache, p.workspace, raw, tokens, positions, rows)
}

func (p *productionPrefixRuntime) Close() error {
	if err := p.cache.Close(); err != nil {
		return err
	}
	p.workspace.Free()
	return nil
}

func (p *productionPrefixRuntime) Stats() productionPrefixStats {
	s := p.cache.Stats()
	w := p.workspace.stats()
	return productionPrefixStats{Entries: s.Entries, Blocks: s.Blocks, Bytes: s.Bytes, Hits: s.Hits, Misses: s.Misses, SavedTokens: s.SavedTokens, WorkspaceBytes: w.Bytes, DescriptorAddress: w.DescriptorAddress, RowAddress: w.RowAddress, DeviceReadBytes: s.DeviceReadBytes, DeviceWriteBytes: s.DeviceWriteBytes}
}

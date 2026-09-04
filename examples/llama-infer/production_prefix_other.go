//go:build !darwin

package main

import "context"

type productionPrefixLease struct{}

func (*productionPrefixLease) Tokens() int    { return 0 }
func (*productionPrefixLease) Current() int32 { return 0 }
func (*productionPrefixLease) Release()       {}

type productionPrefixRuntime struct{}

type productionPrefixStats struct {
	Entries, Blocks                   int
	Bytes                             int64
	Hits, Misses                      int64
	SavedTokens                       int64
	WorkspaceBytes                    int64
	DescriptorAddress, RowAddress     uint64
	DeviceReadBytes, DeviceWriteBytes uint64
}

func newProductionPrefixRuntime(*ModelEngine, Config, int) (*productionPrefixRuntime, error) {
	return &productionPrefixRuntime{}, nil
}

func (*productionPrefixRuntime) Acquire([]int32) (*productionPrefixLease, bool, error) {
	return nil, false, nil
}

func (*productionPrefixRuntime) AcquirePublished([]int32) (*productionPrefixLease, error) {
	return nil, nil
}

func (*productionPrefixRuntime) Put(context.Context, []int32, int32, *BatchedAttentionCache, int) error {
	return nil
}

func (*productionPrefixRuntime) Extend(context.Context, *productionPrefixLease, []int32, int32, *BatchedAttentionCache, int) error {
	return nil
}

func (*productionPrefixRuntime) Run(_ context.Context, engine *ModelEngine, cfg Config, cache *BatchedAttentionCache, _ []*productionPrefixLease, tokens []int32, positions, rows []int) (BatchedLayerChainResult, error) {
	return RunBatchedLayerChain(engine, cfg, cache, tokens, positions, rows)
}

func (p *productionPrefixRuntime) Decode(ctx context.Context, engine *ModelEngine, cfg Config, cache *BatchedAttentionCache, leases []*productionPrefixLease, tail *BatchedModelTail, samplers []SamplerConfig, tokens []int32, positions, rows []int) (BatchedLayerChainResult, BatchedModelTailResult, error) {
	chain, err := p.Run(ctx, engine, cfg, cache, leases, tokens, positions, rows)
	if err != nil {
		return BatchedLayerChainResult{}, BatchedModelTailResult{}, err
	}
	result, err := RunBatchedModelTailFromChain(engine, cfg, tail, chain, samplers)
	return chain, result, err
}

func (*productionPrefixRuntime) Close() error                 { return nil }
func (*productionPrefixRuntime) Stats() productionPrefixStats { return productionPrefixStats{} }

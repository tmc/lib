package main

import (
	"fmt"
	"math"

	"github.com/tmc/lib/examples/ds4-min/dsv4shape"
)

type shapeCacheBufferPlan struct {
	layer    int
	name     string
	elements int
	initial  uint32
}

type shapeCachePlan struct {
	layers  int
	bytes   uint64
	buffers []shapeCacheBufferPlan
}

// This describes persistent cache storage only. It excludes weights, graph
// resources and forward scratch and does not admit a full-model backend.
func planShapeCache(c dsv4shape.Config, context uint64, available, reserve uint64) (shapeCachePlan, error) {
	if context == 0 || context > math.MaxUint32 {
		return shapeCachePlan{}, fmt.Errorf("cuda shape cache: context %d outside upstream uint32 ABI", context)
	}
	state, err := c.CacheState(context)
	if err != nil {
		return shapeCachePlan{}, err
	}
	if reserve > available || state.CacheBytes > available-reserve {
		return shapeCachePlan{}, fmt.Errorf("cuda shape cache: requires %d bytes plus %d reserve, available %d", state.CacheBytes, reserve, available)
	}
	p := shapeCachePlan{layers: len(state.Layers), bytes: state.CacheBytes}
	for i, l := range state.Layers {
		for _, v := range []struct {
			name  string
			n     uint64
			score bool
		}{{"raw", l.RawKV, false}, {"attention-cache", l.CompressedKV, false}, {"attention-kv", l.CompressorKV, false}, {"attention-score", l.CompressorScore, true}, {"attention-pooled", l.Pooled, false}, {"indexer-cache", l.IndexerKV, false}, {"indexer-kv", l.IndexerCompressorKV, false}, {"indexer-score", l.IndexerCompressorScore, true}, {"indexer-pooled", l.IndexerPooled, false}} {
			if v.n == 0 {
				continue
			}
			if v.n > uint64(int(^uint(0)>>1))/4 {
				return shapeCachePlan{}, fmt.Errorf("cuda shape cache: layer %d %s exceeds host allocation extent", i, v.name)
			}
			bits := uint32(0)
			if v.score {
				bits = math.Float32bits(-1.0e30)
			}
			p.buffers = append(p.buffers, shapeCacheBufferPlan{layer: i, name: v.name, elements: int(v.n), initial: bits})
		}
	}
	return p, nil
}

//go:build linux

package main

import (
	_ "embed"
	"errors"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

//go:embed cuda_shape_indexer.cu
var cudaShapeIndexerSource string

// encodeShapeIndexer consumes already projected index queries and weights.
// The 64x128 query/score consumers are common to both pinned families; only
// selection differs (512 Flash, 1024 Pro). This route does not accept the
// Flash-only encodeIndexer shape predicate or fixed-k selector.
func encodeShapeIndexer(state *cudaShapeForward, selectEntry driver.Function, layer int, position uint32, rows int, queryScratch, mask cudamem.Slice[float32], invalid cudamem.Slice[uint32]) error {
	if state == nil || state.closed || selectEntry == 0 || state.cache == nil || state.cache.r != state.r {
		return errors.New("cuda shape indexer: invalid owner or entry")
	}
	if err := state.r.checkOpen(); err != nil {
		return err
	}
	c := state.plan.config
	l, err := c.Layer(layer)
	if err != nil {
		return err
	}
	if !l.Indexer || uint64(position) >= state.plan.context || rows < 0 || mask.Len() < rows || invalid.Len() != 1 {
		return errors.New("cuda shape indexer: invalid request")
	}
	cache, err := shapeCacheLayerBuffer(state.cache, layer, "indexer-cache")
	if err != nil {
		return err
	}
	if rows > cache.Len()/c.IndexerHeadDim || rows > int((position+1)/uint32(l.CompressRatio)) {
		return errors.New("cuda shape indexer: invalid cache count")
	}
	q, weights, scores := state.scratch["index_q"], state.scratch["index_weights"], state.scratch["index_scores"]
	if c.IndexerHeads != 64 || c.IndexerHeadDim != 128 || q.Len() != c.IndexerHeads*c.IndexerHeadDim || weights.Len() != c.IndexerHeads || queryScratch.Len() != q.Len() || scores.Len() < rows {
		return errors.New("cuda shape indexer: truncated operands")
	}
	var ranges []mxfp4DeviceRange
	for _, v := range []cudamem.Slice[float32]{cache, q, weights, scores, queryScratch, mask} {
		ranges = append(ranges, mxfp4DeviceRange{uint64(v.Ptr()), uint64(v.Len()) * 4})
	}
	ranges = append(ranges, mxfp4DeviceRange{uint64(invalid.Ptr()), 4})
	if err := checkMXFP4DeviceRanges(ranges...); err != nil {
		return err
	}
	if err := state.r.withContext(func() error {
		if err := state.r.loadMathKernels(); err != nil {
			return err
		}
		if rows > c.IndexerTopK {
			if err := cudalaunch.Kernel(state.r.indexerQueryFn).Grid(uint32(c.IndexerHeads), 1, 1).Block(uint32(c.IndexerHeadDim), 1, 1).On(state.r.stream()).Launch(q, queryScratch, position); err != nil {
				return err
			}
			if err := cudalaunch.Kernel(state.r.indexerScoreFn).Grid(uint32((rows+cudaIndexRowsPerBlock-1)/cudaIndexRowsPerBlock), 1, 1).Block(32*cudaIndexRowsPerBlock, 1, 1).On(state.r.stream()).Launch(queryScratch, weights, cache, scores, uint32(rows)); err != nil {
				return err
			}
		}
		return cudalaunch.Kernel(selectEntry).Grid(1, 1, 1).Block(32, 1, 1).On(state.r.stream()).Launch(scores, mask, invalid, uint32(rows), uint32(c.IndexerTopK))
	}); err != nil {
		return err
	}
	// Read the flag before exposing the mask to the attention consumer.
	if err := state.r.withContext(func() error { return syncStream(state.r.stream()) }); err != nil {
		return err
	}
	var flag [1]uint32
	if err := invalid.CopyToHost(flag[:]); err != nil {
		return err
	}
	if flag[0] != 0 {
		return errors.New("cuda shape indexer: nonfinite score or invalid selection")
	}
	return nil
}

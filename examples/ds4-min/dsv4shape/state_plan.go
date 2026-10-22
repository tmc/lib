package dsv4shape

import "fmt"

// LayerStatePlan gives persistent ring/compressor element counts. All counts
// describe float32 storage in current ds4-min CUDA and Metal state code; scratch,
// graphs and batch resources require separate backend plans.
type LayerStatePlan struct {
	Ratio                                                                 uint64
	RawKV, CompressedKV, CompressorKV, CompressorScore, Pooled            uint64
	IndexerKV, IndexerCompressorKV, IndexerCompressorScore, IndexerPooled uint64
}

// StatePlan describes the persistent float32 caches, not total device memory.
type StatePlan struct {
	Layers     []LayerStatePlan
	CacheBytes uint64
}

func mulBytes(a, b uint64) (uint64, error) {
	if a != 0 && b > ^uint64(0)/a {
		return 0, fmt.Errorf("shape: element count overflow")
	}
	return a * b, nil
}

// CacheState sizes persistent caches for the complete known layer schedule.
// The two spare compressed rows and ratio-4 double-width accumulators match
// current ds4-min newCUDACompressedState and Metal allocState layouts.
func (c Config) CacheState(context uint64) (*StatePlan, error) {
	if err := c.Validate(); err != nil {
		return nil, err
	}
	if context == 0 {
		return nil, fmt.Errorf("shape: context is zero")
	}
	p := &StatePlan{Layers: make([]LayerStatePlan, c.Layers)}
	for i := range p.Layers {
		l, _ := c.Layer(i)
		v := &p.Layers[i]
		v.Ratio = uint64(l.CompressRatio)
		v.RawKV = min(context, uint64(c.SlidingWindow)) * uint64(c.HeadDim)
		if v.Ratio != 0 {
			cap, err := addBytes(context/v.Ratio, 2)
			if err != nil {
				return nil, err
			}
			v.CompressedKV, err = mulBytes(cap, uint64(c.HeadDim))
			if err != nil {
				return nil, err
			}
			coff := uint64(1)
			if v.Ratio == 4 {
				coff = 2
			}
			v.CompressorKV = v.Ratio * coff * coff * uint64(c.HeadDim)
			v.CompressorScore = v.CompressorKV
			v.Pooled = uint64(c.HeadDim)
			if l.Indexer {
				v.IndexerKV, err = mulBytes(cap, uint64(c.IndexerHeadDim))
				if err != nil {
					return nil, err
				}
				v.IndexerCompressorKV = 4 * v.Ratio * uint64(c.IndexerHeadDim)
				v.IndexerCompressorScore = v.IndexerCompressorKV
				v.IndexerPooled = uint64(c.IndexerHeadDim)
			}
		}
		for _, n := range []uint64{v.RawKV, v.CompressedKV, v.CompressorKV, v.CompressorScore, v.Pooled, v.IndexerKV, v.IndexerCompressorKV, v.IndexerCompressorScore, v.IndexerPooled} {
			bytes, err := mulBytes(n, 4)
			if err != nil {
				return nil, err
			}
			p.CacheBytes, err = addBytes(p.CacheBytes, bytes)
			if err != nil {
				return nil, err
			}
		}
	}
	return p, nil
}

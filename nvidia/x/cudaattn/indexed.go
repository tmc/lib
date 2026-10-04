package cudaattn

import (
	"fmt"
	"math"
)

// IndexedSelection declares who owns a selected set. No head dimension is inferred.
type IndexedSelection int

const (
	// IndexedQueryHeads uses an independent set for every query head.
	IndexedQueryHeads IndexedSelection = iota + 1
	// IndexedKVHeads shares one explicitly supplied set across a KV head's query heads.
	IndexedKVHeads
)

// IndexedConfig describes ordinary FP32 attention. Q and output are [queries,
// query heads, dimension]. Physical K and V are [physical blocks, block size,
// KV heads, dimension]. IDs name logical blocks, not physical pages. Pages maps
// [batch, maximum logical blocks] to physical blocks. Each query names a sequence
// and its actual position; causal masking compares actual KV token positions.
// Query values and every loaded KV value must be finite, and scale must be
// finite and positive. Unvisited KV storage is outside the numerical domain.
// Empty or fully
// masked rows produce zero output and negative infinity LSE.
type IndexedConfig struct {
	Batch, Queries, QueryHeads, KVHeads, HeadDim           int
	BlockSize, MaxSequence, PhysicalBlocks, SelectedBlocks int
	Selection                                              IndexedSelection
	Causal                                                 bool
	Scale                                                  float32
}

// IndexedExtents contains exact element counts for caller allocations.
type IndexedExtents struct {
	Query, KV, Pages, Sets, IDs, Rows, Scores, MaxBlocks int
}

// Extents checks the finite baseline domain before allocations or native calls.
// It supports batch 1..2, queries 1..64, heads 1..32, dimension 1..128,
// block size 1..128, sequence capacity 1..4096, physical blocks 1..256, and
// selected blocks 1..32. FP32 is the only admitted storage/accumulation type.
func (c IndexedConfig) Extents() (IndexedExtents, error) {
	if c.Batch < 1 || c.Batch > 2 || c.Queries < 1 || c.Queries > 64 || c.QueryHeads < 1 || c.QueryHeads > 32 || c.KVHeads < 1 || c.KVHeads > c.QueryHeads || c.QueryHeads%c.KVHeads != 0 || c.HeadDim < 1 || c.HeadDim > 128 || c.BlockSize < 1 || c.BlockSize > 128 || c.MaxSequence < 1 || c.MaxSequence > 4096 || c.PhysicalBlocks < 1 || c.PhysicalBlocks > 256 || c.SelectedBlocks < 1 || c.SelectedBlocks > 32 || !(c.Scale > 0) || math.IsNaN(float64(c.Scale)) || math.IsInf(float64(c.Scale), 0) {
		return IndexedExtents{}, fmt.Errorf("cudaattn: unsupported indexed configuration")
	}
	h := c.QueryHeads
	if c.Selection == IndexedKVHeads {
		h = c.KVHeads
	} else if c.Selection != IndexedQueryHeads {
		return IndexedExtents{}, fmt.Errorf("cudaattn: ambiguous indexed selection heads")
	}
	b := (c.MaxSequence + c.BlockSize - 1) / c.BlockSize
	return IndexedExtents{Query: c.Queries * c.QueryHeads * c.HeadDim, KV: c.PhysicalBlocks * c.BlockSize * c.KVHeads * c.HeadDim, Pages: c.Batch * b, Sets: c.Queries * h, IDs: c.Queries * h * c.SelectedBlocks, Rows: c.Queries * c.QueryHeads, Scores: c.Queries * c.QueryHeads * b, MaxBlocks: b}, nil
}

// IndexedRequest is host-owned metadata with exact extents. Live IDs occupy the
// prefix specified by Counts; every remaining slot must be -1. Duplicate IDs
// and invalid physical page mappings are refused. Positions may be -1 to admit
// a fully masked causal query. A position otherwise lies within its sequence.
type IndexedRequest struct {
	Sequences, Positions, Lengths, Pages, IDs, Counts []int32
}

// ValidateIndexedRequest checks metadata without using a device. Execute repeats
// this admission on device, including device-produced IDs, before reading KV.
func ValidateIndexedRequest(c IndexedConfig, r IndexedRequest) error {
	e, err := c.Extents()
	if err != nil {
		return err
	}
	if len(r.Sequences) != c.Queries || len(r.Positions) != c.Queries || len(r.Lengths) != c.Batch || len(r.Pages) != e.Pages || len(r.IDs) != e.IDs || len(r.Counts) != e.Sets {
		return fmt.Errorf("cudaattn: indexed metadata extent mismatch")
	}
	for s, n := range r.Lengths {
		if n < 0 || int(n) > c.MaxSequence {
			return fmt.Errorf("cudaattn: indexed sequence %d length out of range", s)
		}
		for b := 0; b < e.MaxBlocks; b++ {
			p := r.Pages[s*e.MaxBlocks+b]
			if b < (int(n)+c.BlockSize-1)/c.BlockSize {
				if p < 0 || int(p) >= c.PhysicalBlocks {
					return fmt.Errorf("cudaattn: indexed page out of range")
				}
			} else if p != -1 {
				return fmt.Errorf("cudaattn: indexed unused page must be -1")
			}
		}
	}
	h := e.Sets / c.Queries
	for q, s := range r.Sequences {
		if s < 0 || int(s) >= c.Batch {
			return fmt.Errorf("cudaattn: indexed query sequence out of range")
		}
		n := r.Lengths[s]
		if r.Positions[q] < -1 || r.Positions[q] >= n && !(n == 0 && r.Positions[q] == -1) {
			return fmt.Errorf("cudaattn: indexed query position out of range")
		}
		for head := 0; head < h; head++ {
			row := q*h + head
			count := r.Counts[row]
			if count < 0 || int(count) > c.SelectedBlocks {
				return fmt.Errorf("cudaattn: indexed count out of range")
			}
			ids := r.IDs[row*c.SelectedBlocks : (row+1)*c.SelectedBlocks]
			for j, id := range ids {
				if j >= int(count) {
					if id != -1 {
						return fmt.Errorf("cudaattn: indexed padding must be -1")
					}
					continue
				}
				if id < 0 || int(id) >= (int(n)+c.BlockSize-1)/c.BlockSize {
					return fmt.Errorf("cudaattn: indexed block out of range")
				}
				for _, prev := range ids[:j] {
					if prev == id {
						return fmt.Errorf("cudaattn: duplicate indexed block")
					}
				}
			}
		}
	}
	return nil
}

// CheckIndexedStatus requires successful evidence from every expected row.
// Status 1 is metadata/ID damage, 2 is nonfinite input or arithmetic.
func CheckIndexedStatus(status []int32, rows int) error {
	if rows < 1 || len(status) != rows {
		return fmt.Errorf("cudaattn: missing indexed device status")
	}
	for row, s := range status {
		if s != 0 {
			return fmt.Errorf("cudaattn: indexed row %d device status %d", row, s)
		}
	}
	return nil
}

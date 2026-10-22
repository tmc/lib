package cudakv

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudaerrors"
)

// LayerKVRole identifies whether a decoder layer produces its own KV cache
// or consumes a shared source layer's KV cache.
type LayerKVRole int

const (
	// LayerKVProducer indicates a layer that computes and stores its own KV cache.
	LayerKVProducer LayerKVRole = iota + 1
	// LayerKVSharedConsumer indicates a layer that reuses the KV cache produced by an earlier SourceLayerID.
	LayerKVSharedConsumer
)

// SharedLayerSpec describes the KV cache sharing role of a single transformer layer.
type SharedLayerSpec struct {
	LayerID       int
	SourceLayerID int
	Role          LayerKVRole
	IsSWA         bool
	WindowSize    int
}

// SharedLayerTopology maps a multi-layer decoder stack to compressed physical KV cache pools.
type SharedLayerTopology struct {
	specs       []SharedLayerSpec
	producers   []int
	layerToSlot []int
}

// NewSharedLayerTopology validates and constructs a SharedLayerTopology from per-layer specs.
func NewSharedLayerTopology(specs []SharedLayerSpec) (*SharedLayerTopology, error) {
	if len(specs) == 0 {
		return nil, fmt.Errorf("cudakv: empty shared layer topology: %w", cudaerrors.ErrUnsupported)
	}
	n := len(specs)
	ordered := make([]SharedLayerSpec, n)
	seen := make([]bool, n)
	for _, s := range specs {
		if s.LayerID < 0 || s.LayerID >= n {
			return nil, fmt.Errorf("cudakv: layer id %d out of range [0, %d): %w", s.LayerID, n, cudaerrors.ErrUnsupported)
		}
		if seen[s.LayerID] {
			return nil, fmt.Errorf("cudakv: duplicate layer id %d: %w", s.LayerID, cudaerrors.ErrUnsupported)
		}
		seen[s.LayerID] = true
		ordered[s.LayerID] = s
	}

	var producers []int
	producerSlot := make(map[int]int, n)
	layerToSlot := make([]int, n)

	for i := 0; i < n; i++ {
		s := ordered[i]
		if s.IsSWA && s.WindowSize <= 0 {
			return nil, fmt.Errorf("cudakv: layer %d enables SWA with non-positive window size %d: %w", i, s.WindowSize, cudaerrors.ErrUnsupported)
		}
		switch s.Role {
		case LayerKVProducer:
			if s.SourceLayerID != i {
				return nil, fmt.Errorf("cudakv: producer layer %d must have SourceLayerID %d, got %d: %w", i, i, s.SourceLayerID, cudaerrors.ErrUnsupported)
			}
			slot := len(producers)
			producers = append(producers, i)
			producerSlot[i] = slot
			layerToSlot[i] = slot
		case LayerKVSharedConsumer:
			if s.SourceLayerID < 0 || s.SourceLayerID >= i {
				return nil, fmt.Errorf("cudakv: consumer layer %d must reference an earlier producer [0, %d), got %d: %w", i, i, s.SourceLayerID, cudaerrors.ErrUnsupported)
			}
			slot, ok := producerSlot[s.SourceLayerID]
			if !ok {
				return nil, fmt.Errorf("cudakv: consumer layer %d references non-producer layer %d: %w", i, s.SourceLayerID, cudaerrors.ErrUnsupported)
			}
			srcSpec := ordered[s.SourceLayerID]
			if s.IsSWA != srcSpec.IsSWA {
				return nil, fmt.Errorf("cudakv: consumer layer %d SWA mismatch with source layer %d: %w", i, s.SourceLayerID, cudaerrors.ErrUnsupported)
			}
			layerToSlot[i] = slot
		default:
			return nil, fmt.Errorf("cudakv: layer %d has invalid role %d: %w", i, int(s.Role), cudaerrors.ErrUnsupported)
		}
	}

	return &SharedLayerTopology{
		specs:       ordered,
		producers:   producers,
		layerToSlot: layerToSlot,
	}, nil
}

// NumLayers returns the total number of layers in the topology.
func (t *SharedLayerTopology) NumLayers() int {
	if t == nil {
		return 0
	}
	return len(t.specs)
}

// ProducerLayers returns the ordered slice of producer layer IDs.
func (t *SharedLayerTopology) ProducerLayers() []int {
	if t == nil {
		return nil
	}
	out := make([]int, len(t.producers))
	copy(out, t.producers)
	return out
}

// PhysicalSlotForLayer returns the compressed physical KV pool slot index [0..len(ProducerLayers)-1] for layerID.
func (t *SharedLayerTopology) PhysicalSlotForLayer(layerID int) (int, error) {
	if t == nil || layerID < 0 || layerID >= len(t.layerToSlot) {
		return 0, fmt.Errorf("cudakv: layer id %d out of range: %w", layerID, cudaerrors.ErrUnsupported)
	}
	return t.layerToSlot[layerID], nil
}

// SpecForLayer returns the SharedLayerSpec for layerID.
func (t *SharedLayerTopology) SpecForLayer(layerID int) (SharedLayerSpec, error) {
	if t == nil || layerID < 0 || layerID >= len(t.specs) {
		return SharedLayerSpec{}, fmt.Errorf("cudakv: layer id %d out of range: %w", layerID, cudaerrors.ErrUnsupported)
	}
	return t.specs[layerID], nil
}

// MemorySavingsRatio returns the fraction of KV cache layers saved by cross-layer sharing (1 - producers/total).
func (t *SharedLayerTopology) MemorySavingsRatio() float64 {
	if t == nil || len(t.specs) == 0 {
		return 0
	}
	return 1.0 - float64(len(t.producers))/float64(len(t.specs))
}

// YOCOTopology is an alias for [SharedLayerTopology].
type YOCOTopology = SharedLayerTopology

// IsConsumer reports whether layerID is a shared KV cache consumer layer.
func (t *SharedLayerTopology) IsConsumer(layerID int) bool {
	if t == nil || layerID < 0 || layerID >= len(t.specs) {
		return false
	}
	return t.specs[layerID].Role == LayerKVSharedConsumer
}

// SharedHiddenRequired reports whether layerID is a producer layer whose KV
// representation is consumed by at least one downstream consumer layer.
func (t *SharedLayerTopology) SharedHiddenRequired(layerID int) bool {
	if t == nil || layerID < 0 || layerID >= len(t.specs) {
		return false
	}
	if t.specs[layerID].Role != LayerKVProducer {
		return false
	}
	for _, s := range t.specs {
		if s.Role == LayerKVSharedConsumer && s.SourceLayerID == layerID {
			return true
		}
	}
	return false
}

// CompactPrefillRange returns the [start, end) token range required by layerID
// during prefill when SWA window compaction is active.
func (t *SharedLayerTopology) CompactPrefillRange(layerID, totalTokens, windowTokens int) (start, end int) {
	if totalTokens <= 0 {
		return 0, 0
	}
	end = totalTokens
	if t == nil || layerID < 0 || layerID >= len(t.specs) {
		return 0, end
	}
	spec := t.specs[layerID]
	if !spec.IsSWA {
		return 0, end
	}
	win := spec.WindowSize
	if windowTokens > 0 && (win <= 0 || windowTokens < win) {
		win = windowTokens
	}
	if win > 0 && totalTokens > win {
		start = totalTokens - win
	}
	return start, end
}

// YoCoBoundaryIndices holds precomputed query-to-KV index tables for cross-layer YoCo SWA attention.
type YoCoBoundaryIndices struct {
	BoundaryIndices []int32
	FullIndices     []int32
	NumTokens       int
	RecentCap       int
}

// BuildYoCoBoundary computes yoco_boundary_indices ([numTokens, recentCap]) and yoco_full_indices
// matching MLAState.prepare_aux, mapping each query token to the most recent min(seqLenQ, recentCap)
// KV positions for cross-layer YoCo SWA attention.
func BuildYoCoBoundary(cuSeqlensQ []int32, seqlensK []int32, recentCap int) (YoCoBoundaryIndices, error) {
	if recentCap == 0 {
		recentCap = 6
	}
	if recentCap < 0 {
		return YoCoBoundaryIndices{}, fmt.Errorf("cudakv: recentCap %d must be positive: %w", recentCap, cudaerrors.ErrUnsupported)
	}
	if len(cuSeqlensQ) < 2 {
		return YoCoBoundaryIndices{}, fmt.Errorf("cudakv: cuSeqlensQ length %d must be at least 2: %w", len(cuSeqlensQ), cudaerrors.ErrUnsupported)
	}
	if cuSeqlensQ[0] != 0 {
		return YoCoBoundaryIndices{}, fmt.Errorf("cudakv: cuSeqlensQ[0] must be 0, got %d: %w", cuSeqlensQ[0], cudaerrors.ErrUnsupported)
	}
	numSeqs := len(cuSeqlensQ) - 1
	if len(seqlensK) != numSeqs {
		return YoCoBoundaryIndices{}, fmt.Errorf("cudakv: seqlensK length %d must equal numSeqs %d: %w", len(seqlensK), numSeqs, cudaerrors.ErrUnsupported)
	}
	for s := 0; s < numSeqs; s++ {
		if cuSeqlensQ[s+1] < cuSeqlensQ[s] {
			return YoCoBoundaryIndices{}, fmt.Errorf("cudakv: cuSeqlensQ not monotonic at seq %d (%d < %d): %w", s, cuSeqlensQ[s+1], cuSeqlensQ[s], cudaerrors.ErrUnsupported)
		}
		seqLenQ := cuSeqlensQ[s+1] - cuSeqlensQ[s]
		if seqlensK[s] < seqLenQ {
			return YoCoBoundaryIndices{}, fmt.Errorf("cudakv: seqlensK[%d]=%d smaller than seqLenQ=%d: %w", s, seqlensK[s], seqLenQ, cudaerrors.ErrUnsupported)
		}
	}

	numTokens := int(cuSeqlensQ[numSeqs])
	boundary := make([]int32, numTokens*recentCap)
	for i := range boundary {
		boundary[i] = -1
	}
	full := make([]int32, numTokens)

	kvBase := int32(0)
	for s := 0; s < numSeqs; s++ {
		qStart := int(cuSeqlensQ[s])
		qEnd := int(cuSeqlensQ[s+1])
		seqLenQ := qEnd - qStart
		seqLenK := int(seqlensK[s])
		prefixLen := seqLenK - seqLenQ
		winLen := seqLenQ
		if winLen > recentCap {
			winLen = recentCap
		}
		winStartInSeqQ := seqLenQ - winLen

		for qIdx := 0; qIdx < seqLenQ; qIdx++ {
			tok := qStart + qIdx
			full[tok] = kvBase + int32(prefixLen+qIdx)
			row := boundary[tok*recentCap : (tok+1)*recentCap]
			for j := 0; j < winLen; j++ {
				qPos := winStartInSeqQ + j
				if qPos <= qIdx {
					row[j] = kvBase + int32(prefixLen+qPos)
				}
			}
		}
		kvBase += int32(seqLenK)
	}

	return YoCoBoundaryIndices{
		BoundaryIndices: boundary,
		FullIndices:     full,
		NumTokens:       numTokens,
		RecentCap:       recentCap,
	}, nil
}

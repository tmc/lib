package cudacollective

import "fmt"

// CPBucketSpec configures per-rank and all-gather bucket buffer sizing for
// Context Parallelism (cp_group).
type CPBucketSpec struct {
	MaxBatchTokens int
	CPSize         int
	HiddenDim      int
	ElemBytes      int
	PadMultiple    int
}

// ComputeCPBucketBytes computes the padded per-rank and all-gather buffer byte
// sizes for a Context Parallelism communication bucket.
func ComputeCPBucketBytes(spec CPBucketSpec) (perRankBytes int64, gatheredBytes int64, err error) {
	if spec.MaxBatchTokens <= 0 {
		return 0, 0, fmt.Errorf("cudacollective: MaxBatchTokens %d must be positive", spec.MaxBatchTokens)
	}
	if spec.CPSize <= 0 {
		return 0, 0, fmt.Errorf("cudacollective: CPSize %d must be positive", spec.CPSize)
	}
	if spec.HiddenDim <= 0 {
		return 0, 0, fmt.Errorf("cudacollective: HiddenDim %d must be positive", spec.HiddenDim)
	}
	if spec.ElemBytes <= 0 {
		return 0, 0, fmt.Errorf("cudacollective: ElemBytes %d must be positive", spec.ElemBytes)
	}
	if spec.PadMultiple < 0 {
		return 0, 0, fmt.Errorf("cudacollective: PadMultiple %d must be non-negative", spec.PadMultiple)
	}

	perRankTokens := (spec.MaxBatchTokens + spec.CPSize - 1) / spec.CPSize
	if spec.PadMultiple > 1 {
		rem := perRankTokens % spec.PadMultiple
		if rem != 0 {
			perRankTokens += spec.PadMultiple - rem
		}
	}

	perRankBytes = int64(perRankTokens) * int64(spec.HiddenDim) * int64(spec.ElemBytes)
	gatheredBytes = perRankBytes * int64(spec.CPSize)
	return perRankBytes, gatheredBytes, nil
}

// CPMirrorPlan holds the symmetric front and back token chunk indices assigned
// to one Context Parallelism rank.
type CPMirrorPlan struct {
	Rank          int
	CPSize        int
	LocalIndices  []int32
	MirrorIndices []int32
	ChunkSize     int
}

// PrepareCPMirrorIndices partitions totalSeqLen into 2*cpSize equal chunks and
// assigns rank r chunk r from the front (LocalIndices) and mirror chunk
// 2*cpSize - 1 - r from the back (MirrorIndices) so causal attention work is
// balanced across all CP ranks.
func PrepareCPMirrorIndices(totalSeqLen int, cpSize int, rank int) (CPMirrorPlan, error) {
	if totalSeqLen <= 0 {
		return CPMirrorPlan{}, fmt.Errorf("cudacollective: totalSeqLen %d must be positive", totalSeqLen)
	}
	if cpSize <= 0 {
		return CPMirrorPlan{}, fmt.Errorf("cudacollective: cpSize %d must be positive", cpSize)
	}
	if rank < 0 || rank >= cpSize {
		return CPMirrorPlan{}, fmt.Errorf("cudacollective: rank %d outside [0, %d)", rank, cpSize)
	}
	numChunks := 2 * cpSize
	if totalSeqLen%numChunks != 0 {
		return CPMirrorPlan{}, fmt.Errorf("cudacollective: totalSeqLen %d must be divisible by 2*cpSize (%d)", totalSeqLen, numChunks)
	}

	chunkSize := totalSeqLen / numChunks
	localStart := rank * chunkSize
	mirrorChunk := numChunks - 1 - rank
	mirrorStart := mirrorChunk * chunkSize

	localIdx := make([]int32, chunkSize)
	mirrorIdx := make([]int32, chunkSize)
	for i := 0; i < chunkSize; i++ {
		localIdx[i] = int32(localStart + i)
		mirrorIdx[i] = int32(mirrorStart + i)
	}

	return CPMirrorPlan{
		Rank:          rank,
		CPSize:        cpSize,
		LocalIndices:  localIdx,
		MirrorIndices: mirrorIdx,
		ChunkSize:     chunkSize,
	}, nil
}

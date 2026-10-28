package cudaattn

import (
	"fmt"
	"math"
	"sort"

	"github.com/tmc/lib/nvidia/x/cudaerrors"
)

// CandidateBlockConfig configures the DeepSelect two-stage block-level top-k indexer.
type CandidateBlockConfig struct {
	BlockSize        int
	BlockTopK        int
	TokenTopK        int
	RecentBlocks     int
	ChunkBlocks      int
	ComputeChunkSize int
}

// Validate checks that c satisfies the DeepSelect two-stage indexer constraints:
// ComputeChunkSize > 0 && ComputeChunkSize <= 8192, BlockSize > 0, BlockTopK > 0, TokenTopK > 0.
func (c CandidateBlockConfig) Validate() error {
	if c.ComputeChunkSize <= 0 || c.ComputeChunkSize > 8192 {
		return fmt.Errorf("cudaattn: ComputeChunkSize %d must be in [1, 8192]: %w", c.ComputeChunkSize, cudaerrors.ErrUnsupported)
	}
	if c.BlockSize <= 0 {
		return fmt.Errorf("cudaattn: BlockSize %d must be positive: %w", c.BlockSize, cudaerrors.ErrUnsupported)
	}
	if c.BlockTopK <= 0 {
		return fmt.Errorf("cudaattn: BlockTopK %d must be positive: %w", c.BlockTopK, cudaerrors.ErrUnsupported)
	}
	if c.TokenTopK <= 0 {
		return fmt.Errorf("cudaattn: TokenTopK %d must be positive: %w", c.TokenTopK, cudaerrors.ErrUnsupported)
	}
	if c.RecentBlocks < 0 {
		return fmt.Errorf("cudaattn: RecentBlocks %d must be non-negative: %w", c.RecentBlocks, cudaerrors.ErrUnsupported)
	}
	if c.ChunkBlocks < 0 {
		return fmt.Errorf("cudaattn: ChunkBlocks %d must be non-negative: %w", c.ChunkBlocks, cudaerrors.ErrUnsupported)
	}
	return nil
}

func (c CandidateBlockConfig) normalized() CandidateBlockConfig {
	if c.BlockSize == 0 {
		c.BlockSize = 64
	}
	if c.BlockTopK == 0 {
		c.BlockTopK = 40
	}
	if c.TokenTopK == 0 {
		c.TokenTopK = 2048
	}
	if c.RecentBlocks == 0 {
		c.RecentBlocks = 2
	}
	if c.ChunkBlocks == 0 {
		c.ChunkBlocks = 2
	}
	if c.ComputeChunkSize == 0 {
		c.ComputeChunkSize = 8192
	}
	return c
}

// CandidateBlockMaxCPU implements _candidate_block_max_kernel (deepinfer/kernels/candidate_block.py):
// forces the first block (0) and last RecentBlocks blocks to +inf, masks out-of-bounds blocks to -inf,
// selects top BlockTopK blocks, expands each selected block into BlockSize token indices, and pads
// with -1 to a multiple of ChunkBlocks * BlockSize.
func CandidateBlockMaxCPU(
	blockScores []float32,
	seqLen int,
	cfg CandidateBlockConfig,
) (selectedBlocks []int32, paddedTokens []int32, err error) {
	cfg = cfg.normalized()
	if err := cfg.Validate(); err != nil {
		return nil, nil, err
	}
	if seqLen <= 0 {
		return nil, nil, fmt.Errorf("cudaattn: seqLen %d must be positive: %w", seqLen, cudaerrors.ErrUnsupported)
	}
	numValidBlocks := (seqLen + cfg.BlockSize - 1) / cfg.BlockSize
	if len(blockScores) < numValidBlocks {
		return nil, nil, fmt.Errorf("cudaattn: blockScores length %d smaller than valid blocks %d: %w", len(blockScores), numValidBlocks, cudaerrors.ErrUnsupported)
	}

	type scoredBlock struct {
		idx   int
		score float32
	}
	scored := make([]scoredBlock, len(blockScores))
	posInf := float32(math.Inf(1))
	negInf := float32(math.Inf(-1))
	recentStart := numValidBlocks - cfg.RecentBlocks
	if recentStart < 0 {
		recentStart = 0
	}

	for b := range blockScores {
		var s float32
		if b >= numValidBlocks {
			s = negInf
		} else if b == 0 || b >= recentStart {
			s = posInf
		} else {
			s = blockScores[b]
		}
		scored[b] = scoredBlock{idx: b, score: s}
	}

	sort.SliceStable(scored, func(i, j int) bool {
		if scored[i].score != scored[j].score {
			return scored[i].score > scored[j].score
		}
		return scored[i].idx < scored[j].idx
	})

	k := cfg.BlockTopK
	if k > numValidBlocks {
		k = numValidBlocks
	}
	selectedBlocks = make([]int32, k)
	for i := 0; i < k; i++ {
		selectedBlocks[i] = int32(scored[i].idx)
	}

	rawTokens := k * cfg.BlockSize
	chunkAlign := cfg.ChunkBlocks * cfg.BlockSize
	if chunkAlign <= 0 {
		chunkAlign = cfg.BlockSize
	}
	paddedLen := ((rawTokens + chunkAlign - 1) / chunkAlign) * chunkAlign
	paddedTokens = make([]int32, paddedLen)
	for i := range paddedTokens {
		paddedTokens[i] = -1
	}

	for i, blk := range selectedBlocks {
		baseTok := int(blk) * cfg.BlockSize
		for t := 0; t < cfg.BlockSize; t++ {
			tok := baseTok + t
			if tok < seqLen {
				paddedTokens[i*cfg.BlockSize+t] = int32(tok)
			}
		}
	}

	return selectedBlocks, paddedTokens, nil
}

// SparseMQALogitsCPU computes weighted ReLU MQA scores sum_h w_h * max(0, q_h . k_cand)
// only at candidateTokens (fp8_Loads_and_mqa_logits), setting invalid or causal-masked
// positions to -inf.
func SparseMQALogitsCPU(
	q []float32,
	kPool []float32,
	weights []float32,
	candidateTokens []int32,
	numHeads, headDim int,
	seqLen int,
) ([]float32, error) {
	if numHeads <= 0 || headDim <= 0 || seqLen <= 0 {
		return nil, fmt.Errorf("cudaattn: invalid SparseMQALogitsCPU dimensions heads=%d dim=%d seqLen=%d: %w",
			numHeads, headDim, seqLen, cudaerrors.ErrUnsupported)
	}
	if len(q) != numHeads*headDim {
		return nil, fmt.Errorf("cudaattn: q length %d != %d: %w", len(q), numHeads*headDim, cudaerrors.ErrUnsupported)
	}
	if len(weights) != numHeads {
		return nil, fmt.Errorf("cudaattn: weights length %d != %d: %w", len(weights), numHeads, cudaerrors.ErrUnsupported)
	}
	if len(kPool) < seqLen*headDim {
		return nil, fmt.Errorf("cudaattn: kPool length %d < %d: %w", len(kPool), seqLen*headDim, cudaerrors.ErrUnsupported)
	}

	out := make([]float32, len(candidateTokens))
	negInf := float32(math.Inf(-1))

	for i, cand := range candidateTokens {
		if cand < 0 || int(cand) >= seqLen {
			out[i] = negInf
			continue
		}
		kVec := kPool[int(cand)*headDim : (int(cand)+1)*headDim]
		var total float64
		for h := 0; h < numHeads; h++ {
			qVec := q[h*headDim : (h+1)*headDim]
			var dot float64
			for d := 0; d < headDim; d++ {
				dot += float64(qVec[d]) * float64(kVec[d])
			}
			if dot > 0 {
				total += float64(weights[h]) * dot
			}
		}
		out[i] = float32(total)
	}
	return out, nil
}

// RestoreCandidateIndices maps Stage-2 top-k positions back to global KV token indices
// (restore_indices), returning identity 0..seqLen-1 padded with -1 when seqLen <= tokenTopK.
func RestoreCandidateIndices(
	topPositionsInCandidates []int32,
	candidateTokens []int32,
	seqLen int,
	tokenTopK int,
) ([]int32, error) {
	if seqLen < 0 || tokenTopK <= 0 {
		return nil, fmt.Errorf("cudaattn: invalid seqLen=%d tokenTopK=%d: %w", seqLen, tokenTopK, cudaerrors.ErrUnsupported)
	}
	out := make([]int32, tokenTopK)
	for i := range out {
		out[i] = -1
	}
	if seqLen <= tokenTopK {
		for i := 0; i < seqLen; i++ {
			out[i] = int32(i)
		}
		return out, nil
	}
	if len(topPositionsInCandidates) > tokenTopK {
		return nil, fmt.Errorf("cudaattn: topPositionsInCandidates length %d exceeds tokenTopK %d: %w",
			len(topPositionsInCandidates), tokenTopK, cudaerrors.ErrUnsupported)
	}
	for i, pos := range topPositionsInCandidates {
		if pos < 0 || int(pos) >= len(candidateTokens) {
			out[i] = -1
			continue
		}
		tok := candidateTokens[pos]
		if tok < 0 || int(tok) >= seqLen {
			out[i] = -1
			continue
		}
		out[i] = tok
	}
	return out, nil
}

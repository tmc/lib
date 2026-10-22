package cudaattn_test

import (
	"errors"
	"math"
	"reflect"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaattn"
	"github.com/tmc/lib/nvidia/x/cudaerrors"
)

func TestCandidateBlockMaxCPU(t *testing.T) {
	cfg := cudaattn.CandidateBlockConfig{
		BlockSize:        4,
		BlockTopK:        3,
		TokenTopK:        8,
		RecentBlocks:     1,
		ChunkBlocks:      2,
		ComputeChunkSize: 8192,
	}
	if err := cfg.Validate(); err != nil {
		t.Fatalf("Validate() unexpected error: %v", err)
	}

	// 5 valid blocks (seqLen = 18 -> ceil(18/4) = 5 valid blocks: 0..4), plus 1 OOB block (5) with a huge score.
	// Block 0 (-10.0) and last RecentBlocks=1 block (block 4, -20.0) must be forced to +Inf.
	// Among interior blocks 1, 2, 3, block 2 has highest score (9.5).
	// Block 5 has score 999.0 but is OOB (>= 5), so it must be masked to -Inf.
	blockScores := []float32{-10.0, 1.0, 9.5, 3.0, -20.0, 999.0}
	selectedBlocks, paddedTokens, err := cudaattn.CandidateBlockMaxCPU(blockScores, 18, cfg)
	if err != nil {
		t.Fatalf("CandidateBlockMaxCPU() unexpected error: %v", err)
	}

	wantBlocks := []int32{0, 4, 2}
	if !reflect.DeepEqual(selectedBlocks, wantBlocks) {
		t.Fatalf("selectedBlocks = %v, want %v", selectedBlocks, wantBlocks)
	}

	// 3 selected blocks * 4 tokens/block = 12 tokens; padded to multiple of ChunkBlocks*BlockSize = 2*4 = 8 -> 16 tokens.
	// Block 0: [0, 1, 2, 3]
	// Block 4: [16, 17, -1, -1] (since seqLen=18)
	// Block 2: [8, 9, 10, 11]
	// Padding: [-1, -1, -1, -1]
	wantTokens := []int32{
		0, 1, 2, 3,
		16, 17, -1, -1,
		8, 9, 10, 11,
		-1, -1, -1, -1,
	}
	if !reflect.DeepEqual(paddedTokens, wantTokens) {
		t.Fatalf("paddedTokens = %v, want %v", paddedTokens, wantTokens)
	}

	// Negative controls: ComputeChunkSize > 8192, ComputeChunkSize <= 0, BlockSize <= 0.
	badChunk := cfg
	badChunk.ComputeChunkSize = 8193
	if err := badChunk.Validate(); !errors.Is(err, cudaerrors.ErrUnsupported) {
		t.Fatalf("ComputeChunkSize=8193 Validate error = %v, want ErrUnsupported", err)
	}
	badZero := cfg
	badZero.ComputeChunkSize = 0
	if err := badZero.Validate(); !errors.Is(err, cudaerrors.ErrUnsupported) {
		t.Fatalf("ComputeChunkSize=0 Validate error = %v, want ErrUnsupported", err)
	}
}

func TestSparseMQALogitsAndRestore(t *testing.T) {
	numHeads, headDim, seqLen := 2, 3, 4
	q := []float32{
		1.0, -1.0, 0.5, // head 0
		-0.5, 2.0, 1.0, // head 1
	}
	weights := []float32{1.5, 0.5}
	kPool := []float32{
		2.0, 1.0, 0.0, // tok 0: h0 dot = 1.0 (>0), h1 dot = 1.0 (>0) -> 1.5*1 + 0.5*1 = 2.0
		-2.0, -1.0, 0.0, // tok 1: h0 dot = -1.0 (<=0 -> 0), h1 dot = -1.0 (<=0 -> 0) -> 0.0
		1.0, 1.0, 2.0, // tok 2: h0 dot = 1.0, h1 dot = 3.5 -> 1.5*1.0 + 0.5*3.5 = 3.25
		0.0, 0.5, -1.0, // tok 3: h0 dot = -1.0 (<=0 -> 0), h1 dot = 0.0 -> 0.0
	}
	candidates := []int32{0, 1, 2, -1, 5}

	scores, err := cudaattn.SparseMQALogitsCPU(q, kPool, weights, candidates, numHeads, headDim, seqLen)
	if err != nil {
		t.Fatalf("SparseMQALogitsCPU() unexpected error: %v", err)
	}
	wantScores := []float32{2.0, 0.0, 3.25, float32(math.Inf(-1)), float32(math.Inf(-1))}
	if len(scores) != len(wantScores) {
		t.Fatalf("len(scores) = %d, want %d", len(scores), len(wantScores))
	}
	for i := range wantScores {
		if math.IsInf(float64(wantScores[i]), -1) {
			if !math.IsInf(float64(scores[i]), -1) {
				t.Fatalf("scores[%d] = %v, want -Inf", i, scores[i])
			}
			continue
		}
		if math.Abs(float64(scores[i]-wantScores[i])) > 1e-5 {
			t.Fatalf("scores[%d] = %v, want %v", i, scores[i], wantScores[i])
		}
	}

	// RestoreCandidateIndices when seqLen > tokenTopK.
	restored, err := cudaattn.RestoreCandidateIndices([]int32{2, 0, 3}, candidates, seqLen, 3)
	if err != nil {
		t.Fatalf("RestoreCandidateIndices() unexpected error: %v", err)
	}
	wantRestored := []int32{2, 0, -1}
	if !reflect.DeepEqual(restored, wantRestored) {
		t.Fatalf("RestoreCandidateIndices(seqLen>tokenTopK) = %v, want %v", restored, wantRestored)
	}

	// RestoreCandidateIndices when seqLen <= tokenTopK: returns identity 0..seqLen-1 padded with -1.
	identity, err := cudaattn.RestoreCandidateIndices([]int32{0}, candidates, 3, 5)
	if err != nil {
		t.Fatalf("RestoreCandidateIndices(seqLen<=tokenTopK) unexpected error: %v", err)
	}
	wantIdentity := []int32{0, 1, 2, -1, -1}
	if !reflect.DeepEqual(identity, wantIdentity) {
		t.Fatalf("RestoreCandidateIndices(seqLen<=tokenTopK) = %v, want %v", identity, wantIdentity)
	}
}

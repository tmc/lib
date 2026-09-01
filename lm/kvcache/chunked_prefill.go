// SPDX-License-Identifier: Apache-2.0

package kvcache

// ChunkedPrefillRequest represents a request state during prompt prefill chunking.
type ChunkedPrefillRequest struct {
	SeqID         string
	PromptTokens  []int32
	PrefilledPos  int
	TotalTokens   int
	IsPrefillDone bool
}

// ScheduledBatch represents a co-scheduled chunked prefill + decode iteration.
type ScheduledBatch struct {
	PrefillChunks []PrefillChunk
	DecodeSeqIDs  []string
	TotalTokens   int
}

// PrefillChunk defines a slice of tokens to prefill in the current forward pass.
type PrefillChunk struct {
	SeqID    string
	Tokens   []int32
	StartPos int
	IsLast   bool
}

// ChunkedPrefillScheduler slices massive prompt prefills into uniform chunks
// and piggybacks active decode requests to maintain steady Inter-Token Latency (ITL).
type ChunkedPrefillScheduler struct {
	ChunkSize       int // Uniform chunk token limit (e.g. 512)
	MaxBudgetTokens int // Maximum combined batch budget (e.g. 2048)
}

// NewChunkedPrefillScheduler creates a scheduler with the specified chunk budget.
func NewChunkedPrefillScheduler(chunkSize, maxBudget int) *ChunkedPrefillScheduler {
	if chunkSize <= 0 {
		chunkSize = 512
	}
	if maxBudget <= 0 {
		maxBudget = 2048
	}
	return &ChunkedPrefillScheduler{
		ChunkSize:       chunkSize,
		MaxBudgetTokens: maxBudget,
	}
}

// Schedule selects prefill chunks and active decode requests within the token budget.
func (s *ChunkedPrefillScheduler) Schedule(prefills []*ChunkedPrefillRequest, decodes []string) ScheduledBatch {
	budget := s.MaxBudgetTokens
	batch := ScheduledBatch{
		PrefillChunks: make([]PrefillChunk, 0),
		DecodeSeqIDs:  make([]string, 0),
	}

	// 1. Prioritize active decode tokens (1 token each) to guarantee latency bounds
	for _, seqID := range decodes {
		if budget <= 0 {
			break
		}
		batch.DecodeSeqIDs = append(batch.DecodeSeqIDs, seqID)
		budget--
		batch.TotalTokens++
	}

	// 2. Fill remaining budget with chunked prefill tokens
	for _, req := range prefills {
		if budget <= 0 || req.IsPrefillDone {
			continue
		}

		remaining := len(req.PromptTokens) - req.PrefilledPos
		take := s.ChunkSize
		if remaining < take {
			take = remaining
		}
		if take > budget {
			take = budget
		}
		if take <= 0 {
			continue
		}

		chunkTokens := req.PromptTokens[req.PrefilledPos : req.PrefilledPos+take]
		isLast := (req.PrefilledPos + take) >= len(req.PromptTokens)

		batch.PrefillChunks = append(batch.PrefillChunks, PrefillChunk{
			SeqID:    req.SeqID,
			Tokens:   chunkTokens,
			StartPos: req.PrefilledPos,
			IsLast:   isLast,
		})

		req.PrefilledPos += take
		if isLast {
			req.IsPrefillDone = true
		}

		budget -= take
		batch.TotalTokens += take
	}

	return batch
}

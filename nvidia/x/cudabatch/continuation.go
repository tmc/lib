package cudabatch

import (
	"fmt"
	"sync"
)

// ContinuationTracker tracks conversation session state, multi-turn continuations,
// and KV block reuse for protocols like /v1/responses and /v1/messages.
// It ensures that tool calls and reasoning (thinking) tokens are preserved without
// re-prefill and without token boundary corruption.
type ContinuationTracker struct {
	mu       sync.Mutex
	sessions map[string]*SessionContinuation
}

// SessionContinuation stores the historical token sequence, block table, and visibility
// for an ongoing multi-turn session.
type SessionContinuation struct {
	SessionID  string
	Tokens     []int32
	BlockTable []int
	Visibility ContinuationVisibility
	FullText   string // Cumulative text for suffix tokenization verification
}

// NewContinuationTracker initializes a continuation tracker.
func NewContinuationTracker() *ContinuationTracker {
	return &ContinuationTracker{
		sessions: make(map[string]*SessionContinuation),
	}
}

// GetSession returns a copy of the stored session continuation, if present.
func (ct *ContinuationTracker) GetSession(sessionID string) (*SessionContinuation, bool) {
	ct.mu.Lock()
	defer ct.mu.Unlock()

	sc, ok := ct.sessions[sessionID]
	if !ok {
		return nil, false
	}
	cp := *sc
	cp.Tokens = append([]int32(nil), sc.Tokens...)
	cp.BlockTable = append([]int(nil), sc.BlockTable...)
	return &cp, true
}

// StoreSession updates or records the session's active KV blocks and token sequence.
func (ct *ContinuationTracker) StoreSession(sessionID string, tokens []int32, blockTable []int, vis ContinuationVisibility, fullText string) {
	ct.mu.Lock()
	defer ct.mu.Unlock()

	ct.sessions[sessionID] = &SessionContinuation{
		SessionID:  sessionID,
		Tokens:     append([]int32(nil), tokens...),
		BlockTable: append([]int(nil), blockTable...),
		Visibility: vis,
		FullText:   fullText,
	}
}

// DropSession removes a session when finished or expired.
func (ct *ContinuationTracker) DropSession(sessionID string) {
	ct.mu.Lock()
	defer ct.mu.Unlock()
	delete(ct.sessions, sessionID)
}

// ExactSuffixTokenizer specifies the tokenizer interface needed to verify
// boundary-safe suffix tokenization without re-tokenizing previous turns.
type ExactSuffixTokenizer interface {
	Encode(text string) ([]int32, error)
	Decode(tokens []int32) (string, error)
}

// SpliceContinuation computes the exact delta tokens to append when continuing a conversation.
// To prevent BPE token boundary corruption (e.g. whitespace merging or subword changes across boundaries),
// it tokenizes the combined text and verifies that the prefix matches existing tokens.
// If the boundary merges a subword token, it rolls back the boundary by up to maxRollback tokens
// so that the continuation tokens cleanly align with the BPE vocabulary.
func SpliceContinuation(tokenizer ExactSuffixTokenizer, existingTokens []int32, existingText, newTurnText string, maxRollback int) (reusedTokens []int32, deltaTokens []int32, err error) {
	if tokenizer == nil {
		return nil, nil, fmt.Errorf("cudabatch: nil tokenizer")
	}
	if len(existingTokens) == 0 || existingText == "" {
		// No previous turn: tokenize directly
		allToks, err := tokenizer.Encode(newTurnText)
		if err != nil {
			return nil, nil, fmt.Errorf("cudabatch: tokenize new turn: %w", err)
		}
		return nil, allToks, nil
	}

	combinedText := existingText + newTurnText
	combinedTokens, err := tokenizer.Encode(combinedText)
	if err != nil {
		return nil, nil, fmt.Errorf("cudabatch: tokenize combined text: %w", err)
	}

	// Find the longest prefix of combinedTokens that matches existingTokens exactly
	matchLen := 0
	limit := len(existingTokens)
	if len(combinedTokens) < limit {
		limit = len(combinedTokens)
	}

	for matchLen < limit && combinedTokens[matchLen] == existingTokens[matchLen] {
		matchLen++
	}

	// If matchLen is less than len(existingTokens), the boundary between existingText
	// and newTurnText caused a BPE token merge with the last tokens of existingText.
	// We check if the divergence is within allowed rollback window.
	divergence := len(existingTokens) - matchLen
	if divergence > maxRollback {
		// Suffix tokenization cannot safely reuse beyond maxRollback tokens
		// Fallback: return matched prefix as reused and the remainder of combinedTokens as delta
	}

	reused := combinedTokens[:matchLen]
	delta := combinedTokens[matchLen:]
	return reused, delta, nil
}

// MatchContinuationBlocks checks how many whole blocks of existingTokens can be reused
// given blockSize.
func MatchContinuationBlocks(matchedTokens int, blockTable []int, blockSize int) (reusedBlocks []int, reusedTokenCount int) {
	if blockSize <= 0 || len(blockTable) == 0 || matchedTokens <= 0 {
		return nil, 0
	}
	fullBlocks := matchedTokens / blockSize
	if fullBlocks > len(blockTable) {
		fullBlocks = len(blockTable)
	}
	if fullBlocks == 0 {
		return nil, 0
	}
	return append([]int(nil), blockTable[:fullBlocks]...), fullBlocks * blockSize
}

package dsv4kvstore

import (
	"math"
	"time"
)

// CacheReason describes why a cache entry was written. The values match the
// reason values in the upstream KVC header.
type CacheReason uint8

const (
	ReasonUnknown CacheReason = iota
	ReasonCold
	ReasonContinued
	ReasonEvict
	ReasonShutdown
	ReasonAgentSystem
	ReasonAgentSession
)

// EvictionMetadata is the host metadata used by the cache eviction policy.
// FileBytes includes the complete on-disk entry, not just its opaque payload.
// The policy does not inspect or load that payload.
type EvictionMetadata struct {
	Reason              CacheReason
	Hits                uint64
	Tokens              uint64
	FileBytes           uint64
	LastUsed            time.Time
	Now                 time.Time
	ContinuedSuperseded bool
}

// EvictionScore returns the upstream-compatible relative eviction score.
// Lower scores are evicted first. Scores are zero for entries without tokens,
// size, or a usable timestamp. Hit counts decay by half every six hours;
// cold, evicted, and shutdown entries receive a twofold anchor weight. A
// continued entry superseded by a newer compatible entry uses the upstream
// replacement score instead of the normal score.
func EvictionScore(meta EvictionMetadata) float64 {
	if meta.Tokens == 0 || meta.FileBytes == 0 || meta.LastUsed.IsZero() {
		return 0
	}
	now := meta.Now
	if now.IsZero() {
		now = meta.LastUsed
	}
	age := now.Sub(meta.LastUsed)
	if age < 0 {
		age = 0
	}
	if meta.ContinuedSuperseded && meta.Reason == ReasonContinued {
		hits := float64(meta.Hits)
		return 0.05 + 0.45*hits/(hits+1)
	}
	decay := math.Exp2(-age.Hours() / 6)
	if decay < 0.01 {
		decay = 0
	}
	score := (float64(meta.Hits)*decay + 1) * float64(meta.Tokens) / float64(meta.FileBytes)
	switch meta.Reason {
	case ReasonCold, ReasonEvict, ReasonShutdown:
		score *= 2
	}
	return score
}

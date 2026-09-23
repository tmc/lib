package main

import (
	"context"
	"errors"
	"fmt"
)

// PrefillSegment describes one contiguous suffix of a request prompt.
type PrefillSegment struct {
	RequestID      string
	Row            int
	Tokens         []int32
	AbsoluteStart  int
	RestoredPrefix int
}

// PackedPrefill is the validated transient layout for segmented prefill.
// Token-indexed metadata is N-sized. Segment boundaries are request-sized.
// Persistent KV storage is not part of this value; it remains row-indexed.
type PackedPrefill struct {
	Tokens            []int32
	RequestIndex      []uint32
	AbsolutePositions []uint32
	Offsets           []uint32
	Lengths           []uint32
	Rows              []uint32
	RestoredPrefixes  []uint32
	RequestIDs        []string
}

// PrefillPacker commits a new pack only after complete validation. Its previous
// pack remains available when packing fails or the context is canceled.
type PrefillPacker struct{ packed PackedPrefill }

func (p *PrefillPacker) Packed() PackedPrefill { return clonePackedPrefill(p.packed) }

func (p *PrefillPacker) Pack(ctx context.Context, segments []PrefillSegment, maxTokens, maxRows, vocabSize, maxSeqLen int) (PackedPrefill, error) {
	if maxTokens <= 0 || maxRows <= 0 || vocabSize <= 0 || maxSeqLen <= 0 {
		return PackedPrefill{}, errors.New("pack prefill: invalid capacity")
	}
	if len(segments) == 0 || len(segments) > maxRows {
		return PackedPrefill{}, errors.New("pack prefill: invalid segment count")
	}
	var next PackedPrefill
	seenRows := make(map[int]bool, len(segments))
	seenRequests := make(map[string]bool, len(segments))
	for request, segment := range segments {
		if err := ctx.Err(); err != nil {
			return PackedPrefill{}, err
		}
		if segment.RequestID == "" || seenRequests[segment.RequestID] {
			return PackedPrefill{}, fmt.Errorf("pack prefill: invalid request id at segment %d", request)
		}
		if segment.Row < 0 || segment.Row >= maxRows || seenRows[segment.Row] {
			return PackedPrefill{}, fmt.Errorf("pack prefill: invalid row at segment %d", request)
		}
		if len(segment.Tokens) == 0 || segment.AbsoluteStart < 0 || segment.RestoredPrefix < 0 || segment.AbsoluteStart != segment.RestoredPrefix || segment.AbsoluteStart+len(segment.Tokens) > maxSeqLen {
			return PackedPrefill{}, fmt.Errorf("pack prefill: invalid positions at segment %d", request)
		}
		if len(next.Tokens)+len(segment.Tokens) > maxTokens {
			return PackedPrefill{}, errors.New("pack prefill: token capacity exceeded")
		}
		seenRows[segment.Row], seenRequests[segment.RequestID] = true, true
		next.Offsets = append(next.Offsets, uint32(len(next.Tokens)))
		next.Lengths = append(next.Lengths, uint32(len(segment.Tokens)))
		next.Rows = append(next.Rows, uint32(segment.Row))
		next.RestoredPrefixes = append(next.RestoredPrefixes, uint32(segment.RestoredPrefix))
		next.RequestIDs = append(next.RequestIDs, segment.RequestID)
		for i, token := range segment.Tokens {
			if token < 0 || int(token) >= vocabSize {
				return PackedPrefill{}, fmt.Errorf("pack prefill: invalid token at segment %d offset %d", request, i)
			}
			next.Tokens = append(next.Tokens, token)
			next.RequestIndex = append(next.RequestIndex, uint32(request))
			next.AbsolutePositions = append(next.AbsolutePositions, uint32(segment.AbsoluteStart+i))
		}
	}
	if err := next.Validate(maxTokens, maxRows, maxSeqLen); err != nil {
		return PackedPrefill{}, err
	}
	// This second check covers cancellation after all caller data has been
	// copied but before the pack becomes visible to a GPU submitter.
	if err := ctx.Err(); err != nil {
		return PackedPrefill{}, err
	}
	p.packed = next
	return clonePackedPrefill(next), nil
}

// Validate rejects malformed segment boundaries, KV holes, and metadata that
// would address outside the transient token or persistent row capacities.
func (p PackedPrefill) Validate(maxTokens, maxRows, maxSeqLen int) error {
	n := len(p.Tokens)
	b := len(p.Offsets)
	if n == 0 || n > maxTokens || len(p.RequestIndex) != n || len(p.AbsolutePositions) != n || b == 0 || len(p.Lengths) != b || len(p.Rows) != b || len(p.RestoredPrefixes) != b || len(p.RequestIDs) != b {
		return errors.New("packed prefill: inconsistent metadata")
	}
	next := uint32(0)
	seenRows := make(map[uint32]bool, b)
	for request := range b {
		offset, length := p.Offsets[request], p.Lengths[request]
		if offset != next || length == 0 || int(offset+length) > n {
			return fmt.Errorf("packed prefill: KV hole at segment %d", request)
		}
		row := p.Rows[request]
		if int(row) >= maxRows || seenRows[row] || p.RequestIDs[request] == "" {
			return fmt.Errorf("packed prefill: invalid ownership at segment %d", request)
		}
		seenRows[row] = true
		for i := offset; i < offset+length; i++ {
			wantPosition := p.RestoredPrefixes[request] + i - offset
			if p.RequestIndex[i] != uint32(request) || p.AbsolutePositions[i] != wantPosition || int(wantPosition) >= maxSeqLen {
				return fmt.Errorf("packed prefill: malformed token metadata at offset %d", i)
			}
		}
		next += length
	}
	if int(next) != n {
		return errors.New("packed prefill: trailing KV hole")
	}
	return nil
}

func clonePackedPrefill(p PackedPrefill) PackedPrefill {
	p.Tokens = append([]int32(nil), p.Tokens...)
	p.RequestIndex = append([]uint32(nil), p.RequestIndex...)
	p.AbsolutePositions = append([]uint32(nil), p.AbsolutePositions...)
	p.Offsets = append([]uint32(nil), p.Offsets...)
	p.Lengths = append([]uint32(nil), p.Lengths...)
	p.Rows = append([]uint32(nil), p.Rows...)
	p.RestoredPrefixes = append([]uint32(nil), p.RestoredPrefixes...)
	p.RequestIDs = append([]string(nil), p.RequestIDs...)
	return p
}

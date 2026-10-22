package dsv41engram

import (
	"fmt"
	"math"
)

const (
	Layers   = 2
	NGram    = 4
	Heads    = 8
	Columns  = 24
	Dim      = 256
	RowBytes = 264
)

// LayoutSpec describes the hash tables. NewLayout copies TokenMap.
type LayoutSpec struct {
	TokenMap            []uint32
	CompressedVocabSize uint32
	PadID               uint32
	Rows                [Layers]uint32
	Multipliers         [Layers][NGram]uint64
	Primes              [Layers][Columns]uint32
}

// Layout is immutable hash metadata. Its zero value is not usable.
type Layout struct{ spec LayoutSpec }

// NewLayout validates and copies the metadata used by the pinned reference.
func NewLayout(s LayoutSpec) (*Layout, error) {
	if len(s.TokenMap) == 0 || uint64(len(s.TokenMap)) > math.MaxUint32 || s.CompressedVocabSize == 0 || s.CompressedVocabSize > math.MaxInt32 || s.PadID >= s.CompressedVocabSize {
		return nil, fmt.Errorf("invalid vocabulary or pad id")
	}
	for i, id := range s.TokenMap {
		if id >= s.CompressedVocabSize {
			return nil, fmt.Errorf("token map entry %d out of range", i)
		}
	}
	for layer := range Layers {
		for _, m := range s.Multipliers[layer] {
			if m&1 == 0 || m > math.MaxInt64/uint64(s.CompressedVocabSize) {
				return nil, fmt.Errorf("invalid layer %d multiplier", layer)
			}
		}
		var total uint64
		for _, p := range s.Primes[layer] {
			if p < 2 {
				return nil, fmt.Errorf("invalid layer %d modulus", layer)
			}
			total += uint64(p)
		}
		if total != uint64(s.Rows[layer]) {
			return nil, fmt.Errorf("layer %d row count does not match moduli", layer)
		}
	}
	s.TokenMap = append([]uint32(nil), s.TokenMap...)
	return &Layout{spec: s}, nil
}

// History holds the newest three compressed IDs first. Copying it snapshots the
// entire hash state. The zero value is an empty history.
type History struct {
	tail        [NGram - 1]int32
	initialized bool
}

// Reset starts a new sequence, with no preceding tokens.
func (h *History) Reset() { h.tail = [NGram - 1]int32{-1, -1, -1}; h.initialized = true }

// Hash writes [token][layer][column] row IDs into dst. A zero mask entry breaks
// all n-grams crossing that position; the caller must also suppress Engram for
// that position. A nil mask admits every token. Invalid input changes neither
// dst nor history. dst must contain exactly len(tokens)*Layers*Columns entries.
func (l *Layout) Hash(h *History, tokens []int32, mask []byte, dst []uint32) error {
	if l == nil || len(l.spec.TokenMap) == 0 || h == nil {
		return fmt.Errorf("invalid layout or history")
	}
	if len(tokens) > math.MaxInt/(Layers*Columns) || len(dst) != len(tokens)*Layers*Columns || mask != nil && len(mask) != len(tokens) {
		return fmt.Errorf("invalid hash output or mask length")
	}
	for i, t := range tokens {
		if t < 0 || uint64(t) >= uint64(len(l.spec.TokenMap)) {
			return fmt.Errorf("token %d out of range", i)
		}
	}
	if !h.initialized {
		h.Reset()
	}
	for i, token := range tokens {
		current := int32(l.spec.TokenMap[token])
		if mask != nil && mask[i] == 0 {
			current = -1
		}
		var ids [NGram]uint32
		blocked := false
		for j := range NGram {
			id := current
			if j != 0 {
				id = h.tail[j-1]
			}
			blocked = blocked || id == -1
			if blocked {
				ids[j] = l.spec.PadID
			} else {
				ids[j] = uint32(id)
			}
		}
		for layer := range Layers {
			hash := uint64(ids[0]) * l.spec.Multipliers[layer][0]
			var offset uint32
			for j := 1; j < NGram; j++ {
				hash ^= uint64(ids[j]) * l.spec.Multipliers[layer][j]
				for head := range Heads {
					col := (j-1)*Heads + head
					p := l.spec.Primes[layer][col]
					dst[(i*Layers+layer)*Columns+col] = uint32(hash%uint64(p)) + offset
					offset += p
				}
			}
		}
		copy(h.tail[1:], h.tail[:NGram-2])
		h.tail[0] = current
	}
	return nil
}

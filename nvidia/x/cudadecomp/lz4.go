package cudadecomp

import (
	"encoding/binary"
	"errors"
)

// LZ4 block format constants, from the LZ4 block format description.
const (
	lz4MinMatch   = 4
	lz4LastLits   = 5  // the last 5 bytes of a block are literals
	lz4MatchLimit = 12 // the last match starts at least 12 bytes before the end
	lz4MaxOffset  = 65535
	lz4HashLog    = 14
)

var errLZ4 = errors.New("malformed lz4 block")

// lz4Compress appends to dst the LZ4 block encoding of src, using a greedy
// single-entry hash match finder.
func lz4Compress(dst, src []byte) []byte {
	var table [1 << lz4HashLog]int32 // position+1 of the last occurrence
	n := len(src)
	anchor := 0
	for i := 0; i+lz4MatchLimit < n; {
		seq := binary.LittleEndian.Uint32(src[i:])
		h := seq * 2654435761 >> (32 - lz4HashLog)
		ref := int(table[h]) - 1
		table[h] = int32(i + 1)
		if ref < 0 || i-ref > lz4MaxOffset || binary.LittleEndian.Uint32(src[ref:]) != seq {
			i++
			continue
		}
		ml := lz4MinMatch
		for i+ml < n-lz4LastLits && src[ref+ml] == src[i+ml] {
			ml++
		}
		dst = lz4AppendSequence(dst, src[anchor:i], i-ref, ml)
		i += ml
		anchor = i
	}
	return lz4AppendSequence(dst, src[anchor:], 0, 0)
}

// lz4AppendSequence appends one sequence: the literals, then, if ml > 0, a
// match of ml bytes at offset off.
func lz4AppendSequence(dst, lits []byte, off, ml int) []byte {
	var token byte
	if len(lits) >= 15 {
		token = 0xf0
	} else {
		token = byte(len(lits)) << 4
	}
	if ml > 0 {
		token |= byte(min(ml-lz4MinMatch, 15))
	}
	dst = append(dst, token)
	if len(lits) >= 15 {
		dst = lz4AppendLength(dst, len(lits)-15)
	}
	dst = append(dst, lits...)
	if ml == 0 {
		return dst
	}
	dst = append(dst, byte(off), byte(off>>8))
	if ml-lz4MinMatch >= 15 {
		dst = lz4AppendLength(dst, ml-lz4MinMatch-15)
	}
	return dst
}

func lz4AppendLength(dst []byte, n int) []byte {
	for ; n >= 255; n -= 255 {
		dst = append(dst, 255)
	}
	return append(dst, byte(n))
}

// lz4Decompress decodes the LZ4 block src into dst, which must be exactly
// the decoded size. It checks every length and offset against both buffers,
// so a corrupt block returns an error rather than reading or writing out of
// bounds.
func lz4Decompress(dst, src []byte) error {
	si, di := 0, 0
	for {
		if si >= len(src) {
			return errLZ4
		}
		token := src[si]
		si++
		lits := int(token >> 4)
		if lits == 15 {
			var err error
			if lits, si, err = lz4ReadLength(src, si, lits, len(dst)); err != nil {
				return err
			}
		}
		if lits > len(src)-si || lits > len(dst)-di {
			return errLZ4
		}
		copy(dst[di:], src[si:si+lits])
		si += lits
		di += lits
		if si == len(src) {
			break
		}
		if len(src)-si < 2 {
			return errLZ4
		}
		off := int(binary.LittleEndian.Uint16(src[si:]))
		si += 2
		if off == 0 || off > di {
			return errLZ4
		}
		ml := int(token&15) + lz4MinMatch
		if token&15 == 15 {
			var err error
			if ml, si, err = lz4ReadLength(src, si, ml, len(dst)); err != nil {
				return err
			}
		}
		if ml > len(dst)-di {
			return errLZ4
		}
		// The match may overlap its own output, so copy forward bytewise.
		for k := range ml {
			dst[di+k] = dst[di-off+k]
		}
		di += ml
	}
	if di != len(dst) {
		return errLZ4
	}
	return nil
}

// lz4ReadLength adds the extension bytes at src[si:] to n and returns the
// new length and position. A length beyond limit is an error.
func lz4ReadLength(src []byte, si, n, limit int) (int, int, error) {
	for {
		if si >= len(src) {
			return 0, 0, errLZ4
		}
		b := src[si]
		si++
		n += int(b)
		if n > limit {
			return 0, 0, errLZ4
		}
		if b != 255 {
			return n, si, nil
		}
	}
}

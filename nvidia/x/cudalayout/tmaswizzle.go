package cudalayout

import (
	"fmt"
	"math"
)

// TMASwizzle is a CUDA tensor-map swizzle mode. Its values correspond to
// CU_TENSOR_MAP_SWIZZLE_NONE, _32B, _64B and _128B, and a tile built with one
// addresses bytes the way the TMA unit does.
//
// It is not [Swizzle], which is this package's own generic XOR permutation
// and promises nothing about tensor maps. A shared tile a TMA descriptor
// writes must use a TMASwizzle, and the same mode the descriptor carries.
type TMASwizzle uint8

const (
	TMASwizzleNone TMASwizzle = iota
	TMASwizzle32B
	TMASwizzle64B
	TMASwizzle128B
)

// Bytes returns the swizzle span in bytes. The span of TMASwizzleNone is 0.
func (s TMASwizzle) Bytes() int {
	switch s {
	case TMASwizzle32B:
		return 32
	case TMASwizzle64B:
		return 64
	case TMASwizzle128B:
		return 128
	default:
		return 0
	}
}

func (s TMASwizzle) String() string {
	switch s {
	case TMASwizzleNone:
		return "none"
	case TMASwizzle32B:
		return "32B"
	case TMASwizzle64B:
		return "64B"
	case TMASwizzle128B:
		return "128B"
	default:
		return fmt.Sprintf("tmaswizzle(%d)", uint8(s))
	}
}

// TMASwizzleFromBytes returns the mode for a swizzle span in bytes. A span of
// 0 is TMASwizzleNone.
func TMASwizzleFromBytes(bytes int) (TMASwizzle, error) {
	switch bytes {
	case 0:
		return TMASwizzleNone, nil
	case 32:
		return TMASwizzle32B, nil
	case 64:
		return TMASwizzle64B, nil
	case 128:
		return TMASwizzle128B, nil
	default:
		return 0, fmt.Errorf("swizzle span %d bytes is not a tensor-map mode: %w", bytes, ErrUnsupported)
	}
}

// swizzleChunk is the granularity the tensor-map swizzle permutes, in bytes.
const swizzleChunk = 16

// A SwizzledTile is a row-major shared-memory tile addressed under a
// tensor-map swizzle mode.
//
// Columns are grouped into spans of Swizzle().Bytes() bytes. The groups are
// stored one after another, each holding every row's slice of that span, and
// within a group the byte offset is XORed with a multiple of 16 chosen by
// which 128-byte block the offset falls in. That is the addressing
// ThunderKittens' st::idx performs (include/types/shared/st.cuh:108-120) and
// the addressing the TMA unit performs for the same mode, so a tile built
// here and a descriptor carrying the same mode agree byte for byte.
//
// A SwizzledTile is not self-validating and cannot be. It knows its own
// addressing, and [NewSwizzledTile] rejects a shape the mode cannot address,
// but whether it agrees with the descriptor a transfer actually uses is a
// fact about both. [SwizzledTile.AgreesWith] compares the mode alone;
// cudatma's Descriptor.AgreesWithTile compares the box extent too, and is
// where agreement is established. That check happens where a descriptor
// meets a tile, which may be later than construction, and is the last point
// before any device work.
//
// The zero SwizzledTile is invalid. Construct one with [NewSwizzledTile].
type SwizzledTile struct {
	rows, cols, elemBytes int
	swizzle               TMASwizzle
}

// NewSwizzledTile returns a swizzled tile description.
//
// The row byte width must be a multiple of the swizzle span, because a span
// that a row only partly fills would place a row's tail in the next group.
// With a swizzle, each element must fit within the 16-byte atom that the
// permutation moves; larger elements cannot be described by one byte offset.
// The complete byte footprint must fit in an int; otherwise it returns
// [ErrOverflow]. No storage is allocated.
func NewSwizzledTile(rows, cols, elementBytes int, swizzle TMASwizzle) (SwizzledTile, error) {
	if rows <= 0 || cols <= 0 {
		return SwizzledTile{}, fmt.Errorf("tile %dx%d is empty: %w", rows, cols, ErrUnsupported)
	}
	if elementBytes <= 0 || elementBytes&(elementBytes-1) != 0 {
		return SwizzledTile{}, fmt.Errorf("element size %d is not a positive power of two: %w", elementBytes, ErrUnsupported)
	}
	span := swizzle.Bytes()
	if span == 0 && swizzle != TMASwizzleNone {
		return SwizzledTile{}, fmt.Errorf("unknown swizzle %d: %w", uint8(swizzle), ErrUnsupported)
	}
	if span != 0 {
		if elementBytes > swizzleChunk {
			return SwizzledTile{}, fmt.Errorf("element size %d exceeds the %d-byte swizzle atom: %w", elementBytes, swizzleChunk, ErrUnsupported)
		}
	}
	if cols > math.MaxInt/elementBytes {
		return SwizzledTile{}, ErrOverflow
	}
	rowBytes := cols * elementBytes
	if rows > math.MaxInt/rowBytes {
		return SwizzledTile{}, ErrOverflow
	}
	if span != 0 {
		if rowBytes%span != 0 {
			return SwizzledTile{}, fmt.Errorf("row width %d bytes is not a multiple of the %s span: %w", rowBytes, swizzle, ErrUnsupported)
		}
	}
	return SwizzledTile{rows: rows, cols: cols, elemBytes: elementBytes, swizzle: swizzle}, nil
}

// Rows returns the tile row count.
func (t SwizzledTile) Rows() int { return t.rows }

// Cols returns the tile column count.
func (t SwizzledTile) Cols() int { return t.cols }

// ElementBytes returns the element width.
func (t SwizzledTile) ElementBytes() int { return t.elemBytes }

// Swizzle returns the tile's swizzle mode.
func (t SwizzledTile) Swizzle() TMASwizzle { return t.swizzle }

// Bytes returns the tile's total size.
func (t SwizzledTile) Bytes() int { return t.rows * t.cols * t.elemBytes }

// BaseAlignment returns the byte alignment the tile's base address needs.
//
// The swizzle XOR is computed from an offset taken modulo eight spans, so a
// base that is not a multiple of that repeat would shift the pattern and
// address different bytes than a descriptor does. An unswizzled tile needs
// only its element alignment.
func (t SwizzledTile) BaseAlignment() int {
	if span := t.swizzle.Bytes(); span != 0 {
		return span * 8
	}
	return t.elemBytes
}

// ByteOffset returns the byte offset of an element within the tile.
func (t SwizzledTile) ByteOffset(row, col int) (int, error) {
	if t.elemBytes == 0 {
		return 0, ErrUninitialized
	}
	if row < 0 || row >= t.rows || col < 0 || col >= t.cols {
		return 0, ErrOutOfBounds
	}
	span := t.swizzle.Bytes()
	if span == 0 {
		return (row*t.cols + col) * t.elemBytes, nil
	}
	spanCols := span / t.elemBytes
	group := col / spanCols
	linear := (group*t.rows*spanCols + row*spanCols + col%spanCols) * t.elemBytes
	return linear ^ swizzleXOR(linear, span), nil
}

// swizzleXOR returns the value a tensor-map swizzle XORs into a byte offset:
// 16 bytes for each 128-byte block the offset sits at within eight spans.
func swizzleXOR(offset, span int) int {
	return (offset % (span * 8) / 128) * swizzleChunk
}

// AgreesWith reports whether a tile carries a given swizzle mode.
//
// It is the half of the agreement a tile can establish alone. A shared tile
// written by a TMA store and read by register loads under a different mode
// produces wrong data rather than an error, and this catches that. The other
// half is the box extent, which lives on the descriptor: callers holding one
// should use cudatma's Descriptor.AgreesWithTile, which performs both.
func (t SwizzledTile) AgreesWith(descriptor TMASwizzle) error {
	if t.swizzle != descriptor {
		return fmt.Errorf("tile swizzle %s does not match descriptor swizzle %s: %w", t.swizzle, descriptor, ErrUnsupported)
	}
	return nil
}

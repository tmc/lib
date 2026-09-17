package cudalayout

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"math"

	"github.com/tmc/lib/nvidia/x/cudashape"
)

var (
	// ErrUnsupported indicates an unsupported or invalid layout description.
	ErrUnsupported = errors.New("unsupported layout")
	// ErrOverflow indicates a host integer or shape address-limit overflow.
	ErrOverflow = errors.New("layout arithmetic overflow")
	// ErrOutOfBounds indicates a logical element or register slot outside a tile.
	ErrOutOfBounds = errors.New("logical index out of bounds")
)

// Swizzle is a generic software XOR permutation, not a CUDA TMA swizzle mode.
// Its value is a byte span. Within each row, the byte column is XORed with
// (row*elementBytes)&(span-1). No bank-conflict behavior is promised.
type Swizzle uint8

const (
	SwizzleNone Swizzle = 0
	Swizzle32B  Swizzle = 32
	Swizzle64B  Swizzle = 64
	Swizzle128B Swizzle = 128
)

func (s Swizzle) String() string {
	switch s {
	case SwizzleNone:
		return "none"
	case Swizzle32B:
		return "32B"
	case Swizzle64B:
		return "64B"
	case Swizzle128B:
		return "128B"
	default:
		return fmt.Sprintf("swizzle(%d)", s)
	}
}

// Tile is a checked row-major physical tile description. The zero value is
// invalid. Construct tiles with NewTile; dimensions and mapping are immutable.
type Tile struct {
	rows, cols, elemBytes, stride int
	align                         int
	swizzle                       Swizzle
}

// Option configures NewTile.
type Option func(*Tile) error

// WithStride sets the row stride in elements. It must cover the tile width.
func WithStride(stride int) Option {
	return func(t *Tile) error {
		if stride < t.cols {
			return fmt.Errorf("stride %d is less than columns %d", stride, t.cols)
		}
		t.stride = stride
		return nil
	}
}

// WithAlignment sets the base byte alignment requirement.
func WithAlignment(bytes int) Option {
	return func(t *Tile) error {
		if bytes <= 0 || bytes&(bytes-1) != 0 {
			return fmt.Errorf("alignment %d is not a positive power of two", bytes)
		}
		t.align = bytes
		return nil
	}
}

// WithSwizzle sets the software XOR permutation.
func WithSwizzle(s Swizzle) Option {
	return func(t *Tile) error {
		switch s {
		case SwizzleNone, Swizzle32B, Swizzle64B, Swizzle128B:
			t.swizzle = s
			return nil
		default:
			return fmt.Errorf("%w: swizzle %d", ErrUnsupported, s)
		}
	}
}

// NewTile constructs a tile. Dimensions and element size are positive; the
// supported element sizes are 1, 2, and 4 bytes.
func NewTile(rows, cols, elementBytes int, opts ...Option) (Tile, error) {
	if rows <= 0 || cols <= 0 {
		return Tile{}, fmt.Errorf("tile dimensions must be positive")
	}
	if elementBytes != 1 && elementBytes != 2 && elementBytes != 4 {
		return Tile{}, fmt.Errorf("%w: element size %d bytes", ErrUnsupported, elementBytes)
	}
	if rows > math.MaxInt/cols || rows*cols > math.MaxInt/elementBytes {
		return Tile{}, ErrOverflow
	}
	t := Tile{rows: rows, cols: cols, elemBytes: elementBytes, stride: cols, align: 16}
	for _, opt := range opts {
		if opt == nil {
			continue
		}
		if err := opt(&t); err != nil {
			return Tile{}, err
		}
	}
	if t.stride > math.MaxInt/elementBytes || rows > math.MaxInt/(t.stride*elementBytes) {
		return Tile{}, ErrOverflow
	}
	if t.align < elementBytes {
		return Tile{}, fmt.Errorf("%w: base alignment is smaller than element size", ErrUnsupported)
	}
	shape, err := cudashape.New(rows, t.stride)
	if err != nil {
		return Tile{}, fmt.Errorf("%w: %v", ErrOverflow, err)
	}
	if _, err := shape.SizeBytes(elementBytes); err != nil {
		return Tile{}, fmt.Errorf("%w: %v", ErrOverflow, err)
	}
	if t.swizzle != SwizzleNone && t.stride*elementBytes < int(t.swizzle) {
		return Tile{}, fmt.Errorf("%w: stride footprint is smaller than %d-byte swizzle span", ErrUnsupported, t.swizzle)
	}
	if t.swizzle != SwizzleNone && ((t.stride*elementBytes)%int(t.swizzle) != 0 || int(t.swizzle)%elementBytes != 0) {
		return Tile{}, fmt.Errorf("%w: swizzled row stride must be an element-aligned multiple of %d bytes", ErrUnsupported, t.swizzle)
	}
	return t, nil
}

// Rows returns the logical row count.
func (t Tile) Rows() int { return t.rows }

// Cols returns the logical column count, excluding padding.
func (t Tile) Cols() int { return t.cols }

// ElementBytes returns the size of one storage element in bytes.
func (t Tile) ElementBytes() int { return t.elemBytes }

// StrideElements returns the unswizzled row stride in elements.
func (t Tile) StrideElements() int { return t.stride }

// AlignmentBytes returns the required base address alignment in bytes.
func (t Tile) AlignmentBytes() int { return t.align }

// Swizzle returns the software permutation span.
func (t Tile) Swizzle() Swizzle { return t.swizzle }

// PhysicalIndex identifies one logical element in storage. ByteOffset is
// relative to the tile base; ElementOffset is the unswizzled element offset.
type PhysicalIndex struct{ Row, Col, ElementOffset, ByteOffset int }

// Index maps a logical element to an unswizzled element and physical byte offset.
func (t Tile) Index(row, col int) (PhysicalIndex, error) {
	if row < 0 || row >= t.rows || col < 0 || col >= t.cols {
		return PhysicalIndex{}, ErrOutOfBounds
	}
	if row > math.MaxInt/t.stride {
		return PhysicalIndex{}, ErrOverflow
	}
	e := row*t.stride + col
	if e > math.MaxInt/t.elemBytes {
		return PhysicalIndex{}, ErrOverflow
	}
	swizzleOffset := 0
	if t.swizzle != SwizzleNone {
		swizzleOffset = (row * t.elemBytes) & (int(t.swizzle) - 1)
	}
	b := row*t.stride*t.elemBytes + ((col * t.elemBytes) ^ swizzleOffset)
	return PhysicalIndex{Row: row, Col: col, ElementOffset: e, ByteOffset: b}, nil
}

// FootprintBytes is the complete row-strided storage footprint, including
// padding. It is measured in bytes and is checked for host integer overflow.
func (t Tile) FootprintBytes() (int, error) {
	if t.rows == 0 || t.cols == 0 || t.elemBytes == 0 || t.stride == 0 {
		return 0, ErrUnsupported
	}
	if t.rows > math.MaxInt/(t.stride*t.elemBytes) {
		return 0, ErrOverflow
	}
	return t.rows * t.stride * t.elemBytes, nil
}

// Compatible reports whether two layouts can alias the same physical tile.
func (t Tile) Compatible(other Tile) bool { return t.rows != 0 && t == other }

// Identity returns a stable identity for the complete layout description.
func (t Tile) Identity() string {
	h := sha256.Sum256([]byte(fmt.Sprintf("cudalayout/v1/%d/%d/%d/%d/%d/%d", t.rows, t.cols, t.elemBytes, t.stride, t.align, t.swizzle)))
	return hex.EncodeToString(h[:])
}

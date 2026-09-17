package cudashape

import (
	"errors"
	"fmt"
	"math"
	"strings"
)

var (
	// ErrNegativeDimension is returned when a shape dimension is negative.
	ErrNegativeDimension = errors.New("cudashape: negative dimension")

	// ErrShapeOverflow is returned when total elements or byte count overflows uint64 or memory limit.
	ErrShapeOverflow = errors.New("cudashape: shape byte count overflow")

	// ErrInvalidStride is returned when a stride configuration is incompatible with dimensions.
	ErrInvalidStride = errors.New("cudashape: invalid stride")

	// ErrRankMismatch is returned when operations require matching ranks but find differences.
	ErrRankMismatch = errors.New("cudashape: rank mismatch")
)

// MaxByteLimit is the maximum addressable memory limit for device/host allocations (256 TiB, 48-bit address space).
const MaxByteLimit uint64 = 1 << 48

// SafeMul multiplies two uint64 values, returning the product and whether overflow occurred.
func SafeMul(a, b uint64) (uint64, bool) {
	if a == 0 || b == 0 {
		return 0, false
	}
	c := a * b
	if c/a != b {
		return 0, true
	}
	return c, false
}

// Shape represents concrete multi-dimensional tensor extents.
// The zero value of Shape represents a 0-dimensional scalar (rank 0, 1 element).
type Shape struct {
	dims []int
}

// New creates a new checked Shape.
// It returns ErrNegativeDimension if any extent is negative, or ErrShapeOverflow if the
// total element count overflows uint64 or exceeds MaxByteLimit.
func New(dims ...int) (Shape, error) {
	for i, d := range dims {
		if d < 0 {
			return Shape{}, fmt.Errorf("%w: axis %d has extent %d", ErrNegativeDimension, i, d)
		}
	}
	s := Shape{dims: append([]int(nil), dims...)}
	if _, err := s.NumElementsSafe(); err != nil {
		return Shape{}, err
	}
	return s, nil
}

// Dims returns a copy of the shape extents.
func (s Shape) Dims() []int {
	if len(s.dims) == 0 {
		return nil
	}
	return append([]int(nil), s.dims...)
}

// Rank returns the number of dimensions.
func (s Shape) Rank() int {
	return len(s.dims)
}

// Dim returns the extent at axis i. If i is out of bounds, it returns 0.
func (s Shape) Dim(i int) int {
	if i < 0 || i >= len(s.dims) {
		return 0
	}
	return s.dims[i]
}

// NumElements returns total elements as an int64.
// Returns ErrShapeOverflow if the element count overflows int64 or MaxByteLimit.
func (s Shape) NumElements() (int64, error) {
	safe, err := s.NumElementsSafe()
	if err != nil {
		return 0, err
	}
	if safe > math.MaxInt64 {
		return 0, fmt.Errorf("%w: element count %d exceeds int64 max", ErrShapeOverflow, safe)
	}
	return int64(safe), nil
}

// NumElementsSafe returns total elements as uint64 with overflow checking.
func (s Shape) NumElementsSafe() (uint64, error) {
	if len(s.dims) == 0 {
		return 1, nil
	}
	var total uint64 = 1
	for i, d := range s.dims {
		if d == 0 {
			return 0, nil
		}
		var overflow bool
		total, overflow = SafeMul(total, uint64(d))
		if overflow || total > MaxByteLimit {
			return 0, fmt.Errorf("%w: element count overflow at axis %d (extent %d)", ErrShapeOverflow, i, d)
		}
	}
	return total, nil
}

// SizeBytes returns the total memory in bytes needed for the shape given an element size in bytes.
// Returns ErrShapeOverflow if the size exceeds MaxByteLimit or overflows uint64.
func (s Shape) SizeBytes(elementSizeBytes int) (uint64, error) {
	if elementSizeBytes <= 0 {
		return 0, errors.New("cudashape: element size must be positive")
	}
	numElems, err := s.NumElementsSafe()
	if err != nil {
		return 0, err
	}
	if numElems == 0 {
		return 0, nil
	}
	bytes, overflow := SafeMul(numElems, uint64(elementSizeBytes))
	if overflow || bytes > MaxByteLimit {
		return 0, fmt.Errorf("%w: requested %d bytes exceeds limit %d", ErrShapeOverflow, bytes, MaxByteLimit)
	}
	return bytes, nil
}

// Equal reports whether two shapes have identical extents.
func (s Shape) Equal(other Shape) bool {
	if len(s.dims) != len(other.dims) {
		return false
	}
	for i := range s.dims {
		if s.dims[i] != other.dims[i] {
			return false
		}
	}
	return true
}

// String returns a human-readable representation, e.g. "[2, 3, 4]".
func (s Shape) String() string {
	if len(s.dims) == 0 {
		return "[]"
	}
	var b strings.Builder
	b.WriteString("[")
	for i, d := range s.dims {
		if i > 0 {
			b.WriteString(", ")
		}
		fmt.Fprintf(&b, "%d", d)
	}
	b.WriteString("]")
	return b.String()
}

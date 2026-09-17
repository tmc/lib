package cudashape

import (
	"fmt"
)

// Layout describes the memory layout order of tensor elements.
type Layout int

const (
	// LayoutRowMajor denotes standard C contiguous order (last dimension has stride 1).
	LayoutRowMajor Layout = iota
	// LayoutColumnMajor denotes Fortran contiguous order (first dimension has stride 1).
	LayoutColumnMajor
	// LayoutStrided denotes an arbitrary or sliced strided layout.
	LayoutStrided
)

func (l Layout) String() string {
	switch l {
	case LayoutRowMajor:
		return "row-major"
	case LayoutColumnMajor:
		return "column-major"
	case LayoutStrided:
		return "strided"
	default:
		return "unknown"
	}
}

// RowMajorStrides computes standard contiguous row-major strides for the given extents.
func RowMajorStrides(dims []int) ([]int, error) {
	if len(dims) == 0 {
		return nil, nil
	}
	for i, d := range dims {
		if d < 0 {
			return nil, fmt.Errorf("%w: axis %d has extent %d", ErrNegativeDimension, i, d)
		}
	}
	strides := make([]int, len(dims))
	stride := 1
	for i := len(dims) - 1; i >= 0; i-- {
		strides[i] = stride
		d := dims[i]
		if d == 0 {
			continue
		}
		safeProd, overflow := SafeMul(uint64(stride), uint64(d))
		if overflow || safeProd > MaxByteLimit {
			return nil, fmt.Errorf("%w: stride calculation overflow at axis %d", ErrShapeOverflow, i)
		}
		stride = int(safeProd)
	}
	return strides, nil
}

// ColumnMajorStrides computes standard column-major strides for the given extents.
func ColumnMajorStrides(dims []int) ([]int, error) {
	if len(dims) == 0 {
		return nil, nil
	}
	for i, d := range dims {
		if d < 0 {
			return nil, fmt.Errorf("%w: axis %d has extent %d", ErrNegativeDimension, i, d)
		}
	}
	strides := make([]int, len(dims))
	stride := 1
	for i := 0; i < len(dims); i++ {
		strides[i] = stride
		d := dims[i]
		if d == 0 {
			continue
		}
		safeProd, overflow := SafeMul(uint64(stride), uint64(d))
		if overflow || safeProd > MaxByteLimit {
			return nil, fmt.Errorf("%w: stride calculation overflow at axis %d", ErrShapeOverflow, i)
		}
		stride = int(safeProd)
	}
	return strides, nil
}

// IsContiguousRowMajor checks if the given strides match contiguous row-major order for dims.
func IsContiguousRowMajor(dims, strides []int) bool {
	if len(dims) != len(strides) {
		return false
	}
	if len(dims) == 0 {
		return true
	}
	expected, err := RowMajorStrides(dims)
	if err != nil {
		return false
	}
	for i := range expected {
		if strides[i] != expected[i] {
			return false
		}
	}
	return true
}

// IsContiguousColumnMajor checks if the given strides match contiguous column-major order for dims.
func IsContiguousColumnMajor(dims, strides []int) bool {
	if len(dims) != len(strides) {
		return false
	}
	if len(dims) == 0 {
		return true
	}
	expected, err := ColumnMajorStrides(dims)
	if err != nil {
		return false
	}
	for i := range expected {
		if strides[i] != expected[i] {
			return false
		}
	}
	return true
}

// LayoutFromStrides identifies the Layout variant for the given dims and strides.
func LayoutFromStrides(dims, strides []int) Layout {
	if IsContiguousRowMajor(dims, strides) {
		return LayoutRowMajor
	}
	if IsContiguousColumnMajor(dims, strides) {
		return LayoutColumnMajor
	}
	return LayoutStrided
}

// ByteStrides converts element strides to byte strides.
func ByteStrides(elementSizeBytes int, strides []int) ([]int64, error) {
	if elementSizeBytes <= 0 {
		return nil, fmt.Errorf("cudashape: invalid element size %d", elementSizeBytes)
	}
	byteStrides := make([]int64, len(strides))
	for i, s := range strides {
		if s < 0 {
			return nil, fmt.Errorf("%w: negative stride %d at index %d", ErrInvalidStride, s, i)
		}
		bProd, overflow := SafeMul(uint64(s), uint64(elementSizeBytes))
		if overflow || bProd > MaxByteLimit {
			return nil, fmt.Errorf("%w: byte stride overflow at index %d", ErrShapeOverflow, i)
		}
		byteStrides[i] = int64(bProd)
	}
	return byteStrides, nil
}

// ValidateStrides verifies that strides match the dimensions rank and have valid values.
func ValidateStrides(dims, strides []int) error {
	if len(dims) != len(strides) {
		return fmt.Errorf("%w: dims rank %d vs strides rank %d", ErrRankMismatch, len(dims), len(strides))
	}
	for i, s := range strides {
		if s < 0 {
			return fmt.Errorf("%w: stride at axis %d is negative (%d)", ErrInvalidStride, i, s)
		}
	}
	return nil
}

package cudalayout

import (
	"errors"
	"fmt"
	"math"
	"sort"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudashape"
)

var (
	// ErrWrappingPartition indicates a partition shape whose bounds wrap into adjacent dimensions.
	ErrWrappingPartition = errors.New("cudalayout: wrapping partition shape")

	// ErrOverlappingPartition indicates partition strides that permit overlapping tile offsets across blocks.
	ErrOverlappingPartition = errors.New("cudalayout: overlapping partition tiles")

	// ErrInvalidPartition indicates an invalid partition configuration.
	ErrInvalidPartition = errors.New("cudalayout: invalid partition")
)

// Partition wraps a physical tensor layout with an explicit tile partition shape,
// defining how the tensor is mapped to thread blocks in a CUDA launch grid.
type Partition struct {
	ptr            driver.DevicePtr
	tensorShape    cudashape.Shape
	partitionShape cudashape.Shape
	strides        []int
	elemBytes      int
	mutable        bool
}

// PartitionMut represents a mutable partitioned tensor view.
type PartitionMut struct {
	Partition
}

// PartitionOption configures NewPartition.
type PartitionOption func(*Partition) error

// WithPartitionStrides sets explicit element strides for the tensor.
func WithPartitionStrides(strides []int) PartitionOption {
	return func(p *Partition) error {
		if len(strides) != p.tensorShape.Rank() {
			return fmt.Errorf("%w: strides length %d does not match rank %d", ErrInvalidPartition, len(strides), p.tensorShape.Rank())
		}
		for i, s := range strides {
			if s <= 0 {
				return fmt.Errorf("%w: stride at axis %d must be positive (got %d)", ErrInvalidPartition, i, s)
			}
		}
		for i, s := range strides {
			if _, ok := checkedMul(s, p.tensorShape.Dim(i)-1); !ok {
				return overflowPartition("stride extent", i)
			}
		}
		p.strides = append([]int(nil), strides...)
		return nil
	}
}

// NewPartition creates a checked immutable tensor partition.
func NewPartition(ptr driver.DevicePtr, tensorShape, partitionShape cudashape.Shape, elemBytes int, opts ...PartitionOption) (Partition, error) {
	if elemBytes <= 0 {
		return Partition{}, fmt.Errorf("%w: element size must be positive", ErrInvalidPartition)
	}
	tRank := tensorShape.Rank()
	pRank := partitionShape.Rank()
	if tRank == 0 || pRank == 0 {
		return Partition{}, fmt.Errorf("%w: rank must be at least 1", ErrInvalidPartition)
	}
	if tRank != pRank {
		return Partition{}, fmt.Errorf("%w: tensor rank %d does not match partition rank %d", ErrInvalidPartition, tRank, pRank)
	}
	for i := 0; i < tRank; i++ {
		if tensorShape.Dim(i) <= 0 {
			return Partition{}, fmt.Errorf("%w: tensor extent at axis %d must be positive", ErrInvalidPartition, i)
		}
		if partitionShape.Dim(i) <= 0 {
			return Partition{}, fmt.Errorf("%w: partition extent at axis %d must be positive", ErrInvalidPartition, i)
		}
	}

	defaultStrides, err := cudashape.RowMajorStrides(tensorShape.Dims())
	if err != nil {
		return Partition{}, fmt.Errorf("%w: compute row-major strides: %w", ErrInvalidPartition, err)
	}

	p := Partition{
		ptr:            ptr,
		tensorShape:    tensorShape,
		partitionShape: partitionShape,
		strides:        defaultStrides,
		elemBytes:      elemBytes,
		mutable:        false,
	}

	for _, opt := range opts {
		if opt == nil {
			continue
		}
		if err := opt(&p); err != nil {
			return Partition{}, err
		}
	}

	return p, nil
}

// NewPartitionMut creates a checked mutable tensor partition.
func NewPartitionMut(ptr driver.DevicePtr, tensorShape, partitionShape cudashape.Shape, elemBytes int, opts ...PartitionOption) (PartitionMut, error) {
	p, err := NewPartition(ptr, tensorShape, partitionShape, elemBytes, opts...)
	if err != nil {
		return PartitionMut{}, err
	}
	p.mutable = true
	return PartitionMut{Partition: p}, nil
}

// NewPartitionFromTile creates a 2D Partition from an existing Tile and partition dimensions.
func NewPartitionFromTile(ptr driver.DevicePtr, t Tile, partitionRows, partitionCols int) (Partition, error) {
	if t.rows <= 0 || t.cols <= 0 {
		return Partition{}, ErrUninitialized
	}
	if t.Swizzle() != SwizzleNone {
		return Partition{}, fmt.Errorf("%w: swizzled tiles cannot define a partition", ErrInvalidPartition)
	}
	tShape, err := cudashape.New(t.rows, t.cols)
	if err != nil {
		return Partition{}, err
	}
	pShape, err := cudashape.New(partitionRows, partitionCols)
	if err != nil {
		return Partition{}, err
	}
	return NewPartition(ptr, tShape, pShape, t.elemBytes, WithPartitionStrides([]int{t.stride, 1}))
}

// Ptr returns the underlying base device pointer.
func (p Partition) Ptr() driver.DevicePtr {
	return p.ptr
}

// TensorShape returns the full tensor shape.
func (p Partition) TensorShape() cudashape.Shape {
	return p.tensorShape
}

// PartitionShape returns the tile partition shape.
func (p Partition) PartitionShape() cudashape.Shape {
	return p.partitionShape
}

// Strides returns a copy of the tensor strides.
func (p Partition) Strides() []int {
	if len(p.strides) == 0 {
		return nil
	}
	return append([]int(nil), p.strides...)
}

// ElementBytes returns the size of each element in bytes.
func (p Partition) ElementBytes() int {
	return p.elemBytes
}

// TileStride returns the leading axis stride in elements (strides[0]).
func (p Partition) TileStride() int {
	if len(p.strides) == 0 {
		return 0
	}
	return p.strides[0]
}

// TileStride32 returns the leading stride when it fits in int32.
func (p Partition) TileStride32() (int32, error) {
	v := p.TileStride()
	if v < 0 || uint64(v) > math.MaxInt32 {
		return 0, fmt.Errorf("%w: stride %d exceeds int32", ErrInvalidPartition, v)
	}
	return int32(v), nil
}

// PartitionStrides returns the stride in elements between adjacent block tiles along each axis.
func (p Partition) PartitionStrides() []int {
	rank := p.tensorShape.Rank()
	out := make([]int, rank)
	for i := 0; i < rank; i++ {
		out[i] = p.partitionShape.Dim(i) * p.strides[i]
	}
	return out
}

// Grid calculates the CUDA launch grid needed for this partition.
func (p Partition) Grid() (cudashape.Grid3D, error) {
	return cudashape.InferGrid(p.tensorShape, p.partitionShape)
}

// IsMutable reports whether this partition allows write access.
func (p Partition) IsMutable() bool {
	return p.mutable
}

// AsReadOnly returns an immutable copy of this partition view.
func (m PartitionMut) AsReadOnly() Partition {
	p := m.Partition
	p.mutable = false
	return p
}

// ValidateDisjoint verifies that strides guarantee non-overlapping tile
// offsets. Kernels must mask partial tiles. For a partial physical inner axis,
// the next outer stride must contain the padded grid-by-tile envelope so that
// masked lanes cannot wrap into an adjacent outer coordinate.
func (p Partition) ValidateDisjoint() error {
	rank := p.tensorShape.Rank()
	if rank < 1 || rank > 3 || rank != p.partitionShape.Rank() {
		return fmt.Errorf("%w: rank must be between 1 and 3 (got %d)", ErrInvalidPartition, rank)
	}

	for i := 0; i < rank; i++ {
		d := p.tensorShape.Dim(i)
		part := p.partitionShape.Dim(i)
		s := p.strides[i]
		if d <= 0 || part <= 0 || s <= 0 {
			return fmt.Errorf("%w: non-positive dimension or stride at axis %d", ErrInvalidPartition, i)
		}
	}

	grid, err := p.Grid()
	if err != nil {
		return fmt.Errorf("%w: %w", ErrInvalidPartition, err)
	}

	// Kernels mask partial tiles. Order axes by physical stride so any stride
	// permutation is checked consistently. Padded inner tiles must fit before
	// the next physical axis; the outermost axis has no neighbor to wrap into.
	axes := make([]int, rank)
	for i := range axes {
		axes[i] = i
	}
	sort.Slice(axes, func(i, j int) bool { return p.strides[axes[i]] < p.strides[axes[j]] })
	span := 1
	for pos, axis := range axes {
		d, part, stride := p.tensorShape.Dim(axis), p.partitionShape.Dim(axis), p.strides[axis]
		if stride < span {
			return fmt.Errorf("%w: stride %d at axis %d is smaller than occupied span %d", ErrOverlappingPartition, stride, axis, span)
		}
		gridDim := []uint32{grid.X, grid.Y, grid.Z}[axis]
		covered, ok := checkedMul(int(gridDim), part)
		if !ok {
			return overflowPartition("grid tile", axis)
		}
		covered, ok = checkedMul(covered, stride)
		if !ok {
			return overflowPartition("grid tile stride", axis)
		}
		if pos+1 < rank && d%part != 0 {
			outerStride := p.strides[axes[pos+1]]
			if outerStride < covered {
				return fmt.Errorf("%w: partial tile at axis %d wraps into adjacent axis", ErrWrappingPartition, axis)
			}
		}
		term, ok := checkedMul(d-1, stride)
		if !ok {
			return overflowPartition("stride extent", axis)
		}
		span, ok = checkedAdd(span, term)
		if !ok {
			return overflowPartition("occupied span", axis)
		}
	}
	return nil
}

func checkedMul(a, b int) (int, bool) {
	if a < 0 || b < 0 || (a != 0 && b > math.MaxInt/a) {
		return 0, false
	}
	return a * b, true
}
func checkedAdd(a, b int) (int, bool) {
	if b > math.MaxInt-a {
		return 0, false
	}
	return a + b, true
}
func overflowPartition(what string, axis int) error {
	return fmt.Errorf("%w: %w: %s at axis %d", ErrInvalidPartition, cudashape.ErrShapeOverflow, what, axis)
}

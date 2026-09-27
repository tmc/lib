package cudalayout

import (
	"errors"
	"fmt"

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
		return Partition{}, fmt.Errorf("%w: compute row-major strides: %v", ErrInvalidPartition, err)
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

// TileStride returns the leading tile stride in elements (strides[0]).
func (p Partition) TileStride() int {
	if len(p.strides) == 0 {
		return 0
	}
	return p.strides[0]
}

// TileStride32 returns the leading tile stride as an int32.
func (p Partition) TileStride32() int32 {
	return int32(p.TileStride())
}

// KernelArgs returns the default kernel arguments for this partition:
// the base device pointer followed by the tile stride as an int.
func (p Partition) KernelArgs() []any {
	return []any{p.Ptr(), p.TileStride()}
}

// KernelArgs32 returns the kernel arguments for this partition with a 32-bit stride:
// the base device pointer followed by the tile stride as an int32.
func (p Partition) KernelArgs32() []any {
	return []any{p.Ptr(), p.TileStride32()}
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

// ValidateDisjoint verifies that the tile shape evenly or ceiling-covers the
// tensor bounds and that stride calculations guarantee non-overlapping tile offsets per block ID.
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
		return fmt.Errorf("%w: %v", ErrInvalidPartition, err)
	}

	switch rank {
	case 1:
		// 1D partition tiles are disjoint if stride is positive (verified above).
		return nil

	case 2:
		D0, D1 := p.tensorShape.Dim(0), p.tensorShape.Dim(1)
		P0, P1 := p.partitionShape.Dim(0), p.partitionShape.Dim(1)
		S0, S1 := p.strides[0], p.strides[1]

		if S0 >= S1 {
			// Row-major order
			if S0 < P1*S1 {
				return fmt.Errorf("%w: tile column span %d exceeds row stride %d", ErrWrappingPartition, P1*S1, S0)
			}
			if S0 < D1*S1 {
				return fmt.Errorf("%w: row stride %d is smaller than tensor row width %d", ErrOverlappingPartition, S0, D1*S1)
			}
			if D1%P1 != 0 && S0 < int(grid.Y)*P1*S1 {
				return fmt.Errorf("%w: non-divisible column partition (%d not divisible by %d) with packed stride %d wraps into next row", ErrWrappingPartition, D1, P1, S0)
			}
		} else {
			// Column-major order
			if S1 < P0*S0 {
				return fmt.Errorf("%w: tile row span %d exceeds column stride %d", ErrWrappingPartition, P0*S0, S1)
			}
			if S1 < D0*S0 {
				return fmt.Errorf("%w: column stride %d is smaller than tensor column height %d", ErrOverlappingPartition, S1, D0*S0)
			}
			if D0%P0 != 0 && S1 < int(grid.X)*P0*S0 {
				return fmt.Errorf("%w: non-divisible row partition (%d not divisible by %d) with packed stride %d wraps into next column", ErrWrappingPartition, D0, P0, S1)
			}
		}
		return nil

	case 3:
		_, D1, D2 := p.tensorShape.Dim(0), p.tensorShape.Dim(1), p.tensorShape.Dim(2)
		_, P1, P2 := p.partitionShape.Dim(0), p.partitionShape.Dim(1), p.partitionShape.Dim(2)
		S0, S1, S2 := p.strides[0], p.strides[1], p.strides[2]

		// Check Axis 2 vs Axis 1
		if S1 < P2*S2 {
			return fmt.Errorf("%w: tile axis 2 span %d exceeds stride %d", ErrWrappingPartition, P2*S2, S1)
		}
		if S1 < D2*S2 {
			return fmt.Errorf("%w: stride 1 (%d) is smaller than dimension 2 extent %d", ErrOverlappingPartition, S1, D2*S2)
		}
		if D2%P2 != 0 && S1 < int(grid.Z)*P2*S2 {
			return fmt.Errorf("%w: non-divisible axis 2 partition (%d not divisible by %d) wraps into axis 1", ErrWrappingPartition, D2, P2)
		}

		// Check Axis 1 vs Axis 0
		if S0 < P1*S1 {
			return fmt.Errorf("%w: tile axis 1 span %d exceeds stride %d", ErrWrappingPartition, P1*S1, S0)
		}
		if S0 < D1*S1 {
			return fmt.Errorf("%w: stride 0 (%d) is smaller than dimension 1 extent %d", ErrOverlappingPartition, S0, D1*S1)
		}
		if D1%P1 != 0 && S0 < int(grid.Y)*P1*S1 {
			return fmt.Errorf("%w: non-divisible axis 1 partition (%d not divisible by %d) wraps into axis 0", ErrWrappingPartition, D1, P1)
		}
		return nil

	default:
		return fmt.Errorf("%w: unsupported rank %d", ErrInvalidPartition, rank)
	}
}

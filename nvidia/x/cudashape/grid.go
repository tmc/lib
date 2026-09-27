package cudashape

import (
	"fmt"
	"math"
)

// Grid3D represents a 3-dimensional launch grid extent (in blocks).
type Grid3D struct {
	X, Y, Z uint32
}

// String returns a human-readable representation of Grid3D.
func (g Grid3D) String() string {
	return fmt.Sprintf("(%d, %d, %d)", g.X, g.Y, g.Z)
}

// Equal reports whether two launch grids have identical extents.
func (g Grid3D) Equal(other Grid3D) bool {
	return g.X == other.X && g.Y == other.Y && g.Z == other.Z
}

// IsZero reports whether all grid dimensions are zero.
func (g Grid3D) IsZero() bool {
	return g.X == 0 && g.Y == 0 && g.Z == 0
}

// Dims returns the grid dimensions as an array of 3 uint32s.
func (g Grid3D) Dims() [3]uint32 {
	return [3]uint32{g.X, g.Y, g.Z}
}

// InferGrid calculates the multi-dimensional launch grid dimensions needed to
// cover tensorShape using tiles of partitionShape.
//
// It performs ceiling division along each dimension:
//
//	grid[i] = ceil(tensorShape[i] / partitionShape[i])
//
// For rank-1 shapes, Y and Z are set to 1. For rank-2 shapes, Z is set to 1.
// Returns ErrRankMismatch if the ranks differ or are not in the range [1, 3].
// Returns ErrNegativeDimension if any extent is non-positive.
// Returns ErrShapeOverflow if a grid dimension exceeds CUDA's limits: X is at
// most 2^31-1, and Y and Z are at most 65535.
func InferGrid(tensorShape, partitionShape Shape) (Grid3D, error) {
	tRank := tensorShape.Rank()
	pRank := partitionShape.Rank()
	if tRank != pRank {
		return Grid3D{}, fmt.Errorf("%w: tensor rank %d vs partition rank %d", ErrRankMismatch, tRank, pRank)
	}
	if tRank < 1 || tRank > 3 {
		return Grid3D{}, fmt.Errorf("%w: unsupported rank %d (want 1, 2, or 3)", ErrRankMismatch, tRank)
	}

	var dims [3]uint32
	for i := 0; i < tRank; i++ {
		tDim := tensorShape.Dim(i)
		pDim := partitionShape.Dim(i)
		if tDim <= 0 {
			return Grid3D{}, fmt.Errorf("%w: tensor dimension at axis %d is non-positive (%d)", ErrNegativeDimension, i, tDim)
		}
		if pDim <= 0 {
			return Grid3D{}, fmt.Errorf("%w: partition dimension at axis %d is non-positive (%d)", ErrNegativeDimension, i, pDim)
		}

		t := uint64(tDim)
		p := uint64(pDim)
		div := t / p
		if t%p != 0 {
			div++
		}
		limit := uint64(math.MaxUint32)
		if i == 0 {
			limit = 1<<31 - 1
		} else {
			limit = 65535
		}
		if div > limit {
			return Grid3D{}, fmt.Errorf("%w: grid dimension at axis %d (%d) exceeds CUDA limit %d", ErrShapeOverflow, i, div, limit)
		}
		dims[i] = uint32(div)
	}

	grid := Grid3D{X: dims[0], Y: 1, Z: 1}
	if tRank >= 2 {
		grid.Y = dims[1]
	}
	if tRank == 3 {
		grid.Z = dims[2]
	}
	return grid, nil
}

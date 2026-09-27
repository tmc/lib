package cudalaunch

import (
	"errors"
	"fmt"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudalayout"
	"github.com/tmc/lib/nvidia/x/cudashape"
)

var (
	// ErrGridMismatch is returned when inferred partition grids or explicit grids do not match.
	ErrGridMismatch = errors.New("cudalaunch: grid dimension mismatch")
	// ErrClusterGridMismatch is returned when the inferred grid is not divisible by the cluster dimension.
	ErrClusterGridMismatch = errors.New("cudalaunch: grid is not divisible by cluster dimension")
)

// Config defines execution configuration for a kernel launch, including
// grid and block dimensions, dynamic shared memory, and stream.
type Config struct {
	Grid      cudashape.Grid3D
	Block     driver.Dim3
	SharedMem uint32
	Stream    *cuda.Stream
}

// AutoGrid infers and sets the launch grid from one or more partitions.
// It verifies that all partitions agree and sets c.Grid to the inferred grid.
func (c *Config) AutoGrid(partitions ...cudalayout.Partition) error {
	grid, err := InferLaunchGrid(c.Grid, partitions)
	if err != nil {
		return err
	}
	c.Grid = grid
	return nil
}

// NewConfig creates a Config with launch grid inferred from the provided partitions.
func NewConfig(partitions ...cudalayout.Partition) (Config, error) {
	var c Config
	if err := c.AutoGrid(partitions...); err != nil {
		return Config{}, err
	}
	c.Block = driver.Dims(1, 1, 1)
	return c, nil
}

// InferLaunchGrid reconciles an optional caller-specified explicit grid with grids
// inferred from one or more tensor partitions. It returns ErrGridMismatch if partitions
// disagree with each other or if a non-zero explicit grid conflicts with the inferred grid.
func InferLaunchGrid(explicit cudashape.Grid3D, partitions []cudalayout.Partition) (cudashape.Grid3D, error) {
	return inferLaunchGrid(explicit, partitions)
}

func inferLaunchGrid(explicit cudashape.Grid3D, partitions []cudalayout.Partition) (cudashape.Grid3D, error) {
	if len(partitions) == 0 {
		if !explicit.IsZero() {
			return explicit, nil
		}
		return cudashape.Grid3D{X: 1, Y: 1, Z: 1}, nil
	}

	var inferred cudashape.Grid3D
	for i, p := range partitions {
		if err := p.ValidateDisjoint(); err != nil {
			return cudashape.Grid3D{}, fmt.Errorf("cudalaunch: partition %d validation failed: %w", i, err)
		}
		g, err := p.Grid()
		if err != nil {
			return cudashape.Grid3D{}, fmt.Errorf("cudalaunch: partition %d grid inference failed: %w", i, err)
		}
		if i == 0 {
			inferred = g
		} else if !g.Equal(inferred) {
			return cudashape.Grid3D{}, fmt.Errorf("%w: partition 0 grid %v does not match partition %d grid %v", ErrGridMismatch, inferred, i, g)
		}
	}

	if !explicit.IsZero() && !explicit.Equal(inferred) {
		return cudashape.Grid3D{}, fmt.Errorf("%w: explicit grid %v does not match inferred partition grid %v", ErrGridMismatch, explicit, inferred)
	}

	return inferred, nil
}

// BindPartitionArgs returns the pointer, all non-innermost strides in axis
// order, then every partition extent in axis order. The physical innermost
// axis is omitted only when it has unit stride; if no axis has unit stride,
// all strides are included.
func BindPartitionArgs(p cudalayout.Partition) []any {
	return partitionArgs(p)
}

// BindPartitionArgs32 returns the same arguments as BindPartitionArgs using
// int32 strides and extents. It errors if any value does not fit.
func BindPartitionArgs32(p cudalayout.Partition) ([]any, error) {
	return partitionArgs32(p)
}

// BindPartitionMutArgs returns the pointer, non-innermost strides, and tile extents.
func BindPartitionMutArgs(p cudalayout.PartitionMut) []any {
	return partitionArgs(p.Partition)
}

// BindPartitionMutArgs32 returns the same arguments using int32 values and errors
// if any stride or extent does not fit.
func BindPartitionMutArgs32(p cudalayout.PartitionMut) ([]any, error) {
	return partitionArgs32(p.Partition)
}

func partitionArgs(p cudalayout.Partition) []any {
	args := []any{p.Ptr()}
	strides := p.Strides()
	inner := -1
	for i := range strides {
		if strides[i] == 1 && (inner < 0 || strides[i] < strides[inner]) {
			inner = i
		}
	}
	for i, stride := range strides {
		if i != inner {
			args = append(args, stride)
		}
	}
	for i := 0; i < p.PartitionShape().Rank(); i++ {
		args = append(args, p.PartitionShape().Dim(i))
	}
	return args
}

func partitionArgs32(p cudalayout.Partition) ([]any, error) {
	args := []any{p.Ptr()}
	strides := p.Strides()
	inner := -1
	for i := range strides {
		if strides[i] == 1 && (inner < 0 || strides[i] < strides[inner]) {
			inner = i
		}
	}
	for i, stride := range strides {
		if i != inner {
			if uint64(stride) > 1<<31-1 {
				return nil, fmt.Errorf("cudalaunch: stride %d exceeds int32", stride)
			}
			args = append(args, int32(stride))
		}
	}
	for i := 0; i < p.PartitionShape().Rank(); i++ {
		d := p.PartitionShape().Dim(i)
		if uint64(d) > 1<<31-1 {
			return nil, fmt.Errorf("cudalaunch: partition extent %d exceeds int32", d)
		}
		args = append(args, int32(d))
	}
	return args, nil
}

// expandPartitionArgs unpacks any cudalayout.Partition or cudalayout.PartitionMut
// in args into base device pointer and tile stride arguments, while collecting
// all partitions for grid inference.
func expandPartitionArgs(args []any) ([]any, []cudalayout.Partition) {
	var out []any
	var partitions []cudalayout.Partition
	for _, arg := range args {
		switch p := arg.(type) {
		case cudalayout.Partition:
			partitions = append(partitions, p)
			out = append(out, partitionArgs(p)...)
		case cudalayout.PartitionMut:
			partitions = append(partitions, p.Partition)
			out = append(out, partitionArgs(p.Partition)...)
		default:
			out = append(out, arg)
		}
	}
	return out, partitions
}

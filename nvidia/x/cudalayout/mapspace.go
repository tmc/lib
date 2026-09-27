package cudalayout

import (
	"fmt"
	"slices"
)

var (
	// ErrCapacityExceeded indicates a tile factorization exceeds hardware storage limits.
	ErrCapacityExceeded = fmt.Errorf("%w: capacity constraint exceeded", ErrUnsupported)

	// ErrAlignmentFailed indicates a tile factorization violates hardware alignment requirements.
	ErrAlignmentFailed = fmt.Errorf("%w: alignment constraint failed", ErrUnsupported)
)

// StorageLevel identifies a tier in the CUDA memory hierarchy.
type StorageLevel int

const (
	LevelRegister StorageLevel = iota
	LevelShared
	LevelL2
	LevelDRAM
)

func (s StorageLevel) String() string {
	switch s {
	case LevelRegister:
		return "Register"
	case LevelShared:
		return "Shared"
	case LevelL2:
		return "L2"
	case LevelDRAM:
		return "DRAM"
	default:
		return fmt.Sprintf("StorageLevel(%d)", s)
	}
}

// TensorRole classifies tensor arguments in a contraction or loop nest.
type TensorRole int

const (
	TensorA TensorRole = iota // Inputs (e.g. left matrix A)
	TensorB                   // Weights (e.g. right matrix B)
	TensorC                   // Outputs (e.g. accumulator/result matrix C)
)

func (r TensorRole) String() string {
	switch r {
	case TensorA:
		return "TensorA"
	case TensorB:
		return "TensorB"
	case TensorC:
		return "TensorC"
	default:
		return fmt.Sprintf("TensorRole(%d)", r)
	}
}

// IndexFactorization maps problem dimensions to tiling factors across memory hierarchy levels.
type IndexFactorization struct {
	Levels  []StorageLevel
	Factors map[StorageLevel]map[string]int
}

// NewIndexFactorization creates an index factorization with the given levels and factors.
func NewIndexFactorization(levels []StorageLevel, factors map[StorageLevel]map[string]int) IndexFactorization {
	fCopy := make(map[StorageLevel]map[string]int, len(factors))
	for lvl, dims := range factors {
		dCopy := make(map[string]int, len(dims))
		for d, val := range dims {
			dCopy[d] = val
		}
		fCopy[lvl] = dCopy
	}
	return IndexFactorization{
		Levels:  append([]StorageLevel(nil), levels...),
		Factors: fCopy,
	}
}

// TileShape returns the cumulative tile size along each dimension at or below level.
func (f IndexFactorization) TileShape(level StorageLevel) map[string]int {
	tile := make(map[string]int)
	for _, lvl := range f.Levels {
		dims, ok := f.Factors[lvl]
		if !ok {
			continue
		}
		for d, factor := range dims {
			cur, exists := tile[d]
			if !exists {
				cur = 1
			}
			tile[d] = cur * factor
		}
		if lvl == level {
			break
		}
	}
	return tile
}

// Validate checks that all factors are positive and their product equals problemExtents.
func (f IndexFactorization) Validate(problemExtents map[string]int) error {
	products := make(map[string]int, len(problemExtents))
	for _, lvl := range f.Levels {
		dims, ok := f.Factors[lvl]
		if !ok {
			continue
		}
		for d, factor := range dims {
			if factor <= 0 {
				return fmt.Errorf("factor for dimension %s at level %s is non-positive (%d): %w", d, lvl, factor, ErrUnsupported)
			}
			cur, exists := products[d]
			if !exists {
				cur = 1
			}
			products[d] = cur * factor
		}
	}
	for d, want := range problemExtents {
		got, ok := products[d]
		if !ok {
			return fmt.Errorf("dimension %s missing in factorization: %w", d, ErrUnsupported)
		}
		if got != want {
			return fmt.Errorf("factorization product for %s is %d, want %d: %w", d, got, want, ErrUnsupported)
		}
	}
	return nil
}

// LoopPermutation represents loop nest ordering at each storage level.
type LoopPermutation struct {
	Orders map[StorageLevel][]string
}

// NewLoopPermutation creates a loop permutation specification.
func NewLoopPermutation(orders map[StorageLevel][]string) LoopPermutation {
	oCopy := make(map[StorageLevel][]string, len(orders))
	for lvl, ord := range orders {
		oCopy[lvl] = append([]string(nil), ord...)
	}
	return LoopPermutation{Orders: oCopy}
}

// Validate checks that each level's order contains the specified dimensions.
func (p LoopPermutation) Validate(dimensions []string) error {
	for lvl, order := range p.Orders {
		if len(order) != len(dimensions) {
			return fmt.Errorf("loop order at level %s has %d dimensions, want %d: %w", lvl, len(order), len(dimensions), ErrUnsupported)
		}
		seen := make(map[string]bool, len(order))
		for _, d := range order {
			if !slices.Contains(dimensions, d) {
				return fmt.Errorf("unknown dimension %s in loop order at %s: %w", d, lvl, ErrUnsupported)
			}
			if seen[d] {
				return fmt.Errorf("duplicate dimension %s in loop order at %s: %w", d, lvl, ErrUnsupported)
			}
			seen[d] = true
		}
	}
	return nil
}

// SpatialTiling describes distribution of loop iterations across parallel hardware units.
type SpatialTiling struct {
	CTAMesh  map[string]int // Spatial distribution across threadblocks (grid)
	WarpMesh map[string]int // Spatial distribution across warps (block)
}

// NewSpatialTiling constructs a spatial tiling specification.
func NewSpatialTiling(ctaMesh, warpMesh map[string]int) SpatialTiling {
	cCopy := make(map[string]int, len(ctaMesh))
	for k, v := range ctaMesh {
		cCopy[k] = v
	}
	wCopy := make(map[string]int, len(warpMesh))
	for k, v := range warpMesh {
		wCopy[k] = v
	}
	return SpatialTiling{CTAMesh: cCopy, WarpMesh: wCopy}
}

// TotalWarps returns the number of warps in the threadblock spatial tiling.
func (s SpatialTiling) TotalWarps() int {
	total := 1
	for _, count := range s.WarpMesh {
		if count > 0 {
			total *= count
		}
	}
	return total
}

// DatatypeBypass specifies per-level tensor retention versus bypass flags.
type DatatypeBypass struct {
	Bypassed map[StorageLevel]map[TensorRole]bool
}

// NewDatatypeBypass creates an empty or initialized datatype bypass map.
func NewDatatypeBypass() DatatypeBypass {
	return DatatypeBypass{
		Bypassed: make(map[StorageLevel]map[TensorRole]bool),
	}
}

// IsBypassed reports whether the given tensor is bypassed at storage level.
func (b DatatypeBypass) IsBypassed(level StorageLevel, role TensorRole) bool {
	if b.Bypassed == nil {
		return false
	}
	lvl, ok := b.Bypassed[level]
	if !ok {
		return false
	}
	return lvl[role]
}

// SetBypass sets whether the given tensor is bypassed at storage level.
func (b *DatatypeBypass) SetBypass(level StorageLevel, role TensorRole, bypassed bool) {
	if b.Bypassed == nil {
		b.Bypassed = make(map[StorageLevel]map[TensorRole]bool)
	}
	lvl, ok := b.Bypassed[level]
	if !ok {
		lvl = make(map[TensorRole]bool)
		b.Bypassed[level] = lvl
	}
	lvl[role] = bypassed
}

// MapSpace represents a four-axis loop nest decomposition.
type MapSpace struct {
	Problem       map[string]int
	Factorization IndexFactorization
	Permutation   LoopPermutation
	Spatial       SpatialTiling
	Bypass        DatatypeBypass
}

// HardwareConstraints describes hardware limits for pruning.
type HardwareConstraints struct {
	MaxSharedMemoryPerBlock int
	MaxRegistersPerBlock    int
	MaxThreadsPerBlock      int
	TMAAlignmentBytes       int
}

// FootprintBytes computes the memory footprint in bytes at level for the given factorization and bypass.
func FootprintBytes(factors IndexFactorization, bypass DatatypeBypass, dtypes map[TensorRole]int, level StorageLevel) (int, error) {
	tile := factors.TileShape(level)
	b := tile["B"]
	if b <= 0 {
		b = 1
	}
	m := tile["M"]
	if m <= 0 {
		m = 1
	}
	n := tile["N"]
	if n <= 0 {
		n = 1
	}
	k := tile["K"]
	if k <= 0 {
		k = 1
	}

	total := 0
	// TensorA: B x M x K
	if !bypass.IsBypassed(level, TensorA) {
		elemBytes, ok := dtypes[TensorA]
		if !ok || elemBytes <= 0 {
			elemBytes = 2
		}
		total += b * m * k * elemBytes
	}
	// TensorB: B x K x N
	if !bypass.IsBypassed(level, TensorB) {
		elemBytes, ok := dtypes[TensorB]
		if !ok || elemBytes <= 0 {
			elemBytes = 2
		}
		total += b * k * n * elemBytes
	}
	// TensorC: B x M x N
	if !bypass.IsBypassed(level, TensorC) {
		elemBytes, ok := dtypes[TensorC]
		if !ok || elemBytes <= 0 {
			elemBytes = 4
		}
		total += b * m * n * elemBytes
	}
	return total, nil
}

// ValidateCapacity checks tile factorizations against target shared memory,
// register bounds, and TMA alignment constraints.
func ValidateCapacity(factors IndexFactorization, bypass DatatypeBypass, dtypes map[TensorRole]int, limits HardwareConstraints) error {
	// Check shared memory capacity at LevelShared
	if limits.MaxSharedMemoryPerBlock > 0 {
		smem, err := FootprintBytes(factors, bypass, dtypes, LevelShared)
		if err != nil {
			return err
		}
		if smem > limits.MaxSharedMemoryPerBlock {
			return fmt.Errorf("shared memory footprint %d exceeds max %d: %w", smem, limits.MaxSharedMemoryPerBlock, ErrCapacityExceeded)
		}
	}

	// Check register footprint at LevelRegister
	if limits.MaxRegistersPerBlock > 0 {
		regs, err := FootprintBytes(factors, bypass, dtypes, LevelRegister)
		if err != nil {
			return err
		}
		if regs > limits.MaxRegistersPerBlock {
			return fmt.Errorf("register footprint %d exceeds max %d: %w", regs, limits.MaxRegistersPerBlock, ErrCapacityExceeded)
		}
	}

	// Check TMA alignment
	if limits.TMAAlignmentBytes > 0 {
		tile := factors.TileShape(LevelShared)
		m := tile["M"]
		n := tile["N"]
		k := tile["K"]

		if !bypass.IsBypassed(LevelShared, TensorA) {
			elemBytes := dtypes[TensorA]
			if elemBytes <= 0 {
				elemBytes = 2
			}
			rowBytes := k * elemBytes
			if rowBytes%limits.TMAAlignmentBytes != 0 {
				return fmt.Errorf("tensor A row stride %d bytes is not aligned to TMA %d bytes: %w", rowBytes, limits.TMAAlignmentBytes, ErrAlignmentFailed)
			}
		}
		if !bypass.IsBypassed(LevelShared, TensorB) {
			elemBytes := dtypes[TensorB]
			if elemBytes <= 0 {
				elemBytes = 2
			}
			rowBytes := n * elemBytes
			if rowBytes%limits.TMAAlignmentBytes != 0 {
				return fmt.Errorf("tensor B row stride %d bytes is not aligned to TMA %d bytes: %w", rowBytes, limits.TMAAlignmentBytes, ErrAlignmentFailed)
			}
		}
		if !bypass.IsBypassed(LevelShared, TensorC) {
			elemBytes := dtypes[TensorC]
			if elemBytes <= 0 {
				elemBytes = 4
			}
			rowBytes := n * elemBytes
			if rowBytes%limits.TMAAlignmentBytes != 0 {
				return fmt.Errorf("tensor C row stride %d bytes is not aligned to TMA %d bytes: %w", rowBytes, limits.TMAAlignmentBytes, ErrAlignmentFailed)
			}
		}
		_ = m
	}

	return nil
}

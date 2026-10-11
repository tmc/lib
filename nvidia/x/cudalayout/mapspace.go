package cudalayout

import (
	"fmt"
	"math"
	"slices"
)

var (
	// ErrCapacityExceeded indicates a tile factorization exceeds hardware storage limits.
	ErrCapacityExceeded = fmt.Errorf("%w: capacity constraint exceeded", ErrUnsupported)

	// ErrAlignmentFailed indicates a tile factorization violates hardware alignment requirements.
	ErrAlignmentFailed = fmt.Errorf("%w: alignment constraint failed", ErrUnsupported)
)

// StorageLevel identifies a tier in the CUDA memory hierarchy. Mapspace
// factors are multiplied from the first listed level through the selected
// level when computing that level's tile.
type StorageLevel int

const (
	// LevelRegister is the per-thread register tile.
	LevelRegister StorageLevel = iota
	// LevelShared is the block's shared-memory tile.
	LevelShared
	// LevelL2 is the L2-resident tile.
	LevelL2
	// LevelDRAM is the full problem tile in device memory.
	LevelDRAM
)

// String returns the stable name of s.
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
	// TensorA is the left input matrix of a contraction.
	TensorA TensorRole = iota
	// TensorB is the right input matrix of a contraction.
	TensorB
	// TensorC is the output or accumulator matrix of a contraction.
	TensorC
)

// String returns the stable name of r.
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
	found := false
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
			if factor <= 0 || cur > math.MaxInt/factor {
				return nil
			}
			tile[d] = cur * factor
		}
		if lvl == level {
			found = true
			break
		}
	}
	if !found {
		return nil
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
			if _, ok := problemExtents[d]; !ok {
				return fmt.Errorf("unknown dimension %s in factorization: %w", d, ErrUnsupported)
			}
			if factor <= 0 {
				return fmt.Errorf("factor for dimension %s at level %s is non-positive (%d): %w", d, lvl, factor, ErrUnsupported)
			}
			cur, exists := products[d]
			if !exists {
				cur = 1
			}
			if cur > math.MaxInt/factor {
				return fmt.Errorf("factorization product for %s overflows: %w", d, ErrOverflow)
			}
			products[d] = cur * factor
		}
	}
	for d, want := range problemExtents {
		if want <= 0 {
			return fmt.Errorf("problem extent for %s is not positive (%d): %w", d, want, ErrUnsupported)
		}
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

// TotalWarps returns the number of warps in the threadblock spatial tiling,
// or zero for a non-positive factor or product overflow. Use Validate to
// check that mesh dimensions and subdivisions match a factorization.
func (s SpatialTiling) TotalWarps() int {
	total, _ := spatialProduct(s.WarpMesh)
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
	MaxSharedMemoryPerBlock int // bytes per block
	MaxRegistersPerBlock    int // 32-bit registers per block
	MaxThreadsPerBlock      int // threads used to scale the per-thread register tile
	TMAAlignmentBytes       int // row-stride alignment in bytes
}

// FootprintBytes computes the memory footprint in bytes at level for the given
// factorization and bypass. It requires exactly the B, M, N and K dimensions
// and a positive byte size for each tensor role, including bypassed tensors.
func FootprintBytes(factors IndexFactorization, bypass DatatypeBypass, dtypes map[TensorRole]int, level StorageLevel) (int, error) {
	tile := factors.TileShape(level)
	if tile == nil {
		return 0, fmt.Errorf("storage level %s is absent from factorization: %w", level, ErrUnsupported)
	}
	for dim := range tile {
		if dim != "B" && dim != "M" && dim != "N" && dim != "K" {
			return 0, fmt.Errorf("unknown dimension %s in tile: %w", dim, ErrUnsupported)
		}
	}
	for _, dim := range []string{"B", "M", "N", "K"} {
		if tile[dim] <= 0 {
			return 0, fmt.Errorf("missing or non-positive tile extent for %s: %w", dim, ErrUnsupported)
		}
	}
	b, m, n, k := tile["B"], tile["M"], tile["N"], tile["K"]
	for _, role := range []TensorRole{TensorA, TensorB, TensorC} {
		if dtypes[role] <= 0 {
			return 0, fmt.Errorf("missing or non-positive dtype size for %s: %w", role, ErrUnsupported)
		}
	}

	total := 0
	product := func(extents ...int) (int, error) {
		elements := 1
		for _, extent := range extents {
			if elements > math.MaxInt/extent {
				return 0, ErrOverflow
			}
			elements *= extent
		}
		return elements, nil
	}
	add := func(bytes int) error {
		if bytes > math.MaxInt-total {
			return ErrOverflow
		}
		total += bytes
		return nil
	}
	// TensorA: B x M x K
	if !bypass.IsBypassed(level, TensorA) {
		elemBytes := dtypes[TensorA]
		bytes, err := product(b, m, k, elemBytes)
		if err != nil {
			return 0, fmt.Errorf("tensor A footprint: %w", err)
		}
		if err := add(bytes); err != nil {
			return 0, fmt.Errorf("tensor A footprint: %w", err)
		}
	}
	// TensorB: B x K x N
	if !bypass.IsBypassed(level, TensorB) {
		elemBytes := dtypes[TensorB]
		bytes, err := product(b, k, n, elemBytes)
		if err != nil {
			return 0, fmt.Errorf("tensor B footprint: %w", err)
		}
		if err := add(bytes); err != nil {
			return 0, fmt.Errorf("tensor B footprint: %w", err)
		}
	}
	// TensorC: B x M x N
	if !bypass.IsBypassed(level, TensorC) {
		elemBytes := dtypes[TensorC]
		bytes, err := product(b, m, n, elemBytes)
		if err != nil {
			return 0, fmt.Errorf("tensor C footprint: %w", err)
		}
		if err := add(bytes); err != nil {
			return 0, fmt.Errorf("tensor C footprint: %w", err)
		}
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
		if limits.MaxThreadsPerBlock <= 0 {
			return fmt.Errorf("register limit requires MaxThreadsPerBlock: %w", ErrUnsupported)
		}
		registerBytes, err := FootprintBytes(factors, bypass, dtypes, LevelRegister)
		if err != nil {
			return err
		}
		// LevelRegister describes bytes held per thread. CUDA register
		// accounting is in 32-bit registers per block.
		regsPerThread := registerBytes / 4
		if registerBytes%4 != 0 {
			regsPerThread++
		}
		if regsPerThread > math.MaxInt/limits.MaxThreadsPerBlock {
			return ErrOverflow
		}
		regs := regsPerThread * limits.MaxThreadsPerBlock
		if regs > limits.MaxRegistersPerBlock {
			return fmt.Errorf("register count %d registers exceeds max %d registers per block: %w", regs, limits.MaxRegistersPerBlock, ErrCapacityExceeded)
		}
	}

	// Check TMA alignment
	if limits.TMAAlignmentBytes > 0 {
		if _, err := FootprintBytes(factors, bypass, dtypes, LevelShared); err != nil {
			return err
		}
		tile := factors.TileShape(LevelShared)
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
	}

	return nil
}

package cudaroofline

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudalayout"
)

// GEMMSpatialProfile selects a logical owner-major retention model: global L2,
// per-CTA Shared, and per-CTA/warp/lane Register. LaneMesh partitions remaining
// Shared register-tile digits after WarpMesh. Omitted dimensions are one; its
// positive product must not exceed 32. The zero value selects one logical lane.
// These are logical owners, not an SM residency or physical multicast model.
type GEMMSpatialProfile struct {
	LaneMesh map[string]int
}

// SpatialTensorMovement reports bytes at one logical instance. ParentFanout is
// the number of these retained instances per nearest retained parent; it is
// zero for DRAM or bypass. MaxResidentUnionBytes is the maximum distinct data
// retained across owners over the declared event stream, not replicated capacity.
type SpatialTensorMovement struct {
	PerInstance           TensorMovement
	MaxResidentUnionBytes uint64
	ParentFanout          uint64
}

// SpatialHierarchyLevel reports logical storage-owner count and per-role data.
// Instances is independent of bypass; a bypassed role has zero bytes and fanout.
// Per-role union maxima must not be summed as a simultaneous level maximum.
type SpatialHierarchyLevel struct {
	Level     cudalayout.StorageLevel
	Instances uint64
	Tensors   map[cudalayout.TensorRole]SpatialTensorMovement
}

// SpatialHierarchyMovement separates aggregate traffic/replicated capacity in
// Movement from per-instance footprints, union and fanout in Levels.
// ComputeInstances counts logical CTA/warp/lane participants, not active SM units.
type SpatialHierarchyMovement struct {
	Movement         HierarchyMovement
	Levels           []SpatialHierarchyLevel
	ComputeInstances uint64
}

type spatialLoop struct {
	gemmLoop
	owner int // 0 temporal, 1 CTA, 2 warp, 3 lane
}

// AnalyticalGEMMSpatial analyzes the fixed logical profile without enumerating
// MACs or instances. CTA mesh digits divide factors above Shared, lower level
// first using gcd; each original digit is local*part+owner. Lower-level CTA
// owner digits are least significant. At Shared, the register-tile digit is
// ((local*lanes+lane)*warps+warp), per dimension.
//
// Owner coordinates iterate B/M/N/K lexicographically, CTA then warp then lane;
// each participant preserves its supplied temporal order. Retained buffers use
// the same full projected tiles, C initialization/revisit and nearest-parent
// rules as AnalyticalGEMMHierarchy. Identity meshes reproduce that analysis.
// Spatial K reductions are explicitly unmodeled; refusing them does not supply
// a partial-C reduction protocol. Counts imply no physical execution or timing.
func AnalyticalGEMMSpatial(mapping cudalayout.MapSpace, dtypes map[cudalayout.TensorRole]int, profile GEMMSpatialProfile) (SpatialHierarchyMovement, error) {
	return analyticalGEMMSpatial(mapping, dtypes, profile, false)
}

// splitK tags C tiles by their spatial K owner. Only AnalyticalGEMMSplitK
// enables it; its caller must also account for the declared merge phase.
func analyticalGEMMSpatial(mapping cudalayout.MapSpace, dtypes map[cudalayout.TensorRole]int, profile GEMMSpatialProfile, splitK bool) (SpatialHierarchyMovement, error) {
	var result SpatialHierarchyMovement
	temporal, err := validateGEMMHierarchy(mapping, dtypes, true)
	if err != nil {
		return result, err
	}
	loops, counts, err := spatialGEMMLoops(mapping, temporal, profile, splitK)
	if err != nil {
		return result, err
	}
	result.ComputeInstances, err = checkedProduct(counts[0], counts[1], counts[2])
	if err != nil {
		return result, err
	}
	result.Movement, err = gemmFootprints(mapping, dtypes)
	if err != nil {
		return SpatialHierarchyMovement{}, err
	}
	partials := uint64(1)
	if splitK {
		partials, err = spatialKParts(mapping, profile)
		if err != nil {
			return SpatialHierarchyMovement{}, err
		}
		last := len(result.Movement.Levels) - 1
		entry := result.Movement.Levels[last].Tensors[cudalayout.TensorC]
		entry.ResidentBytes, err = checkedProduct(entry.ResidentBytes, partials)
		if err != nil {
			return SpatialHierarchyMovement{}, err
		}
		result.Movement.Levels[last].Tensors[cudalayout.TensorC] = entry
	}
	levels := mapping.Factorization.Levels
	roles := []cudalayout.TensorRole{cudalayout.TensorA, cudalayout.TensorB, cudalayout.TensorC}
	for _, level := range levels {
		instances := uint64(1)
		if level == cudalayout.LevelRegister {
			instances = result.ComputeInstances
		} else if level == cudalayout.LevelShared {
			instances = counts[0]
		}
		result.Levels = append(result.Levels, SpatialHierarchyLevel{Level: level, Instances: instances, Tensors: make(map[cudalayout.TensorRole]SpatialTensorMovement)})
	}
	owned := func(level cudalayout.StorageLevel, owner int) bool {
		return owner != 0 && (level == cudalayout.LevelRegister || (level == cudalayout.LevelShared && owner == 1))
	}
	keyDepends := func(loop spatialLoop, i int, role cudalayout.TensorRole) bool {
		return (loop.level > i && gemmProjects(role, loop.dim)) ||
			(splitK && role == cudalayout.TensorC && loop.owner != 0 && loop.dim == "K")
	}
	for i, level := range levels {
		for _, role := range roles {
			result.Levels[i].Tensors[role] = SpatialTensorMovement{}
			bytes := result.Movement.Levels[i].Tensors[role].ResidentBytes
			if bytes == 0 {
				continue
			}
			// Fixed owner coordinates leave uniform temporal streams. Distinct
			// spatial keys count projected owner digits; temporal keys count
			// projected free digits above this buffer.
			distinct, temporalKeys := uint64(1), uint64(1)
			for _, loop := range loops {
				if !keyDepends(loop, i, role) {
					continue
				}
				if owned(level, loop.owner) {
					distinct, err = checkedProduct(distinct, loop.size)
				} else {
					temporalKeys, err = checkedProduct(temporalKeys, loop.size)
				}
				if err != nil {
					return SpatialHierarchyMovement{}, err
				}
			}
			union := distinct
			if result.Levels[i].Instances > distinct && temporalKeys > 1 {
				union++ // At most one owner advances; older owners retain final keys.
			}
			union, err = checkedProduct(union, bytes)
			if err != nil {
				return SpatialHierarchyMovement{}, err
			}
			entry := SpatialTensorMovement{PerInstance: TensorMovement{ResidentBytes: bytes}, MaxResidentUnionBytes: union}
			for parent := i + 1; parent < len(levels); parent++ {
				if !mapping.Bypass.IsBypassed(levels[parent], role) {
					entry.ParentFanout = result.Levels[i].Instances / result.Levels[parent].Instances
					break
				}
			}
			result.Levels[i].Tensors[role] = entry
		}
	}
	charge := func(child int, instances uint64, role cudalayout.TensorRole, read, write uint64) error {
		for parent := child + 1; parent < len(levels); parent++ {
			if mapping.Bypass.IsBypassed(levels[parent], role) {
				continue
			}
			fanout := instances / result.Levels[parent].Instances
			read, err = checkedProduct(read, fanout)
			if err != nil {
				return err
			}
			write, err = checkedProduct(write, fanout)
			if err != nil {
				return err
			}
			entry := result.Levels[parent].Tensors[role]
			entry.PerInstance.ReadBytes, err = checkedSum(entry.PerInstance.ReadBytes, read)
			if err != nil {
				return err
			}
			entry.PerInstance.WriteBytes, err = checkedSum(entry.PerInstance.WriteBytes, write)
			result.Levels[parent].Tensors[role] = entry
			return err
		}
		return fmt.Errorf("spatial GEMM has no retained parent for %s", role)
	}
	for i, level := range levels[:len(levels)-1] {
		for _, role := range roles {
			bytes := result.Levels[i].Tensors[role].PerInstance.ResidentBytes
			if bytes == 0 {
				continue
			}
			distinct, runs, last := uint64(1), uint64(1), -1
			for j, loop := range loops {
				if !owned(level, loop.owner) && keyDepends(loop, i, role) {
					distinct, err = checkedProduct(distinct, loop.size)
					if err != nil {
						return SpatialHierarchyMovement{}, err
					}
					last = j
				}
			}
			for j := 0; j <= last; j++ {
				if !owned(level, loops[j].owner) {
					runs, err = checkedProduct(runs, loops[j].size)
					if err != nil {
						return SpatialHierarchyMovement{}, err
					}
				}
			}
			read, err := checkedProduct(runs, bytes)
			if err != nil {
				return SpatialHierarchyMovement{}, err
			}
			var write uint64
			if role == cudalayout.TensorC {
				write = read
				read, err = checkedProduct(runs-distinct, bytes)
				if err != nil {
					return SpatialHierarchyMovement{}, err
				}
			}
			if err := charge(i, result.Levels[i].Instances, role, read, write); err != nil {
				return SpatialHierarchyMovement{}, err
			}
		}
	}
	macs, err := checkedProduct(uint64(mapping.Problem["B"]), uint64(mapping.Problem["M"]), uint64(mapping.Problem["N"]), uint64(mapping.Problem["K"]))
	if err != nil {
		return SpatialHierarchyMovement{}, err
	}
	outputs, err := checkedProduct(uint64(mapping.Problem["B"]), uint64(mapping.Problem["M"]), uint64(mapping.Problem["N"]))
	if err != nil {
		return SpatialHierarchyMovement{}, err
	}
	outputs, err = checkedProduct(outputs, partials)
	if err != nil {
		return SpatialHierarchyMovement{}, err
	}
	for _, role := range roles {
		read, err := checkedProduct(macs/result.ComputeInstances, uint64(dtypes[role]))
		if err != nil {
			return SpatialHierarchyMovement{}, err
		}
		var write uint64
		if role == cudalayout.TensorC {
			write = read
			read, err = checkedProduct((macs-outputs)/result.ComputeInstances, uint64(dtypes[role]))
			if err != nil {
				return SpatialHierarchyMovement{}, err
			}
		}
		if err := charge(-1, result.ComputeInstances, role, read, write); err != nil {
			return SpatialHierarchyMovement{}, err
		}
	}
	for i := range result.Levels {
		result.Movement.Levels[i].ResidentBytes = 0
		for _, role := range roles {
			per := result.Levels[i].Tensors[role].PerInstance
			n := result.Levels[i].Instances
			var aggregate TensorMovement
			aggregate.ResidentBytes, err = checkedProduct(per.ResidentBytes, n)
			if err != nil {
				return SpatialHierarchyMovement{}, err
			}
			aggregate.ReadBytes, err = checkedProduct(per.ReadBytes, n)
			if err != nil {
				return SpatialHierarchyMovement{}, err
			}
			aggregate.WriteBytes, err = checkedProduct(per.WriteBytes, n)
			if err != nil {
				return SpatialHierarchyMovement{}, err
			}
			result.Movement.Levels[i].Tensors[role] = aggregate
			result.Movement.Levels[i].ResidentBytes, err = checkedSum(result.Movement.Levels[i].ResidentBytes, aggregate.ResidentBytes)
			if err != nil {
				return SpatialHierarchyMovement{}, err
			}
		}
	}
	dram := &result.Movement.DRAM
	dram.TensorTransfers = make(map[string]uint64)
	for r, role := range roles {
		entry := result.Movement.Levels[len(levels)-1].Tensors[role]
		dram.TensorTransfers[[]string{"A", "B", "C"}[r]], err = checkedSum(entry.ReadBytes, entry.WriteBytes)
		if err != nil {
			return SpatialHierarchyMovement{}, err
		}
		dram.ReadBytes, err = checkedSum(dram.ReadBytes, entry.ReadBytes)
		if err != nil {
			return SpatialHierarchyMovement{}, err
		}
		dram.WriteBytes, err = checkedSum(dram.WriteBytes, entry.WriteBytes)
		if err != nil {
			return SpatialHierarchyMovement{}, err
		}
	}
	dram.TotalBytes, err = checkedSum(dram.ReadBytes, dram.WriteBytes)
	if err != nil {
		return SpatialHierarchyMovement{}, err
	}
	flops, err := checkedProduct(2, macs)
	if err != nil {
		return SpatialHierarchyMovement{}, err
	}
	dram.OperationalIntensity = float64(flops) / float64(dram.TotalBytes)
	return result, nil
}

func spatialGEMMLoops(mapping cudalayout.MapSpace, temporal []gemmLoop, profile GEMMSpatialProfile, splitK bool) ([]spatialLoop, [3]uint64, error) {
	dims := []string{"B", "M", "N", "K"}
	meshes := []map[string]int{mapping.Spatial.CTAMesh, mapping.Spatial.WarpMesh, profile.LaneMesh}
	counts := [3]uint64{1, 1, 1}
	value := func(m map[string]int, dim string) uint64 {
		if n, ok := m[dim]; ok {
			return uint64(n)
		}
		return 1
	}
	for i, mesh := range meshes {
		for dim, n := range mesh {
			if mapping.Problem[dim] <= 0 || n <= 0 {
				return nil, counts, fmt.Errorf("invalid spatial mesh: %w", cudalayout.ErrUnsupported)
			}
			var err error
			counts[i], err = checkedProduct(counts[i], uint64(n))
			if err != nil || counts[i] > uint64(^uint(0)>>1) {
				return nil, counts, cudalayout.ErrOverflow
			}
		}
	}
	if counts[2] > 32 {
		return nil, counts, fmt.Errorf("lane mesh exceeds 32: %w", cudalayout.ErrUnsupported)
	}
	shared, register := mapping.Factorization.TileShape(cudalayout.LevelShared), mapping.Factorization.TileShape(cudalayout.LevelRegister)
	for _, dim := range dims {
		if n := value(meshes[0], dim); n > 1 && (shared[dim] <= 0 || uint64(mapping.Problem[dim]/shared[dim])%n != 0) {
			return nil, counts, fmt.Errorf("CTA mesh does not divide shared tile count: %w", cudalayout.ErrUnsupported)
		}
		wl, err := checkedProduct(value(meshes[1], dim), value(meshes[2], dim))
		if err != nil {
			return nil, counts, err
		}
		if wl > 1 && (register[dim] <= 0 || shared[dim] <= 0 || uint64(shared[dim]/register[dim])%wl != 0) {
			return nil, counts, fmt.Errorf("warp/lane mesh does not divide register tile count: %w", cudalayout.ErrUnsupported)
		}
	}
	for _, mesh := range meshes {
		if !splitK && value(mesh, "K") != 1 {
			return nil, counts, fmt.Errorf("spatial K reduction: %w", cudalayout.ErrUnmodeled)
		}
	}
	levels := mapping.Factorization.Levels
	parts := make([]map[string]uint64, len(levels))
	sharedIndex := -1
	for i, level := range levels {
		parts[i] = make(map[string]uint64)
		if level == cudalayout.LevelShared {
			sharedIndex = i
		}
	}
	for _, dim := range dims {
		remaining := value(meshes[0], dim)
		for i, level := range levels {
			if level <= cudalayout.LevelShared {
				continue
			}
			f := uint64(mapping.Factorization.Factors[level][dim])
			if f == 0 {
				f = 1
			}
			a, b := remaining, f
			for b != 0 {
				a, b = b, a%b
			}
			parts[i][dim] = a
			remaining /= a
		}
		if remaining != 1 {
			return nil, counts, fmt.Errorf("CTA digit partition incomplete: %w", cudalayout.ErrUnsupported)
		}
	}
	var loops []spatialLoop
	add := func(i int, dim string, n uint64, owner int) {
		if n > 1 {
			loops = append(loops, spatialLoop{gemmLoop{i, dim, n}, owner})
		}
	}
	for _, dim := range dims {
		for i := len(levels) - 1; i >= 0; i-- {
			add(i, dim, parts[i][dim], 1)
		}
	}
	for owner := 2; owner <= 3; owner++ {
		for _, dim := range dims {
			add(sharedIndex, dim, value(meshes[owner-1], dim), owner)
		}
	}
	for _, loop := range temporal {
		divisor := parts[loop.level][loop.dim]
		if divisor == 0 {
			divisor = 1
		}
		if levels[loop.level] == cudalayout.LevelShared {
			divisor = value(meshes[1], loop.dim) * value(meshes[2], loop.dim) // checked above
		}
		add(loop.level, loop.dim, loop.size/divisor, 0)
	}
	return loops, counts, nil
}

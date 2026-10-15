package cudaroofline_test

import (
	"errors"
	"reflect"
	"sort"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudalayout"
	"github.com/tmc/lib/nvidia/x/cudaroofline"
)

// spatialTrace walks original scalar coordinates, assigns each operation an
// owner, then stably sorts by that owner. It tracks independent retained caches
// and actual union sets. It never calls production loop/count/footprint helpers.
func spatialTrace(t *testing.T, mapping cudalayout.MapSpace, sizes map[cudalayout.TensorRole]int, profile cudaroofline.GEMMSpatialProfile) cudaroofline.SpatialHierarchyMovement {
	t.Helper()
	levels := mapping.Factorization.Levels
	dims := []string{"B", "M", "N", "K"}
	roles := []cudalayout.TensorRole{cudalayout.TensorA, cudalayout.TensorB, cudalayout.TensorC}
	projection := [3][]int{{0, 1, 3}, {0, 2, 3}, {0, 1, 2}}
	tiles := make([][4]int, len(levels))
	var shared, register [4]int
	current := [4]int{1, 1, 1, 1}
	for i, level := range levels {
		for d, dim := range dims {
			if f := mapping.Factorization.Factors[level][dim]; f > 0 {
				current[d] *= f
			}
		}
		tiles[i] = current
		if level == cudalayout.LevelShared {
			shared = current
		}
		if level == cudalayout.LevelRegister {
			register = current
		}
	}
	value := func(mesh map[string]int, d int) int {
		if n := mesh[dims[d]]; n != 0 {
			return n
		}
		return 1
	}
	parts := make([][4]int, len(levels))
	for d := range dims {
		remaining := value(mapping.Spatial.CTAMesh, d)
		for i, level := range levels {
			parts[i][d] = 1
			if level <= cudalayout.LevelShared {
				continue
			}
			f := mapping.Factorization.Factors[level][dims[d]]
			if f == 0 {
				f = 1
			}
			// Exhaustive tiny divisor selection, separate from production gcd.
			for candidate := remaining; candidate >= 1; candidate-- {
				if remaining%candidate == 0 && f%candidate == 0 {
					parts[i][d] = candidate
					remaining /= candidate
					break
				}
			}
		}
		if remaining != 1 {
			t.Fatal("invalid CTA fixture")
		}
	}
	type operation struct {
		coordinate [4]int
		owner      [12]int
	}
	var operations []operation
	var coordinate [4]int
	type loop struct{ dimension, count, stride int }
	var nest []loop
	for i := len(levels) - 1; i >= 0; i-- {
		for _, dim := range mapping.Permutation.Orders[levels[i]] {
			for d, name := range dims {
				if name == dim {
					f := mapping.Factorization.Factors[levels[i]][dim]
					if f == 0 {
						f = 1
					}
					stride := 1
					if i > 0 {
						stride = tiles[i-1][d]
					}
					nest = append(nest, loop{d, f, stride})
				}
			}
		}
	}
	var walk func(int)
	walk = func(at int) {
		if at < len(nest) {
			l := nest[at]
			old := coordinate[l.dimension]
			for n := 0; n < l.count; n++ {
				coordinate[l.dimension] = old + n*l.stride
				walk(at + 1)
			}
			coordinate[l.dimension] = old
			return
		}
		op := operation{coordinate: coordinate}
		for d := range dims {
			place := 1
			for i, level := range levels {
				if level > cudalayout.LevelShared {
					stride := 1
					if i > 0 {
						stride = tiles[i-1][d]
					}
					f := mapping.Factorization.Factors[level][dims[d]]
					if f == 0 {
						f = 1
					}
					digit := coordinate[d] / stride % f
					op.owner[d] += digit % parts[i][d] * place
					place *= parts[i][d]
				}
			}
			if shared[d] > 0 && register[d] > 0 {
				r := (coordinate[d] % shared[d]) / register[d]
				w := value(mapping.Spatial.WarpMesh, d)
				op.owner[4+d] = r % w
				op.owner[8+d] = (r / w) % value(profile.LaneMesh, d)
			}
		}
		operations = append(operations, op)
	}
	walk(0)
	sort.SliceStable(operations, func(i, j int) bool {
		a, b := operations[i].owner, operations[j].owner
		for k := range a {
			if a[k] != b[k] {
				return a[k] < b[k]
			}
		}
		return false
	})
	ownerAt := func(level cudalayout.StorageLevel, owner [12]int) [12]int {
		if level == cudalayout.LevelRegister {
			return owner
		}
		if level == cudalayout.LevelShared {
			for i := 4; i < 12; i++ {
				owner[i] = 0
			}
			return owner
		}
		return [12]int{}
	}
	type stateKey struct {
		level int
		owner [12]int
		role  int
	}
	type state struct {
		last  [4]int
		valid bool
		seen  map[[4]int]bool
	}
	states := map[stateKey]*state{}
	accesses := map[stateKey]cudaroofline.TensorMovement{}
	owners := make([]map[[12]int]bool, len(levels))
	for i := range owners {
		owners[i] = map[[12]int]bool{}
	}
	compute := map[[12]int]bool{}
	footprints := make([][3]uint64, len(levels))
	peak := make([][3]uint64, len(levels))
	for i, level := range levels {
		for r, role := range roles {
			if !mapping.Bypass.IsBypassed(level, role) {
				bytes := uint64(sizes[role])
				for _, d := range projection[r] {
					bytes *= uint64(tiles[i][d])
				}
				footprints[i][r] = bytes
			}
		}
	}
	charge := func(child, r int, owner [12]int, read, write uint64) {
		for parent := child + 1; parent < len(levels); parent++ {
			if mapping.Bypass.IsBypassed(levels[parent], roles[r]) {
				continue
			}
			key := stateKey{parent, ownerAt(levels[parent], owner), r}
			entry := accesses[key]
			entry.ReadBytes += read
			entry.WriteBytes += write
			accesses[key] = entry
			return
		}
		t.Fatal("no retained parent")
	}
	outputs := map[[3]int]bool{}
	for _, op := range operations {
		compute[op.owner] = true
		for i, level := range levels {
			owners[i][ownerAt(level, op.owner)] = true
		}
		for i, level := range levels[:len(levels)-1] {
			for r, role := range roles {
				if mapping.Bypass.IsBypassed(level, role) {
					continue
				}
				key := stateKey{i, ownerAt(level, op.owner), r}
				s := states[key]
				if s == nil {
					s = &state{seen: map[[4]int]bool{}}
					states[key] = s
				}
				var tileKey [4]int
				for _, d := range projection[r] {
					tileKey[d] = op.coordinate[d] / tiles[i][d]
				}
				if s.valid && s.last == tileKey {
					continue
				}
				bytes := footprints[i][r]
				if r == 2 {
					read := uint64(0)
					if s.seen[tileKey] {
						read = bytes
					}
					charge(i, r, op.owner, read, bytes)
				} else {
					charge(i, r, op.owner, bytes, 0)
				}
				s.last, s.valid, s.seen[tileKey] = tileKey, true, true
			}
		}
		charge(-1, 0, op.owner, uint64(sizes[roles[0]]), 0)
		charge(-1, 1, op.owner, uint64(sizes[roles[1]]), 0)
		out := [3]int{op.coordinate[0], op.coordinate[1], op.coordinate[2]}
		read := uint64(0)
		if outputs[out] {
			read = uint64(sizes[roles[2]])
		}
		charge(-1, 2, op.owner, read, uint64(sizes[roles[2]]))
		outputs[out] = true
		for i := range levels[:len(levels)-1] {
			for r := range roles {
				keys := map[[4]int]bool{}
				for key, s := range states {
					if key.level == i && key.role == r {
						keys[s.last] = true
					}
				}
				bytes := uint64(len(keys)) * footprints[i][r]
				if bytes > peak[i][r] {
					peak[i][r] = bytes
				}
			}
		}
	}
	result := cudaroofline.SpatialHierarchyMovement{ComputeInstances: uint64(len(compute))}
	for i, level := range levels {
		entry := cudaroofline.SpatialHierarchyLevel{Level: level, Instances: uint64(len(owners[i])), Tensors: map[cudalayout.TensorRole]cudaroofline.SpatialTensorMovement{}}
		aggregate := cudaroofline.HierarchyLevel{Level: level, Tensors: map[cudalayout.TensorRole]cudaroofline.TensorMovement{}}
		for r, role := range roles {
			var per cudaroofline.TensorMovement
			first := true
			for owner := range owners[i] {
				a := accesses[stateKey{i, owner, r}]
				a.ResidentBytes = footprints[i][r]
				if !first && a != per {
					t.Fatalf("nonuniform fixture at %v %v: %+v vs %+v", level, role, a, per)
				}
				per = a
				first = false
			}
			union := peak[i][r]
			if level == cudalayout.LevelDRAM {
				union = footprints[i][r]
			}
			fanout := uint64(0)
			if footprints[i][r] > 0 {
				for parent := i + 1; parent < len(levels); parent++ {
					if !mapping.Bypass.IsBypassed(levels[parent], role) {
						fanout = entry.Instances / uint64(len(owners[parent]))
						break
					}
				}
			}
			entry.Tensors[role] = cudaroofline.SpatialTensorMovement{PerInstance: per, MaxResidentUnionBytes: union, ParentFanout: fanout}
			a := cudaroofline.TensorMovement{ResidentBytes: per.ResidentBytes * entry.Instances, ReadBytes: per.ReadBytes * entry.Instances, WriteBytes: per.WriteBytes * entry.Instances}
			aggregate.Tensors[role] = a
			aggregate.ResidentBytes += a.ResidentBytes
		}
		result.Levels = append(result.Levels, entry)
		result.Movement.Levels = append(result.Movement.Levels, aggregate)
	}
	dram := &result.Movement.DRAM
	dram.TensorTransfers = map[string]uint64{}
	for r, role := range roles {
		a := result.Movement.Levels[len(levels)-1].Tensors[role]
		dram.ReadBytes += a.ReadBytes
		dram.WriteBytes += a.WriteBytes
		dram.TensorTransfers[[]string{"A", "B", "C"}[r]] = a.ReadBytes + a.WriteBytes
	}
	dram.TotalBytes = dram.ReadBytes + dram.WriteBytes
	dram.OperationalIntensity = float64(2*len(operations)) / float64(dram.TotalBytes)
	return result
}

func spatialFixture(kind int) cudalayout.MapSpace {
	levels := []cudalayout.StorageLevel{0, 1, 2, 3}
	factors := map[cudalayout.StorageLevel]map[string]int{}
	for _, l := range levels {
		factors[l] = map[string]int{"B": 1, "M": 1, "N": 1, "K": 1}
	}
	factors[1] = map[string]int{"B": 1, "M": 2, "N": 2, "K": 2}
	factors[2]["M"] = 2
	if kind == 1 {
		factors[3]["M"] = 3
	}
	if kind == 2 {
		factors[0]["M"] = 2
		factors[2]["M"] = 1
		factors[2]["N"] = 2
		factors[3]["B"] = 2
	}
	if kind == 3 {
		factors[1]["B"] = 2
		factors[2]["M"] = 1
		factors[2]["B"] = 2
		factors[3]["N"] = 2
	}
	if kind == 4 {
		factors[3]["M"] = 2
	} // Mesh M2 can bind at either L2 or DRAM.
	problem := map[string]int{"B": 1, "M": 1, "N": 1, "K": 1}
	orders := map[cudalayout.StorageLevel][]string{}
	for _, l := range levels {
		orders[l] = []string{"B", "M", "N", "K"}
		for d, f := range factors[l] {
			problem[d] *= f
		}
	}
	return cudalayout.MapSpace{Problem: problem, Factorization: cudalayout.NewIndexFactorization(levels, factors), Permutation: cudalayout.NewLoopPermutation(orders), Bypass: cudalayout.NewDatatypeBypass()}
}

func TestGEMMSpatialTrace(t *testing.T) {
	sizes := map[cudalayout.TensorRole]int{0: 2, 1: 3, 2: 4}
	meshes := []struct{ cta, warp, lane map[string]int }{
		{nil, nil, nil}, {map[string]int{"M": 2}, nil, nil}, {map[string]int{"M": 3}, nil, nil}, {map[string]int{"M": 6}, nil, nil}, {map[string]int{"B": 2}, nil, nil}, {map[string]int{"N": 2}, nil, nil},
		{nil, map[string]int{"M": 2}, nil}, {nil, map[string]int{"N": 2}, nil}, {nil, nil, map[string]int{"M": 2}}, {nil, nil, map[string]int{"N": 2}},
		{nil, map[string]int{"N": 2}, map[string]int{"M": 2}}, {nil, map[string]int{"M": 2}, map[string]int{"N": 2}},
		{map[string]int{"M": 2}, map[string]int{"N": 2}, map[string]int{"M": 2}}, {map[string]int{"M": 3}, map[string]int{"N": 2}, map[string]int{"M": 2}}, {map[string]int{"M": 6}, map[string]int{"N": 2}, map[string]int{"M": 2}},
		{map[string]int{"B": 2}, map[string]int{"N": 2}, map[string]int{"M": 2}}, {map[string]int{"N": 2}, map[string]int{"M": 2}, map[string]int{"N": 2}}, {nil, map[string]int{"B": 2}, map[string]int{"N": 2}},
	}
	orders := [][]string{{"B", "M", "N", "K"}, {"B", "K", "N", "M"}, {"N", "K", "M", "B"}, {"K", "M", "N", "B"}}
	cases := 0
	for kind := 0; kind < 5; kind++ {
		for m, mesh := range meshes {
			mapping := spatialFixture(kind)
			mapping.Spatial = cudalayout.NewSpatialTiling(mesh.cta, mesh.warp)
			profile := cudaroofline.GEMMSpatialProfile{LaneMesh: mesh.lane}
			if err := mapping.Spatial.Validate(mapping.Factorization, mapping.Problem); err != nil {
				continue
			}
			shared, reg := mapping.Factorization.TileShape(1), mapping.Factorization.TileShape(0)
			valid := true
			for d, n := range mesh.lane {
				w := mesh.warp[d]
				if w == 0 {
					w = 1
				}
				if (shared[d]/reg[d])%(w*n) != 0 {
					valid = false
				}
			}
			if !valid {
				continue
			}
			for o, order := range orders {
				for _, mask := range []int{0, 1, 7, 42, 85, 170, 273, 511} {
					for _, l := range mapping.Factorization.Levels {
						mapping.Permutation.Orders[l] = order
					}
					mapping.Bypass = cudalayout.NewDatatypeBypass()
					for i := 0; i < 9; i++ {
						mapping.Bypass.SetBypass(cudalayout.StorageLevel(i/3), cudalayout.TensorRole(i%3), mask&(1<<i) != 0)
					}
					got, err := cudaroofline.AnalyticalGEMMSpatial(mapping, sizes, profile)
					if err != nil {
						t.Fatalf("kind%d mesh%d order%d mask%d: %v", kind, m, o, mask, err)
					}
					want := spatialTrace(t, mapping, sizes, profile)
					if !reflect.DeepEqual(got, want) {
						t.Fatalf("kind%d mesh%d order%d mask%d:\ngot%+v\nwant%+v", kind, m, o, mask, got, want)
					}
					cases++
				}
			}
		}
	}
	if cases < 1000 {
		t.Fatalf("too few varied cases: %d", cases)
	}
	t.Logf("%d independent owner/cache/union cases", cases)
}

func TestGEMMSpatialAllBypasses(t *testing.T) {
	sizes := map[cudalayout.TensorRole]int{0: 2, 1: 3, 2: 4}
	for _, order := range [][]string{{"B", "M", "N", "K"}, {"K", "N", "M", "B"}} {
		for mask := 0; mask < 512; mask++ {
			m := spatialFixture(0)
			m.Spatial = cudalayout.NewSpatialTiling(map[string]int{"M": 2}, map[string]int{"N": 2})
			p := cudaroofline.GEMMSpatialProfile{LaneMesh: map[string]int{"M": 2}}
			for _, l := range m.Factorization.Levels {
				m.Permutation.Orders[l] = order
			}
			for i := 0; i < 9; i++ {
				m.Bypass.SetBypass(cudalayout.StorageLevel(i/3), cudalayout.TensorRole(i%3), mask&(1<<i) != 0)
			}
			got, err := cudaroofline.AnalyticalGEMMSpatial(m, sizes, p)
			if err != nil {
				t.Fatal(err)
			}
			want := spatialTrace(t, m, sizes, p)
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("order%v mask%d: got%+v want%+v", order, mask, got, want)
			}
		}
	}
}

func TestGEMMSpatialScalableCounts(t *testing.T) {
	m := spatialFixture(0)
	m.Factorization.Factors[1] = map[string]int{"B": 1, "M": 2, "N": 2, "K": 1}
	m.Factorization.Factors[2] = map[string]int{"B": 1, "M": 512, "N": 512, "K": 1}
	m.Factorization.Factors[3] = map[string]int{"B": 1, "M": 1024, "N": 1024, "K": 1}
	m.Problem = map[string]int{"B": 1, "M": 1 << 20, "N": 1 << 20, "K": 1}
	m.Spatial = cudalayout.NewSpatialTiling(map[string]int{"M": 1024, "N": 1024}, map[string]int{"N": 2})
	r, err := cudaroofline.AnalyticalGEMMSpatial(m, map[cudalayout.TensorRole]int{0: 1, 1: 1, 2: 1}, cudaroofline.GEMMSpatialProfile{LaneMesh: map[string]int{"M": 2}})
	// 2^20 CTA owners, four warp/lane owners each, and 512^2 remaining
	// outer temporal pairs cause 2^40 L2 C visits. Each tile is 2^20 bytes.
	// There are 2^20 distinct keys, so writes=2^60, reads=2^60-2^40.
	if err != nil || r.ComputeInstances != 1<<22 || r.Movement.DRAM.TensorTransfers["C"] != (1<<61)-(1<<40) {
		t.Fatalf("large logical counts: %+v %v", r, err)
	}
}

func TestGEMMSpatialMissingLevels(t *testing.T) {
	for _, keepRegister := range []bool{false, true} {
		for _, keepL2 := range []bool{false, true} {
			m := spatialFixture(0)
			levels := []cudalayout.StorageLevel{}
			factors := map[cudalayout.StorageLevel]map[string]int{}
			orders := map[cudalayout.StorageLevel][]string{}
			carry := map[string]int{"B": 1, "M": 1, "N": 1, "K": 1}
			for _, l := range m.Factorization.Levels {
				for d, f := range m.Factorization.Factors[l] {
					carry[d] *= f
				}
				if (l == 0 && !keepRegister) || (l == 2 && !keepL2) {
					continue
				}
				levels = append(levels, l)
				factors[l] = carry
				orders[l] = []string{"N", "K", "B", "M"}
				carry = map[string]int{"B": 1, "M": 1, "N": 1, "K": 1}
			}
			m.Factorization = cudalayout.NewIndexFactorization(levels, factors)
			m.Permutation = cudalayout.NewLoopPermutation(orders)
			m.Spatial = cudalayout.NewSpatialTiling(map[string]int{"M": 2}, nil)
			p := cudaroofline.GEMMSpatialProfile{}
			if keepRegister {
				m.Spatial.WarpMesh = map[string]int{"N": 2}
				p.LaneMesh = map[string]int{"M": 2}
			}
			for mask := 0; mask < 8; mask++ {
				m.Bypass = cudalayout.NewDatatypeBypass()
				for _, l := range levels {
					if l != 3 {
						for r := 0; r < 3; r++ {
							m.Bypass.SetBypass(l, cudalayout.TensorRole(r), mask&(1<<r) != 0)
						}
					}
				}
				sizes := map[cudalayout.TensorRole]int{0: 2, 1: 3, 2: 4}
				got, err := cudaroofline.AnalyticalGEMMSpatial(m, sizes, p)
				if err != nil {
					t.Fatal(err)
				}
				want := spatialTrace(t, m, sizes, p)
				if !reflect.DeepEqual(got, want) {
					t.Fatalf("register%v L2%v mask%d: got%+v want%+v", keepRegister, keepL2, mask, got, want)
				}
			}
		}
	}
}

func TestGEMMSpatialIdentitySubsets(t *testing.T) {
	sizes := map[cudalayout.TensorRole]int{0: 2, 1: 3, 2: 4}
	for subset := 0; subset < 8; subset++ {
		for mask := 0; mask < 512; mask++ {
			mapping := spatialFixture(0)
			levels := []cudalayout.StorageLevel{}
			factors := map[cudalayout.StorageLevel]map[string]int{}
			acc := map[string]int{"B": 1, "M": 1, "N": 1, "K": 1}
			orders := map[cudalayout.StorageLevel][]string{}
			for _, l := range mapping.Factorization.Levels {
				for d, f := range mapping.Factorization.Factors[l] {
					acc[d] *= f
				}
				if l == 3 || subset&(1<<l) != 0 {
					levels = append(levels, l)
					factors[l] = acc
					acc = map[string]int{"B": 1, "M": 1, "N": 1, "K": 1}
					orders[l] = []string{"K", "N", "B", "M"}
				}
			}
			mapping.Factorization = cudalayout.NewIndexFactorization(levels, factors)
			mapping.Permutation = cudalayout.NewLoopPermutation(orders)
			for _, l := range levels {
				if l != 3 {
					for r := 0; r < 3; r++ {
						mapping.Bypass.SetBypass(l, cudalayout.TensorRole(r), mask&(1<<(int(l)*3+r)) != 0)
					}
				}
			}
			mapping.Spatial = cudalayout.NewSpatialTiling(map[string]int{"M": 1}, map[string]int{"N": 1})
			old, err := cudaroofline.AnalyticalGEMMHierarchy(mapping, sizes)
			if err != nil {
				t.Fatal(err)
			}
			got, err := cudaroofline.AnalyticalGEMMSpatial(mapping, sizes, cudaroofline.GEMMSpatialProfile{})
			if err != nil || !reflect.DeepEqual(got.Movement, old) {
				t.Fatalf("subset%d mask%d: %+v %v want %+v", subset, mask, got.Movement, err, old)
			}
		}
	}
}

func TestGEMMSpatialRefusals(t *testing.T) {
	for _, tt := range []struct {
		name      string
		change    func(*cudalayout.MapSpace, *cudaroofline.GEMMSpatialProfile, map[cudalayout.TensorRole]int)
		unmodeled bool
	}{
		{"bad lane", func(m *cudalayout.MapSpace, p *cudaroofline.GEMMSpatialProfile, d map[cudalayout.TensorRole]int) {
			p.LaneMesh = map[string]int{"M": 0}
		}, false},
		{"unknown lane", func(m *cudalayout.MapSpace, p *cudaroofline.GEMMSpatialProfile, d map[cudalayout.TensorRole]int) {
			p.LaneMesh = map[string]int{"Q": 1}
		}, false},
		{"lane ratio", func(m *cudalayout.MapSpace, p *cudaroofline.GEMMSpatialProfile, d map[cudalayout.TensorRole]int) {
			p.LaneMesh = map[string]int{"M": 3}
		}, false},
		{"warp lane overlap", func(m *cudalayout.MapSpace, p *cudaroofline.GEMMSpatialProfile, d map[cudalayout.TensorRole]int) {
			m.Spatial.WarpMesh = map[string]int{"M": 2}
			p.LaneMesh = map[string]int{"M": 2}
		}, false},
		{"CTA ratio", func(m *cudalayout.MapSpace, p *cudaroofline.GEMMSpatialProfile, d map[cudalayout.TensorRole]int) {
			m.Spatial.CTAMesh = map[string]int{"M": 3}
		}, false},
		{"warp K", func(m *cudalayout.MapSpace, p *cudaroofline.GEMMSpatialProfile, d map[cudalayout.TensorRole]int) {
			m.Spatial.WarpMesh = map[string]int{"K": 2}
		}, true},
		{"lane K", func(m *cudalayout.MapSpace, p *cudaroofline.GEMMSpatialProfile, d map[cudalayout.TensorRole]int) {
			p.LaneMesh = map[string]int{"K": 2}
		}, true},
		{"CTA K", func(m *cudalayout.MapSpace, p *cudaroofline.GEMMSpatialProfile, d map[cudalayout.TensorRole]int) {
			m.Factorization.Factors[3]["K"] = 2
			m.Problem["K"] *= 2
			m.Spatial.CTAMesh = map[string]int{"K": 2}
		}, true},
		{"extra dtype", func(m *cudalayout.MapSpace, p *cudaroofline.GEMMSpatialProfile, d map[cudalayout.TensorRole]int) {
			d[9] = 1
		}, false},
		{"unknown bypass", func(m *cudalayout.MapSpace, p *cudaroofline.GEMMSpatialProfile, d map[cudalayout.TensorRole]int) {
			m.Bypass.SetBypass(1, 9, true)
		}, false},
		{"DRAM bypass", func(m *cudalayout.MapSpace, p *cudaroofline.GEMMSpatialProfile, d map[cudalayout.TensorRole]int) {
			m.Bypass.SetBypass(3, 2, true)
		}, false},
		{"overflow", func(m *cudalayout.MapSpace, p *cudaroofline.GEMMSpatialProfile, d map[cudalayout.TensorRole]int) {
			d[0] = int(^uint(0) >> 1)
		}, false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			m := spatialFixture(0)
			p := cudaroofline.GEMMSpatialProfile{}
			d := map[cudalayout.TensorRole]int{0: 2, 1: 3, 2: 4}
			tt.change(&m, &p, d)
			got, err := cudaroofline.AnalyticalGEMMSpatial(m, d, p)
			if err == nil || !reflect.DeepEqual(got, cudaroofline.SpatialHierarchyMovement{}) {
				t.Fatalf("expected empty refusal: %+v %v", got, err)
			}
			if tt.unmodeled && !errors.Is(err, cudalayout.ErrUnmodeled) {
				t.Fatalf("not explicitly unmodeled: %v", err)
			}
		})
	}
}

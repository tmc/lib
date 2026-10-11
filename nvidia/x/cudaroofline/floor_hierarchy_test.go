package cudaroofline_test

import (
	"math"
	"reflect"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudalayout"
	"github.com/tmc/lib/nvidia/x/cudaroofline"
)

var hierarchyRoles = []cudalayout.TensorRole{cudalayout.TensorA, cudalayout.TensorB, cudalayout.TensorC}
var hierarchyDims = []string{"B", "M", "N", "K"}

func hierarchyMapping() cudalayout.MapSpace {
	levels := []cudalayout.StorageLevel{cudalayout.LevelRegister, cudalayout.LevelShared, cudalayout.LevelL2, cudalayout.LevelDRAM}
	factors := map[cudalayout.StorageLevel]map[string]int{
		cudalayout.LevelRegister: {"B": 1, "M": 2, "N": 1, "K": 1},
		cudalayout.LevelShared:   {"B": 1, "M": 1, "N": 2, "K": 2},
		cudalayout.LevelL2:       {"B": 1, "M": 2, "N": 1, "K": 1},
		cudalayout.LevelDRAM:     {"B": 2, "M": 1, "N": 1, "K": 2},
	}
	orders := map[cudalayout.StorageLevel][]string{}
	for _, level := range levels {
		orders[level] = []string{"B", "M", "N", "K"}
	}
	return cudalayout.MapSpace{
		Problem:       map[string]int{"B": 2, "M": 4, "N": 2, "K": 4},
		Factorization: cudalayout.NewIndexFactorization(levels, factors),
		Permutation:   cudalayout.NewLoopPermutation(orders),
		Bypass:        cudalayout.NewDatatypeBypass(),
	}
}

// hierarchyTrace executes scalar iterations, tracks the actual projected tile
// keys and first visits, and charges the nearest retained source. It does not
// use the estimator's algebraic run counts or any production footprint helper.
func hierarchyTrace(mapping cudalayout.MapSpace, dtypes map[cudalayout.TensorRole]int) []cudaroofline.HierarchyLevel {
	levels := mapping.Factorization.Levels
	var result []cudaroofline.HierarchyLevel
	tiles := make([][4]int, len(levels))
	projected := [3][]int{{0, 1, 3}, {0, 2, 3}, {0, 1, 2}}
	cumulative := [4]int{1, 1, 1, 1}
	for i, level := range levels {
		for j, dim := range hierarchyDims {
			if factor := mapping.Factorization.Factors[level][dim]; factor != 0 {
				cumulative[j] *= factor
			}
		}
		tiles[i] = cumulative
		entry := cudaroofline.HierarchyLevel{Level: level, Tensors: map[cudalayout.TensorRole]cudaroofline.TensorMovement{}}
		for roleIndex, role := range hierarchyRoles {
			bytes := uint64(0)
			if !mapping.Bypass.IsBypassed(level, role) {
				bytes = uint64(dtypes[role])
				for _, dim := range projected[roleIndex] {
					bytes *= uint64(cumulative[dim])
				}
			}
			entry.Tensors[role] = cudaroofline.TensorMovement{ResidentBytes: bytes}
			entry.ResidentBytes += bytes
		}
		result = append(result, entry)
	}
	charge := func(child int, role cudalayout.TensorRole, read, write uint64) {
		for i := child + 1; i < len(levels); i++ {
			if mapping.Bypass.IsBypassed(levels[i], role) {
				continue
			}
			entry := result[i].Tensors[role]
			entry.ReadBytes += read
			entry.WriteBytes += write
			result[i].Tensors[role] = entry
			return
		}
		panic("invalid test fixture: no source")
	}
	type cache struct {
		last  [4]int
		valid bool
		seen  map[[4]int]bool
	}
	caches := make([][3]cache, len(levels))
	for i := range caches {
		for j := range hierarchyRoles {
			caches[i][j].seen = map[[4]int]bool{}
		}
	}
	type scalarLoop struct{ dim, count, stride int }
	var loops []scalarLoop
	for i := len(levels) - 1; i >= 0; i-- {
		for _, name := range mapping.Permutation.Orders[levels[i]] {
			for dim, expected := range hierarchyDims {
				if name != expected {
					continue
				}
				stride := 1
				if i > 0 {
					stride = tiles[i-1][dim]
				}
				count := mapping.Factorization.Factors[levels[i]][name]
				if count == 0 {
					count = 1
				}
				loops = append(loops, scalarLoop{dim, count, stride})
			}
		}
	}
	var coords [4]int
	outputs := map[[3]int]bool{}
	var walk func(int)
	walk = func(at int) {
		if at < len(loops) {
			loop := loops[at]
			old := coords[loop.dim]
			for n := 0; n < loop.count; n++ {
				coords[loop.dim] = old + n*loop.stride
				walk(at + 1)
			}
			coords[loop.dim] = old
			return
		}
		for i, level := range levels[:len(levels)-1] {
			for roleIndex, role := range hierarchyRoles {
				if mapping.Bypass.IsBypassed(level, role) {
					continue
				}
				var key [4]int
				for _, dim := range projected[roleIndex] {
					key[dim] = coords[dim] / tiles[i][dim]
				}
				state := &caches[i][roleIndex]
				if state.valid && state.last == key {
					continue
				}
				bytes := result[i].Tensors[role].ResidentBytes
				if role == cudalayout.TensorC {
					read := uint64(0)
					if state.seen[key] {
						read = bytes
					}
					charge(i, role, read, bytes)
				} else {
					charge(i, role, bytes, 0)
				}
				state.last, state.valid, state.seen[key] = key, true, true
			}
		}
		charge(-1, cudalayout.TensorA, uint64(dtypes[cudalayout.TensorA]), 0)
		charge(-1, cudalayout.TensorB, uint64(dtypes[cudalayout.TensorB]), 0)
		key := [3]int{coords[0], coords[1], coords[2]}
		read := uint64(0)
		if outputs[key] {
			read = uint64(dtypes[cudalayout.TensorC])
		}
		charge(-1, cudalayout.TensorC, read, uint64(dtypes[cudalayout.TensorC]))
		outputs[key] = true
	}
	walk(0)
	return result
}

func TestAnalyticalGEMMHierarchyTrace(t *testing.T) {
	dtypes := map[cudalayout.TensorRole]int{cudalayout.TensorA: 2, cudalayout.TensorB: 3, cudalayout.TensorC: 4}
	orders := [][]string{{"B", "M", "N", "K"}, {"B", "K", "N", "M"}}
	for direction, order := range orders {
		for mask := 0; mask < 512; mask++ {
			mapping := hierarchyMapping()
			for _, level := range mapping.Factorization.Levels {
				mapping.Permutation.Orders[level] = order
			}
			for level := 0; level < 3; level++ {
				for role := 0; role < 3; role++ {
					mapping.Bypass.SetBypass(cudalayout.StorageLevel(level), hierarchyRoles[role], mask&(1<<(level*3+role)) != 0)
				}
			}
			got, err := cudaroofline.AnalyticalGEMMHierarchy(mapping, dtypes)
			if err != nil {
				t.Fatalf("direction %d mask %d: %v", direction, mask, err)
			}
			want := hierarchyTrace(mapping, dtypes)
			if !reflect.DeepEqual(got.Levels, want) {
				t.Fatalf("direction %d mask %d:\ngot %+v\nwant %+v", direction, mask, got.Levels, want)
			}
			dram := want[len(want)-1]
			var read, write uint64
			for _, role := range hierarchyRoles {
				read += dram.Tensors[role].ReadBytes
				write += dram.Tensors[role].WriteBytes
			}
			if got.DRAM.ReadBytes != read || got.DRAM.WriteBytes != write || got.DRAM.TotalBytes != read+write || got.DRAM.OperationalIntensity != 128/float64(read+write) {
				t.Fatalf("direction %d mask %d DRAM summary: %+v", direction, mask, got.DRAM)
			}
		}
	}
}

func TestAnalyticalGEMMHierarchyOrderAndSpills(t *testing.T) {
	dtypes := map[cudalayout.TensorRole]int{cudalayout.TensorA: 1, cudalayout.TensorB: 1, cudalayout.TensorC: 1}
	mapping := cudalayout.MapSpace{
		Problem: map[string]int{"B": 1, "M": 2, "N": 2, "K": 2},
		Factorization: cudalayout.NewIndexFactorization([]cudalayout.StorageLevel{cudalayout.LevelL2, cudalayout.LevelDRAM}, map[cudalayout.StorageLevel]map[string]int{
			cudalayout.LevelL2: {"B": 1, "M": 1, "N": 1, "K": 1}, cudalayout.LevelDRAM: {"B": 1, "M": 2, "N": 2, "K": 2},
		}),
		Permutation: cudalayout.NewLoopPermutation(map[cudalayout.StorageLevel][]string{cudalayout.LevelDRAM: {"M", "N", "K"}}),
	}
	a, err := cudaroofline.AnalyticalGEMMHierarchy(mapping, dtypes)
	if err != nil {
		t.Fatal(err)
	}
	mapping.Permutation.Orders[cudalayout.LevelDRAM] = []string{"K", "M", "N"}
	b, err := cudaroofline.AnalyticalGEMMHierarchy(mapping, dtypes)
	if err != nil {
		t.Fatal(err)
	}
	if a.DRAM.TensorTransfers["A"] != 8 || a.DRAM.TensorTransfers["B"] != 8 || a.DRAM.TensorTransfers["C"] != 4 ||
		b.DRAM.TensorTransfers["A"] != 4 || b.DRAM.TensorTransfers["B"] != 8 || b.DRAM.TensorTransfers["C"] != 12 {
		t.Fatalf("order/reduction traffic: MNK=%+v KMN=%+v", a.DRAM, b.DRAM)
	}
}

func TestAnalyticalGEMMHierarchyRefusals(t *testing.T) {
	for _, tt := range []struct {
		name   string
		change func(*cudalayout.MapSpace, map[cudalayout.TensorRole]int)
	}{
		{"unknown dimension", func(m *cudalayout.MapSpace, d map[cudalayout.TensorRole]int) { m.Problem["X"] = 1 }},
		{"missing dimension", func(m *cudalayout.MapSpace, d map[cudalayout.TensorRole]int) { delete(m.Problem, "K") }},
		{"zero extent", func(m *cudalayout.MapSpace, d map[cudalayout.TensorRole]int) { m.Problem["K"] = 0 }},
		{"duplicate level", func(m *cudalayout.MapSpace, d map[cudalayout.TensorRole]int) {
			m.Factorization.Levels[1] = cudalayout.LevelRegister
		}},
		{"unknown level", func(m *cudalayout.MapSpace, d map[cudalayout.TensorRole]int) { m.Factorization.Levels[0] = -1 }},
		{"absent DRAM", func(m *cudalayout.MapSpace, d map[cudalayout.TensorRole]int) {
			m.Factorization.Levels = m.Factorization.Levels[:3]
		}},
		{"extra factors", func(m *cudalayout.MapSpace, d map[cudalayout.TensorRole]int) {
			m.Factorization.Factors[99] = map[string]int{}
		}},
		{"missing order", func(m *cudalayout.MapSpace, d map[cudalayout.TensorRole]int) {
			delete(m.Permutation.Orders, cudalayout.LevelShared)
		}},
		{"duplicate order", func(m *cudalayout.MapSpace, d map[cudalayout.TensorRole]int) {
			m.Permutation.Orders[cudalayout.LevelShared] = []string{"N", "K", "K"}
		}},
		{"unknown order", func(m *cudalayout.MapSpace, d map[cudalayout.TensorRole]int) { m.Permutation.Orders[99] = []string{} }},
		{"unknown rank", func(m *cudalayout.MapSpace, d map[cudalayout.TensorRole]int) {
			m.Permutation.Orders[cudalayout.LevelShared] = []string{"N", "K", "X"}
		}},
		{"spatial mesh", func(m *cudalayout.MapSpace, d map[cudalayout.TensorRole]int) {
			m.Spatial.CTAMesh = map[string]int{"M": 2}
		}},
		{"unknown identity mesh", func(m *cudalayout.MapSpace, d map[cudalayout.TensorRole]int) {
			m.Spatial.WarpMesh = map[string]int{"X": 1}
		}},
		{"missing dtype", func(m *cudalayout.MapSpace, d map[cudalayout.TensorRole]int) { delete(d, cudalayout.TensorA) }},
		{"zero dtype", func(m *cudalayout.MapSpace, d map[cudalayout.TensorRole]int) { d[cudalayout.TensorB] = 0 }},
		{"extra dtype", func(m *cudalayout.MapSpace, d map[cudalayout.TensorRole]int) { d[99] = 1 }},
		{"DRAM bypass", func(m *cudalayout.MapSpace, d map[cudalayout.TensorRole]int) {
			m.Bypass.SetBypass(cudalayout.LevelDRAM, cudalayout.TensorA, true)
		}},
		{"unknown bypass level", func(m *cudalayout.MapSpace, d map[cudalayout.TensorRole]int) {
			m.Bypass.SetBypass(99, cudalayout.TensorA, false)
		}},
		{"unknown bypass role", func(m *cudalayout.MapSpace, d map[cudalayout.TensorRole]int) {
			m.Bypass.SetBypass(cudalayout.LevelL2, 99, false)
		}},
		{"footprint overflow", func(m *cudalayout.MapSpace, d map[cudalayout.TensorRole]int) {
			m.Problem["M"] = math.MaxInt
			m.Problem["K"] = math.MaxInt
			m.Factorization = cudalayout.NewIndexFactorization([]cudalayout.StorageLevel{cudalayout.LevelDRAM}, map[cudalayout.StorageLevel]map[string]int{cudalayout.LevelDRAM: {"B": 2, "M": math.MaxInt, "N": 2, "K": math.MaxInt}})
			m.Permutation = cudalayout.NewLoopPermutation(map[cudalayout.StorageLevel][]string{cudalayout.LevelDRAM: {"B", "M", "N", "K"}})
		}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			m := hierarchyMapping()
			d := map[cudalayout.TensorRole]int{cudalayout.TensorA: 2, cudalayout.TensorB: 2, cudalayout.TensorC: 4}
			tt.change(&m, d)
			if _, err := cudaroofline.AnalyticalGEMMHierarchy(m, d); err == nil {
				t.Fatal("invalid mapping admitted")
			}
		})
	}
}

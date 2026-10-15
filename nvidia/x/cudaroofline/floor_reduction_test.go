package cudaroofline_test

import (
	"errors"
	"fmt"
	"maps"
	"math/bits"
	"reflect"
	"slices"
	"sort"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudalayout"
	"github.com/tmc/lib/nvidia/x/cudaroofline"
)

type reductionCell [4]int
type reductionBuffer struct {
	level cudalayout.StorageLevel
	role  cudalayout.TensorRole
	owner [12]int
}
type reductionTile struct {
	key   reductionCell
	cells map[reductionCell]bool
}

// reductionTrace enumerates original coordinates, assigns prime tokens to
// original digits, and sorts by owner plus original temporal rank. Payloads
// are actual element sets. C writes happen on eviction/termination. Neither
// production loop splitting nor visit/union/fanout/tree count formulas are used.
func reductionTrace(t *testing.T, m cudalayout.MapSpace, widths map[cudalayout.TensorRole]int, p cudaroofline.GEMMSpatialProfile) cudaroofline.GEMMSplitKMovement {
	t.Helper()
	dims := []string{"B", "M", "N", "K"}
	levels := m.Factorization.Levels
	mesh := []map[string]int{m.Spatial.CTAMesh, m.Spatial.WarpMesh, p.LaneMesh}
	value := func(x map[string]int, d string) int {
		if n, ok := x[d]; ok {
			return n
		}
		return 1
	}
	parts := map[cudalayout.StorageLevel][4]int{}
	strides := map[cudalayout.StorageLevel][4]int{}
	tiles := map[cudalayout.StorageLevel][4]int{}
	cumulative := [4]int{1, 1, 1, 1}
	for _, l := range levels {
		parts[l] = [4]int{1, 1, 1, 1}
		strides[l] = cumulative
		for d, dim := range dims {
			cumulative[d] *= value(m.Factorization.Factors[l], dim)
		}
		tiles[l] = cumulative
	}
	for d, dim := range dims {
		remaining := value(mesh[0], dim)
		for prime := 2; remaining > 1; prime++ {
			for remaining%prime == 0 {
				found := false
				for _, l := range levels {
					a := parts[l]
					if l > cudalayout.LevelShared && value(m.Factorization.Factors[l], dim)/a[d]%prime == 0 {
						a[d] *= prime
						parts[l] = a
						remaining /= prime
						found = true
						break
					}
				}
				if !found {
					t.Fatal("invalid prime-token fixture")
				}
			}
		}
	}
	type operation struct {
		c     [4]int
		owner [12]int
		rank  []int
		q     int
	}
	var ops []operation
	for b := 0; b < m.Problem["B"]; b++ {
		for mm := 0; mm < m.Problem["M"]; mm++ {
			for n := 0; n < m.Problem["N"]; n++ {
				for k := 0; k < m.Problem["K"]; k++ {
					o := operation{c: [4]int{b, mm, n, k}}
					digits := map[cudalayout.StorageLevel][4]int{}
					for _, l := range levels {
						var x [4]int
						for d, dim := range dims {
							x[d] = o.c[d] / strides[l][d] % value(m.Factorization.Factors[l], dim)
						}
						digits[l] = x
					}
					for d, dim := range dims {
						place := 1
						for _, l := range levels {
							if l > cudalayout.LevelShared {
								o.owner[d] += digits[l][d] % parts[l][d] * place
								place *= parts[l][d]
							}
						}
						if slices.Contains(levels, cudalayout.LevelShared) && slices.Contains(levels, cudalayout.LevelRegister) {
							x := digits[cudalayout.LevelShared][d]
							w := value(mesh[1], dim)
							o.owner[4+d] = x % w
							o.owner[8+d] = x / w % value(mesh[2], dim)
						}
					}
					for i := len(levels) - 1; i >= 0; i-- {
						for _, dim := range m.Permutation.Orders[levels[i]] {
							o.rank = append(o.rank, digits[levels[i]][slices.Index(dims, dim)])
						}
					}
					o.q = (o.owner[3]*value(mesh[1], "K")+o.owner[7])*value(mesh[2], "K") + o.owner[11]
					ops = append(ops, o)
				}
			}
		}
	}
	if len(ops) > 512 {
		t.Fatal("oracle exceeds tiny bound")
	}
	sort.Slice(ops, func(i, j int) bool {
		if c := slices.Compare(ops[i].owner[:], ops[j].owner[:]); c != 0 {
			return c < 0
		}
		return slices.Compare(ops[i].rank, ops[j].rank) < 0
	})
	partials := value(mesh[0], "K") * value(mesh[1], "K") * value(mesh[2], "K")
	owner := func(l cudalayout.StorageLevel, o [12]int) [12]int {
		if l == cudalayout.LevelRegister {
			return o
		}
		var x [12]int
		if l == cudalayout.LevelShared {
			copy(x[:4], o[:4])
		}
		return x
	}
	owners := map[cudalayout.StorageLevel]map[[12]int]bool{}
	for _, l := range levels {
		owners[l] = map[[12]int]bool{}
		for _, o := range ops {
			owners[l][owner(l, o.owner)] = true
		}
	}
	parent := func(l cudalayout.StorageLevel, r cudalayout.TensorRole) cudalayout.StorageLevel {
		for _, p := range levels {
			if p > l && !m.Bypass.IsBypassed(p, r) {
				return p
			}
		}
		t.Fatal("no parent")
		return 0
	}
	projection := [3][]int{{0, 1, 3}, {0, 3, 2}, {0, 1, 2}}
	tileCache := map[reductionBuffer]map[reductionCell]reductionTile{}
	payload := func(l cudalayout.StorageLevel, r cudalayout.TensorRole, c [4]int, q int) reductionTile {
		var key reductionCell
		for i, d := range projection[r] {
			key[i] = c[d] / tiles[l][d] * tiles[l][d]
		}
		if r == 2 {
			key = reductionCell{q, key[0], key[1], key[2]}
			if l == cudalayout.LevelDRAM {
				key[0] = 0
			}
		}
		index := reductionBuffer{level: l, role: r}
		if tileCache[index] == nil {
			tileCache[index] = map[reductionCell]reductionTile{}
		}
		if old, ok := tileCache[index][key]; ok {
			return old
		}
		x := reductionTile{key: key, cells: map[reductionCell]bool{}}
		base := [3]int{}
		for i, d := range projection[r] {
			base[i] = c[d] / tiles[l][d] * tiles[l][d]
		}
		for a := base[0]; a < base[0]+tiles[l][projection[r][0]]; a++ {
			for b := base[1]; b < base[1]+tiles[l][projection[r][1]]; b++ {
				for c := base[2]; c < base[2]+tiles[l][projection[r][2]]; c++ {
					if r != 2 {
						x.cells[reductionCell{a, b, c, 0}] = true
						continue
					}
					if l == cudalayout.LevelDRAM {
						for tag := 0; tag < partials; tag++ {
							x.cells[reductionCell{tag, a, b, c}] = true
						}
					} else {
						x.cells[reductionCell{q, a, b, c}] = true
					}
				}
			}
		}
		tileCache[index][key] = x
		return x
	}
	current := map[reductionBuffer]reductionTile{}
	seen := map[reductionBuffer]map[reductionCell]bool{}
	reads, writes := map[reductionBuffer]uint64{}, map[reductionBuffer]uint64{}
	peak := map[[2]int]uint64{}
	fan := map[reductionBuffer]map[[12]int]bool{}
	charge := func(l cudalayout.StorageLevel, r cudalayout.TensorRole, o [12]int, read, write uint64) {
		p := parent(l, r)
		k := reductionBuffer{p, r, owner(p, o)}
		reads[k] += read
		writes[k] += write
	}
	scalarSeen := map[reductionCell]bool{}
	values := make([]int64, partials*m.Problem["B"]*m.Problem["M"]*m.Problem["N"])
	outputs := len(values) / partials
	reference := make([]int64, outputs)
	for _, o := range ops {
		for _, l := range levels {
			for r := cudalayout.TensorA; r <= cudalayout.TensorC; r++ {
				if m.Bypass.IsBypassed(l, r) {
					continue
				}
				k := reductionBuffer{l, r, owner(l, o.owner)}
				tile := payload(l, r, o.c, o.q)
				if l != cudalayout.LevelDRAM {
					p := parent(l, r)
					group := reductionBuffer{l, r, owner(p, o.owner)}
					if fan[group] == nil {
						fan[group] = map[[12]int]bool{}
					}
					fan[group][owner(l, o.owner)] = true
					old, initialized := current[k]
					if !initialized || old.key != tile.key {
						if r == cudalayout.TensorC {
							if initialized {
								charge(l, r, o.owner, 0, uint64(len(old.cells)*widths[r]))
							}
							if seen[k][tile.key] {
								charge(l, r, o.owner, uint64(len(tile.cells)*widths[r]), 0)
							}
						} else {
							charge(l, r, o.owner, uint64(len(tile.cells)*widths[r]), 0)
						}
						if seen[k] == nil {
							seen[k] = map[reductionCell]bool{}
						}
						seen[k][tile.key] = true
					}
				}
				current[k] = tile
			}
		}
		output := reductionCell{o.q, o.c[0], o.c[1], o.c[2]}
		for r := cudalayout.TensorA; r <= cudalayout.TensorC; r++ {
			read, write := uint64(widths[r]), uint64(0)
			if r == cudalayout.TensorC {
				write = read
				if !scalarSeen[output] {
					read = 0
				}
			}
			charge(-1, r, o.owner, read, write)
		}
		scalarSeen[output] = true
		for _, l := range levels {
			for r := 0; r < 3; r++ {
				union := map[reductionCell]bool{}
				for own := range owners[l] {
					for c := range current[reductionBuffer{l, cudalayout.TensorRole(r), own}].cells {
						union[c] = true
					}
				}
				index := [2]int{int(l), r}
				peak[index] = max(peak[index], uint64(len(union)*widths[cudalayout.TensorRole(r)]))
			}
		}
		p := (o.c[0]*m.Problem["M"]+o.c[1])*m.Problem["N"] + o.c[2]
		a, b := int64((o.c[0]+2*o.c[1]+3*o.c[3])%7-3), int64((o.c[0]+5*o.c[3]+o.c[2])%5-2)
		values[o.q*outputs+p] += a * b
		reference[p] += a * b
	}
	for k, tile := range current {
		if k.role == cudalayout.TensorC && k.level != cudalayout.LevelDRAM {
			for _, o := range ops {
				if owner(k.level, o.owner) == k.owner {
					charge(k.level, k.role, o.owner, 0, uint64(len(tile.cells)*widths[k.role]))
					break
				}
			}
		}
	}
	result := cudaroofline.GEMMSplitKMovement{Partials: uint64(partials), OutputBytes: uint64(outputs * widths[2]), ExecutedFLOPs: 2 * uint64(len(ops))}
	allOwners := map[[12]int]bool{}
	for _, o := range ops {
		allOwners[o.owner] = true
	}
	result.GEMM.ComputeInstances = uint64(len(allOwners))
	for _, l := range levels {
		agg := cudaroofline.HierarchyLevel{Level: l, Tensors: map[cudalayout.TensorRole]cudaroofline.TensorMovement{}}
		sp := cudaroofline.SpatialHierarchyLevel{Level: l, Instances: uint64(len(owners[l])), Tensors: map[cudalayout.TensorRole]cudaroofline.SpatialTensorMovement{}}
		for r := cudalayout.TensorA; r <= cudalayout.TensorC; r++ {
			var first *cudaroofline.TensorMovement
			var total cudaroofline.TensorMovement
			for own := range owners[l] {
				k := reductionBuffer{l, r, own}
				v := cudaroofline.TensorMovement{ResidentBytes: uint64(len(current[k].cells) * widths[r]), ReadBytes: reads[k], WriteBytes: writes[k]}
				if first == nil {
					first = &v
				} else if *first != v {
					t.Fatalf("nonuniform oracle %s %s: %+v %+v", l, r, *first, v)
				}
				total.ResidentBytes += v.ResidentBytes
				total.ReadBytes += v.ReadBytes
				total.WriteBytes += v.WriteBytes
			}
			var fanout uint64
			for k, group := range fan {
				if k.level == l && k.role == r {
					if fanout != 0 && fanout != uint64(len(group)) {
						t.Fatal("nonuniform fanout")
					}
					fanout = uint64(len(group))
				}
			}
			agg.Tensors[r] = total
			agg.ResidentBytes += total.ResidentBytes
			sp.Tensors[r] = cudaroofline.SpatialTensorMovement{PerInstance: *first, MaxResidentUnionBytes: peak[[2]int{int(l), int(r)}], ParentFanout: fanout}
		}
		result.GEMM.Movement.Levels = append(result.GEMM.Movement.Levels, agg)
		result.GEMM.Levels = append(result.GEMM.Levels, sp)
	}
	dram := cudaroofline.CompoundDataMovement{TensorTransfers: map[string]uint64{}}
	for r := 0; r < 3; r++ {
		v := result.GEMM.Movement.Levels[len(levels)-1].Tensors[cudalayout.TensorRole(r)]
		dram.ReadBytes += v.ReadBytes
		dram.WriteBytes += v.WriteBytes
		dram.TensorTransfers[[]string{"A", "B", "C"}[r]] = v.ReadBytes + v.WriteBytes
	}
	dram.TotalBytes = dram.ReadBytes + dram.WriteBytes
	dram.OperationalIntensity = float64(2*len(ops)) / float64(dram.TotalBytes)
	result.GEMM.Movement.DRAM = dram
	result.DRAM = dram
	result.DRAM.TensorTransfers = maps.Clone(dram.TensorTransfers)
	if partials == 1 {
		if !slices.Equal(values, reference) {
			t.Fatal("identity numeric mismatch")
		}
		return result
	}
	// Materialize both backing arrays and every tree event; no recurrence count
	// formula supplies the expected workspace, odd copies, additions or bytes.
	pool1 := slices.Clone(values)
	var pool2 []int64
	if partials > 2 {
		for j := 0; j < partials; j += 2 {
			pool2 = append(pool2, make([]int64, outputs)...)
		}
	}
	result.WorkspaceBytes = uint64((len(pool1) + len(pool2)) * widths[2])
	result.MergeRegisterBytes = uint64(2 * widths[2])
	type node struct{ round, q, p int }
	live := map[node]bool{}
	for q := 0; q < partials; q++ {
		for p := 0; p < outputs; p++ {
			live[node{0, q, p}] = true
		}
	}
	result.PeakLivePartialBytes = uint64(len(live) * widths[2])
	input := pool1
	final := make([]int64, outputs)
	for n, roundIndex := partials, 0; n > 1; roundIndex++ {
		var groups [][2]int
		for j := 0; j < n; j += 2 {
			right := j + 1
			if right == n {
				right = -1
			}
			groups = append(groups, [2]int{j, right})
		}
		output := pool2
		if roundIndex%2 == 1 {
			output = pool1
		}
		if len(groups) == 1 {
			output = final
		}
		round := cudaroofline.GEMMReductionRound{InputPartials: uint64(n), OutputPartials: uint64(len(groups))}
		newLive := map[node]bool{}
		for p := 0; p < outputs; p++ {
			for j, g := range groups {
				v := input[g[0]*outputs+p]
				round.ReadBytes += uint64(widths[2])
				if g[1] >= 0 {
					v += input[g[1]*outputs+p]
					round.ReadBytes += uint64(widths[2])
					round.Additions++
				}
				output[j*outputs+p] = v
				round.WriteBytes += uint64(widths[2])
				if len(groups) > 1 {
					id := node{roundIndex + 1, j, p}
					newLive[id] = true
					live[id] = true
					result.PeakLivePartialBytes = max(result.PeakLivePartialBytes, uint64(len(live)*widths[2]))
				}
			}
		}
		result.Rounds = append(result.Rounds, round)
		result.ReductionAdditions += round.Additions
		result.ExecutedFLOPs += round.Additions
		result.DRAM.ReadBytes += round.ReadBytes
		result.DRAM.WriteBytes += round.WriteBytes
		result.DRAM.TensorTransfers["C"] += round.ReadBytes + round.WriteBytes
		input = output
		live = newLive
		n = len(groups)
	}
	if !slices.Equal(final, reference) {
		t.Fatalf("numeric tree mismatch: %v != %v", final, reference)
	}
	result.DRAM.TotalBytes = result.DRAM.ReadBytes + result.DRAM.WriteBytes
	result.DRAM.OperationalIntensity = float64(2*len(ops)) / float64(result.DRAM.TotalBytes)
	return result
}

func reductionMapping(factors [4][4]int, levels []cudalayout.StorageLevel, order string, mask int) cudalayout.MapSpace {
	if levels == nil {
		levels = []cudalayout.StorageLevel{0, 1, 2, 3}
	}
	m := cudalayout.MapSpace{Problem: map[string]int{"B": 1, "M": 1, "N": 1, "K": 1}, Bypass: cudalayout.NewDatatypeBypass()}
	fs := map[cudalayout.StorageLevel]map[string]int{}
	orders := map[cudalayout.StorageLevel][]string{}
	for _, l := range levels {
		fs[l] = map[string]int{}
		for d, dim := range []string{"B", "M", "N", "K"} {
			fs[l][dim] = factors[l][d]
			m.Problem[dim] *= factors[l][d]
			orders[l] = append(orders[l], string(order[(d+int(l))%4]))
		}
		for r := cudalayout.TensorA; r <= cudalayout.TensorC; r++ {
			if l != cudalayout.LevelDRAM && mask&(1<<(3*int(l)+int(r))) != 0 {
				m.Bypass.SetBypass(l, r, true)
			}
		}
	}
	m.Factorization = cudalayout.NewIndexFactorization(levels, fs)
	m.Permutation = cudalayout.NewLoopPermutation(orders)
	return m
}

func TestGEMMSplitKTrace(t *testing.T) {
	type fixture struct {
		name            string
		factors         [4][4]int
		levels          []cudalayout.StorageLevel
		cta, warp, lane map[string]int
	}
	f1 := [4][4]int{{1, 1, 1, 1}, {1, 2, 1, 6}, {1, 1, 2, 2}, {1, 1, 1, 2}}
	f2 := [4][4]int{{1, 2, 1, 1}, {2, 1, 2, 5}, {2, 1, 1, 1}, {1, 1, 1, 1}}
	f3 := [4][4]int{{1, 1, 1, 1}, {1, 2, 2, 1}, {1, 1, 1, 3}, {1, 1, 1, 2}}
	cases := []fixture{
		{"identity", f1, nil, nil, nil, nil},
		{"CTA-common-prime", f1, nil, map[string]int{"K": 2}, nil, nil},
		{"CTA-carried", f1, nil, map[string]int{"K": 4}, nil, nil},
		{"warp", f1, nil, nil, map[string]int{"K": 3}, nil},
		{"lane", f1, nil, nil, nil, map[string]int{"K": 2}},
		{"mixed", f1, nil, map[string]int{"K": 2}, map[string]int{"K": 3}, map[string]int{"K": 2}},
		{"non-K-owners", f1, nil, map[string]int{"K": 2, "N": 2}, map[string]int{"M": 2}, map[string]int{"K": 3}},
		{"batch-five", f2, nil, map[string]int{"B": 2}, map[string]int{"B": 2}, map[string]int{"K": 5}},
		{"CTA-three", f3, nil, map[string]int{"K": 3}, nil, nil},
		{"CTA-six", f3, nil, map[string]int{"K": 6}, map[string]int{"N": 2}, map[string]int{"M": 2}},
		{"no-register", f3, []cudalayout.StorageLevel{1, 2, 3}, map[string]int{"K": 3}, nil, nil},
		{"no-L2", [4][4]int{{1, 1, 1, 1}, {1, 2, 2, 1}, {1, 1, 1, 1}, {1, 1, 1, 6}}, []cudalayout.StorageLevel{0, 1, 3}, map[string]int{"K": 3}, map[string]int{"N": 2}, map[string]int{"M": 2}},
	}
	widths := map[cudalayout.TensorRole]int{0: 2, 1: 3, 2: 5}
	count := 0
	for _, c := range cases {
		for _, order := range []string{"BMNK", "KNMB", "NMKB", "KMNB"} {
			for _, mask := range []int{0, 4, 32, 256, 292, 85, 170, 511} {
				t.Run(fmt.Sprintf("%s/%s/%d", c.name, order, mask), func(t *testing.T) {
					m := reductionMapping(c.factors, c.levels, order, mask)
					m.Spatial = cudalayout.NewSpatialTiling(c.cta, c.warp)
					p := cudaroofline.GEMMSpatialProfile{LaneMesh: c.lane}
					want := reductionTrace(t, m, widths, p)
					got, err := cudaroofline.AnalyticalGEMMSplitK(m, widths, p)
					if err != nil {
						t.Fatal(err)
					}
					if !reflect.DeepEqual(got, want) {
						t.Fatalf("trace mismatch\ngot %+v\nwant %+v", got, want)
					}
					if got.Partials == 1 {
						old, err := cudaroofline.AnalyticalGEMMSpatial(m, widths, p)
						if err != nil || !reflect.DeepEqual(got.GEMM, old) {
							t.Fatal("one-partial compatibility", err)
						}
					} else {
						if _, err := cudaroofline.AnalyticalGEMMSpatial(m, widths, p); !errors.Is(err, cudalayout.ErrUnmodeled) {
							t.Fatal("old API changed K admission", err)
						}
					}
				})
				count++
			}
		}
	}
	if count != 384 {
		t.Fatal(count)
	}
}

func TestGEMMSplitKScaleAndOverflow(t *testing.T) {
	m := reductionMapping([4][4]int{{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1 << 20}, {1, 1 << 16, 1, 1}}, nil, "BMNK", 0)
	m.Spatial = cudalayout.NewSpatialTiling(map[string]int{"K": 1 << 20}, nil)
	w := map[cudalayout.TensorRole]int{0: 1, 1: 1, 2: 4}
	got, err := cudaroofline.AnalyticalGEMMSplitK(m, w, cudaroofline.GEMMSpatialProfile{})
	if err != nil {
		t.Fatal(err)
	}
	const s = uint64(1 << 20)
	const e = uint64(1 << 16)
	const f = 4 * e
	var read, write uint64
	for _, r := range got.Rounds {
		read += r.ReadBytes
		write += r.WriteBytes
	}
	if len(got.Rounds) != 20 || read != (2*s-2)*f || write != (s-1)*f || got.WorkspaceBytes != 3*(1<<37) || got.ReductionAdditions != e*(s-1) || got.ExecutedFLOPs != 2*e*s+e*(s-1) {
		t.Fatal(got)
	}
	// Explicit backing overflow while individual extents still fit int.
	m = reductionMapping([4][4]int{{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1 << 30, 1 << 30, 1}, {1, 1, 1, 2}}, nil, "BMNK", 0)
	m.Spatial = cudalayout.NewSpatialTiling(map[string]int{"K": 2}, nil)
	w[2] = 3 // backing exceeds MaxInt while all recipe traffic still fits uint64
	bad, err := cudaroofline.AnalyticalGEMMSplitK(m, w, cudaroofline.GEMMSpatialProfile{})
	if !errors.Is(err, cudalayout.ErrOverflow) || !reflect.DeepEqual(bad, cudaroofline.GEMMSplitKMovement{}) {
		t.Fatalf("backing overflow got=%+v err=%v", bad, err)
	}
	w[2] = 8
	bad, err = cudaroofline.AnalyticalGEMMSplitK(m, w, cudaroofline.GEMMSpatialProfile{})
	if err == nil || !strings.Contains(err.Error(), "overflows uint64") || !reflect.DeepEqual(bad, cudaroofline.GEMMSplitKMovement{}) {
		t.Fatalf("byte overflow got=%+v err=%v", bad, err)
	}
	// Useful FLOPs fit; the separately counted tree additions overflow.
	k := (1 << 61) - 2 // divisible by five
	m = reductionMapping([4][4]int{{1, 2, 2, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, k}}, nil, "BMNK", 0)
	m.Spatial = cudalayout.NewSpatialTiling(map[string]int{"K": 5}, nil)
	bad, err = cudaroofline.AnalyticalGEMMSplitK(m, map[cudalayout.TensorRole]int{0: 1, 1: 1, 2: 1}, cudaroofline.GEMMSpatialProfile{})
	if err == nil || !strings.Contains(err.Error(), "overflows uint64") || !reflect.DeepEqual(bad, cudaroofline.GEMMSplitKMovement{}) {
		t.Fatalf("executed-FLOP overflow got=%+v err=%v", bad, err)
	}
}

func TestGEMMSplitKUnevenLarge(t *testing.T) {
	for _, s := range []uint64{(1 << 20) - 1, (1 << 20) + 1} {
		m := reductionMapping([4][4]int{{1, 2, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, int(s)}}, nil, "BMNK", 0)
		m.Spatial = cudalayout.NewSpatialTiling(map[string]int{"K": int(s)}, nil)
		got, err := cudaroofline.AnalyticalGEMMSplitK(m, map[cudalayout.TensorRole]int{0: 1, 1: 1, 2: 4}, cudaroofline.GEMMSpatialProfile{})
		if err != nil {
			t.Fatal(err)
		}
		// Count nonempty original-leaf blocks of width 2^r, independently of
		// the production recurrence over the preceding round's partial count.
		var want []cudaroofline.GEMMReductionRound
		for r := 0; r < bits.Len64(s-1); r++ {
			in, out := ((s-1)>>r)+1, ((s-1)>>(r+1))+1
			want = append(want, cudaroofline.GEMMReductionRound{InputPartials: in, OutputPartials: out, ReadBytes: in * 8, WriteBytes: out * 8, Additions: (in - out) * 2})
		}
		if !reflect.DeepEqual(got.Rounds, want) || got.WorkspaceBytes != (s+((s-1)>>1)+1)*8 || got.ReductionAdditions != 2*(s-1) {
			t.Fatal(s, got)
		}
	}
}

func TestGEMMSplitKOnePartialLevels(t *testing.T) {
	original := [4][4]int{{1, 2, 1, 1}, {2, 1, 2, 1}, {1, 1, 1, 2}, {1, 2, 1, 1}}
	widths := map[cudalayout.TensorRole]int{0: 2, 1: 3, 2: 5}
	for subset := 0; subset < 8; subset++ {
		var levels []cudalayout.StorageLevel
		var factors [4][4]int
		carry := [4]int{1, 1, 1, 1}
		for l := 0; l < 4; l++ {
			for d := 0; d < 4; d++ {
				carry[d] *= original[l][d]
			}
			if l == 3 || subset&(1<<l) != 0 {
				levels = append(levels, cudalayout.StorageLevel(l))
				factors[l] = carry
				carry = [4]int{1, 1, 1, 1}
			}
		}
		for _, order := range []string{"BMNK", "KNMB"} {
			for _, mask := range []int{0, 4, 85, 170, 511} {
				m := reductionMapping(factors, levels, order, mask)
				p := cudaroofline.GEMMSpatialProfile{LaneMesh: map[string]int{"K": 1}}
				got, err := cudaroofline.AnalyticalGEMMSplitK(m, widths, p)
				if err != nil {
					t.Fatal(err)
				}
				want := reductionTrace(t, m, widths, p)
				if !reflect.DeepEqual(got, want) {
					t.Fatal("identity trace", subset, order, mask)
				}
				old, err := cudaroofline.AnalyticalGEMMSpatial(m, widths, p)
				if err != nil || !reflect.DeepEqual(got.GEMM, old) || !reflect.DeepEqual(got.DRAM, old.Movement.DRAM) {
					t.Fatal("one-partial compatibility", subset, err)
				}
			}
		}
	}
}

func TestGEMMSplitKRefusalsAndOwnership(t *testing.T) {
	base := func() cudalayout.MapSpace {
		m := reductionMapping([4][4]int{{1, 1, 1, 1}, {1, 2, 2, 6}, {1, 1, 1, 2}, {1, 1, 1, 1}}, nil, "BMNK", 0)
		m.Spatial = cudalayout.NewSpatialTiling(map[string]int{"K": 2}, nil)
		return m
	}
	w := map[cudalayout.TensorRole]int{0: 2, 1: 3, 2: 5}
	for _, kind := range []string{"lane-zero", "lane-unknown", "lane-ratio", "lane-size", "CTA-ratio", "warp-ratio", "order", "bypass"} {
		t.Run(kind, func(t *testing.T) {
			m := base()
			p := cudaroofline.GEMMSpatialProfile{}
			switch kind {
			case "lane-zero":
				p.LaneMesh = map[string]int{"K": 0}
			case "lane-unknown":
				p.LaneMesh = map[string]int{"Q": 1}
			case "lane-ratio":
				p.LaneMesh = map[string]int{"K": 5}
			case "lane-size":
				p.LaneMesh = map[string]int{"K": 33}
			case "CTA-ratio":
				m.Spatial.CTAMesh["K"] = 3
			case "warp-ratio":
				m.Spatial.WarpMesh = map[string]int{"K": 4}
			case "order":
				m.Permutation.Orders[1] = []string{"M", "N"}
			case "bypass":
				m.Bypass.SetBypass(3, 2, true)
			}
			got, err := cudaroofline.AnalyticalGEMMSplitK(m, w, p)
			if !errors.Is(err, cudalayout.ErrUnsupported) || !reflect.DeepEqual(got, cudaroofline.GEMMSplitKMovement{}) {
				t.Fatalf("got %+v err=%v", got, err)
			}
		})
	}
	m := base()
	p := cudaroofline.GEMMSpatialProfile{LaneMesh: map[string]int{"K": 3}}
	want, err := cudaroofline.AnalyticalGEMMSplitK(m, w, p)
	if err != nil {
		t.Fatal(err)
	}
	got, err := cudaroofline.AnalyticalGEMMSplitK(m, w, p)
	if err != nil {
		t.Fatal(err)
	}
	got.Rounds[0].ReadBytes = 0
	got.GEMM.Levels[0].Tensors[2] = cudaroofline.SpatialTensorMovement{}
	got.GEMM.Movement.DRAM.TensorTransfers["C"] = 0
	got.DRAM.TensorTransfers["A"] = 0
	again, err := cudaroofline.AnalyticalGEMMSplitK(m, w, p)
	if err != nil || !reflect.DeepEqual(again, want) || p.LaneMesh["K"] != 3 || m.Spatial.CTAMesh["K"] != 2 {
		t.Fatal("returned evidence aliases inputs/state", err)
	}
}

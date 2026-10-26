package cudasearch_test

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudalayout"
	"github.com/tmc/lib/nvidia/x/cudasearch"
	"github.com/tmc/lib/nvidia/x/cudashape"
	"github.com/tmc/lib/nvidia/x/cudatensor"
)

var (
	timeloopRoles = []cudalayout.TensorRole{cudalayout.TensorA, cudalayout.TensorB, cudalayout.TensorC}
	timeloopMNK   = []string{"M", "N", "K"}
	timeloopKMN   = []string{"K", "M", "N"}
	timeloopMKN   = []string{"M", "K", "N"}
	timeloopBMNK  = []string{"B", "M", "N", "K"}

	// timeloopProblem is the problem of timeloopShape("4"), and timeloopTile
	// is the Shared tile of timeloopFit.
	timeloopProblem = map[string]int{"B": 1, "M": 4, "N": 6, "K": 8}
	timeloopTile    = map[string]int{"B": 1, "M": 2, "N": 3, "K": 4}
	// timeloopWidths are the act, wt and out element sizes of timeloopOptions.
	timeloopWidths = [3]int{1, 2, 4}
	// timeloopRanks maps the exporter's labels for mk,kn->mn.
	timeloopRanks = map[cudashape.DimToken]string{"A": "M", "B": "N", "C": "K"}
	timeloopBound = map[string]string{"A": "4", "B": "6", "C": "8"}
)

// timeloopGraph returns one GEMM, mm, reading lhs and rhs and writing dst.
func timeloopGraph(shape map[string]string, lhs, rhs, dst string) *cudatensor.Graph {
	return &cudatensor.Graph{
		Nodes: []cudatensor.EinsumNode{timeloopNode(lhs, rhs, dst)},
		Shape: shape,
	}
}

func timeloopNode(lhs, rhs, dst string) cudatensor.EinsumNode {
	return cudatensor.EinsumNode{
		ID: "mm", Equation: "mk,kn->mn", Inputs: []string{"", ""},
		InputTensors:  []cudatensor.TensorInfo{{Name: lhs}, {Name: rhs}},
		OutputTensors: []cudatensor.TensorInfo{{Name: dst}},
	}
}

func timeloopShape(m string) map[string]string {
	return map[string]string{"M": m, "N": "6", "K": "8"}
}

// timeloopExport returns the bytes cudatensor.ExportTimeloopWorkload writes
// for g after rank canonicalization.
func timeloopExport(t *testing.T, g *cudatensor.Graph) []byte {
	t.Helper()
	canonical, err := cudatensor.CanonicalizeRanks(g)
	if err != nil {
		t.Fatal(err)
	}
	var b bytes.Buffer
	if err := cudatensor.ExportTimeloopWorkload(canonical, &b); err != nil {
		t.Fatal(err)
	}
	return b.Bytes()
}

// timeloopCompact returns doc without insignificant space, for editing.
func timeloopCompact(t *testing.T, doc []byte) string {
	t.Helper()
	var b bytes.Buffer
	if err := json.Compact(&b, doc); err != nil {
		t.Fatal(err)
	}
	return b.String()
}

// timeloopEdit replaces the first occurrence of each old, new pair in doc.
func timeloopEdit(t *testing.T, doc string, pairs ...string) []byte {
	t.Helper()
	for i := 0; i+1 < len(pairs); i += 2 {
		if !strings.Contains(doc, pairs[i]) {
			t.Fatalf("workload has no %s: %s", pairs[i], doc)
		}
		doc = strings.Replace(doc, pairs[i], pairs[i+1], 1)
	}
	return []byte(doc)
}

func timeloopFactorization(shared, dram map[string]int) cudalayout.IndexFactorization {
	return cudalayout.NewIndexFactorization(
		[]cudalayout.StorageLevel{cudalayout.LevelShared, cudalayout.LevelDRAM},
		map[cudalayout.StorageLevel]map[string]int{cudalayout.LevelShared: shared, cudalayout.LevelDRAM: dram},
	)
}

// timeloopFit holds 8+24+24 Shared bytes at widths 1, 2 and 4.
func timeloopFit() cudalayout.IndexFactorization {
	return timeloopFactorization(timeloopTile, map[string]int{"B": 1, "M": 2, "N": 2, "K": 2})
}

// timeloopOvercap holds the whole problem in Shared: 32+96+96 bytes.
func timeloopOvercap() cudalayout.IndexFactorization {
	return timeloopFactorization(map[string]int{"B": 1, "M": 4, "N": 6, "K": 8}, map[string]int{"B": 1, "M": 1, "N": 1, "K": 1})
}

func timeloopOrder(dram []string) cudalayout.LoopPermutation {
	return cudalayout.NewLoopPermutation(map[cudalayout.StorageLevel][]string{
		cudalayout.LevelShared: {"M", "N", "K"},
		cudalayout.LevelDRAM:   dram,
	})
}

func timeloopBypassA() cudalayout.DatatypeBypass {
	b := cudalayout.NewDatatypeBypass()
	b.SetBypass(cudalayout.LevelShared, cudalayout.TensorA, true)
	return b
}

// timeloopOptions searches mm with a 64-byte Shared limit, which prunes
// timeloopOvercap with and without bypassing A and admits timeloopFit.
func timeloopOptions(perms ...cudalayout.LoopPermutation) cudasearch.TimeloopGEMMOptions {
	return cudasearch.TimeloopGEMMOptions{
		Einsum:         "mm",
		MaxBytes:       1 << 20,
		ElementBytes:   map[string]int{"act": 1, "wt": 2, "out": 4},
		Factorizations: []cudalayout.IndexFactorization{timeloopOvercap(), timeloopFit()},
		Permutations:   perms,
		Bypasses:       []cudalayout.DatatypeBypass{cudalayout.NewDatatypeBypass(), timeloopBypassA()},
		Constraints:    cudalayout.HardwareConstraints{MaxSharedMemoryPerBlock: 64},
	}
}

func timeloopSearch(t *testing.T, doc []byte, opts cudasearch.TimeloopGEMMOptions) cudasearch.TimeloopGEMMResult {
	t.Helper()
	res, err := cudasearch.SearchTimeloopGEMM(context.Background(), bytes.NewReader(doc), opts)
	if err != nil {
		t.Fatalf("SearchTimeloopGEMM: %v", err)
	}
	return res
}

func timeloopCounts(t *testing.T, stats cudasearch.MapSpaceSearchStats, tested, pruned, skipped, evaluated int) {
	t.Helper()
	got := [5]int{stats.FactorizationsTested, stats.FactorizationsPruned, stats.DownstreamSkipped, stats.EvaluatedCandidates, stats.UnsupportedCandidates}
	if want := [5]int{tested, pruned, skipped, evaluated, 0}; got != want {
		t.Fatalf("tested, pruned, skipped, evaluated, unsupported = %v, want %v", got, want)
	}
}

func timeloopCheckIdentity(t *testing.T, res cudasearch.TimeloopGEMMResult, shape map[string]string, ranks map[cudashape.DimToken]string, a, b, c string) {
	t.Helper()
	if res.Einsum != "mm" {
		t.Errorf("Einsum = %q, want mm", res.Einsum)
	}
	if !reflect.DeepEqual(res.ShapeBounds, shape) {
		t.Errorf("ShapeBounds = %v, want %v", res.ShapeBounds, shape)
	}
	if !reflect.DeepEqual(res.RankNames, ranks) {
		t.Errorf("RankNames = %v, want %v", res.RankNames, ranks)
	}
	names := map[cudalayout.TensorRole]string{cudalayout.TensorA: a, cudalayout.TensorB: b, cudalayout.TensorC: c}
	if !reflect.DeepEqual(res.TensorNames, names) {
		t.Errorf("TensorNames = %v, want %v", res.TensorNames, names)
	}
}

// timeloopTraffic is the traffic of a scalar replay of a Shared/DRAM nest.
// Indexes 0, 1 and 2 are A, B and C.
type timeloopTraffic struct {
	dramRead, dramWrite     [3]uint64
	sharedRead, sharedWrite [3]uint64
	fills                   [2]int // Shared A and B tile loads
	evictions, reloads      int    // Shared C tile write-backs and reloads
}

func (tr timeloopTraffic) total() uint64 {
	var total uint64
	for r := range 3 {
		total += tr.dramRead[r] + tr.dramWrite[r]
	}
	return total
}

// timeloopReplay visits every scalar MAC in loop order: the DRAM loops over
// Shared tiles in order, outermost first, then the scalars of one tile. Each
// role keeps one Shared tile and loads another when the scalar's tile
// changes. C tiles start at zero, are written back when replaced and at the
// end, and are reloaded only when revisited. Every access costs the role's
// element width.
func timeloopReplay(t *testing.T, problem, tile map[string]int, order []string, width [3]int) timeloopTraffic {
	t.Helper()
	dims := []string{"B", "M", "N", "K"}
	projects := [3][]string{{"B", "M", "K"}, {"B", "K", "N"}, {"B", "M", "N"}}
	type loop struct {
		dim   string
		count int
		outer bool
	}
	var loops []loop
	for _, dim := range dims {
		if problem[dim]/tile[dim] > 1 && !slices.Contains(order, dim) {
			t.Fatalf("order %v omits %s", order, dim)
		}
	}
	for _, dim := range order {
		loops = append(loops, loop{dim, problem[dim] / tile[dim], true})
	}
	for _, dim := range dims {
		loops = append(loops, loop{dim, tile[dim], false})
	}
	var tileBytes [3]uint64
	for r := range 3 {
		tileBytes[r] = uint64(width[r])
		for _, dim := range projects[r] {
			tileBytes[r] *= uint64(tile[dim])
		}
	}
	var tr timeloopTraffic
	var held [3]string
	visited := make(map[string]bool) // C tiles
	written := make(map[string]bool) // C scalars
	idx := make([]int, len(loops))
	for {
		coord := make(map[string]int)
		for i, l := range loops {
			if l.outer {
				coord[l.dim] += idx[i] * tile[l.dim]
			} else {
				coord[l.dim] += idx[i]
			}
		}
		scalarC := fmt.Sprint(coord["B"], coord["M"], coord["N"])
		for r := range 3 {
			var key []int
			for _, dim := range projects[r] {
				key = append(key, coord[dim]/tile[dim])
			}
			if k := fmt.Sprint(key); k != held[r] {
				if r < 2 {
					tr.dramRead[r] += tileBytes[r]
					tr.fills[r]++
				} else {
					if held[r] != "" {
						tr.dramWrite[r] += tileBytes[r]
						tr.evictions++
					}
					if visited[k] {
						tr.dramRead[r] += tileBytes[r]
						tr.reloads++
					}
					visited[k] = true
				}
				held[r] = k
			}
			w := uint64(width[r])
			if r < 2 {
				tr.sharedRead[r] += w
				continue
			}
			if written[scalarC] {
				tr.sharedRead[r] += w
			}
			tr.sharedWrite[r] += w
		}
		written[scalarC] = true
		i := len(loops) - 1
		for ; i >= 0; i-- {
			idx[i]++
			if idx[i] < loops[i].count {
				break
			}
			idx[i] = 0
		}
		if i < 0 {
			break
		}
	}
	if held[2] != "" {
		tr.dramWrite[2] += tileBytes[2]
		tr.evictions++
	}
	return tr
}

func timeloopCheckMovement(t *testing.T, best *cudasearch.MapSpaceSearchResult, want timeloopTraffic) {
	t.Helper()
	if best == nil {
		t.Fatal("search has no best candidate")
	}
	levels := best.Movement.Levels
	if len(levels) != 2 || levels[0].Level != cudalayout.LevelShared || levels[1].Level != cudalayout.LevelDRAM {
		t.Fatalf("levels = %+v, want Shared and DRAM", levels)
	}
	for r, role := range timeloopRoles {
		shared, dram := levels[0].Tensors[role], levels[1].Tensors[role]
		if dram.ReadBytes != want.dramRead[r] || dram.WriteBytes != want.dramWrite[r] {
			t.Errorf("%v DRAM read, write = %d, %d, want %d, %d", role, dram.ReadBytes, dram.WriteBytes, want.dramRead[r], want.dramWrite[r])
		}
		if shared.ReadBytes != want.sharedRead[r] || shared.WriteBytes != want.sharedWrite[r] {
			t.Errorf("%v Shared read, write = %d, %d, want %d, %d", role, shared.ReadBytes, shared.WriteBytes, want.sharedRead[r], want.sharedWrite[r])
		}
	}
	if got := best.TotalDRAM().TotalBytes; got != want.total() {
		t.Errorf("TotalDRAM = %d, want %d", got, want.total())
	}
}

func TestSearchTimeloopGEMMExporterPruning(t *testing.T) {
	doc := timeloopExport(t, timeloopGraph(timeloopShape("4"), "act", "wt", "out"))
	// Over the 2x2x2 DRAM loops with K innermost, A and B load a tile on every
	// iteration and C writes back 4 tiles without reloading one.
	want := timeloopReplay(t, timeloopProblem, timeloopTile, timeloopMNK, timeloopWidths)
	if want.fills != [2]int{8, 8} || want.evictions != 4 || want.reloads != 0 {
		t.Fatalf("replay events = %+v", want)
	}
	if total := uint64(8*(2*4)*1 + 8*(4*3)*2 + 4*(2*3)*4); want.total() != total {
		t.Fatalf("replay total = %d, want %d", want.total(), total)
	}
	for _, test := range []struct {
		name               string
		perms              []cudalayout.LoopPermutation
		skipped, evaluated int
		order              []string
	}{
		{"explicit", []cudalayout.LoopPermutation{timeloopOrder(timeloopKMN), timeloopOrder(timeloopMNK)}, 4, 4, timeloopMNK},
		{"default", nil, 2, 2, timeloopBMNK},
	} {
		t.Run(test.name, func(t *testing.T) {
			res := timeloopSearch(t, doc, timeloopOptions(test.perms...))
			timeloopCounts(t, res.Stats, 2, 1, test.skipped, test.evaluated)
			best := res.Stats.Best
			timeloopCheckMovement(t, best, want)
			if got := best.Mapping.Permutation.Orders[cudalayout.LevelDRAM]; !slices.Equal(got, test.order) {
				t.Errorf("DRAM order = %v, want %v", got, test.order)
			}
			if best.Mapping.Bypass.IsBypassed(cudalayout.LevelShared, cudalayout.TensorA) {
				t.Error("best bypasses A")
			}
			if !reflect.DeepEqual(best.Mapping.Problem, timeloopProblem) {
				t.Errorf("problem = %v, want %v", best.Mapping.Problem, timeloopProblem)
			}
			if got := best.Mapping.Factorization.Factors[cudalayout.LevelShared]; !reflect.DeepEqual(got, timeloopTile) {
				t.Errorf("Shared tile = %v, want %v", got, timeloopTile)
			}
			if intensity := float64(2*4*6*8) / float64(want.total()); res.Stats.BestIntensity != intensity {
				t.Errorf("BestIntensity = %v, want %v", res.Stats.BestIntensity, intensity)
			}
			if best.SpatialProfile != nil || best.SpatialMovement != nil || best.SplitK != nil {
				t.Errorf("temporal best has spatial or split-K results: %+v", best)
			}
			timeloopCheckIdentity(t, res, timeloopBound, timeloopRanks, "act", "wt", "out")
		})
	}
}

func TestSearchTimeloopGEMMOrdering(t *testing.T) {
	doc := timeloopExport(t, timeloopGraph(timeloopShape("4"), "act", "wt", "out"))
	replay := func(order []string) timeloopTraffic {
		return timeloopReplay(t, timeloopProblem, timeloopTile, order, timeloopWidths)
	}
	mnk, kmn, mkn := replay(timeloopMNK), replay(timeloopKMN), replay(timeloopMKN)
	// With K outside N, each of the 4 C tiles is written back twice and
	// reloaded once.
	if mnk.total() != 352 || kmn.total() != 512 || mkn.total() != 512 || kmn.reloads != 4 || mkn.reloads != 4 {
		t.Fatalf("replay totals = %d, %d, %d; reloads = %d, %d", mnk.total(), kmn.total(), mkn.total(), kmn.reloads, mkn.reloads)
	}
	for _, test := range []struct {
		name    string
		orders  [][]string
		want    []string
		traffic timeloopTraffic
	}{
		{"cheaper", [][]string{timeloopKMN, timeloopMNK}, timeloopMNK, mnk},
		{"tieFirst", [][]string{timeloopMKN, timeloopKMN}, timeloopMKN, mkn},
		{"tieReversed", [][]string{timeloopKMN, timeloopMKN}, timeloopKMN, kmn},
	} {
		t.Run(test.name, func(t *testing.T) {
			opts := timeloopOptions()
			opts.Factorizations = []cudalayout.IndexFactorization{timeloopFit()}
			opts.Bypasses = nil
			for _, order := range test.orders {
				opts.Permutations = append(opts.Permutations, timeloopOrder(order))
			}
			res := timeloopSearch(t, doc, opts)
			timeloopCounts(t, res.Stats, 1, 0, 0, 2)
			timeloopCheckMovement(t, res.Stats.Best, test.traffic)
			if got := res.Stats.Best.Mapping.Permutation.Orders[cudalayout.LevelDRAM]; !slices.Equal(got, test.want) {
				t.Errorf("DRAM order = %v, want %v", got, test.want)
			}
		})
	}
}

func TestSearchTimeloopGEMMExtentBinding(t *testing.T) {
	doc := timeloopExport(t, timeloopGraph(timeloopShape("12"), "act", "wt", "out"))
	opts := timeloopOptions(timeloopOrder(timeloopKMN), timeloopOrder(timeloopMNK))
	res := timeloopSearch(t, doc, opts)
	// Both factorizations multiply to M=4, not the emitted M=12.
	timeloopCounts(t, res.Stats, 2, 2, 8, 0)
	if res.Stats.Best != nil {
		t.Fatalf("best = %+v, want none", res.Stats.Best)
	}
	timeloopCheckIdentity(t, res, map[string]string{"A": "12", "B": "6", "C": "8"}, timeloopRanks, "act", "wt", "out")

	problem := map[string]int{"B": 1, "M": 12, "N": 6, "K": 8}
	opts.Factorizations = []cudalayout.IndexFactorization{
		timeloopFactorization(problem, map[string]int{"B": 1, "M": 1, "N": 1, "K": 1}),
		timeloopFactorization(timeloopTile, map[string]int{"B": 1, "M": 6, "N": 2, "K": 2}),
	}
	res = timeloopSearch(t, doc, opts)
	timeloopCounts(t, res.Stats, 2, 1, 4, 4)
	want := timeloopReplay(t, problem, timeloopTile, timeloopMNK, timeloopWidths)
	if total := uint64(24*(2*4)*1 + 24*(4*3)*2 + 12*(2*3)*4); want.total() != total {
		t.Fatalf("replay total = %d, want %d", want.total(), total)
	}
	timeloopCheckMovement(t, res.Stats.Best, want)
	if !reflect.DeepEqual(res.Stats.Best.Mapping.Problem, problem) {
		t.Errorf("problem = %v, want %v", res.Stats.Best.Mapping.Problem, problem)
	}
}

func TestSearchTimeloopGEMMNames(t *testing.T) {
	base := timeloopSearch(t, timeloopExport(t, timeloopGraph(timeloopShape("4"), "act", "wt", "out")), timeloopOptions())
	doc := timeloopExport(t, timeloopGraph(timeloopShape("4"), "lhs", "rhs", "dst"))
	opts := timeloopOptions()
	opts.ElementBytes = map[string]int{"lhs": 1, "rhs": 2, "dst": 4}
	res := timeloopSearch(t, doc, opts)
	if !reflect.DeepEqual(res.Stats, base.Stats) {
		t.Errorf("renamed stats = %+v, want %+v", res.Stats, base.Stats)
	}
	timeloopCheckIdentity(t, res, timeloopBound, timeloopRanks, "lhs", "rhs", "dst")

	// Exchanging the input widths changes the bytes routed to A and B.
	opts.ElementBytes = map[string]int{"lhs": 2, "rhs": 1, "dst": 4}
	res = timeloopSearch(t, doc, opts)
	timeloopCounts(t, res.Stats, 2, 1, 2, 2)
	want := timeloopReplay(t, timeloopProblem, timeloopTile, timeloopMNK, [3]int{2, 1, 4})
	if total := uint64(8*(2*4)*2 + 8*(4*3)*1 + 4*(2*3)*4); want.total() != total {
		t.Fatalf("replay total = %d, want %d", want.total(), total)
	}
	timeloopCheckMovement(t, res.Stats.Best, want)
}

func TestSearchTimeloopGEMMBatched(t *testing.T) {
	g := &cudatensor.Graph{
		Nodes: []cudatensor.EinsumNode{{
			ID: "mm", Equation: "bmk,bkn->bmn", Inputs: []string{"", ""},
			InputTensors:  []cudatensor.TensorInfo{{Name: "act"}, {Name: "wt"}},
			OutputTensors: []cudatensor.TensorInfo{{Name: "out"}},
		}},
		Shape: map[string]string{"B": "2", "M": "4", "N": "6", "K": "8"},
	}
	problem := map[string]int{"B": 2, "M": 4, "N": 6, "K": 8}
	opts := timeloopOptions()
	opts.Factorizations = []cudalayout.IndexFactorization{
		timeloopFactorization(problem, map[string]int{"B": 1, "M": 1, "N": 1, "K": 1}),
		timeloopFactorization(timeloopTile, map[string]int{"B": 2, "M": 2, "N": 2, "K": 2}),
	}
	opts.Bypasses = nil
	res := timeloopSearch(t, timeloopExport(t, g), opts)
	timeloopCounts(t, res.Stats, 2, 1, 1, 1)
	want := timeloopReplay(t, problem, timeloopTile, timeloopBMNK, timeloopWidths)
	if total := uint64(16*(2*4)*1 + 16*(4*3)*2 + 8*(2*3)*4); want.total() != total {
		t.Fatalf("replay total = %d, want %d", want.total(), total)
	}
	timeloopCheckMovement(t, res.Stats.Best, want)
	if !reflect.DeepEqual(res.Stats.Best.Mapping.Problem, problem) {
		t.Errorf("problem = %v, want %v", res.Stats.Best.Mapping.Problem, problem)
	}
	ranks := map[cudashape.DimToken]string{"A": "B", "B": "M", "C": "N", "D": "K"}
	timeloopCheckIdentity(t, res, map[string]string{"A": "2", "B": "4", "C": "6", "D": "8"}, ranks, "act", "wt", "out")
}

func TestSearchTimeloopGEMMChain(t *testing.T) {
	g := &cudatensor.Graph{
		Nodes: []cudatensor.EinsumNode{
			timeloopNode("act", "wt", "out"),
			{
				ID: "expand", Equation: "mn->mnp", Inputs: []string{"mm"},
				InputTensors:  []cudatensor.TensorInfo{{Name: "out"}},
				OutputTensors: []cudatensor.TensorInfo{{Name: "y"}},
			},
		},
		Shape: map[string]string{"M": "4", "N": "6", "K": "8", "P": "[1, 64]"},
	}
	doc := timeloopExport(t, g)
	opts := timeloopOptions(timeloopOrder(timeloopKMN), timeloopOrder(timeloopMNK))
	isolated := timeloopSearch(t, timeloopExport(t, timeloopGraph(timeloopShape("4"), "act", "wt", "out")), opts)
	res := timeloopSearch(t, doc, opts)
	if !reflect.DeepEqual(res.Stats, isolated.Stats) {
		t.Errorf("chain stats = %+v, want isolated %+v", res.Stats, isolated.Stats)
	}
	if res.Stats.Best == nil {
		t.Fatal("chain search has no best candidate")
	}
	timeloopCheckIdentity(t, res, map[string]string{"A": "4", "B": "6", "C": "8", "D": "[1, 64]"}, timeloopRanks, "act", "wt", "out")

	c := timeloopCompact(t, doc)
	for _, test := range []struct {
		name, einsum string
		doc          []byte
		want         string
	}{
		{"unary", "expand", doc, `einsum "expand" has 1 input and 1 output accesses`},
		{"unknown", "nope", doc, `einsum "nope" not found`},
		{"duplicate", "mm", timeloopEdit(t, c, `"name":"expand"`, `"name":"mm"`), `duplicate Timeloop einsum "mm"`},
	} {
		t.Run(test.name, func(t *testing.T) {
			opts := timeloopOptions()
			opts.Einsum = test.einsum
			res, err := cudasearch.SearchTimeloopGEMM(context.Background(), bytes.NewReader(test.doc), opts)
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("error = %v, want %q", err, test.want)
			}
			if !reflect.DeepEqual(res, cudasearch.TimeloopGEMMResult{}) {
				t.Fatalf("refused result = %+v, want zero", res)
			}
		})
	}
}

func TestSearchTimeloopGEMMRefusals(t *testing.T) {
	c := timeloopCompact(t, timeloopExport(t, timeloopGraph(timeloopShape("4"), "act", "wt", "out")))
	edit := func(pairs ...string) []byte { return timeloopEdit(t, c, pairs...) }
	base := []byte(c)
	defaults := strings.Index(c, `,"renames":{"einsums"`)
	if defaults < 0 {
		t.Fatalf("workload has no default renames: %s", c)
	}
	bound := func(b string) []byte { return edit(`"A":"4"`, `"A":"`+b+`"`) }
	sizes := func(sizes map[string]int) func(*cudasearch.TimeloopGEMMOptions) {
		return func(o *cudasearch.TimeloopGEMMOptions) { o.ElementBytes = sizes }
	}
	for _, test := range []struct {
		name       string
		doc        []byte
		opts       func(*cudasearch.TimeloopGEMMOptions)
		nilReader  bool
		nilContext bool
		want       string
	}{
		{name: "malformed", doc: []byte(c[:len(c)-1]), want: "Timeloop workload: document: "},
		{name: "trailingValue", doc: []byte(c + "{}"), want: "trailing data"},
		{name: "trailingText", doc: []byte(c + "x"), want: "trailing data"},
		{name: "depth", doc: []byte(`{"workload":[[[[[[[[[]]]]]]]]]}`), want: "nesting deeper than 8"},
		{name: "utf8", doc: edit(`"name":"act"`, "\"name\":\"a\xffct\""), want: "not valid UTF-8"},
		{name: "unknownKey", doc: edit(`{"workload":`, `{"extra":{},"workload":`), want: `unknown key "extra"`},
		{name: "caseAlias", doc: edit(`"version"`, `"Version"`), want: `unknown key "Version"`},
		{name: "duplicateKey", doc: edit(`"version":"0.5"`, `"version":"0.5","version":"0.5"`), want: `duplicate key "version"`},
		{name: "duplicateShape", doc: edit(`"A":"4"`, `"A":"4","A":"4"`), want: `duplicate key "A"`},
		{name: "version", doc: edit(`"0.5"`, `"0.4"`), want: "is not 0.5"},
		{name: "nullOutput", doc: edit(`"output":true`, `"output":null`), want: "got null, want boolean"},
		{name: "nullShape", doc: edit(`"shape":{"A":"4","B":"6","C":"8"}`, `"shape":null`), want: "got null, want object"},
		{name: "nullProjection", doc: edit(`"projection":["A","C"]`, `"projection":null`), want: "got null, want array"},
		{name: "nullBound", doc: edit(`"A":"4"`, `"A":null`), want: "got null, want string"},
		{name: "stringOutput", doc: edit(`"output":true`, `"output":"true"`), want: "got string, want boolean"},
		{name: "stringCount", doc: edit(`"expected_count":1`, `"expected_count":"1"`), want: "got string, want integer"},
		{name: "fractionalCount", doc: edit(`"expected_count":1`, `"expected_count":1.0`), want: "is not an integer"},
		{name: "numberVersion", doc: edit(`"version":"0.5"`, `"version":0.5`), want: "got number, want string"},
		{name: "numberBound", doc: edit(`"A":"4"`, `"A":4`), want: "got number, want string"},
		{name: "arrayName", doc: edit(`"name":"act"`, `"name":["act"]`), want: "got array, want string"},
		{name: "missingVersion", doc: edit(`"version":"0.5",`, ``), want: `missing key "version"`},
		{name: "missingProjection", doc: edit(`{"name":"act","projection":["A","C"]}`, `{"name":"act"}`), want: `missing key "projection"`},
		{name: "missingEinsumRenames", doc: edit(`,"renames":{"output":"out","input":"act","weight":"wt"}`, ``), want: `workload.einsums[0]: missing key "renames"`},
		{name: "missingDefaultRenames", doc: []byte(c[:defaults] + "}"), want: `document: missing key "renames"`},
		{name: "defaultName", doc: edit(`"name":"default"`, `"name":"custom"`), want: "default renames"},
		{name: "defaultSource", doc: edit(`"source":"Outputs()"`, `"source":"Inputs()"`), want: "default renames"},
		{name: "defaultCount", doc: edit(`"expected_count":1`, `"expected_count":2`), want: "default renames"},
		{name: "defaultDuplicateRule", doc: edit(`{"name":"weight"`, `{"name":"input"`), want: "default renames"},
		{name: "defaultMissingRule", doc: edit(`,{"name":"weight","source":"~(input | output)","expected_count":1}`, ``), want: "default renames"},
		{name: "defaultExtraSet", doc: edit(`}]}]}}`, `}]},{"name":"default","tensor_accesses":[]}]}}`), want: "default renames"},
		{name: "swappedRenames", doc: edit(`"input":"act","weight":"wt"`, `"input":"wt","weight":"act"`), want: "renames input"},
		{name: "outputRenames", doc: edit(`"output":"out","input"`, `"output":"act","input"`), want: "renames output"},
		{name: "nothingWeight", doc: edit(`"weight":"wt"`, `"weight":"Nothing()"`), want: "renames weight"},
		{name: "lowerRank", doc: edit(`["A","C"]`, `["a","C"]`), want: "is not a canonical rank"},
		{name: "mergedRank", doc: edit(`["A","C"]`, `["AC"]`), want: "is not a canonical rank"},
		{name: "groupRank", doc: edit(`["A","C"]`, `["(A+C)"]`), want: "is not a canonical rank"},
		{name: "spacedRank", doc: edit(`["A","C"]`, `[" A","C"]`), want: "is not a canonical rank"},
		{name: "unknownRank", doc: edit(`["A","C"]`, `["A","C1"]`), want: `rank "C1" has no shape bound`},
		{name: "repeatedRank", doc: edit(`["A","C"]`, `["A","A"]`), want: `repeats rank "A"`},
		{name: "emptyProjection", doc: edit(`["A","C"]`, `[]`), want: "has no projection"},
		{name: "emptyName", doc: edit(`"name":"act"`, `"name":""`), want: "has no tensor name"},
		{name: "inputAfterOutput", doc: edit(
			`{"name":"act","projection":["A","C"]},{"name":"wt","projection":["C","B"]},{"name":"out","projection":["A","B"],"output":true}`,
			`{"name":"out","projection":["A","B"],"output":true},{"name":"act","projection":["A","C"]},{"name":"wt","projection":["C","B"]}`,
		), want: "is an input after an output"},
		{name: "unprojectedShape", doc: edit(`"C":"8"}`, `"C":"8","D":"1"}`), want: `shape rank "D" is not projected`},
		{name: "missingShapeRank", doc: edit(`,"C":"8"`, ``), want: `rank "C" has no shape bound`},
		{name: "zeroBound", doc: bound("0"), want: "not a positive decimal integer"},
		{name: "negativeBound", doc: bound("-4"), want: "not a positive decimal integer"},
		{name: "plusBound", doc: bound("+4"), want: "not a positive decimal integer"},
		{name: "leadingZeroBound", doc: bound("04"), want: "not a positive decimal integer"},
		{name: "fractionalBound", doc: bound("4.0"), want: "not a positive decimal integer"},
		{name: "exponentBound", doc: bound("4e0"), want: "not a positive decimal integer"},
		{name: "intervalBound", doc: bound("[1, 4]"), want: `rank "A" bound "[1, 4]": not a positive decimal integer`},
		{name: "symbolicBound", doc: bound("M"), want: "not a positive decimal integer"},
		{name: "emptyBound", doc: bound(""), want: "not a positive decimal integer"},
		{name: "spacedBound", doc: bound(" 4"), want: "not a positive decimal integer"},
		{name: "overflowBound", doc: bound("99999999999999999999"), want: "out of range"},
		{name: "selfMultiply", doc: edit(`{"name":"wt"`, `{"name":"act"`, `"weight":"wt"`, `"weight":"act"`), want: "are not distinct"},
		{name: "inPlace", doc: edit(`{"name":"out"`, `{"name":"act"`, `"output":"out"`, `"output":"act"`), want: "are not distinct"},
		{name: "transpose", doc: edit(`["A","C"]`, `["C","A"]`), want: "unsupported GEMM equation"},
		{name: "rankOneOperand", doc: edit(`["C","B"]`, `["C"]`), want: "unsupported GEMM equation"},
		{name: "extraOutput", doc: edit(`"output":true}`, `"output":true},{"name":"out2","projection":["A","B"],"output":true}`), want: "has 2 input and 2 output accesses"},
		{name: "oversize", doc: base, opts: func(o *cudasearch.TimeloopGEMMOptions) { o.MaxBytes = int64(len(c) - 1) }, want: "exceeds"},
		{name: "unselected", doc: base, opts: func(o *cudasearch.TimeloopGEMMOptions) { o.Einsum = "" }, want: "no Timeloop einsum selected"},
		{name: "unknownEinsum", doc: base, opts: func(o *cudasearch.TimeloopGEMMOptions) { o.Einsum = "nope" }, want: `einsum "nope" not found`},
		{name: "zeroMaxBytes", doc: base, opts: func(o *cudasearch.TimeloopGEMMOptions) { o.MaxBytes = 0 }, want: "MaxBytes 0"},
		{name: "negativeMaxBytes", doc: base, opts: func(o *cudasearch.TimeloopGEMMOptions) { o.MaxBytes = -1 }, want: "MaxBytes -1"},
		{name: "hugeMaxBytes", doc: base, opts: func(o *cudasearch.TimeloopGEMMOptions) { o.MaxBytes = 1<<20 + 1 }, want: "MaxBytes 1048577"},
		{name: "nilElementBytes", doc: base, opts: sizes(nil), want: "ElementBytes has 0 names"},
		{name: "missingElementBytes", doc: base, opts: sizes(map[string]int{"act": 1, "wt": 2}), want: "ElementBytes has 2 names"},
		{name: "extraElementBytes", doc: base, opts: sizes(map[string]int{"act": 1, "wt": 2, "out": 4, "x": 1}), want: "ElementBytes has 4 names"},
		{name: "wrongElementName", doc: base, opts: sizes(map[string]int{"act": 1, "wt": 2, "dst": 4}), want: `ElementBytes["out"] is not a positive size`},
		{name: "zeroElementBytes", doc: base, opts: sizes(map[string]int{"act": 1, "wt": 0, "out": 4}), want: `ElementBytes["wt"] is not a positive size`},
		{name: "negativeElementBytes", doc: base, opts: sizes(map[string]int{"act": -1, "wt": 2, "out": 4}), want: `ElementBytes["act"] is not a positive size`},
		{name: "nilReader", nilReader: true, want: "nil Timeloop reader"},
		{name: "nilContext", doc: base, nilContext: true, want: "nil context"},
	} {
		t.Run(test.name, func(t *testing.T) {
			opts := timeloopOptions()
			if test.opts != nil {
				test.opts(&opts)
			}
			var r io.Reader = bytes.NewReader(test.doc)
			if test.nilReader {
				r = nil
			}
			ctx := context.Background()
			if test.nilContext {
				ctx = nil
			}
			res, err := cudasearch.SearchTimeloopGEMM(ctx, r, opts)
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("error = %v, want %q", err, test.want)
			}
			if !reflect.DeepEqual(res, cudasearch.TimeloopGEMMResult{}) {
				t.Fatalf("refused result = %+v, want zero", res)
			}
		})
	}
}

func TestSearchTimeloopGEMMAccepts(t *testing.T) {
	doc := timeloopExport(t, timeloopGraph(timeloopShape("4"), "act", "wt", "out"))
	c := timeloopCompact(t, doc)
	edit := func(pairs ...string) []byte { return timeloopEdit(t, c, pairs...) }
	want := timeloopSearch(t, doc, timeloopOptions())
	if want.Stats.Best == nil {
		t.Fatal("search has no best candidate")
	}
	for _, test := range []struct {
		name     string
		doc      []byte
		maxBytes int64
	}{
		{name: "exporter", doc: doc},
		{name: "compact", doc: []byte(c)},
		{name: "whitespace", doc: []byte(c + "\n\t \r\n")},
		{name: "explicitFalse", doc: edit(`{"name":"act","projection":["A","C"]}`, `{"name":"act","projection":["A","C"],"output":false}`)},
		{name: "keyOrder", doc: edit(`"version":"0.5","shape":{"A":"4","B":"6","C":"8"},`, `"shape":{"A":"4","B":"6","C":"8"},"version":"0.5",`)},
		{name: "ruleOrder", doc: edit(
			`{"name":"input","source":"Inputs() \u0026 Intermediates()","expected_count":1},{"name":"output","source":"Outputs()","expected_count":1}`,
			`{"name":"output","source":"Outputs()","expected_count":1},{"name":"input","source":"Inputs() \u0026 Intermediates()","expected_count":1}`,
		)},
		{name: "unescaped", doc: edit(`\u0026`, `&`)},
		{name: "exactMaxBytes", doc: []byte(c), maxBytes: int64(len(c))},
	} {
		t.Run(test.name, func(t *testing.T) {
			opts := timeloopOptions()
			if test.maxBytes != 0 {
				opts.MaxBytes = test.maxBytes
			}
			res := timeloopSearch(t, test.doc, opts)
			if !reflect.DeepEqual(res, want) {
				t.Fatalf("result = %+v, want %+v", res, want)
			}
		})
	}
}

// timeloopCountingReader records the bytes requested from and returned by r.
type timeloopCountingReader struct {
	r        io.Reader
	n, limit int
}

func (c *timeloopCountingReader) Read(p []byte) (int, error) {
	c.limit = max(c.limit, len(p))
	n, err := c.r.Read(p)
	c.n += n
	return n, err
}

func TestSearchTimeloopGEMMReadBound(t *testing.T) {
	doc := timeloopExport(t, timeloopGraph(timeloopShape("4"), "act", "wt", "out"))
	opts := timeloopOptions()
	opts.MaxBytes = int64(len(doc))
	padded := append(slices.Clone(doc), bytes.Repeat([]byte(" "), 4096)...)
	src := &timeloopCountingReader{r: bytes.NewReader(padded)}
	_, err := cudasearch.SearchTimeloopGEMM(context.Background(), src, opts)
	if err == nil || !strings.Contains(err.Error(), "exceeds") {
		t.Fatalf("error = %v, want size refusal", err)
	}
	if src.n != len(doc)+1 || src.limit > len(doc)+1 {
		t.Fatalf("read %d bytes in requests of at most %d, want %d", src.n, src.limit, len(doc)+1)
	}

	src = &timeloopCountingReader{r: bytes.NewReader(doc)}
	if _, err := cudasearch.SearchTimeloopGEMM(context.Background(), src, opts); err != nil {
		t.Fatal(err)
	}
	if src.n != len(doc) || src.limit > len(doc)+1 {
		t.Fatalf("read %d bytes in requests of at most %d, want %d", src.n, src.limit, len(doc))
	}
}

func TestSearchTimeloopGEMMDetached(t *testing.T) {
	doc := timeloopExport(t, timeloopGraph(timeloopShape("4"), "act", "wt", "out"))
	options := func() cudasearch.TimeloopGEMMOptions {
		return timeloopOptions(timeloopOrder(timeloopKMN), timeloopOrder(timeloopMNK))
	}
	want := timeloopSearch(t, doc, options())

	opts := options()
	res := timeloopSearch(t, doc, opts)
	if !reflect.DeepEqual(opts, options()) {
		t.Fatal("SearchTimeloopGEMM modified its options")
	}
	opts.ElementBytes["act"] = 64
	opts.Factorizations[1].Factors[cudalayout.LevelShared]["M"] = 64
	opts.Permutations[1].Orders[cudalayout.LevelDRAM][0] = "N"
	opts.Bypasses[0].SetBypass(cudalayout.LevelShared, cudalayout.TensorB, true)
	if !reflect.DeepEqual(res, want) {
		t.Fatalf("result changed with its options:\n%+v\nwant\n%+v", res, want)
	}

	opts = options()
	res = timeloopSearch(t, doc, opts)
	res.ShapeBounds["A"] = "64"
	res.RankNames["A"] = "K"
	res.TensorNames[cudalayout.TensorA] = "wt"
	res.Stats.Best.Mapping.Problem["M"] = 64
	res.Stats.Best.Mapping.Factorization.Factors[cudalayout.LevelShared]["M"] = 64
	res.Stats.Best.Mapping.Permutation.Orders[cudalayout.LevelDRAM][0] = "K"
	res.Stats.Best.Mapping.Bypass.SetBypass(cudalayout.LevelShared, cudalayout.TensorB, true)
	if !reflect.DeepEqual(opts, options()) {
		t.Fatal("options changed with their result")
	}
	if again := timeloopSearch(t, doc, opts); !reflect.DeepEqual(again, want) {
		t.Fatalf("second result = %+v, want %+v", again, want)
	}
}

// timeloopCancelAfter reports cancellation after n successful Err calls.
type timeloopCancelAfter struct {
	context.Context
	n int
}

func (c *timeloopCancelAfter) Err() error {
	if c.n == 0 {
		return context.Canceled
	}
	c.n--
	return nil
}

func TestSearchTimeloopGEMMCanceled(t *testing.T) {
	doc := timeloopExport(t, timeloopGraph(timeloopShape("4"), "act", "wt", "out"))
	opts := timeloopOptions(timeloopOrder(timeloopKMN), timeloopOrder(timeloopMNK))
	full := timeloopSearch(t, doc, opts)
	var refused, partial bool
	evaluated := 0
	for n := 0; ; n++ {
		if n > 1000 {
			t.Fatal("search did not complete")
		}
		ctx := &timeloopCancelAfter{Context: context.Background(), n: n}
		res, err := cudasearch.SearchTimeloopGEMM(ctx, bytes.NewReader(doc), opts)
		if err == nil {
			if !reflect.DeepEqual(res, full) {
				t.Fatalf("n=%d: result = %+v, want %+v", n, res, full)
			}
			break
		}
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("n=%d: error = %v, want context.Canceled", n, err)
		}
		if res.Einsum == "" {
			if partial || !reflect.DeepEqual(res, cudasearch.TimeloopGEMMResult{}) {
				t.Fatalf("n=%d: canceled before search, result = %+v", n, res)
			}
			refused = true
			continue
		}
		partial = true
		timeloopCheckIdentity(t, res, timeloopBound, timeloopRanks, "act", "wt", "out")
		stats := res.Stats
		if stats.EvaluatedCandidates < evaluated || stats.EvaluatedCandidates > full.Stats.EvaluatedCandidates {
			t.Fatalf("n=%d: evaluated %d after %d of %d", n, stats.EvaluatedCandidates, evaluated, full.Stats.EvaluatedCandidates)
		}
		if (stats.Best == nil) != (stats.EvaluatedCandidates == 0) {
			t.Fatalf("n=%d: best %+v after %d evaluations", n, stats.Best, stats.EvaluatedCandidates)
		}
		evaluated = stats.EvaluatedCandidates
	}
	if !refused || !partial {
		t.Fatalf("canceled before search: %v, during search: %v", refused, partial)
	}
}

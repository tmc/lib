package numopt

import (
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"sort"
	"testing"
)

// TestPruneEnergy checks the 2:4 pruning constant in the noise model: the
// expected share of a unit Gaussian's energy in the two smallest of every
// four magnitudes.
func TestPruneEnergy(t *testing.T) {
	r := rand.New(rand.NewPCG(1, 2))
	var lost, total float64
	for range 200000 {
		var v [4]float64
		for i := range v {
			v[i] = math.Abs(r.NormFloat64())
		}
		sort.Float64s(v[:])
		lost += v[0]*v[0] + v[1]*v[1]
		total += v[0]*v[0] + v[1]*v[1] + v[2]*v[2] + v[3]*v[3]
	}
	if got := lost / total; math.Abs(got-formats[PrecSparse2To4NVFP4].prune) > 0.002 {
		t.Errorf("pruned energy share = %.4f, model uses %.4f", got, formats[PrecSparse2To4NVFP4].prune)
	}
}

func randomTiles(r *rand.Rand, n int) []TileSensitivity {
	tiles := make([]TileSensitivity, n)
	for i := range tiles {
		tiles[i] = TileSensitivity{
			TileID:          fmt.Sprintf("t%d", i),
			HessianTrace:    math.Exp(r.NormFloat64() * 2),
			WeightKurtosis:  3 + r.ExpFloat64()*4,
			ActDynamicRange: 10 + r.Float64()*100,
			NonNegativeAct:  r.IntN(2) == 0,
			Elements:        []int{0, 64 * 128, 128 * 256}[r.IntN(3)],
		}
	}
	return tiles
}

// checkFrontier is the strict dominance oracle: points by strictly
// increasing cycles and strictly decreasing drift, each assignment valid
// and costing what the point says.
func checkFrontier(t *testing.T, tiles []TileSensitivity, arch string, frontier []ParetoPoint) {
	t.Helper()
	if len(frontier) == 0 {
		t.Fatal("empty frontier")
	}
	for i, p := range frontier {
		if err := Validate(p.Assignments, tiles, arch); err != nil {
			t.Fatalf("point %d: %v", i, err)
		}
		e, err := Evaluate(tiles, p.Assignments)
		if err != nil {
			t.Fatal(err)
		}
		if e.TotalCycles != p.TotalCycles || math.Abs(e.ExpectedKLDrift-p.ExpectedKLDrift) > 1e-9*math.Max(1, p.ExpectedKLDrift) {
			t.Fatalf("point %d claims %v, assignment evaluates to %v", i, p, e)
		}
		if i == 0 {
			continue
		}
		q := frontier[i-1]
		if !(q.TotalCycles < p.TotalCycles && q.ExpectedKLDrift > p.ExpectedKLDrift) {
			t.Fatalf("points %d (%v) and %d (%v) are not strictly ordered: one dominates", i-1, q, i, p)
		}
	}
}

func TestFrontierDominance(t *testing.T) {
	r := rand.New(rand.NewPCG(3, 4))
	for _, arch := range []string{"sm_100a", "sm_103a", "sm_107a", "sm_107f"} {
		for trial := range 5 {
			tiles := randomTiles(r, 10+trial*10)
			_, frontier, err := SolveParetoFrontier(tiles, arch, math.Inf(1))
			if err != nil {
				t.Fatalf("%s: %v", arch, err)
			}
			checkFrontier(t, tiles, arch, frontier)
		}
	}
}

// bruteForce returns the exact Pareto frontier by enumerating every
// assignment of arch's allowed formats.
func bruteForce(tiles []TileSensitivity, arch string) [][2]float64 {
	var pts [][2]float64
	var rec func(i int, cyc, drift float64)
	rec = func(i int, cyc, drift float64) {
		if i == len(tiles) {
			pts = append(pts, [2]float64{cyc, drift})
			return
		}
		for _, f := range Formats() {
			if Allowed(f, arch, tiles[i].NonNegativeAct) == nil {
				rec(i+1, cyc+float64(CycleCost(tiles[i], f)), drift+Drift(tiles[i], f))
			}
		}
	}
	rec(0, 0, 0)
	slices.SortFunc(pts, func(a, b [2]float64) int {
		if a[0] != b[0] {
			return cmpFloat(a[0], b[0])
		}
		return cmpFloat(a[1], b[1])
	})
	var out [][2]float64
	for _, p := range pts {
		// Tolerate rounding from summing drifts in another order.
		if len(out) == 0 || p[1] < out[len(out)-1][1]*(1-1e-12) {
			out = append(out, p)
		}
	}
	return out
}

func TestFrontierMatchesBruteForce(t *testing.T) {
	r := rand.New(rand.NewPCG(5, 6))
	for trial := range 20 {
		arch := []string{"sm_100a", "sm_107a"}[trial%2]
		tiles := randomTiles(r, 2+trial%5)
		_, frontier, err := SolveParetoFrontier(tiles, arch, math.Inf(1))
		if err != nil {
			t.Fatal(err)
		}
		want := bruteForce(tiles, arch)
		if len(frontier) != len(want) {
			t.Fatalf("trial %d (%s, %d tiles): frontier has %d points, brute force %d", trial, arch, len(tiles), len(frontier), len(want))
		}
		for i, p := range frontier {
			if p.TotalCycles != want[i][0] || math.Abs(p.ExpectedKLDrift-want[i][1]) > 1e-9*want[i][1] {
				t.Fatalf("trial %d point %d: got (%v, %v), brute force (%v, %v)", trial, i, p.TotalCycles, p.ExpectedKLDrift, want[i][0], want[i][1])
			}
		}
	}
}

// TestLagrangianOnFrontier cross-checks the merge with the Lagrangian
// sweep: every point the sweep finds is weakly dominated by a frontier
// point, and since the sweep's points are themselves Pareto-optimal, by
// one of equal cost.
func TestLagrangianOnFrontier(t *testing.T) {
	r := rand.New(rand.NewPCG(7, 8))
	tiles := randomTiles(r, 40)
	_, frontier, err := SolveParetoFrontier(tiles, "sm_107a", math.Inf(1))
	if err != nil {
		t.Fatal(err)
	}
	var lambdas []float64
	for e := -6.0; e <= 6; e += 0.25 {
		lambdas = append(lambdas, math.Pow(10, e))
	}
	for _, l := range lagrangian(tiles, "sm_107a", lambdas) {
		i := slices.IndexFunc(frontier, func(p ParetoPoint) bool { return p.TotalCycles == l.TotalCycles })
		if i < 0 || frontier[i].ExpectedKLDrift > l.ExpectedKLDrift*(1+1e-12) {
			t.Fatalf("Lagrangian point %v is not on the frontier", l)
		}
	}
}

func TestBestIsFastestWithinBudget(t *testing.T) {
	r := rand.New(rand.NewPCG(9, 10))
	tiles := randomTiles(r, 30)
	_, frontier, err := SolveParetoFrontier(tiles, "sm_100a", math.Inf(1))
	if err != nil {
		t.Fatal(err)
	}
	budget := frontier[len(frontier)/2].ExpectedKLDrift
	best, _, err := SolveParetoFrontier(tiles, "sm_100a", budget)
	if err != nil {
		t.Fatal(err)
	}
	if best.ExpectedKLDrift > budget {
		t.Fatalf("best drift %v over budget %v", best.ExpectedKLDrift, budget)
	}
	for _, p := range frontier {
		if p.ExpectedKLDrift <= budget && p.TotalCycles < best.TotalCycles {
			t.Fatalf("%v is faster than best %v and within budget", p, best)
		}
	}
	_, _, err = SolveParetoFrontier(tiles, "sm_100a", frontier[len(frontier)-1].ExpectedKLDrift/2)
	if !errors.Is(err, ErrInfeasible) {
		t.Fatalf("budget below every point: err = %v, want ErrInfeasible", err)
	}
}

func outlierLayer() []TileSensitivity {
	tiles := make([]TileSensitivity, 64)
	for i := range tiles {
		tiles[i] = TileSensitivity{TileID: fmt.Sprintf("k%02d", i), HessianTrace: 1, WeightKurtosis: 3, ActDynamicRange: 18}
		if i%16 == 5 {
			tiles[i].HessianTrace = 100
			tiles[i].WeightKurtosis = 25
		}
	}
	return tiles
}

// TestOutlierProtection is the spec's outlier oracle: at the cycle budget of
// uniform NVFP4, the four outlier tiles get 8-bit formats and the smooth
// tiles stay in 4-bit, 3-bit LUT or 2:4 sparse formats.
func TestOutlierProtection(t *testing.T) {
	for _, arch := range []string{"sm_100a", "sm_107a"} {
		t.Run(arch, func(t *testing.T) {
			tiles := outlierLayer()
			uniform := map[string]TilePrecisionFormat{}
			for _, tl := range tiles {
				uniform[tl.TileID] = PrecNVFP4ScaleVec4X
			}
			base, err := Evaluate(tiles, uniform)
			if err != nil {
				t.Fatal(err)
			}
			_, frontier, err := SolveParetoFrontier(tiles, arch, math.Inf(1))
			if err != nil {
				t.Fatal(err)
			}
			p, ok := LowestDriftWithin(frontier, base.TotalCycles)
			if !ok {
				t.Fatal("no frontier point within the uniform NVFP4 cycle budget")
			}
			for _, tl := range tiles {
				f := p.Assignments[tl.TileID]
				switch outlier := tl.HessianTrace > 1; {
				case outlier && f != PrecMXFP8E4M3 && f != PrecRubinUE5M3:
					t.Errorf("outlier %s got %v", tl.TileID, f)
				case !outlier && f != PrecNVFP4ScaleVec4X && f != PrecSparse2To4NVFP4 && f != PrecRubin3BitLUT:
					t.Errorf("smooth %s got %v", tl.TileID, f)
				}
			}
			ratio := base.ExpectedKLDrift / p.ExpectedKLDrift
			t.Logf("uniform NVFP4: %v", base)
			t.Logf("mixed:         %v (%.1fx lower drift)", p, ratio)
			if ratio < 8.5 {
				t.Errorf("drift ratio %.2f, want at least 8.5", ratio)
			}
		})
	}
}

func TestValidate(t *testing.T) {
	tiles := []TileSensitivity{
		{TileID: "signed", HessianTrace: 1, WeightKurtosis: 3},
		{TileID: "relu", HessianTrace: 1, WeightKurtosis: 3, NonNegativeAct: true},
	}
	tests := []struct {
		name   string
		arch   string
		assign map[string]TilePrecisionFormat
		ok     bool
	}{
		{"valid", "sm_107a", map[string]TilePrecisionFormat{"signed": PrecRubin3BitLUT, "relu": PrecRubinUE5M3}, true},
		{"ue5m3 on signed activations", "sm_107a", map[string]TilePrecisionFormat{"signed": PrecRubinUE5M3, "relu": PrecBF16}, false},
		{"lut on sm_100a", "sm_100a", map[string]TilePrecisionFormat{"signed": PrecRubin3BitLUT, "relu": PrecBF16}, false},
		{"ue5m3 on sm_103a", "sm_103a", map[string]TilePrecisionFormat{"signed": PrecBF16, "relu": PrecRubinUE5M3}, false},
		{"sm_121a has no tcgen05", "sm_121a", map[string]TilePrecisionFormat{"signed": PrecBF16, "relu": PrecBF16}, false},
		{"missing tile", "sm_100a", map[string]TilePrecisionFormat{"signed": PrecBF16}, false},
		{"unknown tile", "sm_100a", map[string]TilePrecisionFormat{"signed": PrecBF16, "other": PrecBF16}, false},
		{"unknown format", "sm_100a", map[string]TilePrecisionFormat{"signed": PrecBF16, "relu": 99}, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := Validate(tt.assign, tiles, tt.arch)
			if (err == nil) != tt.ok {
				t.Fatalf("Validate = %v, want ok=%v", err, tt.ok)
			}
		})
	}
	if err := Validate(nil, tiles, "sm_121a"); !errors.Is(err, ErrUnsupportedArch) {
		t.Errorf("sm_121a: %v, want ErrUnsupportedArch", err)
	}
}

// TestSolverRespectsTargets checks the solver never proposes what Validate
// refuses: no Rubin format off sm_107a, no UE5M3 on signed activations.
func TestSolverRespectsTargets(t *testing.T) {
	r := rand.New(rand.NewPCG(11, 12))
	tiles := randomTiles(r, 25)
	for _, arch := range []string{"sm_100a", "sm_107a"} {
		_, frontier, err := SolveParetoFrontier(tiles, arch, math.Inf(1))
		if err != nil {
			t.Fatal(err)
		}
		used := map[TilePrecisionFormat]bool{}
		for _, p := range frontier {
			for _, f := range p.Assignments {
				used[f] = true
			}
		}
		if arch == "sm_107a" && !used[PrecRubin3BitLUT] && !used[PrecRubinUE5M3] {
			t.Errorf("%s: no Rubin format on any frontier point; the test does not exercise the gate", arch)
		}
	}
}

func TestSolveErrors(t *testing.T) {
	ok := TileSensitivity{TileID: "a", HessianTrace: 1, WeightKurtosis: 3}
	tests := []struct {
		name  string
		tiles []TileSensitivity
		arch  string
		max   float64
	}{
		{"no tiles", nil, "sm_100a", 1},
		{"duplicate id", []TileSensitivity{ok, ok}, "sm_100a", 1},
		{"empty id", []TileSensitivity{{HessianTrace: 1}}, "sm_100a", 1},
		{"negative trace", []TileSensitivity{{TileID: "a", HessianTrace: -1}}, "sm_100a", 1},
		{"nan kurtosis", []TileSensitivity{{TileID: "a", WeightKurtosis: math.NaN()}}, "sm_100a", 1},
		{"unknown arch", []TileSensitivity{ok}, "sm_90a", 1},
		{"negative budget", []TileSensitivity{ok}, "sm_100a", -1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, _, err := SolveParetoFrontier(tt.tiles, tt.arch, tt.max); err == nil {
				t.Fatal("no error")
			}
		})
	}
}

func TestNoiseModel(t *testing.T) {
	smooth := TileSensitivity{TileID: "s", HessianTrace: 1, WeightKurtosis: 3}
	heavy := smooth
	heavy.WeightKurtosis = 30
	wide := smooth
	wide.ActDynamicRange = 61.6
	tests := []struct {
		name string
		t    TileSensitivity
		f    TilePrecisionFormat
		want float64
	}{
		{"nvfp4 gaussian", smooth, PrecNVFP4ScaleVec4X, 0.012},
		{"nvfp4 kurtosis 30", heavy, PrecNVFP4ScaleVec4X, 0.12},
		{"mxfp8 ignores kurtosis", heavy, PrecMXFP8E4M3, 0x1p-6 / 12},
		{"nvfp4 40 dB past range", wide, PrecNVFP4ScaleVec4X, 0.036},
		{"bf16 has no range limit", wide, PrecBF16, 0x1p-14 / 12},
		{"sparse prune not grown", heavy, PrecSparse2To4NVFP4, 0.1325 + 0.012*(1-0.1325)*10},
	}
	for _, tt := range tests {
		if got := NoiseVariance(tt.t, tt.f); math.Abs(got-tt.want) > 1e-12 {
			t.Errorf("%s: NoiseVariance = %v, want %v", tt.name, got, tt.want)
		}
	}
	// Cycle ordering the model assumes for a weight-streaming tile.
	var prev int64 = math.MaxInt64
	for _, f := range []TilePrecisionFormat{PrecBF16, PrecMXFP8E4M3, PrecNVFP4ScaleVec4X, PrecRubin3BitLUT, PrecSparse2To4NVFP4} {
		c := CycleCost(smooth, f)
		if c >= prev {
			t.Errorf("CycleCost(%v) = %d, not below the previous format's %d", f, c, prev)
		}
		prev = c
	}
}

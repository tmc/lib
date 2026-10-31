package numopt

import (
	"errors"
	"fmt"
	"math"
	"slices"
	"strings"
)

// A TilePrecisionFormat is the operand format one K-tile's tcgen05.mma
// reads. All formats accumulate into the same FP32 TMEM accumulator.
type TilePrecisionFormat uint8

const (
	PrecBF16            TilePrecisionFormat = iota // kind::f16, 2 bytes per element
	PrecRubinUE5M3                                 // unsigned 8-bit, 5 exponent bits; sm_107a only; non-negative operands only
	PrecMXFP8E4M3                                  // kind::mxf8f6f4, E4M3 with a UE8M0 scale per 32 elements
	PrecNVFP4ScaleVec4X                            // kind::mxf4nvf4 scale_vec::4X, E2M1 with a scale per 16 elements
	PrecRubin3BitLUT                               // decompress::lut::b.block32: 3-bit index into 8 E4M3 centroids; sm_107a only
	PrecSparse2To4NVFP4                            // tcgen05.mma.sp kind::mxf4nvf4: 2:4 structured sparse NVFP4

	numFormats = iota
)

// formatModel holds the assumed constants of one format. None is measured;
// see [NoiseVariance] and [CycleCost].
type formatModel struct {
	name    string
	bytes   float64 // operand bytes per dense element, scales and metadata included
	speedup float64 // dense math rate relative to BF16
	relMSE  float64 // relative mean squared error on Gaussian data (kurtosis 3)
	prune   float64 // part of relMSE that is pruning, not rounding; not grown by kurtosis
	tail    float64 // exponent on kurtosis/3 growing the rounding error
	rangeDB float64 // representable dynamic range within one scale block
	rubin   bool    // sm_107a/sm_107f only
}

// Relative MSE: BF16 and the 8-bit formats round to their mantissa width
// (2^-2m/12 with m = 7 and 3). NVFP4 is taken at the 16-level Lloyd-Max
// distortion of a unit Gaussian (0.0095) rounded up to 0.012 for the
// uniform E2M1 grid; LUT3 at the 8-level Lloyd-Max distortion (0.0345),
// which a learned codebook can at best reach. 2:4 pruning of a Gaussian
// discards the two smallest of every four magnitudes, 0.1325 of the energy
// (see TestPruneEnergy), and the kept half is rounded as NVFP4.
var formats = [numFormats]formatModel{
	PrecBF16:            {name: "BF16", bytes: 2, speedup: 1, relMSE: 0x1p-14 / 12, rangeDB: math.Inf(1)},
	PrecRubinUE5M3:      {name: "RubinUE5M3", bytes: 1, speedup: 2, relMSE: 0x1p-6 / 12, rangeDB: 180, rubin: true},
	PrecMXFP8E4M3:       {name: "MXFP8E4M3", bytes: 1 + 1.0/32, speedup: 2, relMSE: 0x1p-6 / 12, rangeDB: 107},
	PrecNVFP4ScaleVec4X: {name: "NVFP4ScaleVec4X", bytes: 0.5 + 1.0/16, speedup: 4, relMSE: 0.012, tail: 1, rangeDB: 21.6},
	PrecRubin3BitLUT:    {name: "Rubin3BitLUT", bytes: 0.375 + 1.0/32, speedup: 2, relMSE: 0.0345, tail: 0.5, rangeDB: 18, rubin: true},
	PrecSparse2To4NVFP4: {name: "Sparse2To4NVFP4", bytes: 0.25 + 1.0/32 + 1.0/16, speedup: 8, relMSE: 0.1325 + 0.012*(1-0.1325), prune: 0.1325, tail: 1, rangeDB: 21.6},
}

func (f TilePrecisionFormat) String() string {
	if int(f) < numFormats {
		return formats[f].name
	}
	return fmt.Sprintf("TilePrecisionFormat(%d)", f)
}

// BytesPerElement returns the operand bytes one dense element costs in f,
// including block scales and sparsity metadata.
func (f TilePrecisionFormat) BytesPerElement() float64 { return formats[f].bytes }

// Formats returns every format in declaration order.
func Formats() []TilePrecisionFormat {
	out := make([]TilePrecisionFormat, numFormats)
	for i := range out {
		out[i] = TilePrecisionFormat(i)
	}
	return out
}

// ErrUnsupportedArch is returned for a target with no tcgen05 tensor cores.
var ErrUnsupportedArch = errors.New("numopt: target has no tcgen05")

// ErrInfeasible is returned by [SolveParetoFrontier] when no assignment meets
// the drift budget.
var ErrInfeasible = errors.New("numopt: no assignment within drift budget")

// rubinArch reports whether arch is a Vera Rubin target, and an error if it
// is not a tcgen05 target at all.
func rubinArch(arch string) (bool, error) {
	switch arch {
	case "sm_100a", "sm_100f", "sm_103a", "sm_103f":
		return false, nil
	case "sm_107a", "sm_107f":
		return true, nil
	case "sm_121a", "sm_121f", "sm_120a", "sm_120f":
		return false, fmt.Errorf("%w: %s (no tcgen05 on the device; emulation only)", ErrUnsupportedArch, arch)
	}
	return false, fmt.Errorf("%w: %q (want sm_100a, sm_103a or sm_107a, or an f variant)", ErrUnsupportedArch, arch)
}

// Allowed reports whether f can run on arch for a tile whose activations
// are non-negative (nonNeg) or signed.
func Allowed(f TilePrecisionFormat, arch string, nonNeg bool) error {
	if int(f) >= numFormats {
		return fmt.Errorf("unknown format %d", f)
	}
	rubin, err := rubinArch(arch)
	if err != nil {
		return err
	}
	if formats[f].rubin && !rubin {
		return fmt.Errorf("%v requires sm_107a or sm_107f, not %s", f, arch)
	}
	if f == PrecRubinUE5M3 && !nonNeg {
		return fmt.Errorf("%v is unsigned and the tile's activations are signed", f)
	}
	return nil
}

// TileSensitivity is the calibration statistics of one K-tile.
type TileSensitivity struct {
	TileID          string
	HessianTrace    float64 // E[‖X_tile‖² · ‖∂L/∂Y_tile‖²]
	WeightKurtosis  float64 // Pearson kurtosis of the weights; 3 for a Gaussian
	ActDynamicRange float64 // 20·log10(max|X| / median|X|), in dB
	NonNegativeAct  bool    // every activation ≥ 0 (after ReLU or softmax)
	Elements        int     // weight elements in the tile; 0 means 128×128
}

// NoiseVariance returns the modelled relative mean squared error of t in f:
//
//	σ² = prune + (relMSE − prune) · (κ/3)^tail · (1 + max(0, DR − range)/20)
//
// where κ is the weight kurtosis (floored at 3), DR the activation dynamic
// range and range the format's dynamic range within one scale block, both in
// dB. The kurtosis term models a block scale set by the block's largest
// value; it applies to the 4-bit and 3-bit block-scaled formats only, and
// at half strength to the LUT, whose centroids adapt to the distribution.
// The range term adds one unit of error for each 20 dB the activations
// exceed the format. Both are modelling choices, not measured laws.
func NoiseVariance(t TileSensitivity, f TilePrecisionFormat) float64 {
	m := formats[f]
	k := max(t.WeightKurtosis, 3) / 3
	r := 1.0
	if !math.IsInf(m.rangeDB, 1) && t.ActDynamicRange > m.rangeDB {
		r += (t.ActDynamicRange - m.rangeDB) / 20
	}
	return m.prune + (m.relMSE-m.prune)*math.Pow(k, m.tail)*r
}

// Drift returns the modelled expected loss perturbation ½·Tr(H)·σ² of t in f.
func Drift(t TileSensitivity, f TilePrecisionFormat) float64 {
	return 0.5 * t.HessianTrace * NoiseVariance(t, f)
}

// Assumed per-SM rates for [CycleCost]: dense BF16 math in FLOP per cycle,
// operand bytes per cycle, and rows of the activation tile. They give
// round numbers near a B200 SM (2.25 PFLOP/s dense BF16 and 8 TB/s over
// 148 SMs at about 1.9 GHz) and are not calibrated.
const (
	bf16FlopsPerCycle = 8192
	bytesPerCycle     = 32
	tileRows          = 128
)

// CycleCost returns the modelled cycles one SM spends on t in f: the larger
// of math time (2·rows·elements FLOP at the format's rate) and weight
// operand time (elements·bytes at a fixed byte rate), rounded up. The rates
// are assumptions: dense math 1×, 2×, 4× BF16 for 16-, 8- and 4-bit
// operands and 8× for 2:4 sparse NVFP4; the LUT format runs at the 8-bit
// rate because it is decompressed into kind::mxf8f6f4.
func CycleCost(t TileSensitivity, f TilePrecisionFormat) int64 {
	n := float64(t.Elements)
	if t.Elements <= 0 {
		n = 128 * 128
	}
	m := formats[f]
	math_ := 2 * tileRows * n / (bf16FlopsPerCycle * m.speedup)
	mem := n * m.bytes / bytesPerCycle
	return int64(math.Ceil(max(math_, mem)))
}

// A ParetoPoint is one assignment of formats to tiles and its modelled cost.
type ParetoPoint struct {
	TotalCycles     float64
	ExpectedKLDrift float64
	Assignments     map[string]TilePrecisionFormat
}

func (p ParetoPoint) String() string {
	count := make([]int, numFormats)
	for _, f := range p.Assignments {
		count[f]++
	}
	var b strings.Builder
	fmt.Fprintf(&b, "cycles=%.0f drift=%.4g", p.TotalCycles, p.ExpectedKLDrift)
	for f, n := range count {
		if n > 0 {
			fmt.Fprintf(&b, " %v=%d", TilePrecisionFormat(f), n)
		}
	}
	return b.String()
}

func checkTiles(tiles []TileSensitivity) error {
	if len(tiles) == 0 {
		return fmt.Errorf("no tiles")
	}
	seen := make(map[string]bool, len(tiles))
	for i, t := range tiles {
		if t.TileID == "" {
			return fmt.Errorf("tile %d has no id", i)
		}
		if seen[t.TileID] {
			return fmt.Errorf("duplicate tile id %q", t.TileID)
		}
		seen[t.TileID] = true
		for _, v := range []float64{t.HessianTrace, t.WeightKurtosis, t.ActDynamicRange} {
			if math.IsNaN(v) || math.IsInf(v, 0) || v < 0 {
				return fmt.Errorf("tile %q: statistic %v is not a finite non-negative number", t.TileID, v)
			}
		}
		if t.Elements < 0 {
			return fmt.Errorf("tile %q: %d elements", t.TileID, t.Elements)
		}
	}
	return nil
}

// Validate reports whether assign gives every tile exactly one format that
// arch can run for that tile.
func Validate(assign map[string]TilePrecisionFormat, tiles []TileSensitivity, arch string) error {
	if err := checkTiles(tiles); err != nil {
		return fmt.Errorf("validate: %w", err)
	}
	if _, err := rubinArch(arch); err != nil {
		return fmt.Errorf("validate: %w", err)
	}
	if len(assign) != len(tiles) {
		return fmt.Errorf("validate: %d assignments for %d tiles", len(assign), len(tiles))
	}
	for _, t := range tiles {
		f, ok := assign[t.TileID]
		if !ok {
			return fmt.Errorf("validate: tile %q unassigned", t.TileID)
		}
		if err := Allowed(f, arch, t.NonNegativeAct); err != nil {
			return fmt.Errorf("validate: tile %q: %w", t.TileID, err)
		}
	}
	return nil
}

// Evaluate returns the modelled cost of assign.
func Evaluate(tiles []TileSensitivity, assign map[string]TilePrecisionFormat) (ParetoPoint, error) {
	if err := checkTiles(tiles); err != nil {
		return ParetoPoint{}, fmt.Errorf("evaluate: %w", err)
	}
	p := ParetoPoint{Assignments: make(map[string]TilePrecisionFormat, len(tiles))}
	for _, t := range tiles {
		f, ok := assign[t.TileID]
		if !ok || int(f) >= numFormats {
			return ParetoPoint{}, fmt.Errorf("evaluate: tile %q has no valid format", t.TileID)
		}
		p.TotalCycles += float64(CycleCost(t, f))
		p.ExpectedKLDrift += Drift(t, f)
		p.Assignments[t.TileID] = f
	}
	return p, nil
}

// An option is one format choice for one tile.
type option struct {
	f     TilePrecisionFormat
	cyc   int64
	drift float64
}

// options returns the non-dominated formats arch allows for t, by
// increasing cycles. Dropping a dominated option loses no frontier point:
// swapping it for the option that dominates it never makes an assignment
// worse in either objective.
func options(t TileSensitivity, arch string) []option {
	var all []option
	for _, f := range Formats() {
		if Allowed(f, arch, t.NonNegativeAct) != nil {
			continue
		}
		all = append(all, option{f, CycleCost(t, f), Drift(t, f)})
	}
	return paretoOptions(all)
}

func paretoOptions(all []option) []option {
	slices.SortStableFunc(all, func(a, b option) int {
		if a.cyc != b.cyc {
			return cmpInt(a.cyc, b.cyc)
		}
		return cmpFloat(a.drift, b.drift)
	})
	var out []option
	for _, o := range all {
		if len(out) == 0 || o.drift < out[len(out)-1].drift {
			out = append(out, o)
		}
	}
	return out
}

func cmpInt(a, b int64) int {
	switch {
	case a < b:
		return -1
	case a > b:
		return 1
	}
	return 0
}

func cmpFloat(a, b float64) int {
	switch {
	case a < b:
		return -1
	case a > b:
		return 1
	}
	return 0
}

// A node is one partial assignment on the frontier after some tiles.
type node struct {
	cyc    int64
	drift  float64
	parent int32 // index in the previous layer
	f      TilePrecisionFormat
}

// prune sorts cand by cycles and keeps the points whose drift is strictly
// below that of every cheaper point: the Pareto frontier of cand.
func prune(cand []node) []node {
	slices.SortStableFunc(cand, func(a, b node) int {
		if a.cyc != b.cyc {
			return cmpInt(a.cyc, b.cyc)
		}
		return cmpFloat(a.drift, b.drift)
	})
	out := cand[:0]
	for _, n := range cand {
		if len(out) == 0 || n.drift < out[len(out)-1].drift {
			out = append(out, n)
		}
	}
	return slices.Clip(out)
}

// SolveParetoFrontier returns the Pareto frontier of (TotalCycles,
// ExpectedKLDrift) over every assignment of arch's formats to tiles, by
// increasing cycles and strictly decreasing drift, and the fastest frontier
// point with drift at most maxKLDrift.
//
// The frontier is exact for the model of [CycleCost] and [Drift]: tiles are
// merged one at a time, each partial frontier extended by every option of
// the next tile and pruned of dominated points, which loses nothing
// because a dominated partial assignment extends only to dominated ones.
// Cycle costs are integers, so a frontier never holds more points than the
// distinct cycle totals.
//
// If no frontier point is within budget, it returns the frontier and an
// error wrapping [ErrInfeasible].
func SolveParetoFrontier(tiles []TileSensitivity, arch string, maxKLDrift float64) (best ParetoPoint, frontier []ParetoPoint, err error) {
	if err := checkTiles(tiles); err != nil {
		return ParetoPoint{}, nil, fmt.Errorf("solve pareto frontier: %w", err)
	}
	if _, err := rubinArch(arch); err != nil {
		return ParetoPoint{}, nil, fmt.Errorf("solve pareto frontier: %w", err)
	}
	if math.IsNaN(maxKLDrift) || maxKLDrift < 0 {
		return ParetoPoint{}, nil, fmt.Errorf("solve pareto frontier: drift budget %v", maxKLDrift)
	}
	layers := make([][]node, len(tiles))
	prev := []node{{parent: -1}}
	for i, t := range tiles {
		opts := options(t, arch)
		cand := make([]node, 0, len(prev)*len(opts))
		for j, p := range prev {
			for _, o := range opts {
				cand = append(cand, node{p.cyc + o.cyc, p.drift + o.drift, int32(j), o.f})
			}
		}
		layers[i] = prune(cand)
		prev = layers[i]
	}
	last := layers[len(layers)-1]
	frontier = make([]ParetoPoint, len(last))
	for j, n := range last {
		p := ParetoPoint{TotalCycles: float64(n.cyc), ExpectedKLDrift: n.drift, Assignments: make(map[string]TilePrecisionFormat, len(tiles))}
		k := int32(j)
		for i := len(tiles) - 1; i >= 0; i-- {
			n := layers[i][k]
			p.Assignments[tiles[i].TileID] = n.f
			k = n.parent
		}
		frontier[j] = p
	}
	for _, p := range frontier {
		if p.ExpectedKLDrift <= maxKLDrift {
			return p, frontier, nil
		}
	}
	return ParetoPoint{}, frontier, fmt.Errorf("solve pareto frontier: lowest drift %.4g exceeds %.4g: %w",
		frontier[len(frontier)-1].ExpectedKLDrift, maxKLDrift, ErrInfeasible)
}

// LowestDriftWithin returns the frontier point with the lowest drift whose
// cycles are at most budget, and false if every point costs more.
func LowestDriftWithin(frontier []ParetoPoint, budget float64) (ParetoPoint, bool) {
	var best ParetoPoint
	found := false
	for _, p := range frontier {
		if p.TotalCycles <= budget && (!found || p.ExpectedKLDrift < best.ExpectedKLDrift) {
			best, found = p, true
		}
	}
	return best, found
}

// lagrangian returns, for each λ, the assignment minimising
// cycles/C₀ + λ·drift/D₀ tile by tile, where C₀ and D₀ normalise the two
// objectives. Each is a supported point of the frontier (one on its convex
// hull); the sweep finds no point in a non-convex part, which is why the
// solver merges frontiers instead. It is kept as an independent check.
func lagrangian(tiles []TileSensitivity, arch string, lambdas []float64) []ParetoPoint {
	var c0, d0 float64
	opts := make([][]option, len(tiles))
	for i, t := range tiles {
		opts[i] = options(t, arch)
		c0 += float64(opts[i][len(opts[i])-1].cyc)
		d0 += opts[i][0].drift
	}
	if d0 == 0 {
		d0 = 1
	}
	var out []ParetoPoint
	for _, l := range lambdas {
		p := ParetoPoint{Assignments: make(map[string]TilePrecisionFormat, len(tiles))}
		for i, t := range tiles {
			bi, bv := 0, math.Inf(1)
			for j, o := range opts[i] {
				if v := float64(o.cyc)/c0 + l*o.drift/d0; v < bv {
					bi, bv = j, v
				}
			}
			o := opts[i][bi]
			p.TotalCycles += float64(o.cyc)
			p.ExpectedKLDrift += o.drift
			p.Assignments[t.TileID] = o.f
		}
		out = append(out, p)
	}
	return out
}

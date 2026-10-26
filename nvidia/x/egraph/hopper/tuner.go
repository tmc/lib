package hopper

import (
	"errors"
	"fmt"
	"math"
	"regexp"
	"slices"
	"strconv"
	"strings"

	"github.com/tmc/lib/nvidia/x/pairedtiming"
	"github.com/tmc/lib/nvidia/x/ptxasinfo"
)

// Hopper SM hardware resource limits used by [FilterVariantsByPTXAS],
// [ResidentCTAsPerSM], and [RankVariants].
const (
	HopperSMRegisters       = 65536
	HopperMaxWarpsPerSM     = 64
	HopperMaxSharedPerSM    = 228 * 1024
	HopperMaxRegistersPerTh = 255
	HopperMaxCTAsPerSM      = 32

	// DefaultMinGain is the campaign promotion threshold: a candidate must
	// exceed a 1.015 median ABBA ratio over the baseline or previous winner
	// to be promoted by [SelectByPairedTiming].
	DefaultMinGain = 1.015
)

// StaticGate specifies the static compiler resource constraints that a
// [Variant] must satisfy in [FilterVariantsByPTXAS].
type StaticGate struct {
	// NoSpills rejects any variant whose entry or device function spills
	// registers to local memory (SpillStores > 0 || SpillLoads > 0).
	// Spills are also rejected by default unless AllowSpills is true.
	NoSpills bool

	// AllowSpills permits register spills when NoSpills is false.
	AllowSpills bool

	// MaxRegisters is the maximum registers per thread (0 defaults to 255).
	MaxRegisters int

	// MinCTAsPerSM is the minimum resident CTAs per SM required on Hopper
	// (0 defaults to 1).
	MinCTAsPerSM int
}

// A RejectedVariant is a [Variant] rejected by [FilterVariantsByPTXAS] along
// with its static compiler resource metrics and rejection reasons.
type RejectedVariant struct {
	Variant     Variant                `json:"variant"`
	Reasons     []string               `json:"reasons"`
	Registers   int                    `json:"registers,omitempty"`
	SharedBytes int                    `json:"shared_bytes,omitempty"`
	SpillStores int                    `json:"spill_stores,omitempty"`
	SpillLoads  int                    `json:"spill_loads,omitempty"`
	CTAsPerSM   int                    `json:"ctas_per_sm"`
	Diagnostics []ptxasinfo.Diagnostic `json:"diagnostics,omitempty"`
}

// Reason returns the semicolon-separated rejection reasons.
func (r RejectedVariant) Reason() string {
	return strings.Join(r.Reasons, "; ")
}

// FilterVariantsByPTXAS filters variants against a parsed ptxas -v report for
// arch, rejecting any variant that spills registers (unless permitted by gate),
// triggers wgmma serialization (such as C7519), exceeds gate.MaxRegisters, or
// falls below gate.MinCTAsPerSM resident CTAs per SM on Hopper.
//
// Surviving variants have their ptxas register and static shared-memory usage
// attached to their Launch metadata for downstream wave-quantized ranking.
func FilterVariantsByPTXAS(variants []Variant, report ptxasinfo.Report, arch string, gate StaticGate) ([]Variant, []RejectedVariant) {
	maxRegs := gate.MaxRegisters
	if maxRegs <= 0 {
		maxRegs = HopperMaxRegistersPerTh
	}
	minCTAs := max(gate.MinCTAsPerSM, 1)
	noSpills := gate.NoSpills || !gate.AllowSpills

	var (
		kept     []Variant
		rejected []RejectedVariant
	)
	for _, v := range variants {
		fns := lookupVariantFunctions(report, v.Name, arch)
		var (
			regs         int
			staticShared int
			spillStores  int
			spillLoads   int
			diags        []ptxasinfo.Diagnostic
			serialDiags  []ptxasinfo.Diagnostic
			spills       bool
		)
		for _, f := range fns {
			if f.Registers > regs {
				regs = f.Registers
			}
			if f.SharedMem > staticShared {
				staticShared = f.SharedMem
			}
			spillStores += f.SpillStores
			spillLoads += f.SpillLoads
			if f.Spills() {
				spills = true
			}
			diags = append(diags, f.Diagnostics...)
			for _, d := range f.Diagnostics {
				if d.WGMMASerialized() || d.Code == "C7519" {
					serialDiags = append(serialDiags, d)
				}
			}
		}

		ctasPerSM := ResidentCTAsPerSM(v, regs, staticShared)
		var reasons []string
		if noSpills && spills {
			reasons = append(reasons, fmt.Sprintf("spills registers (%dB store, %dB load)", spillStores, spillLoads))
		}
		if len(serialDiags) > 0 {
			code := serialDiags[0].Code
			if code == "" {
				code = "C7519"
			}
			reasons = append(reasons, fmt.Sprintf("wgmma serialized (%s)", code))
		}
		if regs > maxRegs {
			reasons = append(reasons, fmt.Sprintf("registers %d > max %d", regs, maxRegs))
		}
		if ctasPerSM < minCTAs {
			reasons = append(reasons, fmt.Sprintf("resident CTAs/SM %d < min %d", ctasPerSM, minCTAs))
		}

		if len(reasons) > 0 {
			rejected = append(rejected, RejectedVariant{
				Variant:     v,
				Reasons:     reasons,
				Registers:   regs,
				SharedBytes: v.Launch.SharedBytes + staticShared,
				SpillStores: spillStores,
				SpillLoads:  spillLoads,
				CTAsPerSM:   ctasPerSM,
				Diagnostics: diags,
			})
			continue
		}
		if regs > 0 {
			v.Launch.Registers = regs
		}
		if staticShared > 0 {
			v.Launch.ptxasShared = staticShared
		}
		kept = append(kept, v)
	}
	return kept, rejected
}

// ResidentCTAsPerSM computes the maximum resident CTAs per SM on Hopper from
// the SM register file (65536 registers), warp limit (64 warps), and shared
// memory capacity (228 KiB) against v.Launch.Block, regs per thread, and
// v.Launch.SharedBytes + staticShared.
func ResidentCTAsPerSM(v Variant, regs, staticShared int) int {
	threads := v.Launch.Block
	if threads <= 0 {
		threads = max(v.Launch.Warps*32, 128)
	}
	warps := (threads + 31) / 32
	if v.Launch.Warps > warps {
		warps = v.Launch.Warps
	}
	ctas := min(HopperMaxCTAsPerSM, HopperMaxWarpsPerSM/max(warps, 1))

	if regs > 0 {
		if regs > HopperMaxRegistersPerTh {
			return 0
		}
		regCTAs := HopperSMRegisters / (regs * threads)
		ctas = min(ctas, regCTAs)
	}

	sharedBytes := v.Launch.SharedBytes + staticShared
	if sharedBytes == 0 && v.Op == "paged_attention" {
		if a, ok := reconstructPagedAttention(v); ok {
			sharedBytes = a.SharedBytes()
		}
	}
	if sharedBytes > HopperMaxSharedPerSM {
		return 0
	}
	if sharedBytes > 0 {
		smemCTAs := HopperMaxSharedPerSM / sharedBytes
		ctas = min(ctas, smemCTAs)
	}
	return max(ctas, 0)
}

func lookupVariantFunctions(report ptxasinfo.Report, variantName, arch string) []*ptxasinfo.Function {
	var fns []*ptxasinfo.Function
	collect := func(matchArch func(string) bool) {
		for _, t := range report.Targets {
			if !matchArch(t.Arch) {
				continue
			}
			for _, f := range t.Functions {
				if matchesVariantFunction(f.Name, variantName) {
					fns = append(fns, f)
				}
			}
		}
	}
	collect(func(tArch string) bool {
		return arch == "" || tArch == "" || tArch == arch
	})
	if len(fns) == 0 && arch != "" {
		baseArch := strings.TrimSuffix(arch, "a")
		collect(func(tArch string) bool {
			return strings.TrimSuffix(tArch, "a") == baseArch
		})
	}
	return fns
}

func matchesVariantFunction(funcName, variantName string) bool {
	if variantName == "" {
		return false
	}
	if funcName == variantName || funcName == variantName+"_launch" || funcName == variantName+"_launch_c" || funcName == variantName+"_wgmma" {
		return true
	}
	mangled := fmt.Sprintf("_Z%d%s", len(variantName), variantName)
	if strings.HasPrefix(funcName, mangled) {
		return true
	}
	mangledWGMMA := fmt.Sprintf("_Z%d%s_wgmma", len(variantName)+6, variantName)
	return strings.HasPrefix(funcName, mangledWGMMA)
}

// TuningShape specifies the concrete runtime dimensions used by [RankVariants]
// to price and rank kernel variants.
type TuningShape struct {
	M       int // activation rows for SkinnyGEMM and TileGEMM
	Batch   int // request batch size for PagedAttention (0 defaults to max(M, 1))
	MaxKeys int // maximum attended sequence length for PagedAttention
	Experts int // active expert count for grouped TileGEMM (0 defaults to 1)
}

// A RankedVariant is a [Variant] scored by [RankVariants] for a concrete
// [TuningShape] and [CostConfig].
type RankedVariant struct {
	Variant           Variant `json:"variant"`
	EstimatedNs       float64 `json:"estimated_ns"`
	NumCTAs           int     `json:"num_ctas"`
	ResidentCTAsPerSM int     `json:"resident_ctas_per_sm"`
	Waves             int     `json:"waves"`
	FractionalWaves   float64 `json:"fractional_waves"`
	WorkloadBytes     uint64  `json:"workload_bytes"`
	CombineBytes      uint64  `json:"combine_bytes"`
	WorkloadFLOPs     uint64  `json:"workload_flops"`
}

// RankVariants ranks variants for shape under cfg using a wave-quantized
// analytical model and returns them sorted from lowest EstimatedNs to highest.
func RankVariants(variants []Variant, shape TuningShape, cfg CostConfig) []RankedVariant {
	normCfg, err := cfg.norm()
	if err != nil {
		normCfg = CostConfig{
			TensorTFLOPS: 778.0,
			SMs:          132,
			LaunchNs:     2000.0,
		}
		normCfg.Ceiling.PeakTFLOPS = 67.0
		normCfg.Ceiling.PeakMemoryGBps = 2740.48
	}
	if shape.MaxKeys <= 0 && normCfg.MaxKeys > 0 {
		shape.MaxKeys = normCfg.MaxKeys
	}

	ranked := make([]RankedVariant, len(variants))
	for i, v := range variants {
		ranked[i] = scoreVariant(v, shape, normCfg)
	}
	slices.SortFunc(ranked, func(a, b RankedVariant) int {
		if a.EstimatedNs < b.EstimatedNs {
			return -1
		}
		if a.EstimatedNs > b.EstimatedNs {
			return 1
		}
		return strings.Compare(a.Variant.Name, b.Variant.Name)
	})
	return ranked
}

var maxMRE = regexp.MustCompile(`\bMAXM\s*=\s*(\d+)`)

func scoreVariant(v Variant, shape TuningShape, cfg CostConfig) RankedVariant {
	regs := v.Launch.Registers
	staticShared := v.Launch.ptxasShared
	if regs <= 0 {
		regs = defaultVariantRegisters(v)
	}
	resCTAs := max(ResidentCTAsPerSM(v, regs, staticShared), 1)
	slots := max(cfg.SMs*resCTAs, 1)

	rv := RankedVariant{
		Variant:           v,
		ResidentCTAsPerSM: resCTAs,
	}

	bw := cfg.Ceiling.PeakMemoryGBps
	tflops := cfg.Ceiling.PeakTFLOPS

	switch v.Op {
	case "skinny_gemm":
		meta := parseGEMMKey(v.Key)
		m := max(shape.M, 1)
		maxM := parseSkinnyMaxM(v.Source)
		rowsPerBlock := max(v.Launch.RowsPerBlock, 1)
		splitK := max(v.Launch.SplitK, 1)
		rv.NumCTAs = max(((meta.n+rowsPerBlock-1)/rowsPerBlock)*splitK, 1)
		rv.FractionalWaves = float64(rv.NumCTAs) / float64(slots)
		rv.Waves = max((rv.NumCTAs+slots-1)/slots, 1)

		wBytes := weightFormatBytes(meta.weight, meta.n, meta.k)
		xBytes := uint64(m*meta.k*2) * uint64(splitK)
		extraBytes := extraTrafficBytes(meta.extras, m, meta.n)
		outBytes := uint64(m * meta.n * meta.out.Bits() / 8)
		if splitK > 1 {
			rv.CombineBytes = uint64(2*splitK*m*meta.n*4 + 4*(meta.n/rowsPerBlock))
		}
		rv.WorkloadBytes = wBytes + xBytes + extraBytes + outBytes + rv.CombineBytes
		rv.WorkloadFLOPs = uint64(2*m*meta.n*meta.k + m*meta.n*max(len(meta.extras)*2, 1))

		if v.Launch.Prefetch > 0 {
			bw *= 1.15
		}
		baseBytes := rv.WorkloadBytes - rv.CombineBytes
		workNs := max(float64(baseBytes)/bw, float64(rv.WorkloadFLOPs)/(tflops*1000.0))
		waveFactor := float64(rv.Waves) / rv.FractionalWaves
		combineNs := float64(rv.CombineBytes) / bw
		rv.EstimatedNs = workNs*waveFactor + combineNs + cfg.LaunchNs
		if maxM > 0 && m > maxM {
			// SkinnyGEMM is only instantiated up to MAXM.
			rv.EstimatedNs += 1e12
		}
		return rv

	case "tile_gemm":
		meta := parseGEMMKey(v.Key)
		m := max(shape.M, 1)
		experts := max(shape.Experts, 1)
		bm := max(v.Launch.BM, 128)
		bn := max(v.Launch.BN, 128)
		tilesM := max((m+bm-1)/bm, 1)
		tilesN := max((meta.n+bn-1)/bn, 1)
		totalTiles := tilesM * tilesN * experts

		if v.Launch.StreamK > 0 {
			rv.NumCTAs = v.Launch.StreamK
		} else {
			rv.NumCTAs = totalTiles
		}
		rv.FractionalWaves = float64(rv.NumCTAs) / float64(slots)
		rv.Waves = max((rv.NumCTAs+slots-1)/slots, 1)

		nout := meta.n
		if meta.glu {
			nout = meta.n / 2
		}
		wBytes := uint64(experts) * weightFormatBytes(meta.weight, meta.n, meta.k)
		xElem := 2
		if meta.weight == WeightE4M3 {
			xElem = 1
		}
		xBytes := uint64(m * meta.k * xElem)
		if meta.gather {
			xBytes += uint64(m * 4)
		}
		if meta.grouped {
			xBytes += uint64((experts + 1) * 4)
		}
		extraBytes := extraTrafficBytes(meta.extras, m, nout)
		outBytes := uint64(m * nout * meta.out.Bits() / 8)
		if meta.sumSq {
			outBytes += uint64(m * max(nout/32, 1) * 4)
		}
		if v.Launch.StreamK > 0 {
			splitTiles := min(totalTiles, v.Launch.StreamK)
			rv.CombineBytes = uint64(2*splitTiles*bm*bn*4 + 4*v.Launch.StreamK)
		}
		rv.WorkloadBytes = wBytes + xBytes + extraBytes + outBytes + rv.CombineBytes
		paddedM := tilesM * bm
		epiFlops := 2
		if meta.glu {
			epiFlops += 4
		}
		if meta.sumSq {
			epiFlops += 2
		}
		rv.WorkloadFLOPs = uint64(2*paddedM*meta.n*meta.k + m*nout*epiFlops)

		tflops = cfg.TensorTFLOPS
		if v.Launch.MMA != "wgmma" {
			tflops *= 0.65
		}
		if v.Launch.Swizzle128B {
			bw *= 1.10
		}
		if v.Launch.GroupM > 1 {
			bw *= 1.10
		}
		if v.Launch.WarpSpecialized {
			bw *= 1.05
			tflops *= 1.05
		}

		baseBytes := rv.WorkloadBytes - rv.CombineBytes
		workNs := max(float64(baseBytes)/bw, float64(rv.WorkloadFLOPs)/(tflops*1000.0))
		waveFactor := float64(rv.Waves) / rv.FractionalWaves
		if v.Launch.StreamK > 0 {
			// Persistent Stream-K distributes tile*KT steps evenly across StreamK blocks.
			smFrac := float64(rv.NumCTAs) / float64(max(cfg.SMs, 1))
			smWaves := max((rv.NumCTAs+cfg.SMs-1)/max(cfg.SMs, 1), 1)
			waveFactor = float64(smWaves) / smFrac
		}
		combineNs := float64(rv.CombineBytes) / bw
		rv.EstimatedNs = workNs*waveFactor + combineNs + cfg.LaunchNs
		return rv

	case "paged_attention":
		a, _ := reconstructPagedAttention(v)
		batch := shape.Batch
		if batch <= 0 {
			batch = max(shape.M, 1)
		}
		qlen := max(a.QLen, 1)
		maxKeys := shape.MaxKeys
		if maxKeys <= 0 {
			if a.Window > 0 {
				maxKeys = a.Window + qlen - 1
			} else {
				maxKeys = 1024
			}
		}
		hpb := max(a.KVHeadsPerBlock, 1)
		var splits int
		if a.DeviceSplits {
			_, splits = a.PlanSplits(batch, maxKeys, slots)
		} else {
			splits = a.Splits(maxKeys)
		}
		splits = max(splits, 1)
		rv.NumCTAs = max(batch*(a.Hkv/hpb)*splits, 1)
		rv.FractionalWaves = float64(rv.NumCTAs) / float64(slots)
		rv.Waves = max((rv.NumCTAs+slots-1)/slots, 1)

		tokens := batch * qlen
		qBytes := uint64(tokens * a.Hq * HeadDim * a.Q.Bits() / 8)
		kvBytes := uint64(batch * maxKeys * a.Hkv * HeadDim * 2)
		var gateBytes uint64
		if a.Gate {
			gateBytes = uint64(tokens * a.Hq * HeadDim * 2)
		}
		outBytes := uint64(tokens * a.Hq * HeadDim * a.Out.Bits() / 8)
		if splits > 1 {
			rv.CombineBytes = 2 * uint64(a.WorkspaceBytes(batch, maxKeys))
		}
		rv.WorkloadBytes = qBytes + kvBytes + gateBytes + outBytes + rv.CombineBytes
		rv.WorkloadFLOPs = uint64(tokens * a.Hq * maxKeys * 4 * HeadDim)
		if a.Gate {
			rv.WorkloadFLOPs += uint64(tokens * a.Hq * HeadDim * 6)
		}

		tflops = cfg.TensorTFLOPS
		if v.Launch.Swizzle {
			bw *= 1.10
		}
		if v.Launch.TableAhead >= 2 {
			bw *= 1.05
		}
		if a.QKFP8 {
			tflops *= 1.05
		}
		baseBytes := rv.WorkloadBytes - rv.CombineBytes
		workNs := max(float64(baseBytes)/bw, float64(rv.WorkloadFLOPs)/(tflops*1000.0))
		waveFactor := float64(rv.Waves) / rv.FractionalWaves
		combineNs := float64(rv.CombineBytes) / bw
		rv.EstimatedNs = workNs*waveFactor + combineNs + cfg.LaunchNs
		return rv
	}

	rv.NumCTAs = 1
	rv.Waves = 1
	rv.FractionalWaves = 1.0 / float64(slots)
	rv.EstimatedNs = cfg.LaunchNs
	return rv
}

func defaultVariantRegisters(v Variant) int {
	switch v.Op {
	case "skinny_gemm":
		if v.Launch.Prefetch > 0 {
			return 96
		}
		return 64
	case "tile_gemm":
		if v.Launch.MMA == "wgmma" && v.Launch.BN >= 256 {
			return 192
		}
		return 128
	case "paged_attention":
		if v.Launch.KVDepth >= 2 {
			return 160
		}
		return 128
	default:
		return 128
	}
}

func parseSkinnyMaxM(src string) int {
	if m := maxMRE.FindStringSubmatch(src); m != nil {
		if n, err := strconv.Atoi(m[1]); err == nil {
			return n
		}
	}
	return 16
}

type gemmKeyMeta struct {
	n, k                        int
	weight                      WeightFormat
	out                         DType
	grouped, gather, glu, sumSq bool
	extras                      []Operand
}

func parseGEMMKey(key string) gemmKeyMeta {
	m := gemmKeyMeta{
		n:      128,
		k:      128,
		weight: WeightBF16,
		out:    BF16,
	}
	for _, part := range strings.Split(key, "/") {
		switch {
		case strings.HasPrefix(part, "N="):
			if v, err := strconv.Atoi(part[2:]); err == nil && v > 0 {
				m.n = v
			}
		case strings.HasPrefix(part, "K="):
			if v, err := strconv.Atoi(part[2:]); err == nil && v > 0 {
				m.k = v
			}
		case strings.HasPrefix(part, "w="):
			switch part[2:] {
			case "bf16":
				m.weight = WeightBF16
			case "bf16hilo":
				m.weight = WeightBF16HiLo
			case "nvfp4":
				m.weight = WeightNVFP4
			case "e4m3":
				m.weight = WeightE4M3
			}
		case strings.HasPrefix(part, "out="):
			m.out = parseDTypeName(part[4:])
		case strings.HasPrefix(part, "epi="):
			fields := strings.Split(part[4:], ",")
			for _, f := range fields[1:] {
				toks := strings.Split(f, ":")
				if len(toks) == 3 {
					m.extras = append(m.extras, Operand{
						Name:      toks[0],
						DType:     parseDTypeName(toks[1]),
						Broadcast: parseBroadcastName(toks[2]),
					})
				}
			}
		case part == "grouped=true":
			m.grouped = true
		case part == "gather=true":
			m.gather = true
		case part == "glu":
			m.glu = true
		case part == "sumsq":
			m.sumSq = true
		}
	}
	return m
}

func reconstructPagedAttention(v Variant) (PagedAttention, bool) {
	if v.Op != "paged_attention" {
		return PagedAttention{}, false
	}
	a := PagedAttention{
		Name:            v.Name,
		Hq:              16,
		Hkv:             2,
		QLen:            1,
		Page:            1,
		Q:               E4M3,
		Out:             F32,
		Chunk:           v.Launch.SplitKV,
		Depth:           v.Launch.KVDepth,
		Stages:          v.Launch.Stages,
		TableAhead:      v.Launch.TableAhead,
		KVHeadsPerBlock: v.Launch.KVHeads,
		DeviceSplits:    v.Launch.DeviceSplits,
		Swizzle:         v.Launch.Swizzle,
		ClusterCombine:  v.Launch.ClusterCombine,
	}
	if a.Name == "" {
		a.Name = "attn"
	}
	for _, part := range strings.Split(v.Key, "/") {
		switch {
		case strings.HasPrefix(part, "Hq="):
			if n, err := strconv.Atoi(part[3:]); err == nil && n > 0 {
				a.Hq = n
			}
		case strings.HasPrefix(part, "Hkv="):
			if n, err := strconv.Atoi(part[4:]); err == nil && n > 0 {
				a.Hkv = n
			}
		case strings.HasPrefix(part, "qlen="):
			if n, err := strconv.Atoi(part[5:]); err == nil && n > 0 {
				a.QLen = n
			}
		case strings.HasPrefix(part, "window="):
			if n, err := strconv.Atoi(part[7:]); err == nil && n >= 0 {
				a.Window = n
			}
		case strings.HasPrefix(part, "page="):
			if n, err := strconv.Atoi(part[5:]); err == nil && n > 0 {
				a.Page = n
			}
		case strings.HasPrefix(part, "q="):
			a.Q = parseDTypeName(part[2:])
		case strings.HasPrefix(part, "out="):
			a.Out = parseDTypeName(part[4:])
		case part == "qkfp8":
			a.QKFP8 = true
		case strings.HasPrefix(part, "gate="):
			a.Gate = true
			toks := strings.Split(part[5:], ":")
			if len(toks) >= 2 {
				a.GateStride, _ = strconv.Atoi(toks[0])
				a.GateOffset, _ = strconv.Atoi(toks[1])
			}
		}
	}
	if normed, _, err := a.norm(); err == nil {
		return normed, true
	}
	return a, true
}

func parseDTypeName(s string) DType {
	switch strings.ToLower(s) {
	case "f32":
		return F32
	case "bf16":
		return BF16
	case "e4m3":
		return E4M3
	default:
		return BF16
	}
}

func parseBroadcastName(s string) Broadcast {
	switch strings.ToLower(s) {
	case "row":
		return PerRow
	case "full":
		return Full
	default:
		return PerCol
	}
}

func extraTrafficBytes(extras []Operand, m, n int) uint64 {
	var total uint64
	for _, e := range extras {
		elems := n
		switch e.Broadcast {
		case PerRow:
			elems = m
		case Full:
			elems = m * n
		}
		total += uint64(elems * e.DType.Bits() / 8)
	}
	return total
}

// SelectByPairedTiming evaluates ABBA paired timing blocks for candidate
// variants against a baseline using [pairedtiming.Analyze] and the campaign
// selection screen rule (a candidate is promoted when its median ABBA ratio
// is at least [DefaultMinGain], all blocks are finite and above intercept, and
// it does not regress on blocks).
func SelectByPairedTiming(variants []Variant, timings map[string][]pairedtiming.Block, intercept float64) (Variant, pairedtiming.Summary, error) {
	return selectByPairedTimingWithGain(variants, timings, intercept, DefaultMinGain)
}

func selectByPairedTimingWithGain(variants []Variant, timings map[string][]pairedtiming.Block, intercept, minGain float64) (Variant, pairedtiming.Summary, error) {
	if len(variants) == 0 {
		return Variant{}, pairedtiming.Summary{}, errors.New("select by paired timing: no variants")
	}
	if len(timings) == 0 {
		return Variant{}, pairedtiming.Summary{}, errors.New("select by paired timing: no timing blocks")
	}
	if math.IsNaN(intercept) || math.IsInf(intercept, 0) || intercept < 0 {
		return Variant{}, pairedtiming.Summary{}, fmt.Errorf("select by paired timing: invalid intercept %v", intercept)
	}
	if minGain <= 0 {
		minGain = DefaultMinGain
	}

	var (
		haveWinner  bool
		winner      Variant
		winnerSum   pairedtiming.Summary
		haveBase    bool
		baseVariant Variant
		baseSum     pairedtiming.Summary
		validCount  int
		lastErr     error
	)

	// If variants[0] has no timing blocks in timings, it acts as the implicit
	// baseline A against which candidate variants B are timed.
	if _, ok := timings[variants[0].Name]; !ok {
		haveBase = true
		baseVariant = variants[0]
		baseSum = pairedtiming.Summary{
			Result: pairedtiming.Result{
				Ratio:    1.0,
				MinRatio: 1.0,
			},
		}
	}

	for _, v := range variants {
		blocks, ok := timings[v.Name]
		if !ok || len(blocks) == 0 {
			continue
		}
		sum, err := pairedtiming.Analyze(blocks, intercept)
		if err != nil {
			lastErr = fmt.Errorf("variant %s: %w", v.Name, err)
			continue
		}
		validCount++

		// An exact 1.0 baseline entry in timings records the baseline A.
		if !haveBase && math.Abs(sum.Ratio-1.0) <= 1e-9 && math.Abs(sum.MinRatio-1.0) <= 1e-9 {
			haveBase = true
			baseVariant = v
			baseSum = sum
			continue
		}

		// Campaign selection screen: candidate must beat minGain and have
		// consistent non-regressing ABBA blocks.
		if sum.Ratio < minGain {
			continue
		}
		if sum.Wins <= sum.Losses || sum.MinRatio < 1.0 {
			continue
		}
		if !haveWinner || sum.Ratio > winnerSum.Ratio {
			haveWinner = true
			winner = v
			winnerSum = sum
		}
	}

	if haveWinner {
		return winner, winnerSum, nil
	}
	if validCount == 0 && lastErr != nil {
		return Variant{}, pairedtiming.Summary{}, fmt.Errorf("select by paired timing: %w", lastErr)
	}
	if haveBase && validCount > 0 {
		return baseVariant, baseSum, nil
	}
	return Variant{}, pairedtiming.Summary{}, fmt.Errorf("select by paired timing: no variant met minimum gain %.4f", minGain)
}

// BuildShapeDispatcher emits a self-contained C++ translation unit embedding
// the distinct winning variants across runtime batch/row buckets shapes and a
// shape-routing entry function funcName (plus extern "C" <funcName>_c) that
// dispatches each runtime m to its winning variant.
//
// Positive values in shapes are sorted in ascending order; the largest shape
// bucket serves as the unbounded fallback unless winners[0] (or 0 in shapes)
// explicitly supplies a separate unbounded route.
func BuildShapeDispatcher(funcName string, shapes []int, winners map[int]Variant) (string, error) {
	if !cIdent(funcName) {
		return "", fmt.Errorf("build shape dispatcher: bad function name %q", funcName)
	}
	if len(shapes) == 0 || len(winners) == 0 {
		return "", fmt.Errorf("build shape dispatcher %s: empty shapes or winners", funcName)
	}

	seen := make(map[int]bool, len(shapes))
	var buckets []int
	hasUnbounded := false
	for _, s := range shapes {
		if s < 0 {
			return "", fmt.Errorf("build shape dispatcher %s: negative shape %d", funcName, s)
		}
		if s == 0 {
			hasUnbounded = true
			continue
		}
		if !seen[s] {
			seen[s] = true
			buckets = append(buckets, s)
		}
	}
	slices.Sort(buckets)
	if _, ok := winners[0]; ok {
		hasUnbounded = true
	}
	if len(buckets) == 0 && !hasUnbounded {
		return "", fmt.Errorf("build shape dispatcher %s: no valid shape buckets", funcName)
	}

	var routes []Route
	for i, s := range buckets {
		v, ok := winners[s]
		if !ok {
			return "", fmt.Errorf("build shape dispatcher %s: missing winner for shape %d", funcName, s)
		}
		if v.Name == "" || v.Entry == "" || v.Source == "" {
			return "", fmt.Errorf("build shape dispatcher %s: incomplete winner variant for shape %d", funcName, s)
		}
		maxBound := s
		if i == len(buckets)-1 && !hasUnbounded {
			maxBound = 0
		}
		routes = append(routes, Route{Max: maxBound, Variant: v})
	}
	if hasUnbounded {
		v, ok := winners[0]
		if !ok {
			return "", fmt.Errorf("build shape dispatcher %s: missing unbounded winner for shape 0", funcName)
		}
		if v.Name == "" || v.Entry == "" || v.Source == "" {
			return "", fmt.Errorf("build shape dispatcher %s: incomplete unbounded winner variant", funcName)
		}
		routes = append(routes, Route{Max: 0, Variant: v})
	}
	return Dispatcher(funcName, routes)
}

// A Tuner combines static ptxas filtering, wave-quantized analytical ranking,
// ABBA paired-timing selection, and multi-bucket shape dispatcher generation.
type Tuner struct {
	Arch      string
	Gate      StaticGate
	Cost      CostConfig
	MinGain   float64
	Intercept float64
}

// Filter runs [FilterVariantsByPTXAS] with t.Arch and t.Gate.
func (t Tuner) Filter(variants []Variant, report ptxasinfo.Report) ([]Variant, []RejectedVariant) {
	return FilterVariantsByPTXAS(variants, report, t.Arch, t.Gate)
}

// Rank runs [RankVariants] with t.Cost.
func (t Tuner) Rank(variants []Variant, shape TuningShape) []RankedVariant {
	return RankVariants(variants, shape, t.Cost)
}

// Select runs [SelectByPairedTiming] with t.Intercept and t.MinGain.
func (t Tuner) Select(variants []Variant, timings map[string][]pairedtiming.Block) (Variant, pairedtiming.Summary, error) {
	return selectByPairedTimingWithGain(variants, timings, t.Intercept, t.MinGain)
}

// BuildDispatcher runs [BuildShapeDispatcher].
func (t Tuner) BuildDispatcher(funcName string, shapes []int, winners map[int]Variant) (string, error) {
	return BuildShapeDispatcher(funcName, shapes, winners)
}

// TuneShapes filters variants against report (when report has functions),
// ranks surviving variants at each bucket in shapes, and emits a shape-routed
// C++ dispatcher for funcName.
func (t Tuner) TuneShapes(funcName string, variants []Variant, report ptxasinfo.Report, shapes []int) (string, map[int]Variant, []RejectedVariant, error) {
	surviving := variants
	var rejected []RejectedVariant
	if len(report.Functions()) > 0 {
		surviving, rejected = t.Filter(variants, report)
	}
	if len(surviving) == 0 {
		return "", nil, rejected, fmt.Errorf("tune shapes %s: no surviving variants", funcName)
	}
	winners := make(map[int]Variant, len(shapes))
	for _, s := range shapes {
		m := max(s, 1)
		ranked := t.Rank(surviving, TuningShape{M: m, Batch: m})
		if len(ranked) == 0 {
			return "", nil, rejected, fmt.Errorf("tune shapes %s: no ranked variant for shape %d", funcName, s)
		}
		winners[s] = ranked[0].Variant
	}
	src, err := t.BuildDispatcher(funcName, shapes, winners)
	if err != nil {
		return "", nil, rejected, err
	}
	return src, winners, rejected, nil
}

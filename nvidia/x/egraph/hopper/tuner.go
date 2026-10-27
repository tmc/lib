package hopper

import (
	"strings"

	"github.com/tmc/lib/nvidia/x/egraph/qualify"
	"github.com/tmc/lib/nvidia/x/pairedtiming"
	"github.com/tmc/lib/nvidia/x/ptxasinfo"
)

// Hopper SM hardware resource limits used by [FilterVariantsByPTXAS],
// [ResidentCTAsPerSM], and [RankVariants].
const (
	HopperSMRegisters       = qualify.HopperSMRegisters
	HopperMaxWarpsPerSM     = qualify.HopperMaxWarpsPerSM
	HopperMaxSharedPerSM    = qualify.HopperMaxSharedPerSM
	HopperMaxRegistersPerTh = qualify.HopperMaxRegistersPerTh
	HopperMaxCTAsPerSM      = qualify.HopperMaxCTAsPerSM
	DefaultMinGain          = qualify.DefaultMinGain
)

// StaticGate specifies the static compiler resource constraints that a
// [Variant] must satisfy in [FilterVariantsByPTXAS].
type StaticGate = qualify.StaticGate

// TuningShape specifies the concrete runtime dimensions used by [RankVariants]
// to price and rank kernel variants.
type TuningShape = qualify.TuningShape

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

// FilterVariantsByPTXAS filters variants against a parsed ptxas -v report for
// arch, rejecting any variant that spills registers (unless permitted by gate),
// triggers wgmma serialization (such as C7519), exceeds gate.MaxRegisters, or
// falls below gate.MinCTAsPerSM resident CTAs per SM on Hopper.
func FilterVariantsByPTXAS(variants []Variant, report ptxasinfo.Report, arch string, gate StaticGate) ([]Variant, []RejectedVariant) {
	qKept, qRejected := qualify.FilterVariantsByPTXAS(toQualifyVariants(variants), report, arch, gate)
	kept := make([]Variant, len(qKept))
	for i, qv := range qKept {
		kept[i] = fromQualifyVariant(qv)
	}
	rejected := make([]RejectedVariant, len(qRejected))
	for i, qr := range qRejected {
		rejected[i] = RejectedVariant{
			Variant:     fromQualifyVariant(qr.Variant),
			Reasons:     qr.Reasons,
			Registers:   qr.Registers,
			SharedBytes: qr.SharedBytes,
			SpillStores: qr.SpillStores,
			SpillLoads:  qr.SpillLoads,
			CTAsPerSM:   qr.CTAsPerSM,
			Diagnostics: qr.Diagnostics,
		}
	}
	return kept, rejected
}

// ResidentCTAsPerSM computes the maximum resident CTAs per SM on Hopper.
func ResidentCTAsPerSM(v Variant, regs, staticShared int) int {
	return qualify.ResidentCTAsPerSM(toQualifyVariant(v), regs, staticShared)
}

func matchesVariantFunction(funcName, variantName string) bool {
	return qualify.MatchesVariantFunction(funcName, variantName)
}

// RankVariants ranks variants for shape under cfg using a wave-quantized
// analytical model and returns them sorted from lowest EstimatedNs to highest.
func RankVariants(variants []Variant, shape TuningShape, cfg CostConfig) []RankedVariant {
	qRanked := qualify.RankVariants(toQualifyVariants(variants), shape, toQualifyCost(cfg))
	out := make([]RankedVariant, len(qRanked))
	for i, qr := range qRanked {
		out[i] = RankedVariant{
			Variant:           fromQualifyVariant(qr.Variant),
			EstimatedNs:       qr.EstimatedNs,
			NumCTAs:           qr.NumCTAs,
			ResidentCTAsPerSM: qr.ResidentCTAsPerSM,
			Waves:             qr.Waves,
			FractionalWaves:   qr.FractionalWaves,
			WorkloadBytes:     qr.WorkloadBytes,
			CombineBytes:      qr.CombineBytes,
			WorkloadFLOPs:     qr.WorkloadFLOPs,
		}
	}
	return out
}

// SelectByPairedTiming evaluates ABBA paired timing blocks for candidate
// variants against a baseline using [pairedtiming.Analyze].
func SelectByPairedTiming(variants []Variant, timings map[string][]pairedtiming.Block, intercept float64) (Variant, pairedtiming.Summary, error) {
	qv, sum, err := qualify.SelectByPairedTiming(toQualifyVariants(variants), timings, intercept)
	if err != nil {
		return Variant{}, pairedtiming.Summary{}, err
	}
	return fromQualifyVariant(qv), sum, nil
}

// BuildShapeDispatcher emits a self-contained C++ translation unit embedding
// the distinct winning variants across runtime batch/row buckets shapes and a
// shape-routing entry function funcName.
func BuildShapeDispatcher(funcName string, shapes []int, winners map[int]Variant) (string, error) {
	qWinners := make(map[int]qualify.Variant, len(winners))
	for k, v := range winners {
		qWinners[k] = toQualifyVariant(v)
	}
	return qualify.BuildShapeDispatcher(funcName, shapes, qWinners)
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
	qv, sum, err := qualify.SelectByPairedTimingWithGain(toQualifyVariants(variants), timings, t.Intercept, t.MinGain)
	if err != nil {
		return Variant{}, pairedtiming.Summary{}, err
	}
	return fromQualifyVariant(qv), sum, nil
}

// BuildDispatcher runs [BuildShapeDispatcher].
func (t Tuner) BuildDispatcher(funcName string, shapes []int, winners map[int]Variant) (string, error) {
	return BuildShapeDispatcher(funcName, shapes, winners)
}

// TuneShapes filters variants against report, ranks surviving variants at each
// bucket in shapes, and emits a shape-routed C++ dispatcher for funcName.
func (t Tuner) TuneShapes(funcName string, variants []Variant, report ptxasinfo.Report, shapes []int) (string, map[int]Variant, []RejectedVariant, error) {
	qt := qualify.Tuner{
		Arch:      t.Arch,
		Gate:      t.Gate,
		Cost:      toQualifyCost(t.Cost),
		MinGain:   t.MinGain,
		Intercept: t.Intercept,
	}
	src, qWinners, qRejected, err := qt.TuneShapes(funcName, toQualifyVariants(variants), report, shapes)
	rejected := make([]RejectedVariant, len(qRejected))
	for i, qr := range qRejected {
		rejected[i] = RejectedVariant{
			Variant:     fromQualifyVariant(qr.Variant),
			Reasons:     qr.Reasons,
			Registers:   qr.Registers,
			SharedBytes: qr.SharedBytes,
			SpillStores: qr.SpillStores,
			SpillLoads:  qr.SpillLoads,
			CTAsPerSM:   qr.CTAsPerSM,
			Diagnostics: qr.Diagnostics,
		}
	}
	if err != nil {
		return "", nil, rejected, err
	}
	winners := make(map[int]Variant, len(qWinners))
	for k, qv := range qWinners {
		winners[k] = fromQualifyVariant(qv)
	}
	return src, winners, rejected, nil
}

func toQualifyVariant(v Variant) qualify.Variant {
	return qualify.Variant{
		Name:   v.Name,
		Op:     v.Op,
		Key:    v.Key,
		Config: v.Config,
		Launch: qualify.Launch{
			Block:           v.Launch.Block,
			Warps:           v.Launch.Warps,
			Stages:          v.Launch.Stages,
			TableAhead:      v.Launch.TableAhead,
			SplitK:          v.Launch.SplitK,
			SplitKV:         v.Launch.SplitKV,
			KVDepth:         v.Launch.KVDepth,
			KVHeads:         v.Launch.KVHeads,
			DeviceSplits:    v.Launch.DeviceSplits,
			Swizzle:         v.Launch.Swizzle,
			ClusterCombine:  v.Launch.ClusterCombine,
			BM:              v.Launch.BM,
			BN:              v.Launch.BN,
			GroupM:          v.Launch.GroupM,
			StreamK:         v.Launch.StreamK,
			Swizzle128B:     v.Launch.Swizzle128B,
			WarpSpecialized: v.Launch.WarpSpecialized,
			RowsPerBlock:    v.Launch.RowsPerBlock,
			Unroll:          v.Launch.Unroll,
			Prefetch:        v.Launch.Prefetch,
			MMA:             v.Launch.MMA,
			SharedBytes:     v.Launch.SharedBytes,
			Registers:       v.Launch.Registers,
			PTXASShared:     v.Launch.ptxasShared,
		},
		Entry:  v.Entry,
		Params: v.Params,
		SHA256: v.SHA256,
		Source: v.Source,
	}
}

func fromQualifyVariant(v qualify.Variant) Variant {
	return Variant{
		Name:   v.Name,
		Op:     v.Op,
		Key:    v.Key,
		Config: v.Config,
		Launch: Launch{
			Block:           v.Launch.Block,
			Warps:           v.Launch.Warps,
			Stages:          v.Launch.Stages,
			TableAhead:      v.Launch.TableAhead,
			SplitK:          v.Launch.SplitK,
			SplitKV:         v.Launch.SplitKV,
			KVDepth:         v.Launch.KVDepth,
			KVHeads:         v.Launch.KVHeads,
			DeviceSplits:    v.Launch.DeviceSplits,
			Swizzle:         v.Launch.Swizzle,
			ClusterCombine:  v.Launch.ClusterCombine,
			BM:              v.Launch.BM,
			BN:              v.Launch.BN,
			GroupM:          v.Launch.GroupM,
			StreamK:         v.Launch.StreamK,
			Swizzle128B:     v.Launch.Swizzle128B,
			WarpSpecialized: v.Launch.WarpSpecialized,
			RowsPerBlock:    v.Launch.RowsPerBlock,
			Unroll:          v.Launch.Unroll,
			Prefetch:        v.Launch.Prefetch,
			MMA:             v.Launch.MMA,
			SharedBytes:     v.Launch.SharedBytes,
			Registers:       v.Launch.Registers,
			ptxasShared:     v.Launch.PTXASShared,
		},
		Entry:  v.Entry,
		Params: v.Params,
		SHA256: v.SHA256,
		Source: v.Source,
	}
}

func toQualifyVariants(vs []Variant) []qualify.Variant {
	out := make([]qualify.Variant, len(vs))
	for i, v := range vs {
		out[i] = toQualifyVariant(v)
	}
	return out
}

func toQualifyCost(c CostConfig) qualify.CostConfig {
	return qualify.CostConfig{
		Device:       c.Device,
		Residency:    c.Residency,
		Ceiling:      c.Ceiling,
		TensorTFLOPS: c.TensorTFLOPS,
		SMs:          c.SMs,
		LaunchNs:     c.LaunchNs,
		MaxKeys:      c.MaxKeys,
	}
}

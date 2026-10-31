package qualify

import (
	"strings"

	"github.com/tmc/lib/nvidia/x/egraph/hopper"
	"github.com/tmc/lib/nvidia/x/pairedtiming"
	"github.com/tmc/lib/nvidia/x/ptxasinfo"
)

// A RejectedHopperVariant is a [hopper.Variant] rejected by
// [FilterHopperVariantsByPTXAS] along with its static compiler resource
// metrics and rejection reasons.
type RejectedHopperVariant struct {
	Variant     hopper.Variant         `json:"variant"`
	Reasons     []string               `json:"reasons"`
	Registers   int                    `json:"registers,omitempty"`
	SharedBytes int                    `json:"shared_bytes,omitempty"`
	SpillStores int                    `json:"spill_stores,omitempty"`
	SpillLoads  int                    `json:"spill_loads,omitempty"`
	CTAsPerSM   int                    `json:"ctas_per_sm"`
	Diagnostics []ptxasinfo.Diagnostic `json:"diagnostics,omitempty"`
}

// Reason returns the semicolon-separated rejection reasons.
func (r RejectedHopperVariant) Reason() string {
	return strings.Join(r.Reasons, "; ")
}

// A RankedHopperVariant is a [hopper.Variant] scored by [RankHopperVariants]
// for a concrete [TuningShape] and [hopper.CostConfig].
type RankedHopperVariant struct {
	Variant           hopper.Variant `json:"variant"`
	EstimatedNs       float64        `json:"estimated_ns"`
	NumCTAs           int            `json:"num_ctas"`
	ResidentCTAsPerSM int            `json:"resident_ctas_per_sm"`
	Waves             int            `json:"waves"`
	FractionalWaves   float64        `json:"fractional_waves"`
	WorkloadBytes     uint64         `json:"workload_bytes"`
	CombineBytes      uint64         `json:"combine_bytes"`
	WorkloadFLOPs     uint64         `json:"workload_flops"`
}

// FilterHopperVariantsByPTXAS filters hopper variants against a parsed ptxas -v
// report for arch, rejecting any variant that spills registers (unless
// permitted by gate), triggers wgmma serialization (such as C7519), exceeds
// gate.MaxRegisters, or falls below gate.MinCTAsPerSM resident CTAs per SM on
// Hopper.
func FilterHopperVariantsByPTXAS(variants []hopper.Variant, report ptxasinfo.Report, arch string, gate StaticGate) ([]hopper.Variant, []RejectedHopperVariant) {
	qKept, qRejected := FilterVariantsByPTXAS(ToQualifyVariants(variants), report, arch, gate)
	kept := make([]hopper.Variant, len(qKept))
	for i, qv := range qKept {
		kept[i] = FromQualifyVariant(qv)
	}
	rejected := make([]RejectedHopperVariant, len(qRejected))
	for i, qr := range qRejected {
		rejected[i] = RejectedHopperVariant{
			Variant:     FromQualifyVariant(qr.Variant),
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

// ResidentHopperCTAsPerSM computes the maximum resident CTAs per SM on Hopper
// for a [hopper.Variant].
func ResidentHopperCTAsPerSM(v hopper.Variant, regs, staticShared int) int {
	return ResidentCTAsPerSM(ToQualifyVariant(v), regs, staticShared)
}

// RankHopperVariants ranks hopper variants for shape under cfg using a
// wave-quantized analytical model and returns them sorted from lowest
// EstimatedNs to highest.
func RankHopperVariants(variants []hopper.Variant, shape TuningShape, cfg hopper.CostConfig) []RankedHopperVariant {
	qRanked := RankVariants(ToQualifyVariants(variants), shape, ToQualifyCost(cfg))
	out := make([]RankedHopperVariant, len(qRanked))
	for i, qr := range qRanked {
		out[i] = RankedHopperVariant{
			Variant:           FromQualifyVariant(qr.Variant),
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

// SelectHopperByPairedTiming evaluates ABBA paired timing blocks for candidate
// hopper variants against a baseline using [pairedtiming.Analyze].
func SelectHopperByPairedTiming(variants []hopper.Variant, timings map[string][]pairedtiming.Block, intercept float64) (hopper.Variant, pairedtiming.Summary, error) {
	qv, sum, err := SelectByPairedTiming(ToQualifyVariants(variants), timings, intercept)
	if err != nil {
		return hopper.Variant{}, pairedtiming.Summary{}, err
	}
	return FromQualifyVariant(qv), sum, nil
}

// BuildHopperShapeDispatcher emits a self-contained C++ translation unit
// embedding the distinct winning hopper variants across runtime batch/row
// buckets shapes and a shape-routing entry function funcName.
func BuildHopperShapeDispatcher(funcName string, shapes []int, winners map[int]hopper.Variant) (string, error) {
	qWinners := make(map[int]Variant, len(winners))
	for k, v := range winners {
		qWinners[k] = ToQualifyVariant(v)
	}
	return BuildShapeDispatcher(funcName, shapes, qWinners)
}

// A HopperTuner combines static ptxas filtering, wave-quantized analytical
// ranking, ABBA paired-timing selection, and multi-bucket shape dispatcher
// generation for [hopper.Variant] candidates.
type HopperTuner struct {
	Arch      string
	Gate      StaticGate
	Cost      hopper.CostConfig
	MinGain   float64
	Intercept float64
}

// Filter runs [FilterHopperVariantsByPTXAS] with t.Arch and t.Gate.
func (t HopperTuner) Filter(variants []hopper.Variant, report ptxasinfo.Report) ([]hopper.Variant, []RejectedHopperVariant) {
	return FilterHopperVariantsByPTXAS(variants, report, t.Arch, t.Gate)
}

// Rank runs [RankHopperVariants] with t.Cost.
func (t HopperTuner) Rank(variants []hopper.Variant, shape TuningShape) []RankedHopperVariant {
	return RankHopperVariants(variants, shape, t.Cost)
}

// Select runs [SelectHopperByPairedTiming] with t.Intercept and t.MinGain.
func (t HopperTuner) Select(variants []hopper.Variant, timings map[string][]pairedtiming.Block) (hopper.Variant, pairedtiming.Summary, error) {
	qv, sum, err := SelectByPairedTimingWithGain(ToQualifyVariants(variants), timings, t.Intercept, t.MinGain)
	if err != nil {
		return hopper.Variant{}, pairedtiming.Summary{}, err
	}
	return FromQualifyVariant(qv), sum, nil
}

// BuildDispatcher runs [BuildHopperShapeDispatcher].
func (t HopperTuner) BuildDispatcher(funcName string, shapes []int, winners map[int]hopper.Variant) (string, error) {
	return BuildHopperShapeDispatcher(funcName, shapes, winners)
}

// TuneShapes filters variants against report, ranks surviving variants at each
// bucket in shapes, and emits a shape-routed C++ dispatcher for funcName.
func (t HopperTuner) TuneShapes(funcName string, variants []hopper.Variant, report ptxasinfo.Report, shapes []int) (string, map[int]hopper.Variant, []RejectedHopperVariant, error) {
	qt := Tuner{
		Arch:      t.Arch,
		Gate:      t.Gate,
		Cost:      ToQualifyCost(t.Cost),
		MinGain:   t.MinGain,
		Intercept: t.Intercept,
	}
	src, qWinners, qRejected, err := qt.TuneShapes(funcName, ToQualifyVariants(variants), report, shapes)
	rejected := make([]RejectedHopperVariant, len(qRejected))
	for i, qr := range qRejected {
		rejected[i] = RejectedHopperVariant{
			Variant:     FromQualifyVariant(qr.Variant),
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
	winners := make(map[int]hopper.Variant, len(qWinners))
	for k, qv := range qWinners {
		winners[k] = FromQualifyVariant(qv)
	}
	return src, winners, rejected, nil
}

// ToQualifyVariant converts a [hopper.Variant] into a [Variant].
func ToQualifyVariant(v hopper.Variant) Variant {
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
			PTXASShared:     v.Launch.PTXASSharedBytes(),
		},
		Entry:  v.Entry,
		Params: v.Params,
		SHA256: v.SHA256,
		Source: v.Source,
	}
}

// FromQualifyVariant converts a [Variant] into a [hopper.Variant].
func FromQualifyVariant(v Variant) hopper.Variant {
	out := hopper.Variant{
		Name:   v.Name,
		Op:     v.Op,
		Key:    v.Key,
		Config: v.Config,
		Launch: hopper.Launch{
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
		},
		Entry:  v.Entry,
		Params: v.Params,
		SHA256: v.SHA256,
		Source: v.Source,
	}
	out.Launch.SetPTXASSharedBytes(v.Launch.PTXASShared)
	return out
}

// ToQualifyVariants converts a slice of [hopper.Variant] into [Variant].
func ToQualifyVariants(vs []hopper.Variant) []Variant {
	out := make([]Variant, len(vs))
	for i, v := range vs {
		out[i] = ToQualifyVariant(v)
	}
	return out
}

// ToQualifyCost converts a [hopper.CostConfig] into a [CostConfig].
func ToQualifyCost(c hopper.CostConfig) CostConfig {
	return CostConfig{
		Device:       c.Device,
		Residency:    c.Residency,
		Ceiling:      c.Ceiling,
		TensorTFLOPS: c.TensorTFLOPS,
		SMs:          c.SMs,
		LaunchNs:     c.LaunchNs,
		MaxKeys:      c.MaxKeys,
	}
}

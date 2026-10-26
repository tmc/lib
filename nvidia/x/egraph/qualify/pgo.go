package qualify

import (
	"fmt"
	"math"
	"slices"
	"strings"

	"github.com/tmc/lib/nvidia/x/ptxasinfo"
)

// BottleneckKind classifies a measured or compiled hardware bottleneck.
type BottleneckKind string

const (
	BottleneckBankConflict   BottleneckKind = "shared_bank_conflict"
	BottleneckWGMMASerial    BottleneckKind = "wgmma_serialization"
	BottleneckRegisterSpill  BottleneckKind = "register_spill"
	BottleneckLowOccupancy   BottleneckKind = "low_occupancy"
	BottleneckWaveTail       BottleneckKind = "wave_tail_quantization"
	BottleneckLaunchOverhead BottleneckKind = "launch_overhead"
	BottleneckMemoryBoundKV  BottleneckKind = "memory_bound_kv"
)

// KernelObservation holds measured profiling and compiler telemetry for a
// single kernel or plan step.
type KernelObservation struct {
	Name               string
	DurationNs         float64
	LaunchGapNs        float64 // idle gap before this kernel launched
	SharedBankConflict float64 // ratio of bank-conflict wavefronts (e.g., >= 0.10 triggers swizzle)
	WGMMASerialized    bool    // ptxas C7519 or measured GMMA stall serialization
	SpillStoreBytes    int
	SpillLoadBytes     int
	Registers          int
	AchievedOccupancy  float64 // [0, 1]
	GridBlocks         int
	DeviceSMs          int  // e.g., 132 for H200
	MemoryBoundKV      bool // explicit HBM latency/bandwidth stall on KV or streaming weights
}

// ProfileReceipt aggregates per-step observations and device metadata.
type ProfileReceipt struct {
	Device       string // e.g., "NVIDIA H200", "NVIDIA GB10"
	Observations []KernelObservation
}

// BottleneckReport pairs a [KernelObservation] with its classified bottlenecks
// and recommended schedule rewrites.
type BottleneckReport struct {
	Observation KernelObservation
	Bottlenecks []BottleneckKind
	Actions     []string
}

// ClassifyBottlenecks classifies the hardware bottlenecks present in obs in a
// deterministic order.
func ClassifyBottlenecks(obs KernelObservation) []BottleneckKind {
	var out []BottleneckKind
	if obs.SharedBankConflict >= 0.10 {
		out = append(out, BottleneckBankConflict)
	}
	if obs.WGMMASerialized {
		out = append(out, BottleneckWGMMASerial)
	}
	if obs.SpillStoreBytes > 0 || obs.SpillLoadBytes > 0 {
		out = append(out, BottleneckRegisterSpill)
	}
	if obs.AchievedOccupancy > 0 && obs.AchievedOccupancy < 0.25 && obs.Registers > 128 {
		out = append(out, BottleneckLowOccupancy)
	}
	if obs.MemoryBoundKV || (obs.AchievedOccupancy > 0 && obs.AchievedOccupancy < 0.25 && obs.Registers > 0 && obs.Registers <= 128) {
		out = append(out, BottleneckMemoryBoundKV)
	}
	if obs.GridBlocks > 0 && obs.DeviceSMs > 0 {
		waves := float64(obs.GridBlocks) / float64(obs.DeviceSMs)
		tail := waves - math.Floor(waves)
		if waves > 1.0 && tail > 0 && tail <= 0.35+1e-9 {
			out = append(out, BottleneckWaveTail)
		}
	}
	if obs.DurationNs > 0 && obs.LaunchGapNs/obs.DurationNs >= 0.15-1e-9 {
		out = append(out, BottleneckLaunchOverhead)
	}
	return out
}

// ObservationFromPTXAS constructs a [KernelObservation] from a parsed
// [ptxasinfo.Function] report entry and launch geometry.
func ObservationFromPTXAS(fn ptxasinfo.Function, gridBlocks, deviceSMs int) KernelObservation {
	obs := KernelObservation{
		Name:            fn.Name,
		SpillStoreBytes: fn.SpillStores,
		SpillLoadBytes:  fn.SpillLoads,
		Registers:       fn.Registers,
		GridBlocks:      gridBlocks,
		DeviceSMs:       deviceSMs,
	}
	for _, d := range fn.Diagnostics {
		lower := strings.ToLower(d.Text)
		if d.WGMMASerialized() || d.Code == "C7519" || strings.Contains(lower, "serialized") || strings.Contains(lower, "c7519") {
			obs.WGMMASerialized = true
			break
		}
	}
	return obs
}

// MatchObservation matches a step kernel name and step index against obs,
// marking matched entries in used.
func MatchObservation(kName string, idx int, obs []KernelObservation, used []bool) (KernelObservation, bool) {
	if kName != "" {
		for j, o := range obs {
			if used[j] || o.Name == "" {
				continue
			}
			if o.Name == kName || MatchesVariantFunction(o.Name, kName) {
				used[j] = true
				return o, true
			}
		}
	}
	if idx < len(obs) && !used[idx] {
		used[idx] = true
		return obs[idx], true
	}
	return KernelObservation{}, false
}

// ScheduleKnobs describes generic kernel schedule knobs for [RefineKnobs].
type ScheduleKnobs struct {
	Name            string
	Op              string // "tile_gemm", "skinny_gemm", "paged_attention", "row_kernel"
	BM              int
	BN              int
	Stages          int
	WGMMA           bool
	Swizzle128B     bool
	WarpSpecialized bool
	Repacked        bool
	DequantWords    int
	StreamK         int
	GroupM          int
	RowsPerBlock    int
	Unroll          int
	Prefetch        int
	BlackwellFP4    bool
	Chunk           int
	Depth           int
	TableAhead      int
	QKFP8           bool
	ClusterCombine  bool
	Cluster         bool
}

// RefineKnobs applies deterministic profile-guided refinement rules to knobs
// based on obs.
func RefineKnobs(knobs ScheduleKnobs, obs KernelObservation) (ScheduleKnobs, []string) {
	bns := ClassifyBottlenecks(obs)
	if len(bns) == 0 {
		return knobs, nil
	}
	switch knobs.Op {
	case "tile_gemm":
		tk, reasons := RefineTileGEMMKnobs(TileGEMMKnobs{
			Name:            knobs.Name,
			BM:              knobs.BM,
			BN:              knobs.BN,
			Stages:          knobs.Stages,
			WGMMA:           knobs.WGMMA,
			Swizzle128B:     knobs.Swizzle128B,
			WarpSpecialized: knobs.WarpSpecialized,
			Repacked:        knobs.Repacked,
			DequantWords:    knobs.DequantWords,
			StreamK:         knobs.StreamK,
			GroupM:          knobs.GroupM,
			LinearEpilogue:  true,
		}, obs, bns, nil)
		knobs.BM = tk.BM
		knobs.BN = tk.BN
		knobs.Stages = tk.Stages
		knobs.WGMMA = tk.WGMMA
		knobs.Swizzle128B = tk.Swizzle128B
		knobs.WarpSpecialized = tk.WarpSpecialized
		knobs.Repacked = tk.Repacked
		knobs.DequantWords = tk.DequantWords
		knobs.StreamK = tk.StreamK
		knobs.GroupM = tk.GroupM
		return knobs, reasons
	default:
		return knobs, nil
	}
}

// TileGEMMKnobs holds the schedule parameters of a TileGEMM step for
// [RefineTileGEMMKnobs].
type TileGEMMKnobs struct {
	Name            string
	BM              int
	BN              int
	Stages          int
	WGMMA           bool
	Swizzle128B     bool
	WarpSpecialized bool
	Repacked        bool
	DequantWords    int
	StreamK         int
	GroupM          int
	Grouped         bool
	IsNVFP4         bool
	LinearEpilogue  bool
}

// RefineTileGEMMKnobs deterministically refines TileGEMM schedule knobs to
// cure bottlenecks in bns, consulting validate (when non-nil) before committing
// each candidate transition.
func RefineTileGEMMKnobs(g TileGEMMKnobs, obs KernelObservation, bns []BottleneckKind, validate func(TileGEMMKnobs) bool) (TileGEMMKnobs, []string) {
	if validate == nil {
		validate = func(TileGEMMKnobs) bool { return true }
	}
	var reasons []string
	has := func(k BottleneckKind) bool { return slices.Contains(bns, k) }

	if has(BottleneckBankConflict) && !g.Swizzle128B {
		cand := g
		cand.WGMMA = true
		cand.Swizzle128B = true
		if cand.BM == 0 || (cand.BM == 64 && !(cand.WarpSpecialized && cand.IsNVFP4)) {
			cand.BM = 128
		}
		if validate(cand) {
			g = cand
			reasons = append(reasons, fmt.Sprintf("%s: enable Swizzle128B to eliminate shared memory bank conflicts", g.Name))
		}
	}

	if has(BottleneckWGMMASerial) {
		stages := g.Stages
		if stages == 0 {
			stages = 3
		}
		promoted := false
		if g.WGMMA && stages >= 2 && g.StreamK == 0 && !g.WarpSpecialized {
			cand := g
			cand.Swizzle128B = true
			cand.WarpSpecialized = true
			cand.DequantWords = 0
			if validate(cand) {
				g = cand
				promoted = true
				reasons = append(reasons, fmt.Sprintf("%s: enable WarpSpecialized to cure WGMMA serialization", g.Name))
			}
		}
		if !promoted {
			if g.DequantWords > 1 {
				cand := g
				cand.DequantWords = 1
				if validate(cand) {
					g = cand
					reasons = append(reasons, fmt.Sprintf("%s: step down DequantWords=1 to avoid WGMMA serialization", g.Name))
				}
			} else if g.StreamK > 0 {
				cand := g
				cand.StreamK = 0
				if validate(cand) {
					g = cand
					reasons = append(reasons, fmt.Sprintf("%s: disable StreamK to avoid divergent WGMMA loop serialization", g.Name))
				}
			}
		}
	}

	if has(BottleneckRegisterSpill) || has(BottleneckLowOccupancy) {
		bn := g.BN
		if bn == 0 {
			bn = 128
		}
		stages := g.Stages
		if stages == 0 {
			stages = 3
		}
		if g.DequantWords == 2 {
			cand := g
			cand.DequantWords = 1
			if validate(cand) {
				g = cand
				reasons = append(reasons, fmt.Sprintf("%s: step down DequantWords 2 -> 1 to reduce register pressure", g.Name))
			}
		} else if bn > 64 {
			cand := g
			cand.BN = 64
			if cand.Repacked {
				cand.Repacked = false
				cand.DequantWords = 0
				if cand.WarpSpecialized && cand.IsNVFP4 {
					cand.WarpSpecialized = false
				}
			}
			if validate(cand) {
				g = cand
				reasons = append(reasons, fmt.Sprintf("%s: step down BN %d -> 64 to reduce register pressure", g.Name, bn))
			} else if stages > 2 {
				cand2 := g
				cand2.Stages = 2
				if validate(cand2) {
					g = cand2
					reasons = append(reasons, fmt.Sprintf("%s: step down Stages %d -> 2 to reduce register pressure", g.Name, stages))
				}
			}
		} else if stages > 2 {
			cand := g
			cand.Stages = 2
			if validate(cand) {
				g = cand
				reasons = append(reasons, fmt.Sprintf("%s: step down Stages %d -> 2 to reduce register pressure", g.Name, stages))
			}
		}
	}

	if has(BottleneckWaveTail) {
		sms := obs.DeviceSMs
		if sms <= 0 {
			sms = 132
		}
		if !g.WarpSpecialized && !g.Grouped && g.LinearEpilogue && g.StreamK == 0 {
			cand := g
			cand.StreamK = sms
			if validate(cand) {
				g = cand
				reasons = append(reasons, fmt.Sprintf("%s: enable StreamK=%d to eliminate wave tail quantization", g.Name, sms))
			}
		} else if g.GroupM != 8 {
			cand := g
			cand.GroupM = 8
			if validate(cand) {
				g = cand
				reasons = append(reasons, fmt.Sprintf("%s: set GroupM=8 to improve wave L2 locality", g.Name))
			}
		}
	}

	return g, reasons
}

// SkinnyGEMMKnobs holds the schedule parameters of a SkinnyGEMM step for
// [RefineSkinnyGEMMKnobs].
type SkinnyGEMMKnobs struct {
	Name             string
	N                int
	K                int
	RowsPerBlock     int
	NormRowsPerBlock int
	Warps            int
	NormWarps        int
	SplitK           int
	NormSplitK       int
	Unroll           int
	NormUnroll       int
	Prefetch         int
	BlackwellFP4     bool
	IsNVFP4          bool
	WeightChunk      int
}

// RefineSkinnyGEMMKnobs deterministically refines SkinnyGEMM schedule knobs to
// cure bottlenecks in bns, consulting validate before committing each change.
func RefineSkinnyGEMMKnobs(g SkinnyGEMMKnobs, obs KernelObservation, bns []BottleneckKind, validate func(SkinnyGEMMKnobs) bool) (SkinnyGEMMKnobs, []string) {
	if validate == nil {
		validate = func(SkinnyGEMMKnobs) bool { return true }
	}
	var reasons []string
	has := func(k BottleneckKind) bool { return slices.Contains(bns, k) }

	if has(BottleneckRegisterSpill) {
		if g.Prefetch > 0 {
			cand := g
			cand.Prefetch = 0
			if validate(cand) {
				g = cand
				reasons = append(reasons, fmt.Sprintf("%s: disable Prefetch to eliminate register spills", g.Name))
			}
		}
		if g.NormRowsPerBlock > 1 {
			for _, nextR := range []int{2, 1} {
				if nextR < g.NormRowsPerBlock && g.N%nextR == 0 {
					cand := g
					cand.RowsPerBlock = nextR
					if validate(cand) {
						prev := g.NormRowsPerBlock
						g = cand
						g.NormRowsPerBlock = nextR
						reasons = append(reasons, fmt.Sprintf("%s: step down RowsPerBlock %d -> %d to reduce register pressure", g.Name, prev, nextR))
						break
					}
				}
			}
		} else if g.NormUnroll > 1 {
			cand := g
			cand.Unroll = max(g.NormUnroll/2, 1)
			if validate(cand) {
				prev := g.NormUnroll
				g = cand
				g.NormUnroll = cand.Unroll
				reasons = append(reasons, fmt.Sprintf("%s: step down Unroll %d -> %d to reduce register pressure", g.Name, prev, cand.Unroll))
			}
		}
	}

	if !has(BottleneckRegisterSpill) && (has(BottleneckMemoryBoundKV) || has(BottleneckLowOccupancy)) {
		if g.Prefetch <= 0 {
			cand := g
			cand.Prefetch = 2
			if g.WeightChunk > 0 && g.NormSplitK > 0 && g.NormWarps > 0 {
				steps := g.K / (32 * g.WeightChunk)
				perWarp := (steps/g.NormSplitK + g.NormWarps - 1) / g.NormWarps
				cand.Prefetch = min(4, max(2, perWarp))
			}
			if validate(cand) {
				g = cand
				reasons = append(reasons, fmt.Sprintf("%s: enable Prefetch=%d to hide HBM weight streaming latency", g.Name, g.Prefetch))
			}
		}
		if g.IsNVFP4 && !g.BlackwellFP4 {
			cand := g
			cand.BlackwellFP4 = true
			if validate(cand) {
				g = cand
				reasons = append(reasons, fmt.Sprintf("%s: enable BlackwellFP4 hardware nibble widening for NVFP4", g.Name))
			}
		}
	}

	if has(BottleneckWaveTail) {
		if g.NormRowsPerBlock > 1 && g.N >= 1 {
			cand := g
			cand.RowsPerBlock = 1
			if validate(cand) {
				prev := g.NormRowsPerBlock
				g = cand
				g.NormRowsPerBlock = 1
				reasons = append(reasons, fmt.Sprintf("%s: step down RowsPerBlock %d -> 1 to spread CTAs across SMs", g.Name, prev))
			}
		}
	}

	return g, reasons
}

// PagedAttentionKnobs holds the schedule parameters of a PagedAttention step
// for [RefinePagedAttentionKnobs].
type PagedAttentionKnobs struct {
	Name            string
	Hq              int
	Hkv             int
	QLen            int
	Window          int
	QIsE4M3         bool
	KVHeadsPerBlock int
	Chunk           int
	Depth           int
	Stages          int
	NormStages      int
	TableAhead      int
	WGMMA           bool
	QKFP8           bool
	Swizzle         bool
	ClusterCombine  bool
}

// RefinePagedAttentionKnobs deterministically refines PagedAttention schedule
// knobs to cure bottlenecks in bns, consulting validate before committing each change.
func RefinePagedAttentionKnobs(a PagedAttentionKnobs, batch, maxKeys int, obs KernelObservation, bns []BottleneckKind, validate func(PagedAttentionKnobs) bool) (PagedAttentionKnobs, []string) {
	if validate == nil {
		validate = func(PagedAttentionKnobs) bool { return true }
	}
	var reasons []string
	has := func(k BottleneckKind) bool { return slices.Contains(bns, k) }

	if batch <= 0 {
		batch = 1
	}
	if maxKeys <= 0 {
		if a.Window > 0 {
			maxKeys = a.Window + max(a.QLen, 1) - 1
		} else {
			maxKeys = 1024
		}
	}

	if has(BottleneckBankConflict) && !a.Swizzle {
		cand := a
		cand.Swizzle = true
		if validate(cand) {
			a = cand
			reasons = append(reasons, fmt.Sprintf("%s: enable Swizzle to eliminate shared memory bank conflicts", a.Name))
		}
	}

	if has(BottleneckMemoryBoundKV) || has(BottleneckLowOccupancy) {
		g := a.Hq / max(a.Hkv, 1)
		if !a.WGMMA && g >= 8 && headDim%32 == 0 && a.QIsE4M3 && a.KVHeadsPerBlock <= 1 {
			cand := a
			cand.WGMMA = true
			cand.QKFP8 = true
			if validate(cand) {
				a = cand
				reasons = append(reasons, fmt.Sprintf("%s: enable WGMMA and QKFP8 for G=%d FP8 KV attention", a.Name, g))
			}
		}
		if a.TableAhead < 2 {
			cand := a
			if cand.Stages == 0 {
				if cand.NormStages > 0 {
					cand.Stages = cand.NormStages
				} else {
					cand.Stages = 2
				}
			}
			cand.TableAhead = min(2, max(cand.Stages, 2))
			if validate(cand) {
				a = cand
				reasons = append(reasons, fmt.Sprintf("%s: enable TableAhead=%d to prefetch page-table entries ahead of cp.async", a.Name, a.TableAhead))
			}
		}
	}

	if has(BottleneckWaveTail) {
		sms := obs.DeviceSMs
		if sms <= 0 {
			sms = 132
		}
		curChunk := a.Chunk
		if curChunk == 0 {
			curChunk = 256
		}
		hpb := max(a.KVHeadsPerBlock, 1)
		units := max(batch*(max(a.Hkv, 1)/hpb), 1)

		bestChunk := curChunk
		for _, c := range []int{128, 64} {
			if c >= curChunk {
				continue
			}
			splits := (maxKeys + c - 1) / c
			ctas := units * splits
			waves := float64(ctas) / float64(sms)
			tail := waves - math.Floor(waves)
			bestChunk = c
			if waves >= 1.0 && (tail == 0 || tail > 0.35) {
				break
			}
		}
		if bestChunk != curChunk {
			cand := a
			cand.Chunk = bestChunk
			if validate(cand) {
				a = cand
				reasons = append(reasons, fmt.Sprintf("%s: adjust Chunk %d -> %d to balance SM wave occupancy", a.Name, curChunk, bestChunk))
			}
		}

		effChunk := a.Chunk
		if effChunk == 0 {
			effChunk = 256
		}
		splits := (maxKeys + effChunk - 1) / effChunk
		if splits >= clusterSize && !a.ClusterCombine {
			cand := a
			cand.ClusterCombine = true
			if validate(cand) {
				a = cand
				reasons = append(reasons, fmt.Sprintf("%s: enable ClusterCombine across %d splits", a.Name, splits))
			}
		}
	}

	if has(BottleneckRegisterSpill) {
		if a.Depth > 1 {
			cand := a
			cand.Depth = 1
			if validate(cand) {
				a = cand
				reasons = append(reasons, fmt.Sprintf("%s: step down Depth -> 1 to eliminate register spills", a.Name))
			}
		} else if a.TableAhead > 1 {
			cand := a
			cand.TableAhead = 1
			if validate(cand) {
				a = cand
				reasons = append(reasons, fmt.Sprintf("%s: step down TableAhead -> 1 to reduce register pressure", a.Name))
			}
		}
	}

	return a, reasons
}

// RowKernelKnobs holds the schedule parameters of a RowKernel step for
// [RefineRowKernelKnobs].
type RowKernelKnobs struct {
	Name       string
	Cluster    bool
	CanCluster bool
}

// RefineRowKernelKnobs deterministically refines RowKernel schedule knobs to
// cure bottlenecks in bns, consulting validate before committing each change.
func RefineRowKernelKnobs(r RowKernelKnobs, bns []BottleneckKind, validate func(RowKernelKnobs) bool) (RowKernelKnobs, []string) {
	if validate == nil {
		validate = func(RowKernelKnobs) bool { return true }
	}
	var reasons []string
	has := func(k BottleneckKind) bool { return slices.Contains(bns, k) }

	if has(BottleneckRegisterSpill) && r.Cluster {
		cand := r
		cand.Cluster = false
		if validate(cand) {
			r = cand
			reasons = append(reasons, fmt.Sprintf("%s: disable Cluster to reduce register pressure", r.Name))
		}
		return r, reasons
	}

	if (has(BottleneckWaveTail) || has(BottleneckMemoryBoundKV) || has(BottleneckLowOccupancy)) && !r.Cluster && r.CanCluster {
		cand := r
		cand.Cluster = true
		if validate(cand) {
			r = cand
			reasons = append(reasons, fmt.Sprintf("%s: enable 8-CTA Cluster reduction", r.Name))
		}
	}
	return r, reasons
}

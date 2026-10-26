package hopper

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

// Step is an alias for [PlanKernel] used by profile-guided refinement.
type Step = PlanKernel

// MegakernelSource is an alias for [Megakernel] returned when a plan is
// promoted to a persistent megakernel.
type MegakernelSource = Megakernel

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

// RefineStep deterministically rewrites step's schedule parameters to cure each
// bottleneck classified in obs, returning the refined step and human-readable
// rewrite rationales.
func RefineStep(step Step, obs KernelObservation) (Step, []string) {
	out := cloneStep(step)
	bns := ClassifyBottlenecks(obs)
	if len(bns) == 0 {
		return out, nil
	}
	switch {
	case out.Tile != nil:
		t, reasons := refineTileGEMM(*out.Tile, out.M, obs, bns)
		out.Tile = &t
		return out, reasons
	case out.GEMM != nil:
		g, reasons := refineSkinnyGEMM(*out.GEMM, out.M, obs, bns)
		out.GEMM = &g
		return out, reasons
	case out.Attention != nil:
		a, reasons := refinePagedAttention(*out.Attention, out.Batch, out.MaxKeys, obs, bns)
		out.Attention = &a
		return out, reasons
	case out.Row != nil:
		r, reasons := refineRowKernel(*out.Row, obs, bns)
		out.Row = &r
		return out, reasons
	default:
		return out, nil
	}
}

func refineTileGEMM(g TileGEMM, m int, obs KernelObservation, bns []BottleneckKind) (TileGEMM, []string) {
	var reasons []string
	has := func(k BottleneckKind) bool { return slices.Contains(bns, k) }

	if has(BottleneckBankConflict) && !g.Swizzle128B {
		cand := g
		cand.WGMMA = true
		cand.Swizzle128B = true
		if cand.BM == 0 || (cand.BM == 64 && !(cand.WarpSpecialized && cand.Weight == WeightNVFP4)) {
			cand.BM = 128
		}
		if _, err := cand.norm(); err == nil {
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
			if _, err := cand.norm(); err == nil {
				g = cand
				promoted = true
				reasons = append(reasons, fmt.Sprintf("%s: enable WarpSpecialized to cure WGMMA serialization", g.Name))
			}
		}
		if !promoted {
			if g.DequantWords > 1 {
				cand := g
				cand.DequantWords = 1
				if _, err := cand.norm(); err == nil {
					g = cand
					reasons = append(reasons, fmt.Sprintf("%s: step down DequantWords=1 to avoid WGMMA serialization", g.Name))
				}
			} else if g.StreamK > 0 {
				cand := g
				cand.StreamK = 0
				if _, err := cand.norm(); err == nil {
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
			if _, err := cand.norm(); err == nil {
				g = cand
				reasons = append(reasons, fmt.Sprintf("%s: step down DequantWords 2 -> 1 to reduce register pressure", g.Name))
			}
		} else if bn > 64 {
			cand := g
			cand.BN = 64
			if cand.Repacked {
				cand.Repacked = false
				cand.DequantWords = 0
				if cand.WarpSpecialized && cand.Weight == WeightNVFP4 {
					cand.WarpSpecialized = false
				}
			}
			if _, err := cand.norm(); err == nil {
				g = cand
				reasons = append(reasons, fmt.Sprintf("%s: step down BN %d -> 64 to reduce register pressure", g.Name, bn))
			} else if stages > 2 {
				cand2 := g
				cand2.Stages = 2
				if _, err := cand2.norm(); err == nil {
					g = cand2
					reasons = append(reasons, fmt.Sprintf("%s: step down Stages %d -> 2 to reduce register pressure", g.Name, stages))
				}
			}
		} else if stages > 2 {
			cand := g
			cand.Stages = 2
			if _, err := cand.norm(); err == nil {
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
		linearEpi := g.Epilogue.root == nil || g.Epilogue.IsIdentity()
		if !g.WarpSpecialized && !g.Grouped && linearEpi && g.StreamK == 0 {
			cand := g
			cand.StreamK = sms
			if _, err := cand.norm(); err == nil {
				g = cand
				reasons = append(reasons, fmt.Sprintf("%s: enable StreamK=%d to eliminate wave tail quantization", g.Name, sms))
			}
		} else if g.GroupM != 8 {
			cand := g
			cand.GroupM = 8
			if _, err := cand.norm(); err == nil {
				g = cand
				reasons = append(reasons, fmt.Sprintf("%s: set GroupM=8 to improve wave L2 locality", g.Name))
			}
		}
	}

	return g, reasons
}

func refineSkinnyGEMM(g SkinnyGEMM, m int, obs KernelObservation, bns []BottleneckKind) (SkinnyGEMM, []string) {
	var reasons []string
	has := func(k BottleneckKind) bool { return slices.Contains(bns, k) }

	if has(BottleneckRegisterSpill) {
		if g.Prefetch > 0 {
			cand := g
			cand.Prefetch = 0
			if _, err := cand.norm(); err == nil {
				g = cand
				reasons = append(reasons, fmt.Sprintf("%s: disable Prefetch to eliminate register spills", g.Name))
			}
		}
		normed, err := g.norm()
		if err == nil && normed.RowsPerBlock > 1 {
			for _, nextR := range []int{2, 1} {
				if nextR < normed.RowsPerBlock && g.N%nextR == 0 {
					cand := g
					cand.RowsPerBlock = nextR
					if _, err := cand.norm(); err == nil {
						prev := normed.RowsPerBlock
						g = cand
						reasons = append(reasons, fmt.Sprintf("%s: step down RowsPerBlock %d -> %d to reduce register pressure", g.Name, prev, nextR))
						break
					}
				}
			}
		} else if err == nil && normed.Unroll > 1 {
			cand := g
			cand.Unroll = max(normed.Unroll/2, 1)
			if _, err := cand.norm(); err == nil {
				g = cand
				reasons = append(reasons, fmt.Sprintf("%s: step down Unroll %d -> %d to reduce register pressure", g.Name, normed.Unroll, cand.Unroll))
			}
		}
	}

	if !has(BottleneckRegisterSpill) && (has(BottleneckMemoryBoundKV) || has(BottleneckLowOccupancy)) {
		if g.Prefetch <= 0 {
			cand := g
			cand.Prefetch = 2
			if normed, err := g.norm(); err == nil {
				steps := normed.K / (32 * normed.Weight.chunk())
				perWarp := (steps/normed.SplitK + normed.Warps - 1) / normed.Warps
				cand.Prefetch = min(4, max(2, perWarp))
			}
			if _, err := cand.norm(); err == nil {
				g = cand
				reasons = append(reasons, fmt.Sprintf("%s: enable Prefetch=%d to hide HBM weight streaming latency", g.Name, g.Prefetch))
			}
		}
		if g.Weight == WeightNVFP4 && !g.BlackwellFP4 {
			cand := g
			cand.BlackwellFP4 = true
			if _, err := cand.norm(); err == nil {
				g = cand
				reasons = append(reasons, fmt.Sprintf("%s: enable BlackwellFP4 hardware nibble widening for NVFP4", g.Name))
			}
		}
	}

	if has(BottleneckWaveTail) {
		if normed, err := g.norm(); err == nil && normed.RowsPerBlock > 1 && g.N >= 1 {
			cand := g
			cand.RowsPerBlock = 1
			if _, err := cand.norm(); err == nil {
				g = cand
				reasons = append(reasons, fmt.Sprintf("%s: step down RowsPerBlock %d -> 1 to spread CTAs across SMs", g.Name, normed.RowsPerBlock))
			}
		}
	}

	return g, reasons
}

func refinePagedAttention(a PagedAttention, batch, maxKeys int, obs KernelObservation, bns []BottleneckKind) (PagedAttention, []string) {
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
		if _, _, err := cand.norm(); err == nil {
			a = cand
			reasons = append(reasons, fmt.Sprintf("%s: enable Swizzle to eliminate shared memory bank conflicts", a.Name))
		}
	}

	if has(BottleneckMemoryBoundKV) || has(BottleneckLowOccupancy) {
		g := a.Hq / max(a.Hkv, 1)
		if !a.WGMMA && g >= 8 && HeadDim%32 == 0 && a.Q == E4M3 && a.KVHeadsPerBlock <= 1 {
			cand := a
			cand.WGMMA = true
			cand.QKFP8 = true
			if _, _, err := cand.norm(); err == nil {
				a = cand
				reasons = append(reasons, fmt.Sprintf("%s: enable WGMMA and QKFP8 for G=%d FP8 KV attention", a.Name, g))
			}
		}
		if a.TableAhead < 2 {
			cand := a
			if cand.Stages == 0 {
				if normed, _, err := cand.norm(); err == nil {
					cand.Stages = normed.Stages
				} else {
					cand.Stages = 2
				}
			}
			cand.TableAhead = min(2, max(cand.Stages, 2))
			if _, _, err := cand.norm(); err == nil {
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
			if _, _, err := cand.norm(); err == nil {
				a = cand
				reasons = append(reasons, fmt.Sprintf("%s: adjust Chunk %d -> %d to balance SM wave occupancy", a.Name, curChunk, bestChunk))
			}
		}

		effChunk := a.Chunk
		if effChunk == 0 {
			effChunk = 256
		}
		splits := (maxKeys + effChunk - 1) / effChunk
		if splits >= ClusterSize && !a.ClusterCombine {
			cand := a
			cand.ClusterCombine = true
			if _, _, err := cand.norm(); err == nil {
				a = cand
				reasons = append(reasons, fmt.Sprintf("%s: enable ClusterCombine across %d splits", a.Name, splits))
			}
		}
	}

	if has(BottleneckRegisterSpill) {
		if a.Depth > 1 {
			cand := a
			cand.Depth = 1
			if _, _, err := cand.norm(); err == nil {
				a = cand
				reasons = append(reasons, fmt.Sprintf("%s: step down Depth -> 1 to eliminate register spills", a.Name))
			}
		} else if a.TableAhead > 1 {
			cand := a
			cand.TableAhead = 1
			if _, _, err := cand.norm(); err == nil {
				a = cand
				reasons = append(reasons, fmt.Sprintf("%s: step down TableAhead -> 1 to reduce register pressure", a.Name))
			}
		}
	}

	return a, reasons
}

func refineRowKernel(r RowKernel, obs KernelObservation, bns []BottleneckKind) (RowKernel, []string) {
	var reasons []string
	has := func(k BottleneckKind) bool { return slices.Contains(bns, k) }

	if has(BottleneckRegisterSpill) && r.Cluster {
		cand := r
		cand.Cluster = false
		if _, err := cand.check(); err == nil {
			r = cand
			reasons = append(reasons, fmt.Sprintf("%s: disable Cluster to reduce register pressure", r.Name))
		}
		return r, reasons
	}

	if (has(BottleneckWaveTail) || has(BottleneckMemoryBoundKV) || has(BottleneckLowOccupancy)) && !r.Cluster && r.CanCluster() {
		cand := r
		cand.Cluster = true
		if _, err := cand.check(); err == nil {
			r = cand
			reasons = append(reasons, fmt.Sprintf("%s: enable 8-CTA Cluster reduction", r.Name))
		}
	}
	return r, reasons
}

// RefinePlan matches plan's kernel steps to receipt.Observations (by kernel
// name or step index), applies [RefineStep] to each step, updates any affected
// workspace tensors, and if [BottleneckLaunchOverhead] is present on at least
// one consumer step of a megakernel-compatible plan, lowers the refined plan
// to a persistent [MegakernelSource] named plan.Name+"_mega".
func RefinePlan(plan *Plan, receipt ProfileReceipt) (*Plan, *MegakernelSource, []string, error) {
	if plan == nil {
		return nil, nil, nil, fmt.Errorf("refine plan: nil plan")
	}
	refined := clonePlan(plan)
	defaultSMs := 0
	if receipt.Device != "" {
		if cfg, err := (CostConfig{Device: receipt.Device}).norm(); err == nil {
			defaultSMs = cfg.SMs
		}
	}

	var (
		actions      []string
		wantMega     bool
		consumerHits int
	)
	matched := make([]bool, len(receipt.Observations))
	for i := range refined.Kernels {
		obs, ok := matchObservation(refined.Kernels[i], i, receipt.Observations, matched)
		if !ok {
			continue
		}
		if obs.DeviceSMs <= 0 && defaultSMs > 0 {
			obs.DeviceSMs = defaultSMs
		}
		bns := ClassifyBottlenecks(obs)
		if slices.Contains(bns, BottleneckLaunchOverhead) {
			if i >= 1 || len(refined.Kernels) > 1 {
				consumerHits++
				wantMega = true
			}
		}

		prevTileStreamK := refined.Kernels[i].Tile != nil && refined.Kernels[i].Tile.StreamK > 0
		step, stepLog := RefineStep(refined.Kernels[i], obs)
		refined.Kernels[i] = step
		actions = append(actions, stepLog...)

		// Keep plan workspace buffers consistent with refined schedule parameters.
		if step.Attention != nil && len(step.Args) > 0 {
			maxKeys := step.MaxKeys
			if maxKeys <= 0 {
				if step.Attention.Window > 0 {
					maxKeys = step.Attention.Window + max(step.Attention.QLen, 1) - 1
				} else {
					maxKeys = 1024
				}
			}
			wsBytes := max(step.Attention.WorkspaceBytes(max(step.Batch, 1), maxKeys), 64)
			wsName := step.Args[len(step.Args)-1]
			for j := range refined.Temps {
				if refined.Temps[j].Name == wsName {
					refined.Temps[j].Shape = []int{wsBytes}
				}
			}
		}
		if step.Tile != nil && step.Tile.StreamK > 0 && !prevTileStreamK {
			wsBytes := step.Tile.WorkspaceBytes()
			if wsBytes > 0 {
				wsName := fmt.Sprintf("%s_sk_ws%d", refined.Name, i+1)
				refined.Temps = append(refined.Temps, Tensor{
					Name:  wsName,
					DType: F32,
					Shape: []int{(wsBytes + 3) / 4},
				})
				refined.Kernels[i].Args = append(slices.Clone(refined.Kernels[i].Args), wsName)
			}
		}
	}

	if wantMega && consumerHits >= 1 && len(refined.Kernels) >= 2 {
		megaName := refined.Name + "_mega"
		if refined.Name == "" {
			megaName = "plan_mega"
		}
		mega, err := refined.Megakernel(megaName)
		if err != nil {
			// If a secondary refinement enabled Cluster on a RowKernel or
			// WarpSpecialized on a TileGEMM that prevents megakernel fusion,
			// retry with megakernel-compatible stage flags.
			compat := clonePlan(refined)
			for i := range compat.Kernels {
				if compat.Kernels[i].Row != nil && compat.Kernels[i].Row.Cluster {
					compat.Kernels[i].Row.Cluster = false
				}
				if compat.Kernels[i].Tile != nil && compat.Kernels[i].Tile.WarpSpecialized {
					compat.Kernels[i].Tile.WarpSpecialized = false
				}
			}
			if m2, err2 := compat.Megakernel(megaName); err2 == nil {
				refined = compat
				mega = m2
				err = nil
			}
		}
		if err == nil {
			actions = append(actions, fmt.Sprintf("plan %s: promote %d stages to persistent megakernel %s to eliminate launch overhead", refined.Name, len(refined.Kernels), mega.Name))
			return refined, &mega, actions, nil
		}
	}

	return refined, nil, actions, nil
}

// RefineFromProfile is a convenience alias for [RefinePlan].
func RefineFromProfile(plan *Plan, receipt ProfileReceipt) (*Plan, *MegakernelSource, []string, error) {
	return RefinePlan(plan, receipt)
}

func matchObservation(k PlanKernel, idx int, obs []KernelObservation, used []bool) (KernelObservation, bool) {
	kName := stepKernelName(k)
	if kName != "" {
		for j, o := range obs {
			if used[j] || o.Name == "" {
				continue
			}
			if o.Name == kName || matchesVariantFunction(o.Name, kName) {
				used[j] = true
				return o, true
			}
		}
	}
	if idx < len(obs) && !used[idx] {
		if obs[idx].Name == "" || kName == "" || len(obs) == 1 {
			used[idx] = true
			return obs[idx], true
		}
		// Fall back to positional matching when all observations are positional.
		if idx < len(obs) {
			used[idx] = true
			return obs[idx], true
		}
	}
	return KernelObservation{}, false
}

func stepKernelName(k PlanKernel) string {
	switch {
	case k.GEMM != nil:
		return k.GEMM.Name
	case k.Tile != nil:
		return k.Tile.Name
	case k.Attention != nil:
		return k.Attention.Name
	case k.Row != nil:
		return k.Row.Name
	case k.TopK != nil:
		return k.TopK.Name
	case k.Align != nil:
		return k.Align.Name
	case k.ExpertSum != nil:
		return k.ExpertSum.Name
	default:
		return ""
	}
}

func cloneStep(k PlanKernel) PlanKernel {
	out := k
	out.Args = slices.Clone(k.Args)
	if k.GEMM != nil {
		g := *k.GEMM
		g.Extras = slices.Clone(k.GEMM.Extras)
		out.GEMM = &g
	}
	if k.Tile != nil {
		t := *k.Tile
		t.Extras = slices.Clone(k.Tile.Extras)
		out.Tile = &t
	}
	if k.Attention != nil {
		a := *k.Attention
		out.Attention = &a
	}
	if k.Row != nil {
		r := *k.Row
		r.Inputs = slices.Clone(k.Row.Inputs)
		r.Values = make([]RowValue, len(k.Row.Values))
		for i, v := range k.Row.Values {
			v.Args = slices.Clone(v.Args)
			r.Values[i] = v
		}
		r.Outputs = slices.Clone(k.Row.Outputs)
		out.Row = &r
	}
	if k.TopK != nil {
		tk := *k.TopK
		out.TopK = &tk
	}
	if k.Align != nil {
		al := *k.Align
		out.Align = &al
	}
	if k.ExpertSum != nil {
		es := *k.ExpertSum
		out.ExpertSum = &es
	}
	return out
}

func clonePlan(p *Plan) *Plan {
	if p == nil {
		return nil
	}
	out := &Plan{
		Name:    p.Name,
		Inputs:  make([]Tensor, len(p.Inputs)),
		Temps:   make([]Tensor, len(p.Temps)),
		Output:  Tensor{Name: p.Output.Name, DType: p.Output.DType, Shape: slices.Clone(p.Output.Shape)},
		Kernels: make([]PlanKernel, len(p.Kernels)),
	}
	for i, t := range p.Inputs {
		out.Inputs[i] = Tensor{Name: t.Name, DType: t.DType, Shape: slices.Clone(t.Shape)}
	}
	for i, t := range p.Temps {
		out.Temps[i] = Tensor{Name: t.Name, DType: t.DType, Shape: slices.Clone(t.Shape)}
	}
	for i, k := range p.Kernels {
		out.Kernels[i] = cloneStep(k)
	}
	return out
}

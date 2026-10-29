package hopper

import (
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudadevicetable"
	"github.com/tmc/lib/nvidia/x/cudaroofline"
	"github.com/tmc/lib/nvidia/x/egraph"
)

// CostConfig describes the hardware ceiling and launch parameters used by
// [CompileWithCost], [Plan.SelectSchedules], and [Plan.EstimateNs].
//
// When Ceiling.PeakMemoryGBps is zero, Device (defaulting to "NVIDIA H200") is
// resolved via [cudaroofline.LookupDeviceCeiling] or [cudaroofline.LookupCeiling].
type CostConfig struct {
	// Device is the target GPU product or architecture name (e.g. "NVIDIA H200"
	// or "sm_90"). Empty defaults to "NVIDIA H200".
	Device string

	// Residency is the memory path for cudaroofline ceiling lookups; empty
	// defaults to cudaroofline.DeviceResident.
	Residency cudaroofline.Residency

	// Ceiling overrides the device lookup when PeakMemoryGBps > 0.
	Ceiling cudaroofline.Ceiling

	// TensorTFLOPS is the BF16/FP16 tensor-core compute ceiling in TFLOPS.
	// Zero defaults to 778.0 on Hopper (sm_90 / NVIDIA H200) or
	// Ceiling.PeakTFLOPS.
	TensorTFLOPS float64

	// SMs is the streaming multiprocessor count. Zero uses the device
	// profile, then the ceiling architecture's first profile, or 132.
	SMs int

	// LaunchNs is the per-kernel launch overhead in nanoseconds (0 defaults to 2000).
	LaunchNs float64

	// MaxKeys overrides the sequence length used to price and schedule
	// PagedAttention when > 0.
	MaxKeys int
}

// maxPrefetchWeightBytes is the upper bound on weight bytes for SkinnyGEMM
// register prefetch at small batch sizes (M <= 8).
const maxPrefetchWeightBytes = 128 << 20

func (c CostConfig) isHopper() bool {
	return c.Ceiling.Architecture == "sm_90" ||
		strings.Contains(strings.ToLower(c.Device), "h200") ||
		strings.Contains(strings.ToLower(c.Ceiling.Device), "h200")
}

// profile returns the cudadevicetable profile of the device c describes:
// its product if Device or the ceiling names one, or else the first listed
// product of the ceiling's architecture.
func (c CostConfig) profile() (cudadevicetable.Profile, bool) {
	for _, name := range []string{c.Device, c.Ceiling.Device} {
		if name == "" {
			continue
		}
		if p, err := cudadevicetable.LookupProfile(name); err == nil {
			return p, true
		}
	}
	if c.Ceiling.Architecture == "" {
		return cudadevicetable.Profile{}, false
	}
	p, err := cudadevicetable.LookupProfile(cudadevicetable.Normalize(c.Ceiling.Architecture) + "a")
	return p, err == nil
}

func (c CostConfig) norm() (CostConfig, error) {
	if c.Residency == "" {
		c.Residency = cudaroofline.DeviceResident
	}
	if c.Ceiling.PeakMemoryGBps <= 0 {
		dev := strings.TrimSpace(c.Device)
		if dev == "" {
			dev = "NVIDIA H200"
		}
		ceil, err := cudaroofline.LookupDeviceCeiling(dev, c.Residency)
		if err != nil {
			ceil, err = cudaroofline.LookupCeiling(dev, c.Residency)
			if err != nil {
				return c, err
			}
		}
		c.Ceiling = ceil
	}
	if c.Ceiling.PeakTFLOPS <= 0 {
		c.Ceiling.PeakTFLOPS = 67.0
	}
	isHopper := c.isHopper()
	if c.TensorTFLOPS <= 0 {
		if isHopper {
			c.TensorTFLOPS = 778.0
		} else {
			c.TensorTFLOPS = c.Ceiling.PeakTFLOPS
		}
	}
	if c.SMs <= 0 {
		c.SMs = 132
		if p, ok := c.profile(); ok {
			c.SMs = int(p.SMs.Value)
		}
	}
	if c.LaunchNs <= 0 {
		c.LaunchNs = 2000.0
	}
	return c, nil
}

// CompileWithCost lowers e to a [Plan] and selects the hardware-optimal
// schedule for each kernel step under cfg.
func CompileWithCost(e *egraph.Expr, name string, out DType, cfg CostConfig) (*Plan, error) {
	cfg, err := cfg.norm()
	if err != nil {
		return nil, fmt.Errorf("compile with cost %s: %w", name, err)
	}
	plan, err := Compile(e, name, out)
	if err != nil {
		return nil, err
	}
	if err := plan.SelectSchedules(cfg); err != nil {
		return nil, fmt.Errorf("compile with cost %s: %w", name, err)
	}
	return plan, nil
}

func (c CostConfig) isBlackwell() bool {
	arch := strings.ToLower(strings.TrimSpace(c.Ceiling.Architecture))
	dev := strings.ToLower(strings.TrimSpace(c.Device + " " + c.Ceiling.Device))
	return strings.HasPrefix(arch, "sm_100") ||
		strings.HasPrefix(arch, "sm_120") ||
		strings.HasPrefix(arch, "sm_121") ||
		strings.Contains(dev, "gb10") ||
		strings.Contains(dev, "b200") ||
		strings.Contains(dev, "b100")
}

// SelectSchedules updates each kernel in p to the optimal schedule under cfg:
//   - SkinnyGEMM enables register Prefetch when M <= 8 and weight bytes fit
//     the register-prefetch window (and BlackwellFP4 when targeting Blackwell
//     with WeightNVFP4);
//   - TileGEMM enables BlackwellFP4 (with GroupM = 8 when M >= 64) when
//     targeting Blackwell with WeightNVFP4, or WGMMA, Swizzle128B, and
//     GroupM = 8 when M >= 64 and K is a multiple of 64;
//   - RowKernel enables Cluster when CanCluster holds;
//   - PagedAttention enables Swizzle, TableAhead = Stages - 1, and
//     DeviceSplits (or ClusterCombine when splits are a multiple of 8).
func (p *Plan) SelectSchedules(cfg CostConfig) error {
	cfg, err := cfg.norm()
	if err != nil {
		return err
	}
	blackwell := cfg.isBlackwell()
	for i := range p.Kernels {
		k := &p.Kernels[i]
		switch {
		case k.GEMM != nil:
			g := *k.GEMM
			normed, err := g.norm()
			if err != nil {
				return err
			}
			if blackwell && normed.Weight == WeightNVFP4 {
				g.BlackwellFP4 = true
			}
			wBytes := weightFormatBytes(normed.Weight, normed.N, normed.K)
			if k.M <= 8 && wBytes <= maxPrefetchWeightBytes {
				steps := normed.K / (32 * normed.Weight.chunk())
				perWarp := (steps/normed.SplitK + normed.Warps - 1) / normed.Warps
				g.Prefetch = min(4, max(2, perWarp))
			}
			if _, err := g.norm(); err == nil {
				*k.GEMM = g
			}
		case k.Tile != nil:
			t := *k.Tile
			if blackwell && t.Weight == WeightNVFP4 {
				cand := t
				cand.WGMMA = false
				cand.Swizzle128B = false
				cand.BlackwellFP4 = true
				if k.M >= 64 {
					cand.GroupM = 8
				}
				if _, err := cand.norm(); err == nil {
					*k.Tile = cand
				}
			} else if k.M >= 64 && t.K >= 64 && t.K%64 == 0 && t.Weight != WeightBF16HiLo {
				cand := t
				cand.BM = 128
				if cand.N%128 == 0 && (!cand.GLU || (cand.N/2)%64 == 0) {
					cand.BN = 128
				} else {
					cand.BN = 64
				}
				cand.WGMMA = true
				cand.Swizzle128B = true
				cand.GroupM = 8
				if _, err := cand.norm(); err == nil {
					*k.Tile = cand
				}
			}
		case k.Row != nil:
			if k.Row.CanCluster() {
				k.Row.Cluster = true
			}
		case k.Attention != nil:
			a := *k.Attention
			a.Swizzle = true
			if cfg.isHopper() && a.Q == E4M3 && a.Hkv > 0 && (a.Hq/a.Hkv)*max(a.QLen, 1) >= 16 && a.KVHeadsPerBlock <= 1 {
				a.WGMMA = true
				a.QKFP8 = true
			}
			normed, _, err := a.norm()
			if err != nil {
				return err
			}
			a.Stages = normed.Stages
			a.TableAhead = max(1, a.Stages-1)
			maxKeys := k.MaxKeys
			if cfg.MaxKeys > 0 {
				maxKeys = cfg.MaxKeys
				k.MaxKeys = maxKeys
			}
			if maxKeys <= 0 {
				if a.Window > 0 {
					maxKeys = a.Window + max(a.QLen, 1) - 1
				} else {
					maxKeys = 1024
				}
			}
			chunk := a.Chunk
			if chunk == 0 {
				chunk = 256
			}
			splits := (maxKeys + chunk - 1) / chunk
			if splits >= ClusterSize && splits%ClusterSize == 0 {
				a.ClusterCombine = true
			} else {
				a.DeviceSplits = true
			}
			if normed, _, err := a.norm(); err == nil {
				*k.Attention = normed
				wsBytes := max(normed.WorkspaceBytes(max(k.Batch, 1), maxKeys), 64)
				if len(k.Args) > 0 {
					wsName := k.Args[len(k.Args)-1]
					for j := range p.Temps {
						if p.Temps[j].Name == wsName {
							p.Temps[j].Shape = []int{wsBytes}
						}
					}
				}
			}
		}
	}
	return nil
}

// EstimateNs returns the analytical roofline execution time estimate of p in
// nanoseconds:
//
//	Σ_step (max(bytes / BW, FLOPs / TFLOPS) + LaunchNs)
func (p *Plan) EstimateNs(cfg CostConfig) float64 {
	normed, err := cfg.norm()
	if err != nil {
		normed = CostConfig{
			Ceiling:      cudaroofline.Ceiling{PeakTFLOPS: 67.0, PeakMemoryGBps: 2740.48},
			TensorTFLOPS: 778.0,
			SMs:          132,
			LaunchNs:     2000.0,
		}
	}
	var totalNs float64
	for _, k := range p.Kernels {
		totalNs += k.estimateStepNs(normed)
	}
	return totalNs
}

func (k PlanKernel) estimateStepNs(cfg CostConfig) float64 {
	bytes, flops := k.Workload(cfg)
	bw := cfg.Ceiling.PeakMemoryGBps
	tflops := cfg.Ceiling.PeakTFLOPS
	switch {
	case k.GEMM != nil:
		if k.GEMM.Prefetch > 0 {
			bw *= 1.15
		}
		if k.GEMM.BlackwellFP4 {
			bw *= 1.05
		}
	case k.Tile != nil:
		tflops = cfg.TensorTFLOPS
		if !k.Tile.WGMMA && !k.Tile.BlackwellFP4 {
			tflops *= 0.65
		}
		if k.Tile.Swizzle128B || k.Tile.BlackwellFP4 {
			bw *= 1.10
		}
		if k.Tile.GroupM > 1 {
			bw *= 1.10
		}
	case k.Row != nil:
		if k.Row.Cluster {
			bw *= 1.25
		}
	case k.Attention != nil:
		tflops = cfg.TensorTFLOPS
		if k.Attention.WGMMA {
			tflops *= 1.50
		}
		if k.Attention.Swizzle {
			bw *= 1.10
		}
		if k.Attention.TableAhead >= 2 {
			bw *= 1.05
		}
	}
	memNs := float64(bytes) / bw
	compNs := float64(flops) / (tflops * 1000.0)
	return max(memNs, compNs) + cfg.LaunchNs
}

// Workload returns the estimated DRAM traffic in bytes and floating-point
// operations for a single [PlanKernel] launch.
func (k PlanKernel) Workload(cfg CostConfig) (bytes, flops uint64) {
	switch {
	case k.GEMM != nil:
		g := k.GEMM
		m := max(k.M, 1)
		wBytes := weightFormatBytes(g.Weight, g.N, g.K)
		xBytes := uint64(m * g.K * 2)
		var extraBytes uint64
		for _, e := range g.Extras {
			elems := g.N
			switch e.Broadcast {
			case PerRow:
				elems = m
			case Full:
				elems = m * g.N
			}
			extraBytes += uint64(elems * e.DType.Bits() / 8)
		}
		outBytes := uint64(m * g.N * g.Out.Bits() / 8)
		bytes = wBytes + xBytes + extraBytes + outBytes
		flops = uint64(2*m*g.N*g.K + m*g.N*max(len(g.Extras)*2, 1))
		return bytes, flops

	case k.Tile != nil:
		t := k.Tile
		m := max(k.M, 1)
		experts := max(k.Experts, 1)
		nout := t.N
		if t.GLU {
			nout = t.N / 2
		}
		wBytes := uint64(experts) * weightFormatBytes(t.Weight, t.N, t.K)
		xElemBytes := 2
		if t.Weight == WeightE4M3 {
			xElemBytes = 1
		}
		xBytes := uint64(m * t.K * xElemBytes)
		if t.Gather {
			xBytes += uint64(m * 4)
		}
		if t.Grouped {
			xBytes += uint64((experts + 1) * 4)
		}
		var extraBytes uint64
		for _, e := range t.Extras {
			elems := nout
			switch e.Broadcast {
			case PerRow:
				elems = m
			case Full:
				elems = m * nout
			}
			extraBytes += uint64(elems * e.DType.Bits() / 8)
		}
		outBytes := uint64(m * nout * t.Out.Bits() / 8)
		if t.SumSq {
			outBytes += uint64(m * (nout / 32) * 4)
		}
		bytes = wBytes + xBytes + extraBytes + outBytes
		epiFlops := 2
		if t.GLU {
			epiFlops += 4
		}
		if t.SumSq {
			epiFlops += 2
		}
		flops = uint64(2*m*t.N*t.K + m*nout*epiFlops)
		return bytes, flops

	case k.Row != nil:
		r := k.Row
		rows := max(k.Rows, 1)
		for _, in := range r.Inputs {
			elems := rows * r.D
			switch in.Kind {
			case ColVector:
				elems = r.D
			case RowScalar:
				elems = max(rows/in.group(), 1)
			case Scalar:
				elems = 1
			}
			bytes += uint64(elems * in.DType.Bits() / 8)
		}
		for _, o := range r.Outputs {
			bytes += uint64(rows * r.D * o.DType.Bits() / 8)
		}
		flops = uint64(rows * r.D * max(len(r.Values)*2, 2))
		return bytes, flops

	case k.Attention != nil:
		a := k.Attention
		batch := max(k.Batch, 1)
		qlen := max(a.QLen, 1)
		tokens := batch * qlen
		maxKeys := k.MaxKeys
		if cfg.MaxKeys > 0 {
			maxKeys = cfg.MaxKeys
		}
		if maxKeys <= 0 {
			if a.Window > 0 {
				maxKeys = a.Window + qlen - 1
			} else {
				maxKeys = 1024
			}
		}
		qBytes := uint64(tokens * a.Hq * HeadDim * a.Q.Bits() / 8)
		kvBytes := uint64(batch * maxKeys * a.Hkv * HeadDim * 2)
		var gateBytes uint64
		if a.Gate {
			gateBytes = uint64(tokens * a.Hq * HeadDim * 2)
		}
		outBytes := uint64(tokens * a.Hq * HeadDim * a.Out.Bits() / 8)
		var wsBytes uint64
		if a.Splits(maxKeys) > 1 {
			wsBytes = uint64(a.WorkspaceBytes(batch, maxKeys))
		}
		bytes = qBytes + kvBytes + gateBytes + outBytes + wsBytes
		flops = uint64(tokens * a.Hq * maxKeys * 4 * HeadDim)
		if a.Gate {
			flops += uint64(tokens * a.Hq * HeadDim * 6)
		}
		return bytes, flops

	case k.TopK != nil:
		tk := k.TopK
		rows := max(k.Rows, 1)
		bytes = uint64(rows*tk.D*tk.X.Bits()/8 + rows*tk.K*8)
		if tk.Bias != 0 {
			bytes += uint64(tk.D * tk.Bias.Bits() / 8)
		}
		flops = uint64(rows * tk.D * 4)
		return bytes, flops

	case k.Align != nil:
		al := k.Align
		tokens := max(k.Rows, 1)
		routes := tokens * al.K
		bytes = uint64(routes*12 + (al.Local+1)*4)
		flops = uint64(routes * max(al.Local, 1))
		return bytes, flops

	case k.ExpertSum != nil:
		es := k.ExpertSum
		tokens := max(k.Rows, 1)
		routes := tokens * es.K
		yBytes := uint64(routes * es.Width * es.Y.Bits() / 8)
		var ssBytes uint64
		if !es.PreNormed {
			ssBytes = uint64(routes * (es.Width / 32) * 4)
		}
		metaBytes := uint64(routes * 8)
		outBytes := uint64(tokens * es.Width * es.Out.Bits() / 8)
		bytes = yBytes + ssBytes + metaBytes + outBytes
		flops = uint64(routes * es.Width * 3)
		return bytes, flops
	}
	return 0, 0
}

func weightFormatBytes(f WeightFormat, n, k int) uint64 {
	switch f {
	case WeightBF16:
		return uint64(n * k * 2)
	case WeightBF16HiLo:
		return uint64(n * k * 4)
	case WeightNVFP4:
		return uint64(n*k/2 + n*k/16 + 4)
	case WeightE4M3:
		return uint64(n * k)
	}
	return uint64(n * k * 2)
}

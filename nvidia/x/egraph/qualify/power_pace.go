package qualify

import (
	"errors"
	"fmt"
	"math"
	"strings"
)

// PacingConfig describes a megakernel's idle-to-active power step and the
// staggered start that spreads it out. Cluster c of NumClusters waits
// (c mod RampBuckets)·BucketStepNs before its first Tensor Core phase.
type PacingConfig struct {
	NumClusters      int
	RampBuckets      int     // 1 means no stagger
	BucketStepNs     uint32  // 0 means no stagger
	IdlePowerWatts   float64 // package power while every cluster waits
	ActivePowerWatts float64 // package power once every cluster is in its MMA phase
	CoreVoltage      float64 // volts
	// PkgInductanceNh is the effective inductance, in nanohenries, that
	// the current step sees. It is a model parameter: on-package
	// decoupling supplies most of a fast step, so it is far below the
	// raw package inductance.
	PkgInductanceNh  float64
	MaxAllowedDroopV float64 // droop above which the hardware brakes the clock
	ClusterRiseNs    float64 // time for one cluster's current to ramp; 0 means 10
	MaxRampNs        uint32  // largest stagger SolveOptimalPacing may choose; 0 means 500
}

func (c PacingConfig) rise() float64 {
	if c.ClusterRiseNs == 0 {
		return 10
	}
	return c.ClusterRiseNs
}

func (c PacingConfig) maxRamp() uint32 {
	if c.MaxRampNs == 0 {
		return 500
	}
	return c.MaxRampNs
}

func (c PacingConfig) check() error {
	switch {
	case c.NumClusters < 1:
		return fmt.Errorf("NumClusters %d", c.NumClusters)
	case c.RampBuckets < 1:
		return fmt.Errorf("RampBuckets %d", c.RampBuckets)
	case !(c.CoreVoltage > 0):
		return fmt.Errorf("CoreVoltage %v", c.CoreVoltage)
	case !(c.ActivePowerWatts >= c.IdlePowerWatts) || c.IdlePowerWatts < 0:
		return fmt.Errorf("power %v W idle, %v W active", c.IdlePowerWatts, c.ActivePowerWatts)
	case !(c.PkgInductanceNh >= 0):
		return fmt.Errorf("PkgInductanceNh %v", c.PkgInductanceNh)
	case !(c.rise() > 0):
		return fmt.Errorf("ClusterRiseNs %v", c.ClusterRiseNs)
	}
	return nil
}

// A PowerTransientReport compares the current slew of a synchronized start
// with that of the staggered start.
type PowerTransientReport struct {
	UnpacedPeakDiDtAps   float64 // peak dI/dt in amps per nanosecond, all clusters at once
	PacedPeakDiDtAps     float64 // peak dI/dt with the stagger
	UnpacedVoltageDroopV float64 // PkgInductanceNh · UnpacedPeakDiDtAps
	PacedVoltageDroopV   float64 // PkgInductanceNh · PacedPeakDiDtAps
	TriggersPowerBrake   bool    // PacedVoltageDroopV > MaxAllowedDroopV
	RampOverheadNs       uint32  // (RampBuckets-1) · BucketStepNs
}

// AnalyzePowerTransient models the die current as the sum of per-cluster
// steps of (ActivePowerWatts−IdlePowerWatts)/(CoreVoltage·NumClusters)
// amps, each rising linearly over ClusterRiseNs from its staggered start,
// and reports the peak dI/dt and the inductive droop L·dI/dt with and
// without the stagger.
//
// This is a first-order model: no decoupling capacitance dynamics, no
// voltage-regulator loop, and exact start times. On hardware the stagger
// comes from nanosleep, whose PTX definition allows any sleep in [0, 2t],
// so measured starts spread less evenly than modeled.
func AnalyzePowerTransient(cfg PacingConfig) (PowerTransientReport, error) {
	if err := cfg.check(); err != nil {
		return PowerTransientReport{}, fmt.Errorf("analyze power transient: %w", err)
	}
	unpaced := peakDiDt(cfg.NumClusters, 1, 0, cfg)
	paced := peakDiDt(cfg.NumClusters, cfg.RampBuckets, cfg.BucketStepNs, cfg)
	r := PowerTransientReport{
		UnpacedPeakDiDtAps:   unpaced,
		PacedPeakDiDtAps:     paced,
		UnpacedVoltageDroopV: cfg.PkgInductanceNh * unpaced,
		PacedVoltageDroopV:   cfg.PkgInductanceNh * paced,
		RampOverheadNs:       uint32(cfg.RampBuckets-1) * cfg.BucketStepNs,
	}
	r.TriggersPowerBrake = r.PacedVoltageDroopV > cfg.MaxAllowedDroopV
	return r, nil
}

// peakDiDt returns the largest total slope, in amps per nanosecond, of
// clusters ramping at once: bucket b holds the clusters c with
// c mod buckets == b and ramps over [b·step, b·step+rise).
func peakDiDt(clusters, buckets int, step uint32, cfg PacingConfig) float64 {
	slope := (cfg.ActivePowerWatts - cfg.IdlePowerWatts) / cfg.CoreVoltage / float64(clusters) / cfg.rise()
	count := make([]int, buckets)
	for c := range clusters {
		count[c%buckets]++
	}
	// The total slope is piecewise constant and changes only at ramp
	// starts, so its maximum is at one of them.
	best := 0
	for b := range buckets {
		t := float64(b) * float64(step)
		n := 0
		for o := range buckets {
			if s := float64(o) * float64(step); s <= t && t < s+cfg.rise() {
				n += count[o]
			}
		}
		best = max(best, n)
	}
	return slope * float64(best)
}

// SolveOptimalPacing returns cfg with the RampBuckets and BucketStepNs
// (whole nanoseconds) of least ramp overhead, ties going to fewer buckets,
// whose modeled droop is at most MaxAllowedDroopV, and its report. It fails
// if no stagger up to MaxRampNs is enough.
func SolveOptimalPacing(cfg PacingConfig) (PacingConfig, PowerTransientReport, error) {
	cfg.RampBuckets = 1
	if err := cfg.check(); err != nil {
		return cfg, PowerTransientReport{}, fmt.Errorf("solve pacing: %w", err)
	}
	limit := cfg.maxRamp()
	best := PacingConfig{}
	bestCost := uint32(math.MaxUint32)
	maxStep := uint32(math.Ceil(cfg.rise()))
	for g := 1; g <= cfg.NumClusters; g++ {
		for step := uint32(0); step <= maxStep; step++ {
			cost := uint32(g-1) * step
			if cost > limit || cost >= bestCost {
				break
			}
			if cfg.PkgInductanceNh*peakDiDt(cfg.NumClusters, g, step, cfg) <= cfg.MaxAllowedDroopV {
				best, bestCost = cfg, cost
				best.RampBuckets, best.BucketStepNs = g, step
				break
			}
		}
	}
	if best.RampBuckets == 0 {
		r, _ := AnalyzePowerTransient(cfg)
		return cfg, r, fmt.Errorf("solve pacing: no stagger within %d ns keeps droop at or below %v V (unpaced %.3g V)", limit, cfg.MaxAllowedDroopV, r.UnpacedVoltageDroopV)
	}
	r, err := AnalyzePowerTransient(best)
	return best, r, err
}

// EmitPowerPacedProloguePTX emits a PTX 7.8 device function for sm_90 and
// later (sm_121a included) that sleeps (cluster mod RampBuckets) ·
// BucketStepNs nanoseconds with nanosleep.u32, where cluster is the
// linear index of %clusterid. Without a cluster launch every CTA is its
// own cluster, so %clusterid equals %ctaid. A megakernel calls it in its
// prologue and after each grid-wide barrier. The function touches no data.
func EmitPowerPacedProloguePTX(cfg PacingConfig, funcName string) (string, error) {
	if err := cfg.check(); err != nil {
		return "", fmt.Errorf("emit paced prologue: %w", err)
	}
	if funcName == "" {
		return "", errors.New("emit paced prologue: empty function name")
	}
	return fmt.Sprintf(`.version 7.8
.target sm_90
.address_size 64

// %[1]s staggers cluster start: sleep (cluster %% %[2]d) * %[3]d ns.
.visible .func %[1]s()
{
	.reg .b32 %%x, %%y, %%z, %%nx, %%ny, %%cid, %%d;
	mov.u32 %%x, %%clusterid.x;
	mov.u32 %%y, %%clusterid.y;
	mov.u32 %%z, %%clusterid.z;
	mov.u32 %%nx, %%nclusterid.x;
	mov.u32 %%ny, %%nclusterid.y;
	mad.lo.u32 %%cid, %%z, %%ny, %%y;
	mad.lo.u32 %%cid, %%cid, %%nx, %%x;
	rem.u32 %%d, %%cid, %[2]d;
	mul.lo.u32 %%d, %%d, %[3]d;
	nanosleep.u32 %%d;
	ret;
}
`, funcName, cfg.RampBuckets, cfg.BucketStepNs), nil
}

// InterleaveSchedule merges an MMA issue stream with an independent ALU
// stream (the previous tile's epilogue, say) so the MMA issues are spread
// evenly, keeping each stream's own order. Splitting k MMA issues into
// len(alu)+1 runs bounds the longest back-to-back MMA burst, and with it
// the per-SM current swing at the tile frequency, by ceil(k/(len(alu)+1)).
// The caller must ensure the streams have no dependences on each other.
func InterleaveSchedule(mma, alu []string) []string {
	out := make([]string, 0, len(mma)+len(alu))
	runs := len(alu) + 1
	i := 0
	for r := range runs {
		// Run r gets the MMA issues [r·k/runs, (r+1)·k/runs).
		end := (r + 1) * len(mma) / runs
		out = append(out, mma[i:end]...)
		i = end
		if r < len(alu) {
			out = append(out, alu[r])
		}
	}
	return out
}

// LongestRun returns the length of the longest run of consecutive
// instructions in sched for which isMMA reports true.
func LongestRun(sched []string, isMMA func(string) bool) int {
	best, cur := 0, 0
	for _, s := range sched {
		if isMMA(s) {
			cur++
			best = max(best, cur)
		} else {
			cur = 0
		}
	}
	return best
}

// PacedPrologueCUDA returns CUDA C++ source, for cudahostemu and nvcc, with
// the twin of [EmitPowerPacedProloguePTX] as qp_paced_prologue and a test
// kernel qp_paced_axpy (y = a·x + y after the prologue, one block's
// %globaltimer start time to starts[block] on the device, 0 on the host).
func PacedPrologueCUDA(cfg PacingConfig) (string, error) {
	if err := cfg.check(); err != nil {
		return "", fmt.Errorf("paced prologue cuda: %w", err)
	}
	return strings.NewReplacer("$BUCKETS", fmt.Sprint(cfg.RampBuckets), "$STEP", fmt.Sprint(cfg.BucketStepNs)).Replace(pacedCUDASrc), nil
}

const pacedCUDASrc = `
#include <cstdint>

__device__ __forceinline__ uint32_t qp_cluster_id() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    uint32_t x, y, z, nx, ny;
    asm volatile("mov.u32 %0, %%clusterid.x;" : "=r"(x));
    asm volatile("mov.u32 %0, %%clusterid.y;" : "=r"(y));
    asm volatile("mov.u32 %0, %%clusterid.z;" : "=r"(z));
    asm volatile("mov.u32 %0, %%nclusterid.x;" : "=r"(nx));
    asm volatile("mov.u32 %0, %%nclusterid.y;" : "=r"(ny));
    return (z * ny + y) * nx + x;
#else
    return (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
#endif
}

__device__ __forceinline__ uint64_t qp_globaltimer() {
#if defined(__CUDA_ARCH__)
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
#else
    return 0;
#endif
}

__device__ __forceinline__ void qp_paced_prologue() {
    __nanosleep((qp_cluster_id() % $BUCKETSu) * $STEPu);
}

__global__ void qp_paced_axpy(const float* x, float* y, float a, int n, uint64_t* starts) {
    qp_paced_prologue();
    if (threadIdx.x == 0) starts[blockIdx.x] = qp_globaltimer();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __fadd_rn(__fmul_rn(a, x[i]), y[i]);
}
`

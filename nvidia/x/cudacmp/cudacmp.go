package cudacmp

import (
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"io"
	"os/exec"
	"text/tabwriter"
	"time"

	"github.com/tmc/lib/nvidia/x/cudacensus"
	"github.com/tmc/lib/nvidia/x/cudaroofline"
	"github.com/tmc/lib/nvidia/x/cudatrial"
)

// Phase distinguishes the inference execution phase.
type Phase string

const (
	PhasePrefill Phase = "prefill"
	PhaseDecode  Phase = "decode"
)

// Unmeasured indicates an unmeasured or unavailable timing dimension.
// It is returned instead of zero or host wall-clock time.Since fallbacks.
const Unmeasured int64 = -1

// ErrNaiveSumDoubleCounts indicates an attempt to naively sum non-additive timing channels.
var ErrNaiveSumDoubleCounts = errors.New("cudacmp: naive sum of host, device, queue, and API durations causes double-counting")

// TimingDecomposition separates host elapsed time, summed kernel duration,
// GPU interval union, and API call duration into distinct, non-additive fields.
// Naive additions across these channels are explicitly rejected to prevent double-counting.
type TimingDecomposition struct {
	HostElapsedNs      int64 // Host CPU wall time from submission to synchronization (or Unmeasured)
	SummedKernelNs     int64 // Sum of individual kernel launch durations (or Unmeasured)
	GPUIntervalUnionNs int64 // Non-overlapping device busy duration across streams (or Unmeasured)
	APICallNs          int64 // Duration of host driver/runtime API calls (or Unmeasured)
}

// TotalDuration returns an error to explicitly reject naive addition of overlapping timing channels.
func (td TimingDecomposition) TotalDuration() (time.Duration, error) {
	return 0, ErrNaiveSumDoubleCounts
}

// LatencyDecomposition separates launch overhead from device kernel execution.
type LatencyDecomposition struct {
	HostLaunchNs uint64 // Duration of host cuLaunchKernel driver API call
	QueueNs      uint64 // Interval between enqueue and submission to GPU queue
	WaitNs       uint64 // Interval between queue submission and device execution start
	DeviceNs     uint64 // Active execution duration on streaming multiprocessors (End - Start)
}

// KernelMetric stores aggregated metrics for a single kernel or family.
type KernelMetric struct {
	Name       string
	Family     string
	Launches   int
	Timing     LatencyDecomposition
	BytesMoved uint64
	FLOPs      uint64
	Residency  cudaroofline.Residency
}

// Duration returns the total device execution duration for this kernel or family.
func (k KernelMetric) Duration() time.Duration {
	return time.Duration(k.Timing.DeviceNs)
}

// PhaseCensus stores the census partitioned by workload phase.
type PhaseCensus struct {
	Phase          Phase
	SummedKernelNs int64 // Sum of individual kernel launch durations (or Unmeasured)
	GPUBusyNs      int64 // Duration of the union of GPU execution intervals (or Unmeasured)
	HostElapsedNs  int64 // Duration of host CPU wall time (or Unmeasured)
	APICallNs      int64 // Duration of driver/runtime API calls (or Unmeasured)
	Families       map[string]KernelMetric
	Entries        []cudacensus.Entry
}

// TimingDecomposition returns the decomposed non-additive timing channels for this phase.
func (p PhaseCensus) TimingDecomposition() TimingDecomposition {
	return TimingDecomposition{
		HostElapsedNs:      p.HostElapsedNs,
		SummedKernelNs:     p.SummedKernelNs,
		GPUIntervalUnionNs: p.GPUBusyNs,
		APICallNs:          p.APICallNs,
	}
}

// SummedKernelDuration returns the sum of kernel launch execution durations for this phase.
func (p PhaseCensus) SummedKernelDuration() time.Duration {
	if p.SummedKernelNs == Unmeasured {
		return time.Duration(Unmeasured)
	}
	return time.Duration(p.SummedKernelNs)
}

// GPUBusyDuration returns the duration of the union of GPU execution intervals for this phase.
func (p PhaseCensus) GPUBusyDuration() time.Duration {
	if p.GPUBusyNs == Unmeasured {
		return time.Duration(Unmeasured)
	}
	return time.Duration(p.GPUBusyNs)
}

// Duration returns the summed kernel execution duration for this phase.
func (p PhaseCensus) Duration() time.Duration {
	return p.SummedKernelDuration()
}

// Availability indicates whether an architectural family was observed in both comparison arms.
type Availability string

const (
	AvailableBoth   Availability = "both"
	AvailableCOnly  Availability = "c_only"
	AvailableGoOnly Availability = "go_only"
	AvailableNone   Availability = "none"
)

// DifferentialReport records comparison metrics for one architectural family within a phase.
type DifferentialReport struct {
	Phase              Phase
	Family             string
	Availability       Availability // Observation status across comparison arms
	Available          bool         // true only when observed in both arms
	CObserved          bool         // true if observed in C baseline
	GoObserved         bool         // true if observed in Go candidate
	CDurationNs        int64        // C baseline device execution duration (or Unmeasured)
	GoDurationNs       int64        // Go candidate device execution duration (or Unmeasured)
	DeltaNs            int64        // Go - C duration in nanoseconds (or Unmeasured)
	Speedup            float64      // C / Go speedup factor
	LaunchDeltaNs      int64        // Host launch overhead difference (only computed when Available is true)
	NullDriftNs        int64        // Noise floor from null control basket
	ExceedsDrift       bool         // true if |DeltaNs| > NullDriftNs (only computed when Available is true)
	StatisticallyValid bool         // true if ExceedsDrift is true; exceeding drift threshold alone does not establish statistical validity
	RooflineGo         *cudaroofline.KernelPoint
	RooflineC          *cudaroofline.KernelPoint
}

// CDuration returns the C baseline device execution duration, or Unmeasured if unavailable.
func (r DifferentialReport) CDuration() time.Duration {
	if r.CDurationNs == Unmeasured {
		return time.Duration(Unmeasured)
	}
	return time.Duration(r.CDurationNs)
}

// GoDuration returns the Go candidate device execution duration, or Unmeasured if unavailable.
func (r DifferentialReport) GoDuration() time.Duration {
	if r.GoDurationNs == Unmeasured {
		return time.Duration(Unmeasured)
	}
	return time.Duration(r.GoDurationNs)
}

// Delta returns the duration difference (Go - C), or Unmeasured if unavailable.
func (r DifferentialReport) Delta() time.Duration {
	if r.DeltaNs == Unmeasured {
		return time.Duration(Unmeasured)
	}
	return time.Duration(r.DeltaNs)
}

// ArmRunner executes a workload phase and returns a receipt with measured timings and output digest.
type ArmRunner func(ctx context.Context, phase Phase) (cudatrial.Receipt, error)

// TrialConfig specifies parameters for an interleaved comparison trial.
type TrialConfig struct {
	WorkloadTokensPath string
	ContextLength      int
	PrefillTokens      int
	DecodeTokens       int
	InterleavedRuns    int
	WarmupRuns         int
	TimingKind         cudatrial.TimingKind
	DeviceIndex        int
	Residency          cudaroofline.Residency

	// BaselineArm provides the execution function for the baseline (e.g. C oracle).
	BaselineArm ArmRunner
	// CandidateArm provides the execution function for the candidate (e.g. Go implementation).
	CandidateArm ArmRunner
	// BaselineControlArm provides an independent baseline instance for baseline-vs-baseline null control.
	// If nil, BaselineArm is reused for the null control basket.
	BaselineControlArm ArmRunner

	// ValidateOutput verifies that the output digest produced by an arm is correct.
	// When nil, exact digest equality between baseline and candidate is required.
	ValidateOutput func(phase Phase, digest string) error

	COraclePath string
	COracleArgs []string
}

// TrialResult records the comprehensive results of a completed trial,
// preserving individual receipts, warmups, and null drift controls.
type TrialResult struct {
	Reports             []DifferentialReport
	BaselineReceipts    []cudatrial.Receipt
	CandidateReceipts   []cudatrial.Receipt
	NullControlReceipts []cudatrial.Receipt
	WarmupReceipts      []cudatrial.Receipt
	NullDrift           time.Duration
}

// Trial runs an interleaved differential comparison and produces a Report.
// The zero Trial is not usable; construct one with [NewTrial].
type Trial struct {
	cfg TrialConfig
}

// NewTrial creates a new trial runner with validated configuration.
func NewTrial(cfg TrialConfig) (*Trial, error) {
	if cfg.InterleavedRuns < 0 {
		return nil, fmt.Errorf("cudacmp: interleaved runs must be non-negative: %d", cfg.InterleavedRuns)
	}
	if cfg.WarmupRuns < 0 {
		return nil, fmt.Errorf("cudacmp: warmup runs must be non-negative: %d", cfg.WarmupRuns)
	}
	if cfg.PrefillTokens < 0 {
		return nil, fmt.Errorf("cudacmp: prefill tokens must be non-negative: %d", cfg.PrefillTokens)
	}
	if cfg.DecodeTokens < 0 {
		return nil, fmt.Errorf("cudacmp: decode tokens must be non-negative: %d", cfg.DecodeTokens)
	}
	return &Trial{cfg: cfg}, nil
}

// Config returns the configuration used by the trial.
func (t *Trial) Config() TrialConfig {
	return t.cfg
}

// Run executes the trial across all interleaved blocks and computes the differential reports.
func (t *Trial) Run(ctx context.Context) ([]DifferentialReport, error) {
	res, err := t.RunTrial(ctx)
	if err != nil {
		return nil, err
	}
	return res.Reports, nil
}

// RunTrial executes the trial, running warmup passes, establishing the null drift noise floor
// via baseline-versus-baseline controls, executing interleaved baseline-versus-candidate trials,
// and preserving all receipts.
func (t *Trial) RunTrial(ctx context.Context) (*TrialResult, error) {
	baselineArm := t.cfg.BaselineArm
	if baselineArm == nil && t.cfg.COraclePath != "" {
		baselineArm = CommandArmRunner(t.cfg.COraclePath, t.cfg.COracleArgs...)
	}
	if baselineArm == nil {
		return nil, errors.New("cudacmp: baseline arm not configured")
	}
	if t.cfg.CandidateArm == nil {
		return nil, errors.New("cudacmp: candidate arm not configured")
	}
	if t.cfg.InterleavedRuns <= 0 {
		return nil, fmt.Errorf("cudacmp: interleaved runs must be positive: %d", t.cfg.InterleavedRuns)
	}

	timingKind := t.cfg.TimingKind
	if timingKind == "" {
		timingKind = cudatrial.TimingWallElapsed
	}

	var phases []Phase
	if t.cfg.PrefillTokens > 0 {
		phases = append(phases, PhasePrefill)
	}
	if t.cfg.DecodeTokens > 0 {
		phases = append(phases, PhaseDecode)
	}
	if len(phases) == 0 {
		phases = []Phase{PhasePrefill, PhaseDecode}
	}

	result := &TrialResult{}

	// 1. Warmup passes
	if t.cfg.WarmupRuns > 0 {
		for _, p := range phases {
			for w := 0; w < t.cfg.WarmupRuns; w++ {
				if err := ctx.Err(); err != nil {
					return nil, fmt.Errorf("cudacmp: %w", err)
				}
				rBase, err := baselineArm(ctx, p)
				if err != nil {
					return nil, fmt.Errorf("cudacmp: warmup baseline failed for phase %s: %w", p, err)
				}
				rBase.Round = -(w + 1)
				result.WarmupReceipts = append(result.WarmupReceipts, rBase)

				rCand, err := t.cfg.CandidateArm(ctx, p)
				if err != nil {
					return nil, fmt.Errorf("cudacmp: warmup candidate failed for phase %s: %w", p, err)
				}
				rCand.Round = -(w + 1)
				result.WarmupReceipts = append(result.WarmupReceipts, rCand)
			}
		}
	}

	// 2. Baseline-vs-baseline null control to establish repeatable noise floor
	controlArm := t.cfg.BaselineControlArm
	if controlArm == nil {
		controlArm = baselineArm
	}

	var maxDrift time.Duration
	for _, p := range phases {
		phase := p
		nullArms := []cudatrial.Arm{
			{Name: "baseline_a", Run: func(ctx context.Context) (cudatrial.Receipt, error) { return baselineArm(ctx, phase) }},
			{Name: "baseline_b", Run: func(ctx context.Context) (cudatrial.Receipt, error) { return controlArm(ctx, phase) }},
		}
		nullReceipts, err := cudatrial.InterleaveWithOptions(ctx, nullArms, t.cfg.InterleavedRuns, cudatrial.Options{ContinueOnError: false})
		if err != nil {
			return nil, fmt.Errorf("cudacmp: null control failed for phase %s: %w", phase, err)
		}
		result.NullControlReceipts = append(result.NullControlReceipts, nullReceipts...)

		for _, r := range nullReceipts {
			if !r.Complete {
				return nil, fmt.Errorf("cudacmp: null control arm %q round %d incomplete", r.Name, r.Round)
			}
		}

		byArm := cudatrial.ByArm(nullReceipts)
		armA := byArm["baseline_a"]
		armB := byArm["baseline_b"]
		for i := 0; i < len(armA) && i < len(armB); i++ {
			dA, errA := armA[i].DurationFor(timingKind)
			if errA != nil {
				return nil, fmt.Errorf("cudacmp: null control baseline_a: %w", errA)
			}
			dB, errB := armB[i].DurationFor(timingKind)
			if errB != nil {
				return nil, fmt.Errorf("cudacmp: null control baseline_b: %w", errB)
			}
			diff := dA - dB
			if diff < 0 {
				diff = -diff
			}
			if diff > maxDrift {
				maxDrift = diff
			}
		}
	}
	result.NullDrift = maxDrift

	// 3. Baseline-vs-candidate interleaved evaluation
	for _, p := range phases {
		phase := p
		trialArms := []cudatrial.Arm{
			{Name: "baseline", Run: func(ctx context.Context) (cudatrial.Receipt, error) { return baselineArm(ctx, phase) }},
			{Name: "candidate", Run: func(ctx context.Context) (cudatrial.Receipt, error) { return t.cfg.CandidateArm(ctx, phase) }},
		}
		receipts, err := cudatrial.InterleaveWithOptions(ctx, trialArms, t.cfg.InterleavedRuns, cudatrial.Options{ContinueOnError: false})
		if err != nil {
			return nil, fmt.Errorf("cudacmp: trial failed for phase %s: %w", phase, err)
		}

		byArm := cudatrial.ByArm(receipts)
		result.BaselineReceipts = append(result.BaselineReceipts, byArm["baseline"]...)
		result.CandidateReceipts = append(result.CandidateReceipts, byArm["candidate"]...)

		// Verify completions
		for _, r := range receipts {
			if !r.Complete {
				return nil, fmt.Errorf("cudacmp: arm %q round %d incomplete: cannot yield accepted speedup", r.Name, r.Round)
			}
		}

		// Verify correctness
		if t.cfg.ValidateOutput != nil {
			for _, r := range receipts {
				if err := t.cfg.ValidateOutput(phase, r.Digest); err != nil {
					return nil, fmt.Errorf("cudacmp: output validation failed for arm %q round %d: %w", r.Name, r.Round, err)
				}
			}
		} else {
			exact, msg := cudatrial.SameArmExact(receipts)
			if !exact {
				return nil, fmt.Errorf("cudacmp: candidate execution divergence: %s", msg)
			}
			baseDigests := cudatrial.DistinctDigests(byArm["baseline"])
			candDigests := cudatrial.DistinctDigests(byArm["candidate"])
			if len(baseDigests) == 1 && len(candDigests) == 1 && baseDigests[0] != candDigests[0] {
				return nil, fmt.Errorf("cudacmp: output digest mismatch between baseline (%q) and candidate (%q)", baseDigests[0], candDigests[0])
			}
		}

		// Compute metrics for the phase
		var baseDurs, candDurs []time.Duration
		for _, r := range byArm["baseline"] {
			d, err := r.DurationFor(timingKind)
			if err != nil {
				return nil, fmt.Errorf("cudacmp: baseline: %w", err)
			}
			baseDurs = append(baseDurs, d)
		}
		for _, r := range byArm["candidate"] {
			d, err := r.DurationFor(timingKind)
			if err != nil {
				return nil, fmt.Errorf("cudacmp: candidate: %w", err)
			}
			candDurs = append(candDurs, d)
		}

		sortDurations(baseDurs)
		sortDurations(candDurs)

		baseMed := baseDurs[len(baseDurs)/2]
		candMed := candDurs[len(candDurs)/2]
		delta := candMed - baseMed

		var speedup float64
		if candMed > 0 {
			speedup = float64(baseMed) / float64(candMed)
		}

		absDelta := delta
		if absDelta < 0 {
			absDelta = -absDelta
		}
		exceedsDrift := absDelta > maxDrift

		result.Reports = append(result.Reports, DifferentialReport{
			Phase:              phase,
			Family:             "aggregate",
			Availability:       AvailableBoth,
			Available:          true,
			CObserved:          true,
			GoObserved:         true,
			CDurationNs:        int64(baseMed),
			GoDurationNs:       int64(candMed),
			DeltaNs:            int64(delta),
			Speedup:            speedup,
			NullDriftNs:        int64(maxDrift),
			ExceedsDrift:       exceedsDrift,
			StatisticallyValid: exceedsDrift,
		})
	}

	return result, nil
}

func sortDurations(d []time.Duration) {
	for i := 0; i < len(d); i++ {
		for j := i + 1; j < len(d); j++ {
			if d[j] < d[i] {
				d[i], d[j] = d[j], d[i]
			}
		}
	}
}

// CommandArmRunner returns an ArmRunner that executes an external binary with the specified arguments.
func CommandArmRunner(binPath string, args ...string) ArmRunner {
	return func(ctx context.Context, phase Phase) (cudatrial.Receipt, error) {
		cmdArgs := append(args, string(phase))
		cmd := exec.CommandContext(ctx, binPath, cmdArgs...)
		start := time.Now()
		out, err := cmd.Output()
		finish := time.Now()
		if err != nil {
			return cudatrial.Receipt{
				Complete: false,
				Err:      err,
				Started:  start,
				Finished: finish,
			}, err
		}
		digest := fmt.Sprintf("%x", sha256.Sum256(out))
		r := cudatrial.Receipt{
			Complete: true,
			Digest:   digest,
			Started:  start,
			Finished: finish,
		}
		_ = r.RecordTiming(cudatrial.TimingWallElapsed, finish.Sub(start))
		return r, nil
	}
}

// Render writes a human-readable ANSI terminal summary to w.
func Render(w io.Writer, reports []DifferentialReport) error {
	tw := tabwriter.NewWriter(w, 0, 8, 2, ' ', 0)
	fmt.Fprintln(tw, "PHASE\tFAMILY\tC TIME\tGO TIME\tDELTA\tSPEEDUP\tDRIFT FLOOR\tEXCEEDS DRIFT?")
	for _, r := range reports {
		var statusStr string
		var cTimeStr string
		var goTimeStr string
		var deltaStr string
		var speedupStr string

		if r.CObserved && r.CDurationNs != Unmeasured {
			cTimeStr = formatDurationNs(r.CDurationNs)
		} else {
			cTimeStr = "n/a"
		}

		if r.GoObserved && r.GoDurationNs != Unmeasured {
			goTimeStr = formatDurationNs(r.GoDurationNs)
		} else {
			goTimeStr = "n/a"
		}

		if !r.Available || r.DeltaNs == Unmeasured {
			deltaStr = "n/a"
			speedupStr = "n/a"
			if r.Availability == AvailableCOnly {
				statusStr = "UNAVAILABLE (C only)"
			} else if r.Availability == AvailableGoOnly {
				statusStr = "UNAVAILABLE (Go only)"
			} else {
				statusStr = "UNAVAILABLE"
			}
		} else {
			deltaStr = formatDeltaNs(r.DeltaNs)
			if r.Speedup > 0 {
				speedupStr = fmt.Sprintf("%.2fx", r.Speedup)
			} else {
				speedupStr = "n/a"
			}
			if r.ExceedsDrift {
				statusStr = "YES"
			} else {
				statusStr = "NOISE"
			}
		}

		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
			r.Phase,
			r.Family,
			cTimeStr,
			goTimeStr,
			deltaStr,
			speedupStr,
			formatDurationNs(r.NullDriftNs),
			statusStr,
		)
	}
	return tw.Flush()
}

func formatDurationNs(ns int64) string {
	if ns == Unmeasured {
		return "unmeasured"
	}
	if ns < 0 {
		return "-" + formatDurationNs(-ns)
	}
	d := time.Duration(ns)
	if d < time.Microsecond {
		return fmt.Sprintf("%d ns", ns)
	}
	if d < time.Millisecond {
		return fmt.Sprintf("%.2f µs", float64(ns)/1e3)
	}
	if d < time.Second {
		return fmt.Sprintf("%.2f ms", float64(ns)/1e6)
	}
	return fmt.Sprintf("%.3f s", float64(ns)/1e9)
}

func formatDeltaNs(ns int64) string {
	if ns == Unmeasured {
		return "unmeasured"
	}
	if ns > 0 {
		return "+" + formatDurationNs(ns)
	}
	if ns < 0 {
		return "-" + formatDurationNs(-ns)
	}
	return "0 ns"
}

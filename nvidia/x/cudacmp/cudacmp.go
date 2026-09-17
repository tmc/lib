package cudacmp

import (
	"context"
	"fmt"
	"io"
	"text/tabwriter"
	"time"

	"github.com/tmc/lib/nvidia/x/cudacensus"
	"github.com/tmc/lib/nvidia/x/cudaroofline"
)

// Phase distinguishes the inference execution phase.
type Phase string

const (
	PhasePrefill Phase = "prefill"
	PhaseDecode  Phase = "decode"
)

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
	SummedKernelNs int64 // Sum of individual kernel launch durations
	GPUBusyNs      int64 // Duration of the union of GPU execution intervals
	Families       map[string]KernelMetric
	Entries        []cudacensus.Entry
}

// SummedKernelDuration returns the sum of kernel launch execution durations for this phase.
func (p PhaseCensus) SummedKernelDuration() time.Duration {
	return time.Duration(p.SummedKernelNs)
}

// GPUBusyDuration returns the duration of the union of GPU execution intervals for this phase.
func (p PhaseCensus) GPUBusyDuration() time.Duration {
	return time.Duration(p.GPUBusyNs)
}

// Duration returns the summed kernel execution duration for this phase.
func (p PhaseCensus) Duration() time.Duration {
	return p.SummedKernelDuration()
}

// Availability indicates whether an architectural family was observed in both comparison arms.
type Availability string

const (
	// AvailableBoth indicates the family was observed in both C and Go arms.
	AvailableBoth Availability = "both"
	// AvailableCOnly indicates the family was observed only in the C baseline.
	AvailableCOnly Availability = "c_only"
	// AvailableGoOnly indicates the family was observed only in the Go candidate.
	AvailableGoOnly Availability = "go_only"
)

// DifferentialReport holds pairwise comparison results across arms and control baselines.
type DifferentialReport struct {
	Phase              Phase
	Family             string
	Availability       Availability // Availability status: "both", "c_only", or "go_only"
	Available          bool         // true if observed in both arms and comparable
	CObserved          bool         // true if observed in C arm
	GoObserved         bool         // true if observed in Go arm
	CDurationNs        int64        // C baseline device duration (if CObserved)
	GoDurationNs       int64        // Go candidate device duration (if GoObserved)
	DeltaNs            int64        // Go - C (only computed when Available is true)
	Speedup            float64      // C / Go (only computed when Available is true)
	LaunchDeltaNs      int64        // Host launch overhead difference (only computed when Available is true)
	NullDriftNs        int64        // Noise floor from null control basket
	ExceedsDrift       bool         // true if |DeltaNs| > NullDriftNs (only computed when Available is true)
	StatisticallyValid bool         // true if ExceedsDrift is true; exceeding drift threshold alone does not establish statistical validity
	RooflineGo         *cudaroofline.KernelPoint
	RooflineC          *cudaroofline.KernelPoint
}

// CDuration returns the C baseline device execution duration.
func (r DifferentialReport) CDuration() time.Duration {
	return time.Duration(r.CDurationNs)
}

// GoDuration returns the Go candidate device execution duration.
func (r DifferentialReport) GoDuration() time.Duration {
	return time.Duration(r.GoDurationNs)
}

// Delta returns the duration difference (Go - C).
func (r DifferentialReport) Delta() time.Duration {
	return time.Duration(r.DeltaNs)
}

// TrialConfig specifies parameters for an interleaved comparison trial.
type TrialConfig struct {
	WorkloadTokensPath string
	ContextLength      int
	PrefillTokens      int
	DecodeTokens       int
	InterleavedRuns    int
	WarmupRuns         int
	COraclePath        string
	COracleArgs        []string
	DeviceIndex        int
	Residency          cudaroofline.Residency
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

// Run executes the trial across all interleaved blocks and computes the differential report.
func (t *Trial) Run(ctx context.Context) ([]DifferentialReport, error) {
	if t.cfg.COraclePath == "" {
		return nil, fmt.Errorf("cudacmp: oracle path not configured")
	}
	return nil, fmt.Errorf("cudacmp: trial runner not yet implemented for %s", t.cfg.COraclePath)
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

		if r.CObserved {
			cTimeStr = formatDurationNs(r.CDurationNs)
		} else {
			cTimeStr = "n/a"
		}

		if r.GoObserved {
			goTimeStr = formatDurationNs(r.GoDurationNs)
		} else {
			goTimeStr = "n/a"
		}

		if !r.Available {
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
	if ns > 0 {
		return "+" + formatDurationNs(ns)
	}
	if ns < 0 {
		return "-" + formatDurationNs(-ns)
	}
	return "0 ns"
}

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

// TotalNs reports the sum of launch latency, queue wait time, and device execution.
func (l LatencyDecomposition) TotalNs() uint64 {
	return l.HostLaunchNs + l.QueueNs + l.WaitNs + l.DeviceNs
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
	Phase    Phase
	TotalNs  int64
	Families map[string]KernelMetric
	Entries  []cudacensus.Entry
}

// Duration returns the total device execution duration for this phase.
func (p PhaseCensus) Duration() time.Duration {
	return time.Duration(p.TotalNs)
}

// DifferentialReport holds pairwise comparison results across arms and control baselines.
type DifferentialReport struct {
	Phase              Phase
	Family             string
	CDurationNs        int64
	GoDurationNs       int64
	DeltaNs            int64   // Go - C
	Speedup            float64 // C / Go
	LaunchDeltaNs      int64   // Host launch overhead difference
	NullDriftNs        int64   // Noise floor from null control basket
	StatisticallyValid bool    // true if |DeltaNs| > NullDriftNs
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
	fmt.Fprintln(tw, "PHASE\tFAMILY\tC TIME\tGO TIME\tDELTA\tSPEEDUP\tDRIFT FLOOR\tVALID?")
	for _, r := range reports {
		validStr := "YES"
		if !r.StatisticallyValid {
			validStr = "NOISE"
		}
		var speedupStr string
		if r.Speedup > 0 {
			speedupStr = fmt.Sprintf("%.2fx", r.Speedup)
		} else {
			speedupStr = "n/a"
		}
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
			r.Phase,
			r.Family,
			formatDurationNs(r.CDurationNs),
			formatDurationNs(r.GoDurationNs),
			formatDeltaNs(r.DeltaNs),
			speedupStr,
			formatDurationNs(r.NullDriftNs),
			validStr,
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

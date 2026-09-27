package cudaschedule

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"runtime"
	"time"

	"github.com/tmc/lib/nvidia/x/cudaerrors"
)

// ConfirmationSummary records confirmation runs for a finding.
type ConfirmationSummary struct {
	Attempts  int       `json:"attempts"`
	Findings  int       `json:"findings"`
	Confirmed bool      `json:"confirmed"`
	Outcomes  []Verdict `json:"outcomes"`
}

// SeedResult records the outcome of a single seed in a campaign.
type SeedResult struct {
	Seed         uint64               `json:"seed"`
	Verdict      Verdict              `json:"verdict"`
	Run          RunRecord            `json:"run"`
	Report       RewriteReport        `json:"report"`
	Confirmation *ConfirmationSummary `json:"confirmation,omitempty"`
}

// CampaignOptions configures a perturbation sweep over a seed range.
type CampaignOptions struct {
	// PTXSource is the baseline PTX source text.
	PTXSource string

	// Executable is the path to the executable to run or patch.
	Executable string

	// Args contains additional command-line arguments for the executable.
	Args []string

	// WorkDir is the working directory for process execution.
	WorkDir string

	// SeedStart is the starting seed of the half-open interval.
	SeedStart uint64

	// SeedEnd is the ending seed of the half-open interval.
	SeedEnd uint64

	// Intensity scales perturbation probability and delay durations.
	Intensity float64

	// MaxSleepNs is the delay ceiling in nanoseconds (defaults to 64 µs).
	MaxSleepNs uint32

	// Timeout is the execution deadline per run before triggering Hang.
	Timeout time.Duration

	// Focus is an optional substring to focus perturbation.
	Focus string

	// KeepGoing continues the sweep even if a GpuWedged verdict is encountered.
	KeepGoing bool

	// ConfirmRuns is the total number of executions for a finding (minimum 1).
	ConfirmRuns int

	// CompareOutput treats stdout differences against baseline as OutputChanged.
	CompareOutput bool

	// DeviceProbe checks whether physical GPU hardware is available.
	// Defaults to checking driver device files.
	DeviceProbe func() bool

	// RunVariant is an optional runner hook for testing or external perturbation layers.
	RunVariant func(ctx context.Context, seed uint64, ptx string) (RunRecord, error)
}

// CampaignReport summarizes the results of a schedule perturbation campaign.
type CampaignReport struct {
	SeedStart   uint64       `json:"seed_start"`
	SeedEnd     uint64       `json:"seed_end"`
	Intensity   float64      `json:"intensity"`
	MaxSleepNs  uint32       `json:"max_sleep_ns"`
	Focus       string       `json:"focus,omitempty"`
	ConfirmRuns int          `json:"confirm_runs"`
	SitesTotal  int          `json:"sites_total"`
	Seeds       []SeedResult `json:"seeds"`
}

// FindingCount returns the total number of seeds that produced a schedule finding.
func (cr *CampaignReport) FindingCount() int {
	count := 0
	for _, s := range cr.Seeds {
		if s.Verdict.IsFinding() {
			count++
		}
	}
	return count
}

// RunCampaign sweeps a half-open seed interval and executes a perturbation campaign.
// Without an available device or custom runner, it returns [cudaerrors.ErrHardwarePending].
func RunCampaign(ctx context.Context, opts CampaignOptions) (*CampaignReport, error) {
	if opts.SeedStart >= opts.SeedEnd {
		return nil, fmt.Errorf("cudaschedule: invalid seed range %d..%d: start must be less than end", opts.SeedStart, opts.SeedEnd)
	}
	if opts.ConfirmRuns <= 0 {
		return nil, fmt.Errorf("cudaschedule: confirm_runs must be at least 1, got %d", opts.ConfirmRuns)
	}

	probe := opts.DeviceProbe
	if probe == nil {
		probe = defaultDeviceProbe
	}

	if opts.RunVariant == nil && !probe() {
		return nil, fmt.Errorf("cudaschedule: %w", cudaerrors.ErrHardwarePending)
	}

	analysis, err := Analyze(opts.PTXSource)
	if err != nil {
		return nil, fmt.Errorf("cudaschedule: baseline analysis: %w", err)
	}

	timeout := opts.Timeout
	if timeout <= 0 {
		timeout = 30 * time.Second
	}

	report := &CampaignReport{
		SeedStart:   opts.SeedStart,
		SeedEnd:     opts.SeedEnd,
		Intensity:   opts.Intensity,
		MaxSleepNs:  opts.MaxSleepNs,
		Focus:       opts.Focus,
		ConfirmRuns: opts.ConfirmRuns,
		SitesTotal:  len(analysis),
	}

	var baselineRun RunRecord
	if opts.CompareOutput {
		baselineRun = executeRun(ctx, opts, opts.PTXSource, timeout)
	}

	for seed := opts.SeedStart; seed < opts.SeedEnd; seed++ {
		if err := ctx.Err(); err != nil {
			return report, err
		}

		perturbOpts := Options{
			Seed:       seed,
			Intensity:  opts.Intensity,
			MaxSleepNs: opts.MaxSleepNs,
			Focus:      opts.Focus,
		}
		rewrite, err := Perturb(opts.PTXSource, perturbOpts)
		if err != nil {
			report.Seeds = append(report.Seeds, SeedResult{
				Seed:    seed,
				Verdict: VerdictHarnessError,
				Run:     RunRecord{HarnessErr: err.Error()},
				Report: RewriteReport{
					Seed:      seed,
					Intensity: opts.Intensity,
				},
			})
			continue
		}

		runRec := executeRun(ctx, opts, rewrite.PTX, timeout)
		if opts.CompareOutput {
			runRec.CompareOutput = true
			runRec.Baseline = &baselineRun
		}

		if runRec.TimedOut {
			pristineRec := executeRun(ctx, opts, opts.PTXSource, timeout)
			runRec.Pristine = &pristineRec
		}

		verdict := Classify(runRec)

		seedRes := SeedResult{
			Seed:    seed,
			Verdict: verdict,
			Run:     runRec,
			Report:  rewrite.Report,
		}

		if verdict.IsFinding() {
			outcomes := []Verdict{verdict}
			findings := 1
			for attempt := 2; attempt <= opts.ConfirmRuns; attempt++ {
				confirmRec := executeRun(ctx, opts, rewrite.PTX, timeout)
				if opts.CompareOutput {
					confirmRec.CompareOutput = true
					confirmRec.Baseline = &baselineRun
				}
				if confirmRec.TimedOut {
					pristineRec := executeRun(ctx, opts, opts.PTXSource, timeout)
					confirmRec.Pristine = &pristineRec
				}
				confirmVerdict := Classify(confirmRec)
				outcomes = append(outcomes, confirmVerdict)
				if confirmVerdict == verdict {
					findings++
				}
			}
			seedRes.Confirmation = &ConfirmationSummary{
				Attempts:  opts.ConfirmRuns,
				Findings:  findings,
				Confirmed: findings == opts.ConfirmRuns,
				Outcomes:  outcomes,
			}
		}

		report.Seeds = append(report.Seeds, seedRes)

		if verdict == VerdictGpuWedged && !opts.KeepGoing {
			break
		}
	}

	return report, nil
}

func executeRun(ctx context.Context, opts CampaignOptions, ptx string, timeout time.Duration) RunRecord {
	if opts.RunVariant != nil {
		rec, err := opts.RunVariant(ctx, 0, ptx)
		if err != nil {
			rec.HarnessErr = err.Error()
		}
		return rec
	}

	if opts.Executable == "" {
		return RunRecord{HarnessErr: "no executable specified"}
	}

	runCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	cmd := exec.CommandContext(runCtx, opts.Executable, opts.Args...)
	if opts.WorkDir != "" {
		cmd.Dir = opts.WorkDir
	}

	var stdoutBuf, stderrBuf bytes.Buffer
	cmd.Stdout = &stdoutBuf
	cmd.Stderr = &stderrBuf

	err := cmd.Run()
	stdout := stdoutBuf.String()
	stderr := stderrBuf.String()

	if errors.Is(runCtx.Err(), context.DeadlineExceeded) {
		return RunRecord{
			TimedOut: true,
			Stdout:   stdout,
			Stderr:   stderr,
		}
	}

	if err != nil {
		var exitErr *exec.ExitError
		if errors.As(err, &exitErr) {
			return RunRecord{
				Exited:   true,
				ExitCode: exitErr.ExitCode(),
				Stdout:   stdout,
				Stderr:   stderr,
			}
		}
		return RunRecord{
			Exited:     false,
			HarnessErr: err.Error(),
			Stdout:     stdout,
			Stderr:     stderr,
		}
	}

	return RunRecord{
		Exited:   true,
		ExitCode: 0,
		Stdout:   stdout,
		Stderr:   stderr,
	}
}

func defaultDeviceProbe() bool {
	if runtime.GOOS != "linux" {
		return false
	}
	_, err := os.Stat("/dev/nvidiactl")
	return err == nil
}

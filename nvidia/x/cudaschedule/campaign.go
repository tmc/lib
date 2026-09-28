package cudaschedule

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"syscall"
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
	Status       SeedStatus           `json:"status"`
	Verdict      Verdict              `json:"verdict"`
	Run          *RunRecord           `json:"run,omitempty"`
	Report       RewriteReport        `json:"report"`
	Confirmation *ConfirmationSummary `json:"confirmation,omitempty"`
}

// SeedStatus distinguishes a drawn seed from an executed variant.
type SeedStatus string

const (
	SeedExecuted     SeedStatus = "executed"
	SeedEmptyDraw    SeedStatus = "empty_draw"
	SeedHarnessError SeedStatus = "harness_error"
)

// CampaignOptions configures a perturbation sweep over a seed range.
type CampaignOptions struct {
	// PTXSource is the baseline PTX source text.
	PTXSource string

	// Executable is the path to the executable that consumes the generated PTX.
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

	// PristineTimeout bounds baseline and health-check runs. Zero uses Timeout.
	// A larger value can keep host scheduling noise from classifying a healthy
	// reference run as a hang when variants use a deliberately short watchdog.
	PristineTimeout time.Duration

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

	// BuildVariant prepares a runnable artifact from a PTX file. It receives an
	// explicit pristine flag and returns the executable and arguments used for
	// that run. The campaign appends ptxPath as the final argument. The seed is
	// the campaign's start seed for pristine runs and is meaningful only when
	// pristine is false. Builds have their own BuildTimeout, separate from Timeout.
	BuildVariant func(ctx context.Context, seed uint64, pristine bool, ptxPath string) (string, []string, error)

	// RunVariant is an optional runner hook for testing or external perturbation
	// layers. The seed is meaningful only when pristine is false.
	RunVariant func(ctx context.Context, seed uint64, pristine bool, ptx string) (RunRecord, error)

	// BuildTimeout bounds artifact preparation separately from executable runtime.
	// The default is five minutes.
	BuildTimeout time.Duration
}

// CampaignReport summarizes the results of a schedule perturbation campaign.
type CampaignReport struct {
	SeedStart        uint64       `json:"seed_start"`
	SeedEnd          uint64       `json:"seed_end"`
	Intensity        float64      `json:"intensity"`
	MaxSleepNs       uint32       `json:"max_sleep_ns"`
	Focus            string       `json:"focus,omitempty"`
	ConfirmRuns      int          `json:"confirm_runs"`
	SitesTotal       int          `json:"sites_total"`
	SeedsDrawn       int          `json:"seeds_drawn"`
	VariantsExecuted int          `json:"variants_executed"`
	EmptyDraws       int          `json:"empty_draws"`
	HarnessErrors    int          `json:"harness_errors"`
	Baseline         RunRecord    `json:"baseline"`
	BaselineVerdict  Verdict      `json:"baseline_verdict"`
	BaselineStatus   string       `json:"baseline_status"`
	Seeds            []SeedResult `json:"seeds"`
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
	}

	variantRoot := ""
	var err error
	if opts.RunVariant == nil {
		variantRoot, err = campaignVariantRoot()
		if err != nil {
			return report, fmt.Errorf("cudaschedule: create variant directory: %w", err)
		}
		defer os.RemoveAll(variantRoot)
	}

	baselineRun, err := executeRun(ctx, opts, opts.SeedStart, opts.PTXSource, true, timeout, variantRoot)
	if err != nil {
		return report, err
	}
	report.Baseline = baselineRun
	report.BaselineVerdict = Classify(baselineRun)
	switch report.BaselineVerdict {
	case VerdictPass:
		report.BaselineStatus = "Pass"
	case VerdictSkipped:
		report.BaselineStatus = "Declined"
	default:
		report.BaselineStatus = "Broken"
	}
	if report.BaselineVerdict != VerdictPass {
		return report, nil
	}

	analysis, err := Analyze(opts.PTXSource)
	if err != nil {
		return report, fmt.Errorf("cudaschedule: baseline analysis: %w", err)
	}
	report.SitesTotal = len(analysis)

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
			recordSeed(report, SeedResult{
				Seed:    seed,
				Status:  SeedHarnessError,
				Verdict: VerdictHarnessError,
				Run:     &RunRecord{HarnessErr: err.Error()},
				Report: RewriteReport{
					Seed:      seed,
					Intensity: opts.Intensity,
				},
			})
			continue
		}
		if rewrite.Report.SitesInjected == 0 {
			recordSeed(report, SeedResult{
				Seed: seed, Status: SeedEmptyDraw, Verdict: VerdictEmptyDraw, Report: rewrite.Report,
			})
			continue
		}

		runRec, err := executeRun(ctx, opts, seed, rewrite.PTX, false, timeout, variantRoot)
		if err != nil {
			return report, err
		}
		if opts.CompareOutput {
			runRec.CompareOutput = true
			runRec.Baseline = &report.Baseline
		}

		if runRec.TimedOut {
			pristineRec, err := executeRun(ctx, opts, seed, opts.PTXSource, true, timeout, variantRoot)
			if err != nil {
				return report, err
			}
			runRec.Pristine = &pristineRec
		}

		verdict := Classify(runRec)

		seedRes := SeedResult{
			Seed:    seed,
			Verdict: verdict,
			Run:     &runRec,
			Report:  rewrite.Report,
		}
		if verdict == VerdictHarnessError {
			seedRes.Status = SeedHarnessError
		} else {
			seedRes.Status = SeedExecuted
		}

		if verdict.IsFinding() {
			outcomes := []Verdict{verdict}
			findings := 1
			for attempt := 2; attempt <= opts.ConfirmRuns; attempt++ {
				confirmRec, err := executeRun(ctx, opts, seed, rewrite.PTX, false, timeout, variantRoot)
				if err != nil {
					return report, err
				}
				if opts.CompareOutput {
					confirmRec.CompareOutput = true
					confirmRec.Baseline = &report.Baseline
				}
				if confirmRec.TimedOut {
					pristineRec, err := executeRun(ctx, opts, seed, opts.PTXSource, true, timeout, variantRoot)
					if err != nil {
						return report, err
					}
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

		recordSeed(report, seedRes)

		if verdict == VerdictGpuWedged && !opts.KeepGoing {
			break
		}
	}

	return report, nil
}

func recordSeed(report *CampaignReport, seed SeedResult) {
	report.SeedsDrawn++
	switch seed.Status {
	case SeedEmptyDraw:
		report.EmptyDraws++
	case SeedHarnessError:
		report.HarnessErrors++
	default:
		report.VariantsExecuted++
	}
	report.Seeds = append(report.Seeds, seed)
}

func executeRun(ctx context.Context, opts CampaignOptions, seed uint64, ptx string, pristine bool, timeout time.Duration, variantRoot string) (RunRecord, error) {
	if pristine && opts.PristineTimeout > 0 {
		timeout = opts.PristineTimeout
	}
	if opts.RunVariant != nil {
		runCtx, cancel := context.WithTimeout(ctx, timeout)
		defer cancel()
		rec, err := opts.RunVariant(runCtx, seed, pristine, ptx)
		if ctx.Err() != nil {
			return RunRecord{}, ctx.Err()
		}
		if err != nil {
			rec.HarnessErr = err.Error()
		}
		if errors.Is(runCtx.Err(), context.DeadlineExceeded) {
			rec.TimedOut = true
		}
		rec.PristineRun = pristine
		return rec, nil
	}

	if variantRoot == "" {
		return RunRecord{HarnessErr: "no variant directory specified", PristineRun: pristine}, nil
	}
	role := "variant"
	if pristine {
		role = "pristine"
	}
	variantPath := filepath.Join(variantRoot, role, fmt.Sprintf("seed-%d", seed), "module.ptx")
	if err := os.MkdirAll(filepath.Dir(variantPath), 0o755); err != nil {
		return RunRecord{HarnessErr: err.Error(), PristineRun: pristine}, nil
	}
	if err := os.WriteFile(variantPath, []byte(ptx), 0o600); err != nil {
		return RunRecord{HarnessErr: err.Error(), PristineRun: pristine}, nil
	}

	executable, args := opts.Executable, opts.Args
	if opts.BuildVariant != nil {
		buildTimeout := opts.BuildTimeout
		if buildTimeout <= 0 {
			buildTimeout = 5 * time.Minute
		}
		buildCtx, buildCancel := context.WithTimeout(ctx, buildTimeout)
		var err error
		executable, args, err = opts.BuildVariant(buildCtx, seed, pristine, variantPath)
		buildCancel()
		if ctx.Err() != nil {
			return RunRecord{}, ctx.Err()
		}
		if errors.Is(buildCtx.Err(), context.DeadlineExceeded) {
			return RunRecord{HarnessErr: "build timeout", PristineRun: pristine}, nil
		}
		if err != nil {
			return RunRecord{HarnessErr: err.Error(), PristineRun: pristine}, nil
		}
	}
	if executable == "" {
		return RunRecord{HarnessErr: "no executable specified", PristineRun: pristine}, nil
	}

	runCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	cmd := exec.CommandContext(runCtx, executable, args...)
	cmd.Args = append(cmd.Args, variantPath)
	cmd.WaitDelay = 250 * time.Millisecond
	if opts.WorkDir != "" {
		cmd.Dir = opts.WorkDir
	}

	var stdoutBuf, stderrBuf bytes.Buffer
	cmd.Stdout = &stdoutBuf
	cmd.Stderr = &stderrBuf

	err := cmd.Run()
	stdout := stdoutBuf.String()
	stderr := stderrBuf.String()

	if ctx.Err() != nil {
		return RunRecord{}, ctx.Err()
	}
	if errors.Is(runCtx.Err(), context.DeadlineExceeded) {
		return RunRecord{
			TimedOut:    true,
			PristineRun: pristine,
			Stdout:      stdout,
			Stderr:      stderr,
		}, nil
	}

	if err != nil {
		var exitErr *exec.ExitError
		if errors.As(err, &exitErr) {
			rec := RunRecord{
				Exited:      true,
				ExitCode:    exitErr.ExitCode(),
				PristineRun: pristine,
				Stdout:      stdout,
				Stderr:      stderr,
			}
			if status, ok := exitErr.Sys().(syscall.WaitStatus); ok && status.Signaled() {
				rec.Signal = status.Signal().String()
			}
			return rec, nil
		}
		return RunRecord{
			Exited:      false,
			HarnessErr:  err.Error(),
			PristineRun: pristine,
			Stdout:      stdout,
			Stderr:      stderr,
		}, nil
	}

	return RunRecord{
		Exited:      true,
		ExitCode:    0,
		PristineRun: pristine,
		Stdout:      stdout,
		Stderr:      stderr,
	}, nil
}

func campaignVariantRoot() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	tmp := filepath.Join(home, "tmp")
	if err := os.MkdirAll(tmp, 0o700); err != nil {
		return "", err
	}
	return os.MkdirTemp(tmp, "cudaschedule-")
}

func defaultDeviceProbe() bool {
	if runtime.GOOS != "linux" {
		return false
	}
	_, err := os.Stat("/dev/nvidiactl")
	return err == nil
}

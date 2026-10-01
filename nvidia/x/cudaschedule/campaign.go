package cudaschedule

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/tmc/lib/nvidia/x/cudaerrors"
)

// ConfirmationSummary records confirmation runs for a finding.
type ConfirmationSummary struct {
	// Attempts is the number of completed classifications, including the initial
	// run. A confirmation wedge may stop before the configured run budget.
	Attempts  int       `json:"attempts"`
	Findings  int       `json:"findings"`
	Confirmed bool      `json:"confirmed"`
	Outcomes  []Verdict `json:"outcomes"`
}

// CampaignArms adds a known-racy PTX input and a clean control to a campaign.
// Both arms use the parent campaign's seed range and runner configuration.
type CampaignArms struct {
	PlantedBugPTXSource string `json:"planted_bug_ptx_source"`
	CleanPTXSource      string `json:"clean_ptx_source"`
}

// CampaignArmSummary records whether one calibration arm produced findings.
type CampaignArmSummary struct {
	SourceDigest      string          `json:"source_sha256"`
	SeedsInRange      uint64          `json:"seeds_in_range"`
	SeedsEvaluated    uint64          `json:"seeds_evaluated"`
	Findings          int             `json:"findings"`
	ConfirmedFindings int             `json:"confirmed_findings"`
	VariantsExecuted  int             `json:"variants_executed"`
	HarnessErrors     int             `json:"harness_errors"`
	SkippedVariants   int             `json:"skipped_variants"`
	Outcome           string          `json:"outcome"`
	Error             string          `json:"error,omitempty"`
	Report            *CampaignReport `json:"report,omitempty"`
}

// CampaignArmResults records the planted-bug and clean-control outcomes.
type CampaignArmResults struct {
	PlantedBug   CampaignArmSummary `json:"planted_bug"`
	CleanControl CampaignArmSummary `json:"clean_control"`
}

// SeedResult records the outcome of a single seed in a campaign.
type SeedResult struct {
	Seed                 uint64               `json:"seed"`
	Status               SeedStatus           `json:"status"`
	Verdict              Verdict              `json:"verdict"`
	Run                  *RunRecord           `json:"run,omitempty"`
	Report               RewriteReport        `json:"report"`
	Confirmation         *ConfirmationSummary `json:"confirmation,omitempty"`
	PristineSourceDigest string               `json:"pristine_source_sha256"`
	LaunchGeometry       string               `json:"launch_geometry,omitempty"`
	InputReference       string               `json:"input_reference,omitempty"`
	MaxSleepNs           uint32               `json:"max_sleep_ns"`
	Focus                string               `json:"focus,omitempty"`
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

	// ArtifactDir is the parent directory for campaign-owned run files. The
	// default is ~/tmp.
	ArtifactDir string

	// KeepArtifacts retains PTX, stdout, stderr and replay scripts after the
	// campaign returns. It applies to the default process runner.
	KeepArtifacts bool

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

	// LaunchGeometry records the launch dimensions used by the external
	// workload, for example "grid=4x1x1 block=128x1x1".
	LaunchGeometry string

	// InputReference identifies the input fixture or corpus record used by the
	// external workload.
	InputReference string

	// OracleAttached requires a nonzero oracle liveness receipt from each run.
	// Executables print CUDASCHEDULE_ORACLE_LIVENESS:pid modules accesses.
	// The PID must match the executing process. RunVariant supplies its own
	// trusted liveness record because the campaign does not own that process.
	OracleAttached bool

	// Arms asks RunCampaign to repeat its seed sweep for a known planted bug and
	// a clean control, reporting whether each arm was detected as expected.
	Arms *CampaignArms

	// KeepGoing continues the sweep even if a GpuWedged verdict is encountered.
	KeepGoing bool

	// ConfirmRuns is the total number of executions for a finding (minimum 1).
	// GpuWedged findings are not confirmed because further runs would reuse the
	// device that failed the pristine health check. If a confirmation wedges,
	// KeepGoing controls whether remaining confirmations and seeds run.
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
	SeedStart        uint64              `json:"seed_start"`
	SeedEnd          uint64              `json:"seed_end"`
	Intensity        float64             `json:"intensity"`
	MaxSleepNs       uint32              `json:"max_sleep_ns"`
	Focus            string              `json:"focus,omitempty"`
	ConfirmRuns      int                 `json:"confirm_runs"`
	SitesTotal       int                 `json:"sites_total"`
	SeedsDrawn       int                 `json:"seeds_drawn"`
	VariantsExecuted int                 `json:"variants_executed"`
	EmptyDraws       int                 `json:"empty_draws"`
	HarnessErrors    int                 `json:"harness_errors"`
	Baseline         RunRecord           `json:"baseline"`
	BaselineVerdict  Verdict             `json:"baseline_verdict"`
	BaselineStatus   string              `json:"baseline_status"`
	Seeds            []SeedResult        `json:"seeds"`
	Arms             *CampaignArmResults `json:"arms,omitempty"`
	// CalibrationValidated requires a confirmed planted finding and a complete
	// clean control with no findings, skips or harness errors. False means
	// that other results from this campaign are not calibrated by these arms.
	CalibrationValidated bool `json:"calibration_validated"`
	// ArtifactsDir names the retained run directory when ArtifactsKept is true.
	ArtifactsDir string `json:"artifacts_dir,omitempty"`
	// ArtifactsKept reports whether the campaign left run files on disk.
	ArtifactsKept bool `json:"artifacts_kept,omitempty"`
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
// Optional [CampaignArms] run the same sweep against planted-bug and clean
// control inputs. Without an available device or custom runner, it returns
// [cudaerrors.ErrHardwarePending].
// A failed baseline returns its report and an error before any variants run.
func RunCampaign(ctx context.Context, opts CampaignOptions) (*CampaignReport, error) {
	if opts.Arms == nil {
		return runCampaign(ctx, opts)
	}
	if opts.Arms.PlantedBugPTXSource == "" || opts.Arms.CleanPTXSource == "" {
		return nil, errors.New("cudaschedule: both planted-bug and clean-control PTX are required")
	}
	arms := *opts.Arms
	opts.Arms = nil
	report, err := runCampaign(ctx, opts)
	if err != nil || report == nil {
		return report, err
	}
	if !opts.KeepGoing && campaignWedged(report) {
		report.Arms = &CampaignArmResults{PlantedBug: unrunArm(arms.PlantedBugPTXSource, opts), CleanControl: unrunArm(arms.CleanPTXSource, opts)}
		return report, nil
	}
	plantedOpts := opts
	plantedOpts.PTXSource = arms.PlantedBugPTXSource
	planted, plantedErr := runCampaign(ctx, plantedOpts)
	cleanOpts := opts
	cleanOpts.PTXSource = arms.CleanPTXSource
	var clean *CampaignReport
	var cleanErr error
	if opts.KeepGoing || !campaignWedged(planted) {
		clean, cleanErr = runCampaign(ctx, cleanOpts)
	}
	report.Arms = &CampaignArmResults{
		PlantedBug:   summarizeArm(planted, plantedOpts, plantedErr, true),
		CleanControl: summarizeArm(clean, cleanOpts, cleanErr, false),
	}
	if !opts.KeepGoing && campaignWedged(planted) {
		report.Arms.CleanControl = unrunArm(arms.CleanPTXSource, opts)
	}
	report.CalibrationValidated = report.Arms.PlantedBug.Outcome == "detected" && report.Arms.CleanControl.Outcome == "clean"
	return report, errors.Join(plantedErr, cleanErr)
}

func campaignWedged(report *CampaignReport) bool {
	if report == nil {
		return false
	}
	for _, seed := range report.Seeds {
		if seedWedged(seed) {
			return true
		}
	}
	return false
}

func seedWedged(seed SeedResult) bool {
	if seed.Verdict == VerdictGpuWedged {
		return true
	}
	if seed.Confirmation != nil {
		for _, verdict := range seed.Confirmation.Outcomes {
			if verdict == VerdictGpuWedged {
				return true
			}
		}
	}
	return false
}

func unrunArm(source string, opts CampaignOptions) CampaignArmSummary {
	return CampaignArmSummary{SourceDigest: digestPTX(source), SeedsInRange: opts.SeedEnd - opts.SeedStart, Outcome: "not_run_after_gpu_wedge"}
}

func summarizeArm(report *CampaignReport, opts CampaignOptions, err error, planted bool) CampaignArmSummary {
	summary := CampaignArmSummary{
		SourceDigest: digestPTX(opts.PTXSource),
		SeedsInRange: opts.SeedEnd - opts.SeedStart,
		Outcome:      "baseline_not_passed",
	}
	if err != nil {
		summary.Outcome = "harness_error"
		summary.Error = err.Error()
		return summary
	}
	if report == nil {
		summary.Outcome = "harness_error"
		summary.Error = "campaign returned no report"
		return summary
	}
	summary.SeedsEvaluated = uint64(len(report.Seeds))
	summary.Findings = report.FindingCount()
	summary.VariantsExecuted = report.VariantsExecuted
	summary.HarnessErrors = report.HarnessErrors
	for _, seed := range report.Seeds {
		if seed.Verdict == VerdictSkipped {
			summary.SkippedVariants++
		}
		if seed.Verdict.IsFinding() && seed.Confirmation != nil && seed.Confirmation.Confirmed {
			summary.ConfirmedFindings++
		}
	}
	summary.Report = report
	if report.BaselineVerdict != VerdictPass {
		return summary
	}
	if summary.HarnessErrors > 0 {
		summary.Outcome = "harness_error"
		return summary
	}
	if planted {
		if summary.ConfirmedFindings > 0 {
			summary.Outcome = "detected"
		} else if summary.SeedsEvaluated < summary.SeedsInRange {
			summary.Outcome = fmt.Sprintf("not detected before stop after %d of %d seeds", summary.SeedsEvaluated, summary.SeedsInRange)
		} else {
			unit := "seeds"
			if summary.SeedsInRange == 1 {
				unit = "seed"
			}
			summary.Outcome = fmt.Sprintf("not detected within %d %s", summary.SeedsInRange, unit)
		}
		return summary
	}
	if summary.Findings != 0 {
		summary.Outcome = "unexpected findings"
	} else if summary.SeedsEvaluated < summary.SeedsInRange || summary.VariantsExecuted == 0 || summary.SkippedVariants > 0 {
		summary.Outcome = "incomplete"
	} else {
		summary.Outcome = "clean"
	}
	return summary
}

func runCampaign(ctx context.Context, opts CampaignOptions) (*CampaignReport, error) {
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

	maxSleepNs := effectiveMaxSleepNs(opts.MaxSleepNs)
	pristineSourceDigest := digestPTX(opts.PTXSource)
	report := &CampaignReport{
		SeedStart:   opts.SeedStart,
		SeedEnd:     opts.SeedEnd,
		Intensity:   opts.Intensity,
		MaxSleepNs:  maxSleepNs,
		Focus:       opts.Focus,
		ConfirmRuns: opts.ConfirmRuns,
	}

	variantRoot := ""
	var err error
	if opts.RunVariant == nil {
		variantRoot, err = campaignVariantRoot(opts.ArtifactDir)
		if err != nil {
			return report, fmt.Errorf("cudaschedule: create variant directory: %w", err)
		}
		if opts.KeepArtifacts {
			report.ArtifactsDir = variantRoot
			report.ArtifactsKept = true
		} else {
			defer os.RemoveAll(variantRoot)
		}
	}

	baselineRun, err := executeRun(ctx, opts, opts.SeedStart, opts.PTXSource, true, "baseline", timeout, variantRoot)
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
		if report.BaselineVerdict == VerdictSkipped {
			return report, nil
		}
		return report, fmt.Errorf("cudaschedule: baseline failed (%s, exit %d): harness=%q stderr=%q stdout=%q", report.BaselineVerdict, baselineRun.ExitCode, baselineRun.HarnessErr, baselineRun.Stderr, baselineRun.Stdout)
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
				PristineSourceDigest: pristineSourceDigest,
				LaunchGeometry:       opts.LaunchGeometry,
				InputReference:       opts.InputReference,
				MaxSleepNs:           maxSleepNs,
				Focus:                opts.Focus,
			})
			continue
		}
		if rewrite.Report.SitesInjected == 0 {
			recordSeed(report, SeedResult{
				Seed: seed, Status: SeedEmptyDraw, Verdict: VerdictEmptyDraw, Report: rewrite.Report,
				PristineSourceDigest: pristineSourceDigest,
				LaunchGeometry:       opts.LaunchGeometry,
				InputReference:       opts.InputReference,
				MaxSleepNs:           maxSleepNs,
				Focus:                opts.Focus,
			})
			continue
		}

		runRec, err := executeRun(ctx, opts, seed, rewrite.PTX, false, "variant-1", timeout, variantRoot)
		if err != nil {
			return report, err
		}
		if opts.CompareOutput {
			runRec.CompareOutput = true
			runRec.Baseline = &report.Baseline
		}

		if runRec.TimedOut {
			pristineRec, err := executeRun(ctx, opts, seed, opts.PTXSource, true, "health-1", timeout, variantRoot)
			if err != nil {
				return report, err
			}
			runRec.Pristine = &pristineRec
		}

		verdict := Classify(runRec)

		seedRes := SeedResult{
			Seed:                 seed,
			Verdict:              verdict,
			Run:                  &runRec,
			Report:               rewrite.Report,
			PristineSourceDigest: pristineSourceDigest,
			LaunchGeometry:       opts.LaunchGeometry,
			InputReference:       opts.InputReference,
			MaxSleepNs:           maxSleepNs,
			Focus:                opts.Focus,
		}
		if verdict == VerdictHarnessError {
			seedRes.Status = SeedHarnessError
		} else {
			seedRes.Status = SeedExecuted
		}

		if verdict.IsFinding() && verdict != VerdictGpuWedged {
			outcomes := []Verdict{verdict}
			findings := 1
			for attempt := 2; attempt <= opts.ConfirmRuns; attempt++ {
				runLabel := fmt.Sprintf("variant-confirm-%d", attempt)
				confirmRec, err := executeRun(ctx, opts, seed, rewrite.PTX, false, runLabel, timeout, variantRoot)
				if err != nil {
					return report, err
				}
				if opts.CompareOutput {
					confirmRec.CompareOutput = true
					confirmRec.Baseline = &report.Baseline
				}
				if confirmRec.TimedOut {
					pristineLabel := fmt.Sprintf("health-confirm-%d", attempt)
					pristineRec, err := executeRun(ctx, opts, seed, opts.PTXSource, true, pristineLabel, timeout, variantRoot)
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
				if confirmVerdict == VerdictGpuWedged && !opts.KeepGoing {
					break
				}
			}
			seedRes.Confirmation = &ConfirmationSummary{
				Attempts:  len(outcomes),
				Findings:  findings,
				Confirmed: findings == opts.ConfirmRuns,
				Outcomes:  outcomes,
			}
		}

		recordSeed(report, seedRes)

		if seedWedged(seedRes) && !opts.KeepGoing {
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

func executeRun(ctx context.Context, opts CampaignOptions, seed uint64, ptx string, pristine bool, runLabel string, timeout time.Duration, variantRoot string) (RunRecord, error) {
	expectedDigest := digestPTX(ptx)
	var variantPath string
	var oraclePID, oracleModules int
	var oracleAccesses uint64
	finishRecord := func(rec RunRecord) RunRecord {
		rec.ExpectedPTXDigest = expectedDigest
		rec.OracleAttached = opts.OracleAttached
		if opts.RunVariant == nil {
			rec.OraclePID, rec.OraclePatchedModules, rec.OracleInstrumentedAccesses = oraclePID, oracleModules, oracleAccesses
		}
		if rec.PTXDigest != "" && rec.PTXDigestSource == "" {
			rec.PTXDigestSource = "runner_receipt"
		}
		rec.PristineRun = pristine
		if opts.KeepArtifacts && variantPath != "" {
			rec.PTXPath = variantPath
		}
		return rec
	}
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
		return finishRecord(rec), nil
	}

	if variantRoot == "" {
		return finishRecord(RunRecord{HarnessErr: "no variant directory specified"}), nil
	}
	role := "variant"
	if pristine {
		role = "pristine"
	}
	variantPath = filepath.Join(variantRoot, role, fmt.Sprintf("seed-%d", seed), runLabel, "module.ptx")
	if err := os.MkdirAll(filepath.Dir(variantPath), 0o755); err != nil {
		return finishRecord(RunRecord{HarnessErr: err.Error()}), nil
	}
	if err := os.WriteFile(variantPath, []byte(ptx), 0o600); err != nil {
		return finishRecord(RunRecord{HarnessErr: err.Error()}), nil
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
			return finishRecord(RunRecord{HarnessErr: "build timeout"}), nil
		}
		if err != nil {
			return finishRecord(RunRecord{HarnessErr: err.Error()}), nil
		}
	}
	if executable == "" {
		return finishRecord(RunRecord{HarnessErr: "no executable specified"}), nil
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
	stdout, loadedDigest := takeLoadedPTXDigest(stdout)
	stdout, oraclePID, oracleModules, oracleAccesses = takeOracleLiveness(stdout)
	if opts.OracleAttached && (cmd.Process == nil || oraclePID != cmd.Process.Pid) {
		oraclePID = 0
	}
	var stdoutPath, stderrPath, replayScriptPath string
	if opts.KeepArtifacts {
		var artifactErr error
		stdoutPath, stderrPath, replayScriptPath, artifactErr = writeRunArtifacts(variantPath, cmd.Path, cmd.Args[1:], opts.WorkDir, stdout, stderr)
		if artifactErr != nil {
			return finishRecord(RunRecord{
				HarnessErr: fmt.Sprintf("persist campaign artifacts: %v", artifactErr),
				Stdout:     stdout,
				Stderr:     stderr,
				PTXDigest:  loadedDigest,
			}), nil
		}
	}
	withArtifacts := func(rec RunRecord) RunRecord {
		rec.StdoutPath = stdoutPath
		rec.StderrPath = stderrPath
		rec.ReplayScriptPath = replayScriptPath
		return rec
	}

	if ctx.Err() != nil {
		return RunRecord{}, ctx.Err()
	}
	if errors.Is(runCtx.Err(), context.DeadlineExceeded) {
		return finishRecord(withArtifacts(RunRecord{
			TimedOut:  true,
			Stdout:    stdout,
			Stderr:    stderr,
			PTXDigest: loadedDigest,
		})), nil
	}

	if err != nil {
		var exitErr *exec.ExitError
		if errors.As(err, &exitErr) {
			rec := RunRecord{
				Exited:    true,
				ExitCode:  exitErr.ExitCode(),
				Stdout:    stdout,
				Stderr:    stderr,
				PTXDigest: loadedDigest,
			}
			if status, ok := exitErr.Sys().(syscall.WaitStatus); ok && status.Signaled() {
				rec.Signal = status.Signal().String()
			}
			return finishRecord(withArtifacts(rec)), nil
		}
		return finishRecord(withArtifacts(RunRecord{
			Exited:     false,
			HarnessErr: err.Error(),
			Stdout:     stdout,
			Stderr:     stderr,
			PTXDigest:  loadedDigest,
		})), nil
	}

	return finishRecord(withArtifacts(RunRecord{
		Exited:    true,
		ExitCode:  0,
		Stdout:    stdout,
		Stderr:    stderr,
		PTXDigest: loadedDigest,
	})), nil
}

func writeRunArtifacts(ptxPath, executable string, args []string, workDir, stdout, stderr string) (string, string, string, error) {
	dir := filepath.Dir(ptxPath)
	stdoutPath := filepath.Join(dir, "stdout.txt")
	if err := os.WriteFile(stdoutPath, []byte(stdout), 0o600); err != nil {
		return "", "", "", err
	}
	stderrPath := filepath.Join(dir, "stderr.txt")
	if err := os.WriteFile(stderrPath, []byte(stderr), 0o600); err != nil {
		return "", "", "", err
	}
	if workDir == "" {
		var err error
		workDir, err = os.Getwd()
		if err != nil {
			return "", "", "", err
		}
	}
	workDir, err := filepath.Abs(workDir)
	if err != nil {
		return "", "", "", err
	}
	var script strings.Builder
	script.WriteString("#!/bin/sh\nset -eu\ncd ")
	script.WriteString(shellQuote(workDir))
	script.WriteByte('\n')
	script.WriteString("exec ")
	script.WriteString(shellQuote(executable))
	for _, arg := range args {
		script.WriteByte(' ')
		script.WriteString(shellQuote(arg))
	}
	script.WriteByte('\n')
	replayPath := filepath.Join(dir, "replay.sh")
	if err := os.WriteFile(replayPath, []byte(script.String()), 0o700); err != nil {
		return "", "", "", err
	}
	return stdoutPath, stderrPath, replayPath, nil
}

func shellQuote(value string) string {
	return "'" + strings.ReplaceAll(value, "'", "'\"'\"'") + "'"
}

const loadedPTXDigestPrefix = "CUDASCHEDULE_LOADED_PTX_SHA256:"

func validPTXDigest(text string) bool {
	if len(text) != sha256.Size*2 {
		return false
	}
	_, err := hex.DecodeString(text)
	return err == nil
}

func takeOracleLiveness(stdout string) (string, int, int, uint64) {
	const prefix = "CUDASCHEDULE_ORACLE_LIVENESS:"
	var clean strings.Builder
	var pid, modules int
	var accesses uint64
	seen := false
	for _, line := range strings.SplitAfter(stdout, "\n") {
		text := strings.TrimRight(line, "\r\n")
		if !strings.HasPrefix(text, prefix) {
			clean.WriteString(line)
			continue
		}
		fields := strings.Fields(strings.TrimPrefix(text, prefix))
		if seen || len(fields) != 3 {
			return stdout, 0, 0, 0
		}
		seen = true
		var err error
		if pid, err = strconv.Atoi(fields[0]); err != nil {
			return stdout, 0, 0, 0
		}
		if modules, err = strconv.Atoi(fields[1]); err != nil {
			return stdout, 0, 0, 0
		}
		if accesses, err = strconv.ParseUint(fields[2], 10, 64); err != nil {
			return stdout, 0, 0, 0
		}
	}
	return clean.String(), pid, modules, accesses
}

func takeLoadedPTXDigest(stdout string) (string, string) {
	lines := strings.SplitAfter(stdout, "\n")
	var clean strings.Builder
	var digest string
	seen := false
	for _, line := range lines {
		text := strings.TrimSuffix(line, "\n")
		text = strings.TrimSuffix(text, "\r")
		if strings.HasPrefix(text, loadedPTXDigestPrefix) {
			value := strings.TrimPrefix(text, loadedPTXDigestPrefix)
			if !seen && value != "" {
				digest = value
			} else {
				// Duplicate receipts cannot identify a single loaded input.
				return stdout, "invalid-loaded-PTX-digest"
			}
			seen = true
			continue
		}
		clean.WriteString(line)
	}
	return clean.String(), digest
}

func digestPTX(ptx string) string {
	digest := sha256.Sum256([]byte(ptx))
	return hex.EncodeToString(digest[:])
}

func effectiveMaxSleepNs(maxSleepNs uint32) uint32 {
	if maxSleepNs == 0 {
		return DefaultMaxSleepNs
	}
	return maxSleepNs
}

func campaignVariantRoot(parent string) (string, error) {
	if parent == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			return "", err
		}
		parent = filepath.Join(home, "tmp")
	}
	if err := os.MkdirAll(parent, 0o700); err != nil {
		return "", err
	}
	return os.MkdirTemp(parent, "cudaschedule-")
}

func defaultDeviceProbe() bool {
	return deviceNodeAvailable(runtime.GOOS, func(path string) error {
		_, err := os.Stat(path)
		return err
	})
}

func deviceNodeAvailable(goos string, stat func(string) error) bool {
	if goos != "linux" || stat == nil {
		return false
	}
	return stat("/dev/nvidiactl") == nil
}

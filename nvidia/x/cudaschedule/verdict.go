package cudaschedule

import (
	"bufio"
	"fmt"
	"strings"
)

// Verdict classifies the execution outcome of a PTX variant or baseline run.
type Verdict int

const (
	// VerdictPass indicates the perturbed run behaved like the baseline.
	VerdictPass Verdict = iota

	// VerdictSkipped indicates the run declined to execute (e.g. arch mismatch).
	VerdictSkipped

	// VerdictHang indicates the watchdog timer expired.
	VerdictHang

	// VerdictCrash indicates the process terminated abnormally or with a non-zero exit code.
	VerdictCrash

	// VerdictMismatch indicates an explicit correctness failure was detected in output.
	VerdictMismatch

	// VerdictOutputChanged indicates stdout differed from baseline without explicit failure (opt-in).
	VerdictOutputChanged

	// VerdictGpuWedged indicates the device stopped responding (timeout reproduced on pristine re-run).
	VerdictGpuWedged

	// VerdictHarnessError indicates the test harness failed before or during execution.
	VerdictHarnessError

	// VerdictEmptyDraw indicates the seed selected no perturbation and no run occurred.
	VerdictEmptyDraw
)

// String returns the canonical name of the verdict.
func (v Verdict) String() string {
	switch v {
	case VerdictPass:
		return "Pass"
	case VerdictSkipped:
		return "Skipped"
	case VerdictHang:
		return "Hang"
	case VerdictCrash:
		return "Crash"
	case VerdictMismatch:
		return "Mismatch"
	case VerdictOutputChanged:
		return "OutputChanged"
	case VerdictGpuWedged:
		return "GpuWedged"
	case VerdictHarnessError:
		return "HarnessError"
	case VerdictEmptyDraw:
		return "EmptyDraw"
	default:
		return fmt.Sprintf("Verdict(%d)", int(v))
	}
}

// IsFinding reports whether the verdict represents a schedule finding.
// Exactly five verdicts are findings. Pass, Skipped, HarnessError, and EmptyDraw are not.
func (v Verdict) IsFinding() bool {
	return v.FindingLabel() != ""
}

// FindingLabel returns the human-readable label for schedule findings,
// or an empty string if the verdict is not a finding.
func (v Verdict) FindingLabel() string {
	switch v {
	case VerdictMismatch:
		return "SCHEDULE-SENSITIVE CORRECTNESS FAILURE"
	case VerdictOutputChanged:
		return "SCHEDULE-SENSITIVE OUTPUT CHANGE"
	case VerdictHang:
		return "TIMEOUT CANDIDATE"
	case VerdictCrash:
		return "CRASH CANDIDATE"
	case VerdictGpuWedged:
		return "GPU WEDGE CANDIDATE"
	default:
		return ""
	}
}

// RunRecord captures the recorded outcome of executing a kernel binary.
type RunRecord struct {
	// Mismatch identifies the comparison that failed in the executing runner.
	// It is absent for older runners; consumers must not infer it from prose.
	Mismatch    *MismatchIdentity `json:"mismatch_identity,omitempty"`
	ExitCode    int               `json:"exit_code"`
	Exited      bool              `json:"exited"`
	TimedOut    bool              `json:"timed_out"`
	Signal      string            `json:"signal,omitempty"`
	PristineRun bool              `json:"pristine_run,omitempty"`
	Stdout      string            `json:"stdout"`
	Stderr      string            `json:"stderr"`
	HarnessErr  string            `json:"harness_error,omitempty"`
	// PTXDigest is the runner-reported SHA-256 digest of the loaded PTX.
	PTXDigest string `json:"ptx_sha256,omitempty"`
	// PTXDigestSource identifies the receipt as "runner_receipt".
	PTXDigestSource string `json:"ptx_digest_source,omitempty"`
	// ExpectedPTXDigest is the digest of the PTX source the campaign supplied.
	ExpectedPTXDigest string `json:"expected_ptx_sha256,omitempty"`
	// OracleAttached requires a liveness receipt from the instrumented runner.
	OracleAttached             bool       `json:"oracle_attached,omitempty"`
	OraclePID                  int        `json:"oracle_pid,omitempty"`
	OraclePatchedModules       int        `json:"oracle_patched_modules,omitempty"`
	OracleInstrumentedAccesses uint64     `json:"oracle_instrumented_accesses,omitempty"`
	PTXPath                    string     `json:"ptx_path,omitempty"`
	StdoutPath                 string     `json:"stdout_path,omitempty"`
	StderrPath                 string     `json:"stderr_path,omitempty"`
	ReplayScriptPath           string     `json:"replay_script_path,omitempty"`
	Pristine                   *RunRecord `json:"pristine,omitempty"`
	Baseline                   *RunRecord `json:"baseline,omitempty"`
	CompareOutput              bool       `json:"compare_output,omitempty"`
}

// Classify determines the verdict for a recorded run. It is a pure function
// of the recorded execution outcome with no hardware interaction.
// Missing, invalid or mismatched loaded-PTX receipts are HarnessError,
// including timeouts before a load receipt. Matching SHA-256 digests and
// PTXDigestSource "runner_receipt" are required for every executed outcome.
func Classify(run RunRecord) Verdict {
	if !validPTXDigest(run.PTXDigest) || !validPTXDigest(run.ExpectedPTXDigest) ||
		run.PTXDigest != run.ExpectedPTXDigest || run.PTXDigestSource != "runner_receipt" {
		return VerdictHarnessError
	}
	if run.OracleAttached && (run.OraclePID <= 0 || run.OraclePatchedModules <= 0 || run.OracleInstrumentedAccesses == 0) {
		return VerdictHarnessError
	}
	// 1. Harness error: variant never ran or preparation failed.
	if run.HarnessErr != "" {
		return VerdictHarnessError
	}

	// 2. Watchdog timeout: check pristine re-run if available.
	if run.TimedOut {
		if run.Pristine != nil {
			pristineVerdict := Classify(*run.Pristine)
			if pristineVerdict == VerdictHarnessError {
				return VerdictHarnessError
			}
			// Any non-pass health check, including a graceful skip, leaves the
			// device unable to establish the run's timeout cause.
			if pristineVerdict != VerdictPass {
				return VerdictGpuWedged
			}
		}
		return VerdictHang
	}

	// 3. Process crash or non-zero exit.
	if !run.Exited || run.ExitCode != 0 {
		return VerdictCrash
	}

	combined := run.Stdout
	if run.Stderr != "" {
		combined += "\n" + run.Stderr
	}

	// 4. Correctness mismatch markers take priority over skip markers.
	if HasMismatchMarker(combined) {
		return VerdictMismatch
	}

	// 5. Graceful skip marker.
	if HasSkipMarker(combined) {
		return VerdictSkipped
	}

	// 6. Opt-in output change against baseline.
	if run.CompareOutput && run.Baseline != nil {
		if Classify(*run.Baseline) != VerdictPass {
			return VerdictHarnessError
		}
		if strings.TrimSpace(run.Stdout) != strings.TrimSpace(run.Baseline.Stdout) {
			return VerdictOutputChanged
		}
	}

	return VerdictPass
}

var skipMarkers = []string{
	"skipping:",
	"pass (skipped)",
}

// HasSkipMarker reports whether output contains a recognized test skip marker.
func HasSkipMarker(output string) bool {
	scanner := bufio.NewScanner(strings.NewReader(output))
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		lower := strings.ToLower(line)
		for _, marker := range skipMarkers {
			if strings.HasPrefix(lower, marker) {
				return true
			}
		}
	}
	return false
}

var mismatchMarkers = []string{
	"mismatch",
	"max error too large",
	"barrier sync failed",
	"fail:",
	"failed:",
	"failed!",
	"not unique",
	"incorrect",
	"wrong",
	"wrong answer",
	"does not match",
	"did not match",
	"validation failed",
	"deadlock",
}

// HasMismatchMarker reports whether output contains an explicit failure marker.
func HasMismatchMarker(output string) bool {
	lower := strings.ToLower(output)
	for _, marker := range mismatchMarkers {
		if strings.Contains(lower, marker) {
			return true
		}
	}
	return false
}

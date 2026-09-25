package main

import (
	"encoding/json"
	"errors"
	"fmt"
)

// captureOutcomeTargetFailed reports a capture whose collection succeeded
// from a target that did not: it died by a signal or exited nonzero. The
// journal is finalized and its records are real, but they cover the run only
// until the target stopped, so capture_success is false. A crash under
// --finalize=exit-group used to read as outcome collected with
// capture_success true, and nothing downstream looked at the exit status.
const captureOutcomeTargetFailed = "target-failed"

// markTargetFailed turns a successful capture of a failed target into
// captureOutcomeTargetFailed. A capture that already failed keeps its own
// outcome, which names what went wrong first, and a cancelled one is left
// alone: the controller killed that target itself.
func markTargetFailed(result captureResult) captureResult {
	if !result.CaptureSuccess || result.Cancelled {
		return result
	}
	if result.TargetSignal == "" && result.TargetExitCode == 0 {
		return result
	}
	result.Outcome, result.CaptureSuccess = captureOutcomeTargetFailed, false
	result.CaptureError = targetFailureMessage(result)
	return result
}

// memsetsWarning is printed before a --memsets capture starts, because the
// fault it describes kills the target. The target's use of CUDA graphs is not
// known until it runs.
//
// On GB10 (driver 580.173.02, CUPTI 13.0 through 13.4) an MLX workload faulted
// inside libcuda whenever memset activity was on at the relaunch of a cached
// graph exec after cuGraphExecUpdate, and never with the kind off.
const memsetsWarning = "--memsets: memset collection has crashed CUDA-graph workloads inside libcuda " +
	"(GB10, driver 580, CUPTI 13.0-13.4); if the target launches CUDA graphs, capture without --memsets"

// targetFailureMessage says how the target ended. When the agent collected
// memsets, which it does only for --memsets, it adds the one lever known to
// matter: with memset activity on, cuGraphLaunch has faulted inside libcuda
// for a CUDA-graph workload. A Go
// program reports such a fault as exit status 2, not as a signal, so that
// status gets the hint too.
func targetFailureMessage(result captureResult) string {
	const hint = "; if it uses CUDA graphs, try --memsets=false"
	var c struct {
		Memsets bool `json:"memsets"`
	}
	_ = json.Unmarshal(result.Capability, &c)
	if result.TargetSignal != "" {
		name := signalName(result.TargetSignal)
		msg := "target died by " + name
		if c.Memsets && name == "SIGSEGV" {
			msg += hint
		}
		return msg
	}
	msg := fmt.Sprintf("target exited with status %d", result.TargetExitCode)
	isGo := result.Target != nil && result.Target.Executable != nil && result.Target.Executable.Go != nil
	if c.Memsets && isGo && result.TargetExitCode == 2 {
		msg += ", which is also how a Go program reports a fatal signal such as SIGSEGV" + hint
	}
	return msg
}

// targetFailure reports a capture whose target did not finish. It reads the
// target's own status rather than the outcome, so it also refuses captures
// written before captureOutcomeTargetFailed existed.
func targetFailure(status *captureResult) error {
	if status.TargetSignal == "" && status.TargetExitCode == 0 {
		return nil
	}
	return errors.New(targetFailureMessage(*status) + ": its records cover the run only until it stopped")
}

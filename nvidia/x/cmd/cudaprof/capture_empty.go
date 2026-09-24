package main

import "fmt"

// captureOutcomeGoExit reports a Go target whose journal never finalized and
// holds no activity records because the target ran without
// --finalize=profiler-stop. It is split out of not-finalized because the
// cause is known and so is the remedy: the Go runtime exits through
// exit_group, which runs none of the agent's exit-time flushes, and CUPTI's
// buffers die with the process while the target exits 0.
const captureOutcomeGoExit = "go-exit-not-finalized"

// refuseEmpty withdraws success from a capture that holds no activity
// records. A capture of nothing is not a profile, whatever the producer said
// about why it is empty: a zero-record artifact reads exactly like a workload
// that ran no such kernels, and the two are opposite findings. The outcome,
// which already names why the capture is empty, is left as it is.
func refuseEmpty(result captureResult) captureResult {
	if !result.CaptureSuccess || result.ActivityRecords != 0 {
		return result
	}
	result.CaptureSuccess = false
	if result.CaptureError == "" {
		why := "outcome " + result.Outcome
		if result.Outcome == captureOutcomeObserved {
			why = "the agent finalized and reports that the target never used CUDA"
		}
		result.CaptureError = "the capture holds no activity records: " + why
	}
	return result
}

// goExitRisk reports whether id is a Go binary that will run under an agent
// finalizing on exit. finalize is the CUDAPROF_FINALIZE the target receives.
//
// The build information is what identifyExecutable already read from the
// binary with debug/buildinfo, so a stripped Go binary is still detected; a
// Go program started by a shell script or other non-Go wrapper is not.
func goExitRisk(id *executableIdentity, finalize string) bool {
	return id != nil && id.Go != nil && finalize != finalizeProfilerStop
}

// goExitWarning is printed before a Go target runs without
// --finalize=profiler-stop.
//
// It is a warning, not a refusal, because the launcher cannot see the three
// ways such a run still collects: the target calls cudaprofagent.Finalize
// before it exits, the Go target is a launcher whose CUDA runs in non-Go
// children that exit through libc, or --flush-interval with
// --recover-unfinalized admits what was flushed as a partial capture. A run
// that falls in none of them ends with an unfinalized target journal and no
// activity, which observeSession reports as captureOutcomeGoExit.
func goExitWarning(id *executableIdentity) string {
	return fmt.Sprintf("target %s is a Go binary (%s) and --finalize=%s is not set: "+
		"a Go program exits through exit_group, which runs none of the agent's exit-time flushes, "+
		"so unless it calls cudaprofagent.Finalize() before exiting the capture will hold no records. "+
		"Call driver.ProfilerStop() before the target exits and pass --finalize=%s",
		id.Path, id.Go.GoVersion, finalizeProfilerStop, finalizeProfilerStop)
}

// goExitDiagnosis is the capture error of captureOutcomeGoExit: a Go target
// whose agent loaded and then wrote neither an activity record nor a
// finalized one.
func goExitDiagnosis(id *executableIdentity) string {
	return fmt.Sprintf("target %s is a Go binary (%s) and ran without --finalize=%s: "+
		"the agent loaded, then the Go runtime exited through exit_group, which runs none of the agent's exit-time flushes, "+
		"so CUPTI's buffers were discarded and the capture holds no activity records. "+
		"Call driver.ProfilerStop() (github.com/tmc/lib/nvidia/driver) before the target exits and re-run with --finalize=%s, "+
		"or call cudaprofagent.Finalize() (github.com/tmc/lib/nvidia/x/cudaprofagent) before it exits",
		id.Path, id.Go.GoVersion, finalizeProfilerStop, finalizeProfilerStop)
}

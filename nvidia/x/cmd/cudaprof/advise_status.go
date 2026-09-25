package main

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/internal/capturefiles"
)

// preflightCapture refuses an artifact that records its own failure. A failed
// capture still writes a well-formed pprof -- a couple of kilobytes with zero
// activity records -- and pprof renders it as an empty but valid profile and
// exits 0. Nothing on that path says the numbers are missing rather than
// small, which is the mistake the artifact invites. Reading it fails closed;
// the status is still reachable through advise, which reports the failure as a
// finding rather than pretending to measure.
func preflightCapture(cap *cudacapture.Capture) error {
	reason, failed := capturefiles.Failure(cap)
	if !failed {
		return nil
	}
	return fmt.Errorf("this capture records its own failure: %s\n"+
		"\tIt carries no measurement, and an empty profile is not a fast one.\n"+
		"\tUse 'cudaprof advise' to see the failure as a finding, or re-run the capture.", reason)
}

func auditCaptureStatus(cap *cudacapture.Capture, report *AdviceReport) {
	if cap == nil {
		return
	}
	status := capturefiles.Status(cap)
	if status == nil {
		return
	}
	if errMsg, failed := capturefiles.Failure(cap); failed {
		report.Items = append(report.Items, AdviceItem{
			Severity:                 SeverityCritical,
			Category:                 CategoryArchitecture,
			EvidenceLevel:            EvidenceObserved,
			Finding:                  fmt.Sprintf("Incomplete or failed profile capture: %s", errMsg),
			ActionableRecommendation: "Ensure target workload executes cleanly and does not exceed journal record capacity before analyzing performance.",
			NextMeasurement:          "Re-run capture with increased buffer size or filtered kernel scope to confirm clean execution",
		})
	}
	if dropped, ok := status["dropped_records"].(float64); ok && uint64(dropped) > 0 {
		report.Items = append(report.Items, AdviceItem{
			Severity:                 SeverityWarning,
			Category:                 CategoryArchitecture,
			EvidenceLevel:            EvidenceObserved,
			Finding:                  fmt.Sprintf("Lossy capture: %d records dropped due to buffer pressure", uint64(dropped)),
			ActionableRecommendation: "Whole-run totals and aggregate launch counts are incomplete (a lower bound). Increase buffer pool or reduce profiling overhead to capture a complete trace.",
			NextMeasurement:          "Increase trace buffer pool capacity and re-verify zero dropped records",
		})
	}
}

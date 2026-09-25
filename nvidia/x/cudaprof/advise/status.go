package advise

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/internal/capturefiles"
)

func auditCaptureStatus(cap *cudacapture.Capture, report *Report) {
	if cap == nil {
		return
	}
	status := capturefiles.Status(cap)
	if status == nil {
		return
	}
	if errMsg, failed := capturefiles.Failure(cap); failed {
		report.Items = append(report.Items, Item{
			Severity:                 SeverityReadFirst,
			Category:                 CategoryArchitecture,
			EvidenceLevel:            EvidenceObserved,
			Finding:                  fmt.Sprintf("Incomplete or failed profile capture: %s", errMsg),
			ActionableRecommendation: "Ensure target workload executes cleanly and does not exceed journal record capacity before analyzing performance.",
			NextMeasurement:          "Re-run capture with increased buffer size or filtered kernel scope to confirm clean execution",
		})
	}
	if dropped, ok := status["dropped_records"].(float64); ok && uint64(dropped) > 0 {
		report.Items = append(report.Items, Item{
			Severity:                 SeverityCost,
			Category:                 CategoryArchitecture,
			EvidenceLevel:            EvidenceObserved,
			Finding:                  fmt.Sprintf("Lossy capture: %d records dropped due to buffer pressure", uint64(dropped)),
			ActionableRecommendation: "Whole-run totals and aggregate launch counts are incomplete (a lower bound). Increase buffer pool or reduce profiling overhead to capture a complete trace.",
			NextMeasurement:          "Increase trace buffer pool capacity and re-verify zero dropped records",
		})
	}
}

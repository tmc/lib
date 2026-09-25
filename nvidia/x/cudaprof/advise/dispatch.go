package advise

import (
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaprof/timeline"
)

// auditHostDispatchOverhead compares a complete host API dispatch span with
// the corresponding device kernel work. It requires both correlation-matched
// API timestamps and device activity timestamps; a launch count or a summed
// kernel duration alone cannot establish host overhead.
func auditHostDispatchOverhead(cap *cudacapture.Capture, report *Report) {
	if cap == nil {
		return
	}
	apis := timeline.APIRecords(cap)
	if len(apis) == 0 {
		return
	}
	events, _ := timeline.Events(cap)
	var hostStart, hostEnd uint64
	var busy []timeline.Interval
	seen := make(map[uint32]bool)
	measured := 0
	for _, stream := range events {
		for _, ev := range stream {
			if strings.ToLower(ev.Subtype) != "kernel" && ev.Subtype != "" {
				continue
			}
			if ev.Correlation == 0 || seen[ev.Correlation] || ev.Start == 0 || ev.End <= ev.Start {
				continue
			}
			api, ok := apis[ev.Correlation]
			if !ok || api.Start == 0 || api.End < api.Start {
				continue
			}
			seen[ev.Correlation] = true
			measured++
			busy = append(busy, timeline.Interval{Start: ev.Start, End: ev.End})
			if hostStart == 0 || api.Start < hostStart {
				hostStart = api.Start
			}
			if api.End > hostEnd {
				hostEnd = api.End
			}
		}
	}
	// The device time here is the union of the kernel intervals, not their sum.
	// Summing counts concurrent kernels twice, and the excess is indistinguishable
	// from host overhead in the subtraction below: two streams each busy for the
	// whole span would report the span's worth of overhead on a device that was
	// never idle.
	deviceNanos := timeline.Union(busy)
	if measured == 0 || hostEnd <= hostStart || hostEnd-hostStart <= deviceNanos {
		return
	}
	span := hostEnd - hostStart
	overhead := span - deviceNanos
	if float64(overhead)/float64(span) <= 0.15 {
		return
	}
	// span - deviceNanos is a quantity, not a cause: it counts every nanosecond
	// the device was not running a matched kernel, whether that was a tax paid on
	// each launch or one stall paid once. The two have opposite remedies, so the
	// gap distribution has to decide which this is before the finding recommends
	// anything. The gaps analysis parses the capture journals, so it is consulted
	// only here, once the share is already large enough to report.
	attr, ok := captureIdleAttribution(cap)
	if ok {
		if _, dominant := attr.dominantGap(); dominant {
			// A single gap holds this idle. reportDominantIdleGap already says so,
			// and says what to do about it; repeating it here as dispatch overhead
			// would name a per-launch cause for a cost that was not paid per launch.
			return
		}
	}
	// Batching and CUDA graphs divide a submission cost, so recommend them only
	// where the capture measured one. Absent that measurement the span is still
	// worth reporting, but naming a cause for it would be a guess: the same
	// number is produced by a device that waits on the host every launch and by
	// one that waited once.
	rec := "Establish where this span went before restructuring dispatch: compare it against cudaprof gaps to find which gaps hold the time, then batch launches or capture the loop in a CUDA graph only if the idle is spread across them."
	if share, measured := dispatchSubmissionIdleShare(cap); measured && share >= dominanceSubmissionMinIdleShare {
		rec = fmt.Sprintf("Batch fine-grained launches or capture the dispatch loop in a CUDA Graph and compare end-to-end wall-clock time: the capture attributes %.1f%% of its idle to driver submission, which is the part batching divides. The measured gap is host/driver span, not device kernel time.", share)
	}
	report.Items = append(report.Items, Item{
		Severity:      SeverityReadFirst,
		Category:      CategoryArchitectureDispatch,
		EvidenceLevel: EvidenceObserved,
		Finding: fmt.Sprintf("Host dispatch overhead accounts for %.1f%% of the correlation-matched dispatch span (%d launches, %.3f ms host span, %.3f ms GPU execution)",
			100*float64(overhead)/float64(span), measured, float64(span)/1e6, float64(deviceNanos)/1e6),
		ActionableRecommendation: rec,
		NextMeasurement:          "repeat with CUDA Graph capture or batched launches and retain correlation-matched host/API and device timestamps",
	})
}

package cudadiag

import (
	"fmt"
	"sort"

	"github.com/tmc/lib/nvidia/x/cudaprof/tracecontract"
)

// Observe reports independent timing observations. A missing host return blocks
// that call's duration, not kernel timing. Device busy time requires bounded
// activity loss; scheduler claims remain subject to scheduler admission in Diagnose.
// None of these observations establishes a cause or an operation correlation.
func Observe(in DiagnosticInput) []tracecontract.Finding {
	out := make([]tracecontract.Finding, 0, len(in.HostCalls)+1)
	calls := append([]HostCallEvent(nil), in.HostCalls...)
	sort.Slice(calls, func(i, j int) bool {
		if calls[i].Span.Start != calls[j].Span.Start {
			return calls[i].Span.Start < calls[j].Span.Start
		}
		if calls[i].Span.End != calls[j].Span.End {
			return calls[i].Span.End < calls[j].Span.End
		}
		return calls[i].Name < calls[j].Name
	})
	for i, call := range calls {
		value := float64(call.Span.Duration())
		reqs := []tracecontract.Requirement{{Name: "matched_return", RequireReturn: true}, {Name: "host_clock", ClockDomain: call.Span.Clock}}
		finding := tracecontract.Admit(fmt.Sprintf("host-call-%d", i), "host_api_duration", tracecontract.FindingObserved, "matched host API entry-to-return duration; not GPU execution time", &value, "ns", reqs, tracecontract.EvidenceContext{Receipt: in.Receipt, HasReturn: call.HasReturn, Intervals: []tracecontract.Interval{call.Span}, Artifacts: []string{in.Receipt.InputSHA256}})
		if call.Span.Clock == "" || call.Span.Start < 0 {
			finding.Class = tracecontract.FindingUnavailable
			finding.Value = nil
			finding.Admission = tracecontract.Admission{Status: "withheld", ReasonCode: tracecontract.ReasonEvidenceUnavailable, Reason: "call clock or timestamp unavailable"}
		}
		out = append(out, finding)
	}
	intervals := make([]tracecontract.Interval, 0, len(in.Kernels))
	for _, kernel := range in.Kernels {
		intervals = append(intervals, kernel.Span)
	}
	value := float64(UnionDuration(intervals).Nanoseconds())
	clock := in.Receipt.ClockDomain
	if len(intervals) > 0 {
		clock = intervals[0].Clock
	}
	reqs := []tracecontract.Requirement{{Name: "activity_loss", Family: "activity"}, {Name: "device_clock", ClockDomain: clock}, {Name: "activity_probe", Probe: "cupti_activity", MinProbeCoverage: 1}}
	ctx := tracecontract.EvidenceContext{Receipt: in.Receipt, Intervals: intervals, Artifacts: []string{in.Receipt.InputSHA256}}
	f := tracecontract.Admit("device-busy", "device_busy_duration", tracecontract.FindingObserved, "interval-union duration of captured kernels; not a contention diagnosis", &value, "ns", reqs, ctx)
	if len(intervals) == 0 || clock == "" {
		f.Class = tracecontract.FindingUnavailable
		f.Value = nil
		f.Admission = tracecontract.Admission{Status: "withheld", ReasonCode: tracecontract.ReasonEvidenceUnavailable, Reason: "no timed device activity"}
	}
	out = append(out, f)
	return out
}

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
	for i, p := range in.PreEntries {
		value := float64(p.Span.Duration())
		reqs := []tracecontract.Requirement{{Name: "pre_entry_loss", Family: "pre_entry"}, {Name: "pre_entry_clock", ClockDomain: p.Span.Clock}}
		f := tracecontract.Admit(fmt.Sprintf("pre-entry-%d", i), "host_pre_entry_duration", tracecontract.FindingObserved, "application submission-ready to API entry; not an API duration or scheduler state", &value, "ns", reqs, tracecontract.EvidenceContext{Receipt: in.Receipt, Intervals: []tracecontract.Interval{p.Span}, Artifacts: []string{in.Receipt.InputSHA256}})
		if !p.Complete || p.Span.Start < 0 || p.Span.End < p.Span.Start || p.Span.Clock == "" {
			f.Class, f.Value = tracecontract.FindingUnavailable, nil
			f.Admission = tracecontract.Admission{Status: "withheld", ReasonCode: tracecontract.ReasonEvidenceUnavailable, Reason: "pre-entry endpoints incomplete"}
		} else if p.SourceClass != "native" && f.Admission.Admitted {
			f.Class = tracecontract.FindingAssociated
		}
		out = append(out, f)
	}
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
		if len(in.PreEntries) != 0 && in.PreEntries[0].SourceClass != "native" && finding.Admission.Admitted {
			finding.Class = tracecontract.FindingAssociated
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
	probe := "cupti_activity"
	if len(in.PreEntries) != 0 && in.PreEntries[0].SourceClass != "native" {
		probe = "synthetic_activity"
	}
	reqs := []tracecontract.Requirement{{Name: "activity_loss", Family: "activity"}, {Name: "device_clock", ClockDomain: clock}, {Name: "activity_probe", Probe: probe, MinProbeCoverage: 1}}
	ctx := tracecontract.EvidenceContext{Receipt: in.Receipt, Intervals: intervals, Artifacts: []string{in.Receipt.InputSHA256}}
	f := tracecontract.Admit("device-busy", "device_busy_duration", tracecontract.FindingObserved, "interval-union duration of captured kernels; not a contention diagnosis", &value, "ns", reqs, ctx)
	if len(intervals) == 0 || clock == "" {
		f.Class = tracecontract.FindingUnavailable
		f.Value = nil
		f.Admission = tracecontract.Admission{Status: "withheld", ReasonCode: tracecontract.ReasonEvidenceUnavailable, Reason: "no timed device activity"}
	}
	if len(in.PreEntries) != 0 && in.PreEntries[0].SourceClass != "native" && f.Admission.Admitted {
		f.Class = tracecontract.FindingAssociated
	}
	out = append(out, f)
	return out
}

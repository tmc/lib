//go:build darwin && arm64

package main

import (
	"sort"
	"testing"
)

// Span statistics recovered from a single run.
//
// A workload run records one metalRegion per command buffer: 86 per token for
// prefill, so a 32-token chunk yields 2752 spans in roughly seven seconds. That
// is the sample the benchmarks have been discarding. Averaging the spans into
// one rate per run leaves six samples to reason from, all of them wall-clock
// rates with double-digit variance; the spans themselves describe the structure
// of a single run at a sample size three orders of magnitude larger, and the
// structure is what the command-buffer work changes.
//
// Union, not sum. Command buffers overlap in wall time -- that is the point of
// commit ed187ec, which commits a layer's opening buffer without waiting so the
// closing buffer can be encoded against it. Summing span durations counts the
// overlapped interval twice, so a sum is an upper bound on GPU busy time and
// never an estimate of it. Every occupancy figure here comes from the union of
// the intervals. The difference between the two, reported as overlap, is the
// wall time the overlap actually buys.

// spanStats summarizes the command-buffer spans of one run.
type spanStats struct {
	Count   int     // spans recorded
	Sum     float64 // total span duration, overlap counted once per span
	Union   float64 // wall time covered by at least one span
	Overlap float64 // Sum - Union: time covered by more than one span
	Idle    float64 // time inside the run's extent covered by no span
	Extent  float64 // last end minus first start
}

// analyzeSpans computes the union, overlap and idle time of spans. It reports
// the zero value for an empty input.
func analyzeSpans(spans []metalRegion) spanStats {
	if len(spans) == 0 {
		return spanStats{}
	}
	sorted := make([]metalRegion, len(spans))
	copy(sorted, spans)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i].Start < sorted[j].Start })

	st := spanStats{Count: len(spans)}
	for _, s := range spans {
		st.Sum += s.Seconds()
	}

	// Sweep the sorted spans, merging each into the open interval when it
	// starts before that interval ends.
	start, end := sorted[0].Start, sorted[0].End
	for _, s := range sorted[1:] {
		if s.Start > end {
			st.Union += end - start
			st.Idle += s.Start - end
			start, end = s.Start, s.End
			continue
		}
		if s.End > end {
			end = s.End
		}
	}
	st.Union += end - start
	st.Overlap = st.Sum - st.Union
	st.Extent = end - sorted[0].Start
	return st
}

// spansByRegion groups spans by region label, preserving nothing but membership.
func spansByRegion(spans []metalRegion) map[string][]metalRegion {
	by := make(map[string][]metalRegion)
	for _, s := range spans {
		by[s.Region] = append(by[s.Region], s)
	}
	return by
}

// BenchmarkMetalSpanProbe is the cheap structural measurement.
//
// Cost. The registered prefill baseline took 1274.8s: eighteen samples across
// chunk 32/128/512, of which the model load was about 32s. Compute dominates,
// so amortizing the load buys nothing; repetition and chunk size are the cost.
// This probe runs one 32-token chunk once, for roughly 39s including the load
// -- about 33x cheaper than the baseline.
//
// What that buys, and what it does not. Across all eighteen baseline samples
// cbufs/tok read 86.00 with zero variance, at every chunk size, so a single
// sample of the cheapest chunk carries the same structural verdict as the full
// sweep. Over the same samples %gpu ranged 42.78 to 88.54 and gpu_ms/tok 128.9
// to 179.3. Wall-clock rates therefore need the long run and a quiet host;
// counts and ratios do not. This probe reports only the second kind.
//
// The sample size that matters is inside the run, not across runs. One 32-token
// chunk records 32*86 = 2752 spans, each with a start, an end, a layer and a
// region. The baseline reduced all of that to one mean per run and compared six
// such means. The metrics below describe the structure of a single run at a
// sample size three orders of magnitude larger.
func BenchmarkMetalSpanProbe(b *testing.B) {
	r := benchWorkloadRuntime(b)
	const chunk = 32
	tokens := benchWorkloadTokens(chunk)

	for b.Loop() {
		b.StopTimer()
		if err := r.Reset(); err != nil {
			b.Fatalf("reset: %v", err)
		}
		log := &metalRegionLog{}
		r.ops.regions = log
		b.StartTimer()

		if err := r.EvalBatch(tokens, 0); err != nil {
			b.Fatalf("prefill %d tokens: %v", chunk, err)
		}

		b.StopTimer()
		r.ops.regions = nil
		reportSpanStructure(b, chunk, log.all())
		b.StartTimer()
	}
}

// reportSpanStructure reports the contention-independent structure of one run.
// Every metric is a count or a ratio of intervals measured against each other,
// so none of them is a wall-clock rate against a shared host.
func reportSpanStructure(b *testing.B, tokens int, spans []metalRegion) {
	b.Helper()
	if len(spans) == 0 {
		b.Fatal("no command buffer spans recorded")
	}
	st := analyzeSpans(spans)

	b.ReportMetric(float64(st.Count)/float64(tokens), "cbufs/tok")
	// Occupancy from the union. A sum double-counts buffers that are in
	// flight together and can exceed the extent outright; see
	// TestAnalyzeSpansSumIsUpperBound.
	b.ReportMetric(st.Union/st.Extent*100, "%busy")
	// How much wall time the overlap of ed187ec actually recovers, as a
	// fraction of the summed span time it removes.
	b.ReportMetric(st.Overlap/st.Sum*100, "%overlap")
	// The bubble Phase 1 and Phase 2 exist to remove.
	b.ReportMetric(st.Idle/st.Extent*100, "%idle")

	for region, rs := range spansByRegion(spans) {
		b.ReportMetric(float64(len(rs))/float64(tokens), region+"/tok")
	}
}

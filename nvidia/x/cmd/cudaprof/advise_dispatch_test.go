package main

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/x/cudacapture"
)

func TestAdvice_HostDispatchOverhead(t *testing.T) {
	cap := &cudacapture.Capture{
		Files: map[string][]byte{"capture-status.json": apiRecordsFile(t, map[int]uint64{1: 100, 2: 100100})},
		Profile: &profile.Profile{
			SampleType: []*profile.ValueType{{Type: "kernel_time", Unit: "nanoseconds"}},
			Sample: []*profile.Sample{
				microKernelSample("gather", 1, 100, 200, 1),
				microKernelSample("gather", 1, 100100, 100200, 2),
			},
		},
	}
	report, err := AnalyzeCapture(cap)
	if err != nil {
		t.Fatal(err)
	}
	for _, item := range report.Items {
		if strings.Contains(item.Finding, "Host dispatch overhead accounts for") {
			if item.EvidenceLevel != EvidenceObserved || item.Severity != SeverityCost {
				t.Fatalf("dispatch finding = %+v", item)
			}
			return
		}
	}
	t.Fatal("missing host dispatch overhead finding")
}

// dispatchSpanCapture renders one device timeline into both views advise reads
// it through: the journal the gaps analysis parses, and the profile samples
// plus host API records the dispatch-overhead audit correlates. A fixture that
// wrote only one of them would let the two disagree about the same capture,
// which is the condition the audit under test is supposed to resolve.
//
// The timeline matches concentratedIdleJournal: n kernels of kernelNs, with
// stallNs after the first and smallGapNs after the rest.
//
// submission places each launch call so that it is still open when the device
// goes idle, which is the driver submission arm -- the only idle batching
// recovers. Otherwise the call closes before its kernel starts, so the host was
// already ahead and the idle is something else.
func dispatchSpanCapture(t *testing.T, n int, kernelNs, smallGapNs, stallNs uint64, withJournal, submission bool) *cudacapture.Capture {
	t.Helper()
	b := newJournalBuilder()
	samples := make([]*profile.Sample, 0, n)
	apis := make([]map[string]any, 0, n)
	at := uint64(10_000)
	for i := 1; i <= n; i++ {
		start, end := at, at+kernelNs
		apiStart, apiEnd := start-400, start-100
		if submission && i > 1 {
			apiStart, apiEnd = start-smallGapNs-500, start-1_000
		}
		b.kernel(start, end, uint32(i))
		b.api(apiStart, apiEnd, uint32(i))
		samples = append(samples, microKernelSample("gemm", 1, start, end, i))
		apis = append(apis, map[string]any{"domain": "driver", "correlation": i, "start": apiStart, "end": apiEnd})
		if i == 1 {
			at = end + stallNs
		} else {
			at = end + smallGapNs
		}
	}
	status, err := json.Marshal(map[string]any{"status": completeStatus(), "api_records": apis})
	if err != nil {
		t.Fatalf("marshal status: %v", err)
	}
	files := map[string][]byte{"capture-status.json": status}
	if withJournal {
		files["capture.journal"] = b.finalize()
	}
	return &cudacapture.Capture{
		Files: files,
		Profile: &profile.Profile{
			SampleType: []*profile.ValueType{{Type: "kernel_time", Unit: "nanoseconds"}},
			Sample:     samples,
		},
	}
}

// TestAdvice_DispatchOverheadIsNotASingleStall holds the host dispatch overhead
// finding to the gap distribution.
//
// The finding computes span - deviceTime and calls the difference dispatch
// overhead, then recommends batching or a CUDA graph. That subtraction cannot
// tell a tax paid on every launch from one stall paid once, and on a real
// capture it ranked a 63.81 ms cold start as 57.5% "dispatch overhead" while
// the idle findings in the same report said the per-launch component was 0.5%
// of it. Batching divides the first and recovers nothing of the second.
//
// The middle arm is the control: the same profile with no journal still fires,
// so the suppression in the first arm is the gap data doing work rather than
// the fixture failing to reach the finding at all.
func TestAdvice_DispatchOverheadIsNotASingleStall(t *testing.T) {
	const finding = "Host dispatch overhead accounts for"
	tests := []struct {
		name         string
		smallGapNs   uint64
		stallNs      uint64
		withJournal  bool
		submission   bool
		wantFinding  bool
		wantBatching bool
	}{
		{name: "one stall, gaps measured", smallGapNs: 1_000, stallNs: 500_000, withJournal: true, wantFinding: false},
		{name: "one stall, no gap data", smallGapNs: 1_000, stallNs: 500_000, withJournal: false, wantFinding: true, wantBatching: false},
		// Spread idle, but the host was already ahead at every gap: there is no
		// submission cost, so the span is real and batching still is not the
		// remedy for it.
		{name: "spread idle, host ahead", smallGapNs: 10_000, stallNs: 10_000, withJournal: true, wantFinding: true, wantBatching: false},
		// Spread idle actually spent inside the launch call: the one shape that
		// earns the batching recommendation.
		{name: "spread idle, in submission", smallGapNs: 10_000, stallNs: 10_000, withJournal: true, submission: true, wantFinding: true, wantBatching: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cap := dispatchSpanCapture(t, 12, 1_000, tt.smallGapNs, tt.stallNs, tt.withJournal, tt.submission)
			report, err := AnalyzeCapture(cap)
			if err != nil {
				t.Fatal(err)
			}
			var got *AdviceItem
			for i, item := range report.Items {
				if strings.Contains(item.Finding, finding) {
					got = &report.Items[i]
					break
				}
			}
			if (got != nil) != tt.wantFinding {
				t.Fatalf("dispatch overhead finding present = %v, want %v; items: %+v", got != nil, tt.wantFinding, report.Items)
			}
			if got == nil {
				return
			}
			// Only measured submission idle earns the batching remedy.
			mentionsBatching := strings.Contains(got.ActionableRecommendation, "Batch fine-grained launches")
			if mentionsBatching != tt.wantBatching {
				t.Errorf("recommendation names batching = %v, want %v: %s", mentionsBatching, tt.wantBatching, got.ActionableRecommendation)
			}
		})
	}
}

func TestUnionNanos(t *testing.T) {
	tests := []struct {
		name string
		in   []interval
		want uint64
	}{
		{name: "empty", in: nil, want: 0},
		{name: "single", in: []interval{{10, 20}}, want: 10},
		{name: "disjoint", in: []interval{{10, 20}, {30, 45}}, want: 25},
		{name: "unsorted disjoint", in: []interval{{30, 45}, {10, 20}}, want: 25},
		{name: "touching", in: []interval{{10, 20}, {20, 30}}, want: 20},
		// Concurrent streams: summing would report 20 and charge the excess to
		// the host as dispatch overhead on a device that was never idle.
		{name: "identical overlap", in: []interval{{10, 20}, {10, 20}}, want: 10},
		{name: "partial overlap", in: []interval{{10, 20}, {15, 30}}, want: 20},
		{name: "nested", in: []interval{{10, 50}, {20, 30}}, want: 40},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := unionNanos(tt.in); got != tt.want {
				t.Errorf("unionNanos(%v) = %d, want %d", tt.in, got, tt.want)
			}
		})
	}
}

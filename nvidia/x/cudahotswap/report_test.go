package cudahotswap

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestMedian(t *testing.T) {
	for _, tt := range []struct {
		name string
		in   []int64
		want float64
	}{
		{"empty", nil, 0},
		{"one", []int64{5}, 5},
		{"odd", []int64{9, 1, 5}, 5},
		{"even", []int64{4, 1, 3, 2}, 2.5},
		// One outlier must move the median far less than it moves a mean. The
		// mean here is 2504.5; the median is 3.
		{"outlier", []int64{2, 3, 4, 10009}, 3.5},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if got := median(tt.in); got != tt.want {
				t.Errorf("median(%v) = %v, want %v", tt.in, got, tt.want)
			}
		})
	}
}

func TestMedianDoesNotReorderItsInput(t *testing.T) {
	in := []int64{3, 1, 2}
	median(in)
	if in[0] != 3 || in[1] != 1 || in[2] != 2 {
		t.Fatalf("median sorted its caller's slice: %v", in)
	}
}

// A sample the device clock did not resolve must suppress every rate, not be
// averaged away with the rest. A zero nanosecond span is not a fast kernel.
func TestSummarizeRefusesUnresolvedSamples(t *testing.T) {
	tr := &TimingReport{LaunchesPerSample: 4, SampleNanos: []int64{800, 0, 820}}
	tr.Summarize()
	if tr.Resolved {
		t.Fatal("a zero-length sample was accepted")
	}
	if tr.MedianNanosPerLaunch != 0 {
		t.Fatalf("a rate was derived anyway: %v", tr.MedianNanosPerLaunch)
	}
}

func TestSummarize(t *testing.T) {
	tr := &TimingReport{LaunchesPerSample: 4, SampleNanos: []int64{800, 1200, 1000}}
	tr.Summarize()
	if !tr.Resolved {
		t.Fatal("resolvable samples were rejected")
	}
	if tr.MedianNanosPerLaunch != 250 || tr.MinNanosPerLaunch != 200 || tr.MaxNanosPerLaunch != 300 {
		t.Fatalf("median %v min %v max %v", tr.MedianNanosPerLaunch, tr.MinNanosPerLaunch, tr.MaxNanosPerLaunch)
	}
}

// Every report carries the unmeasured list. A consumer that finds an absent
// field must be able to see that its absence was deliberate.
func TestReportCarriesUnmeasured(t *testing.T) {
	rep := &Report{Task: "axpby", Outcome: OutcomeCompileError, Unmeasured: []string{"occupancy: not queried"}}
	b, err := json.Marshal(rep)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(b), `"unmeasured"`) {
		t.Fatal("the report omitted the unmeasured list")
	}
	if len(rep.Unmeasured) == 0 {
		t.Fatal("the unmeasured list is empty")
	}
	if !strings.Contains(rep.Text(), "unmeasured:") {
		t.Fatal("the text rendering omitted the unmeasured list")
	}
}

// The text rendering must say when the context is poisoned and when the error
// was not recognized; both change what the caller may do next.
func TestTextWarnsOnPoisonAndUnrecognized(t *testing.T) {
	rep := &Report{Outcome: OutcomeDeviceFault, ContextPoisoned: true, Unmeasured: []string{"occupancy: not queried"}}
	if !strings.Contains(rep.Text(), "poisoned") {
		t.Error("a poisoned context was not reported")
	}
	rep = &Report{Outcome: OutcomeHarnessError, Recognized: false, Unmeasured: []string{"occupancy: not queried"}}
	if !strings.Contains(rep.Text(), "did not recognize") {
		t.Error("an unrecognized error was not reported")
	}
}

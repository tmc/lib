package cuptibench

import (
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cuptirange"
)

func wholeResult(metrics map[string]float64) *Result {
	r := &Result{Metadata: map[string]string{"mode": "whole"}, Ranges: []Range{{Name: "operation", Kind: "operation"}}}
	for name, v := range metrics {
		r.Ranges[0].Metrics = append(r.Ranges[0].Metrics, Metric{Name: name, Value: v})
	}
	return r
}

// The measured GB10 values from the probe that established what these counters
// mean. A 50ms host sleep with no kernel launched at all.
var measuredSleepOnly = map[string]float64{
	"gpu__time_duration.sum": 51394784,
	"sm__cycles_active.avg":  0,
	"sm__cycles_elapsed.avg": 126750196,
	"sm__cycles_active.sum":  0,
}

// The same range holding one 20ms busy-wait kernel on a single block.
var measuredKernelOnly = map[string]float64{
	"gpu__time_duration.sum": 20047264,
	"sm__cycles_active.avg":  1075162,
	"sm__cycles_elapsed.avg": 51725294,
	"sm__cycles_active.sum":  51608092,
}

// TestBudgetSeparatesIdleFromBusy is the positive control for the time budget:
// the two inputs differ only in whether a kernel ran, and the reported active
// fraction must separate them. A budget that reads the same for both carries no
// information about idleness, which is the whole reason TimeSet exists.
func TestBudgetSeparatesIdleFromBusy(t *testing.T) {
	idle, err := wholeResult(measuredSleepOnly).Budget()
	if err != nil {
		t.Fatal(err)
	}
	busy, err := wholeResult(measuredKernelOnly).Budget()
	if err != nil {
		t.Fatal(err)
	}
	if idle.ActiveFraction != 0 {
		t.Errorf("host-sleep-only range: active fraction %v, want 0", idle.ActiveFraction)
	}
	if busy.ActiveFraction <= idle.ActiveFraction {
		t.Errorf("kernel range active fraction %v must exceed idle range %v", busy.ActiveFraction, idle.ActiveFraction)
	}
	// One block occupies one SM of 48, so the occupancy-weighted fraction is
	// small even though the device was busy the whole time. The number must not
	// be read as "the device was 2% busy" -- that is what String() guards.
	if got := busy.ActiveFraction; got < 0.01 || got > 0.05 {
		t.Errorf("kernel active fraction %v outside the measured 1/48-of-SMs range", got)
	}
	if !strings.Contains(idle.String(), "no kernel executed") {
		t.Errorf("a zero sm__cycles_active.sum must be stated outright, got %q", idle.String())
	}
}

// TestBudgetRefusesToDeriveIdle pins the negative: the type must not offer an
// idle duration, and must say why where a reader would look for one.
func TestBudgetRefusesToDeriveIdle(t *testing.T) {
	b, err := wholeResult(measuredKernelOnly).Budget()
	if err != nil {
		t.Fatal(err)
	}
	s := b.String()
	if !strings.Contains(s, "not derivable from counters") {
		t.Errorf("String must state that idle is not derivable, got %q", s)
	}
	if !strings.Contains(s, "occupancy-weighted") {
		t.Errorf("the active fraction must name its own weighting, got %q", s)
	}
}

// TestBudgetRejectsSummedModes is the control on the mode restriction: these
// counters are per-range durations, and Regions or Kernels would invite adding
// them across siblings.
func TestBudgetRejectsSummedModes(t *testing.T) {
	for _, mode := range []string{"regions", "kernels"} {
		r := wholeResult(measuredKernelOnly)
		r.Metadata["mode"] = mode
		if _, err := r.Budget(); err == nil {
			t.Errorf("mode %q: Budget must refuse a mode whose ranges would be summed", mode)
		}
	}
	if _, err := wholeResult(measuredKernelOnly).Budget(); err != nil {
		t.Errorf("whole mode must be accepted: %v", err)
	}
}

// TestBudgetMarksAbsentMembers keeps a missing counter from reading as a zero.
func TestBudgetMarksAbsentMembers(t *testing.T) {
	b, err := wholeResult(map[string]float64{"gpu__time_duration.sum": 1000}).Budget()
	if err != nil {
		t.Fatal(err)
	}
	if len(b.Partial) != 3 {
		t.Errorf("Partial = %v, want the three absent members", b.Partial)
	}
	if b.ActiveFraction != 0 {
		t.Errorf("an unmeasured fraction must stay zero, got %v", b.ActiveFraction)
	}
	if !strings.Contains(b.String(), "UNMEASURED") {
		t.Errorf("absent members must render as unmeasured, got %q", b.String())
	}
}

// TestTimeSetResolves checks the curated set is reachable by name.
func TestTimeSetResolves(t *testing.T) {
	got, err := Resolve(TimeSet)
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"gpu__time_duration.sum", "sm__cycles_active.avg", "sm__cycles_elapsed.avg", "sm__cycles_active.sum"}
	if len(got) != len(want) {
		t.Fatalf("Resolve(%q) = %v, want %v", TimeSet, got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("Resolve[%d] = %q, want %q", i, got[i], want[i])
		}
	}
}

// TestWarmCacheTrafficGuard is the positive control for the cache guard: it
// must fire on a traffic metric under a no-reset policy and stay silent
// otherwise. A guard that never fires, or always fires, gates nothing.
func TestWarmCacheTrafficGuard(t *testing.T) {
	for _, tt := range []struct {
		name    string
		policy  cuptirange.CachePolicy
		metrics []string
		want    bool
	}{
		{"l2 read under preserve", cuptirange.CachePreserve, []string{"lts__t_sectors_srcunit_tex_op_read.sum"}, true},
		{"l2 read under default", cuptirange.CacheDefault, []string{"lts__t_sectors_srcunit_tex_op_read.sum"}, true},
		{"default metric set carries l2", cuptirange.CachePreserve, Default(), true},
		{"native dram name", cuptirange.CachePreserve, []string{"dram__bytes_read.sum"}, true},
		{"launch counters only", cuptirange.CachePreserve, []string{"sm__ctas_launched.sum"}, false},
		{"time set", cuptirange.CachePreserve, metricSets[TimeSet], false},
		{"l2 read under flush", cuptirange.CacheFlushL2, []string{"lts__t_sectors_srcunit_tex_op_read.sum"}, false},
	} {
		c := Config{CachePolicy: tt.policy, Metrics: tt.metrics}
		if got := len(c.warmCacheTraffic()) > 0; got != tt.want {
			t.Errorf("%s: warm traffic = %v, want %v (metrics %v)", tt.name, got, tt.want, tt.metrics)
		}
	}
}

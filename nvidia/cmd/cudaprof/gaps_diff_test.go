package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudacapture"
)

// pacedJournal is a five-kernel run with uniform kernels and uniform gaps. The
// two knobs are exactly the two quantities a device-time-only diff conflates.
func pacedJournal(kernelNanos, gapNanos uint64) []byte {
	b := newJournalBuilder()
	start := uint64(10_000)
	for i := uint32(1); i <= 5; i++ {
		b.kernel(start, start+kernelNanos, i)
		b.api(start-3_000, start-2_000, i)
		start += kernelNanos + gapNanos
	}
	return b.finalize()
}

func gapReportFromJournal(t *testing.T, data []byte, warmupMS float64) *GapReport {
	t.Helper()
	cap := testCapture(map[string][]byte{"capture.journal": data}, completeStatus())
	report, err := analyzeGaps(cap, gapOptions{warmupMS: warmupMS, maxRecords: controllerMaxJournalRecords})
	if err != nil {
		t.Fatal(err)
	}
	return report
}

// TestGapsDiffScoresIdleWinAsAWin is the test this whole command exists for. A
// real ds4-min commit raised kernel busy time by 1.504 ms/token and cut
// inter-kernel idle by 1.497, and the run got faster; a profiler summing
// per-kernel device time scored that measured speedup as a 1.9% regression.
// The fixture is that shape in miniature: device-active time rises by 2000 ns
// while idle falls by 2400, so the span falls. Every assertion below fails if
// the diff reports only the active column.
func TestGapsDiffScoresIdleWinAsAWin(t *testing.T) {
	base := gapReportFromJournal(t, pacedJournal(8_000, 1_000), 200)
	target := gapReportFromJournal(t, pacedJournal(8_400, 400), 200)
	diff, err := diffGapReports(base, target)
	if err != nil {
		t.Fatal(err)
	}
	if len(diff.Devices) != 1 {
		t.Fatalf("devices = %d, want 1", len(diff.Devices))
	}
	s := diff.Devices[0].Overall
	if s == nil {
		t.Fatal("overall window missing")
	}
	for _, tt := range []struct {
		name string
		got  GapIntDelta
		want int64
	}{
		{"span", s.Span, -400},
		{"active", s.Active, 2_000},
		{"idle", s.Idle, -2_400},
		{"gap count", s.GapCount, 0},
	} {
		if tt.got.Delta != tt.want {
			t.Errorf("%s delta = %d, want %d (base %d, target %d)", tt.name, tt.got.Delta, tt.want, tt.got.Base, tt.got.Target)
		}
	}
	// Summed per-kernel device time is the active column alone, and it rose.
	if s.Active.Delta <= 0 || s.Span.Delta >= 0 {
		t.Fatalf("fixture no longer poses the problem: active %+d, span %+d", s.Active.Delta, s.Span.Delta)
	}
	if s.DutyCyclePct.Delta <= 0 {
		t.Errorf("duty cycle delta = %+.2f, want positive", s.DutyCyclePct.Delta)
	}
	for _, p := range s.Distribution {
		if p.Delta != -600 {
			t.Errorf("gap %s delta = %d, want -600", p.Name, p.Delta)
		}
	}
	if s.DistMean.Delta != -600 {
		t.Errorf("gap mean delta = %g, want -600", s.DistMean.Delta)
	}
	if !strings.Contains(s.Verdict, "would score this run a regression") {
		t.Errorf("verdict = %q, want it to name the regression a device-time-only diff would report", s.Verdict)
	}
	if !strings.Contains(s.Verdict, "inter-kernel idle") {
		t.Errorf("verdict = %q, want it to attribute the win to inter-kernel idle", s.Verdict)
	}
	out := diff.text()
	for _, want := range []string{"device 0, overall", "timeline span", "device active", "device idle", "verdict:"} {
		if !strings.Contains(out, want) {
			t.Errorf("rendered diff missing %q:\n%s", want, out)
		}
	}
}

func TestGapVerdict(t *testing.T) {
	d := func(base, target int64) GapIntDelta { return intDelta("", base, target) }
	tests := []struct {
		name               string
		span, active, idle GapIntDelta
		want               string
	}{
		{"idle win read as regression", d(100, 90), d(80, 85), d(20, 5), "would score this run a regression"},
		{"idle loss read as speedup", d(90, 100), d(85, 80), d(5, 20), "would score this run a speedup"},
		{"span fell with kernels", d(100, 90), d(80, 75), d(20, 15), "span fell by"},
		{"span rose with kernels", d(90, 100), d(75, 80), d(15, 20), "span rose by"},
		{"nothing moved", d(100, 100), d(80, 80), d(20, 20), "unchanged"},
		{"span held, work traded", d(100, 100), d(80, 85), d(20, 15), "traded busy time for idle time"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := gapVerdict(tt.span, tt.active, tt.idle); !strings.Contains(got, tt.want) {
				t.Errorf("gapVerdict = %q, want it to contain %q", got, tt.want)
			}
		})
	}
}

// TestGapsDiffRefusesMismatchedWarmup keeps two unlike steady-state windows
// from subtracting to a number that means nothing. The command cuts both sides
// at one warmup, so this refusal guards reports that arrive from elsewhere.
func TestGapsDiffRefusesMismatchedWarmup(t *testing.T) {
	base := gapReportFromJournal(t, pacedJournal(8_000, 1_000), 200)
	target := gapReportFromJournal(t, pacedJournal(8_000, 1_000), 50)
	_, err := diffGapReports(base, target)
	if err == nil {
		t.Fatal("diffGapReports accepted mismatched warmup windows")
	}
	if !strings.Contains(err.Error(), "warmup windows differ") {
		t.Errorf("error = %v, want it to name the warmup mismatch", err)
	}
}

// TestGapsDiffSteadyState compares the window that matters for a before/after
// test, and reports the window as absent rather than comparing unlike windows
// when one side has no activity past the warmup.
func TestGapsDiffSteadyState(t *testing.T) {
	// 0.005 ms of warmup leaves the last four kernels of each run in the
	// steady-state section.
	base := gapReportFromJournal(t, pacedJournal(8_000, 1_000), 0.005)
	target := gapReportFromJournal(t, pacedJournal(8_400, 400), 0.005)
	diff, err := diffGapReports(base, target)
	if err != nil {
		t.Fatal(err)
	}
	s := diff.Devices[0].SteadyState
	if s == nil {
		t.Fatalf("steady-state window missing: %v", diff.Devices[0].Missing)
	}
	if s.Label != "steady-state" {
		t.Errorf("label = %q, want steady-state", s.Label)
	}
	if s.Active.Delta <= 0 || s.Idle.Delta >= 0 || s.Span.Delta >= 0 {
		t.Errorf("steady-state deltas = span %+d active %+d idle %+d, want the same shape as overall", s.Span.Delta, s.Active.Delta, s.Idle.Delta)
	}

	// With the default warmup no activity survives, so there is no window.
	wide := gapReportFromJournal(t, pacedJournal(8_000, 1_000), 200)
	wideTarget := gapReportFromJournal(t, pacedJournal(8_400, 400), 200)
	diff, err = diffGapReports(wide, wideTarget)
	if err != nil {
		t.Fatal(err)
	}
	if diff.Devices[0].SteadyState != nil {
		t.Fatal("steady-state compared although neither capture has one")
	}
	if len(diff.Devices[0].Missing) == 0 || !strings.Contains(diff.Devices[0].Missing[0], "no steady-state window") {
		t.Errorf("missing = %v, want the absent steady-state window named", diff.Devices[0].Missing)
	}
}

// TestGapsDiffDeviceOnOneSide keeps a device the other capture never touched
// from being compared against nothing.
func TestGapsDiffDeviceOnOneSide(t *testing.T) {
	b := newJournalBuilder()
	b.activity("kernel", 0, 1_000, 2_000, 1)
	b.activity("kernel", 0, 3_000, 4_000, 2)
	b.activity("kernel", 1, 1_000, 2_000, 3)
	b.activity("kernel", 1, 3_000, 4_000, 4)
	base := gapReportFromJournal(t, b.finalize(), 200)
	target := gapReportFromJournal(t, pacedJournal(8_000, 1_000), 200)
	diff, err := diffGapReports(base, target)
	if err != nil {
		t.Fatal(err)
	}
	if len(diff.Devices) != 2 {
		t.Fatalf("devices = %d, want 2", len(diff.Devices))
	}
	one := diff.Devices[1]
	if one.Device != 1 || one.Overall != nil {
		t.Fatalf("device 1 = %+v, want no comparison", one)
	}
	if len(one.Missing) == 0 || !strings.Contains(one.Missing[0], "target capture recorded no activity") {
		t.Errorf("missing = %v, want the absent side named", one.Missing)
	}
}

func TestGapsDiffJSON(t *testing.T) {
	base := gapReportFromJournal(t, pacedJournal(8_000, 1_000), 200)
	target := gapReportFromJournal(t, pacedJournal(8_400, 400), 200)
	diff, err := diffGapReports(base, target)
	if err != nil {
		t.Fatal(err)
	}
	data, err := json.Marshal(diff)
	if err != nil {
		t.Fatal(err)
	}
	var round struct {
		WarmupMS float64 `json:"warmup_ms"`
		Devices  []struct {
			Overall struct {
				Span    GapIntDelta   `json:"span_ns"`
				Active  GapIntDelta   `json:"active_ns"`
				Idle    GapIntDelta   `json:"idle_ns"`
				Dist    []GapIntDelta `json:"distribution_ns"`
				Verdict string        `json:"verdict"`
			} `json:"overall"`
		} `json:"devices"`
	}
	if err := json.Unmarshal(data, &round); err != nil {
		t.Fatal(err)
	}
	if round.WarmupMS != 200 || len(round.Devices) != 1 {
		t.Fatalf("decoded = %+v", round)
	}
	got := round.Devices[0].Overall
	if got.Span.Delta != -400 || got.Active.Delta != 2_000 || got.Idle.Delta != -2_400 {
		t.Errorf("decoded deltas = span %+d active %+d idle %+d", got.Span.Delta, got.Active.Delta, got.Idle.Delta)
	}
	if len(got.Dist) != 7 || got.Verdict == "" {
		t.Errorf("decoded distribution = %d rows, verdict %q", len(got.Dist), got.Verdict)
	}
}

func TestParseGapsArgsDiffBase(t *testing.T) {
	tests := []struct {
		name     string
		args     []string
		wantBase string
		wantErr  string
	}{
		{"long form", []string{"--diff_base=base.pprof", "target.pprof"}, "base.pprof", ""},
		{"single dash", []string{"-diff_base=base.pprof", "target.pprof"}, "base.pprof", ""},
		{"absent", []string{"target.pprof"}, "", ""},
		{"empty value", []string{"--diff_base=", "target.pprof"}, "", "invalid diff base"},
		{"hyphen spelling is not accepted", []string{"--diff-base=base.pprof", "target.pprof"}, "", "unknown flag"},
		{"no target", []string{"--diff_base=base.pprof"}, "", "usage"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			opts, err := parseGapsArgs(tt.args)
			if tt.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
					t.Fatalf("err = %v, want it to contain %q", err, tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if opts.diffBase != tt.wantBase {
				t.Errorf("diffBase = %q, want %q", opts.diffBase, tt.wantBase)
			}
		})
	}
}

// TestGapsDiffRefusalSurvivesDiffing keeps the lossy-journal refusal in force
// on both sides of a comparison: a duty cycle delta computed from a surviving
// prefix reads exactly like one computed from two whole runs.
func TestGapsDiffRefusesLossyCapture(t *testing.T) {
	dir := t.TempDir()
	good := writeGapCapture(t, filepath.Join(dir, "good.pprof"), pacedJournal(8_000, 1_000), completeStatus())
	lossy := completeStatus()
	lossy["dropped_records"] = 7
	bad := writeGapCapture(t, filepath.Join(dir, "lossy.pprof"), pacedJournal(8_000, 1_000), lossy)

	opts := gapOptions{warmupMS: 200, maxRecords: controllerMaxJournalRecords}
	for _, tt := range []struct{ name, base, target string }{
		{"lossy base", bad, good},
		{"lossy target", good, bad},
	} {
		t.Run(tt.name, func(t *testing.T) {
			o := opts
			o.diffBase, o.capturePath = tt.base, tt.target
			err := runGapsDiff(o)
			if err == nil {
				t.Fatal("runGapsDiff analyzed a capture that dropped records")
			}
			if !strings.Contains(err.Error(), "dropped 7 record(s)") {
				t.Errorf("error = %v, want the dropped records named", err)
			}
		})
	}
}

func writeGapCapture(t *testing.T, path string, journal []byte, status map[string]any) string {
	t.Helper()
	cap := testCapture(map[string][]byte{"capture.journal": journal}, status)
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if err := cudacapture.Write(f, cap); err != nil {
		t.Fatal(err)
	}
	return path
}

// TestGapDecompositionMarkdown checks the section that stands beside the
// summed per-kernel duration delta in "cudaprof diff --explain".
func TestGapDecompositionMarkdown(t *testing.T) {
	base := testCapture(map[string][]byte{"capture.journal": pacedJournal(8_000, 1_000)}, completeStatus())
	target := testCapture(map[string][]byte{"capture.journal": pacedJournal(8_400, 400)}, completeStatus())
	got := gapDecompositionMarkdown(base, target, "base.pprof", "target.pprof")
	for _, want := range []string{"## Device Timeline Decomposition", "timeline span", "device active", "device idle", "duty cycle", "would score this run a regression"} {
		if !strings.Contains(got, want) {
			t.Errorf("decomposition missing %q:\n%s", want, got)
		}
	}
}

// TestGapDecompositionDegrades keeps a capture without a journal from taking
// the rest of the explanation down with it: the reason is named and the
// duration delta above still stands.
func TestGapDecompositionDegrades(t *testing.T) {
	withJournal := testCapture(map[string][]byte{"capture.journal": pacedJournal(8_000, 1_000)}, completeStatus())
	without := testCapture(nil, completeStatus())
	got := gapDecompositionMarkdown(without, withJournal, "base.pprof", "target.pprof")
	if !strings.Contains(got, "## Device Timeline Decomposition") {
		t.Errorf("degraded decomposition lost its heading:\n%s", got)
	}
	if !strings.Contains(got, "`base.pprof`") || !strings.Contains(got, "no capture.journal member") {
		t.Errorf("degraded decomposition does not name the capture and the reason:\n%s", got)
	}
	if strings.Contains(got, "`target.pprof`") {
		t.Errorf("decomposition blamed the capture that does have a journal:\n%s", got)
	}
}

func TestGapDuration(t *testing.T) {
	tests := []struct {
		name         string
		in           int64
		want, signed string
	}{
		{"nanoseconds", 400, "400 ns", "+400 ns"},
		{"negative nanoseconds", -400, "400 ns", "-400 ns"},
		{"microseconds", 2_400, "2.400 us", "+2.400 us"},
		{"milliseconds", 1_504_000, "1.504 ms", "+1.504 ms"},
		{"negative milliseconds", -1_497_000, "1.497 ms", "-1.497 ms"},
		{"zero", 0, "0 ns", "+0 ns"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := gapDuration(tt.in); got != tt.want {
				t.Errorf("gapDuration(%d) = %q, want %q", tt.in, got, tt.want)
			}
			if got := signedDuration(tt.in); got != tt.signed {
				t.Errorf("signedDuration(%d) = %q, want %q", tt.in, got, tt.signed)
			}
		})
	}
}

// TestGapsDiffMeasuredCase replays the nsys numbers that motivated this
// command. A ds4-min commit measured by a six-run ABBA wall-clock test as a
// 5.3% speedup with bit-exact logits showed, per token: span 82.920 -> 82.927
// ms, busy 78.752 -> 80.256, gap 4.168 -> 2.671. Summed per-kernel device time
// is the busy column, and on its own it calls that measured speedup a 1.9%
// regression. The diff must not.
//
// These figures are a shape fixture, not a capture. They are per-token values
// derived by a 32-token difference method across two separate processes, so no
// single capture's span would ever hold them and nothing here should be read as
// a claim about what one capture reports. What the test pins is the direction:
// span and active moving opposite ways, and the verdict that follows.
func TestGapsDiffMeasuredCase(t *testing.T) {
	stats := func(span, active, idle int64) *GapStats {
		s := &GapStats{Label: "steady-state", SpanNanos: span, ActiveNanos: active, IdleNanos: idle}
		s.DutyCyclePct = float64(active) / float64(active+idle) * 100
		return s
	}
	pre := stats(82_920_000, 78_752_000, 4_168_000)
	post := stats(82_927_000, 80_256_000, 2_671_000)
	d := diffGapStats(pre, post)

	// The busy column alone rises by 1.9%: that is the reading being refuted.
	if busyPct := float64(d.Active.Delta) / float64(d.Active.Base) * 100; busyPct < 1.8 || busyPct > 2.0 {
		t.Fatalf("fixture no longer poses the problem: busy rose %.2f%%, want ~1.9%%", busyPct)
	}
	if d.Idle.Delta != -1_497_000 || d.Active.Delta != 1_504_000 {
		t.Fatalf("deltas = active %+d idle %+d, want +1504000 / -1497000", d.Active.Delta, d.Idle.Delta)
	}
	if d.DutyCyclePct.Delta <= 0 {
		t.Errorf("duty cycle delta = %+.2f pp, want positive", d.DutyCyclePct.Delta)
	}
	// Span rose by 7 us here, so the honest verdict is not that the run got
	// faster: it is that the busy column moved the other way from the idle
	// column, and the busy column alone cannot be the headline.
	if !strings.Contains(d.Verdict, "device active +1.504 ms") || !strings.Contains(d.Verdict, "device idle -1.497 ms") {
		t.Errorf("verdict = %q, want both columns reported", d.Verdict)
	}
}

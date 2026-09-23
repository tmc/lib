package main

import (
	"encoding/json"
	"math"
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
		name      string
		args      []string
		wantBase  string
		wantBytes int64
		wantErr   string
	}{
		{"long form", []string{"--diff_base=base.pprof", "target.pprof"}, "base.pprof", 0, ""},
		{"base and target limit", []string{"--diff_base=base.pprof", "--journal-bytes=123", "target.pprof"}, "base.pprof", 123, ""},
		{"single dash", []string{"-diff_base=base.pprof", "target.pprof"}, "base.pprof", 0, ""},
		{"absent", []string{"target.pprof"}, "", 0, ""},
		{"empty value", []string{"--diff_base=", "target.pprof"}, "", 0, "invalid diff base"},
		{"hyphen spelling is not accepted", []string{"--diff-base=base.pprof", "target.pprof"}, "", 0, "unknown flag"},
		{"no target", []string{"--diff_base=base.pprof"}, "", 0, "usage"},
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
			if opts.diffBase != tt.wantBase || (tt.wantBytes != 0 && opts.journalBytes != tt.wantBytes) {
				t.Errorf("options = %+v, want base %q and journal bytes %d", opts, tt.wantBase, tt.wantBytes)
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

// TestCensusTextMarksTheThreeClasses guards the rendering, which is the whole
// deliverable: the classes are only useful if a reader can tell which row
// belongs to which. A column of deltas with no launch counts beside them is
// the view that sent one A/B an hour down the wrong kernel, and a row marked
// "unchanged" on the strength of a count alone is the one that sent the next
// A/B down a window its two arms no longer agreed about.
func TestCensusTextMarksTheThreeClasses(t *testing.T) {
	type row = struct {
		name     string
		duration uint64
		count    int
	}
	base := censusReport(t, []row{{"repack", 1_000, 10}, {"dense", 1_000, 5}, {"tail", 1_000, 3}})
	target := censusReport(t, []row{{"repack", 1_000, 4}, {"dense", 3_000, 5}, {"tail", 1_000, 3}})
	base.Capture, target.Capture = "base.pprof", "target.pprof"
	diff, err := diffGapReports(base, target)
	if err != nil {
		t.Fatal(err)
	}
	k := diff.Kernels
	if len(k.Changed) != 1 || k.Changed[0].Name != "repack" {
		t.Fatalf("changed = %v, want repack alone", names(k.Changed))
	}
	if len(k.Relocated) != 1 || k.Relocated[0].Name != "dense" {
		t.Fatalf("relocated = %v, want dense alone", names(k.Relocated))
	}
	if len(k.Unchanged) != 1 || k.Unchanged[0].Name != "tail" {
		t.Fatalf("unchanged = %v, want tail alone", names(k.Unchanged))
	}
	text := diff.text()
	for _, want := range []string{
		"per-kernel census (whole capture, not a window)",
		"launch count changed",
		"launch count unchanged, position moved",
		"launch count unchanged, position held",
		"repack",
		"dense",
		"tail",
	} {
		if !strings.Contains(text, want) {
			t.Errorf("census text lacks %q:\n%s", want, text)
		}
	}
	// The launch counts have to be printed, not only used to sort: the whole
	// request was for a reader to be able to see that a count held still.
	if !strings.Contains(text, "        5         5") {
		t.Errorf("relocated kernel's launch counts are not printed:\n%s", text)
	}
	md := &strings.Builder{}
	writeKernelCensusMarkdown(md, k)
	for _, want := range []string{
		"| `dense` | 5 -> 5 |",
		"unchanged, position moved |",
		"| `tail` | 3 -> 3 |",
		"unchanged, position held |",
	} {
		if !strings.Contains(md.String(), want) {
			t.Errorf("markdown census lacks %q:\n%s", want, md.String())
		}
	}
}

// TestRelocationIsBlindToDuration is the negative control for the position
// flag, and it is the reason position is measured in launch order rather than
// elapsed time. Both arms run the same kernels in the same order; one of them
// simply got slower. Measured against the device span, that alone slides every
// later kernel's median start backwards -- the span it is divided by grew --
// and the census would report a relocation that nothing relocated. Order
// cannot be perturbed by a duration, so nothing here may be flagged.
func TestRelocationIsBlindToDuration(t *testing.T) {
	type row = struct {
		name     string
		duration uint64
		count    int
	}
	base := censusReport(t, []row{{"repack", 1_000, 10}, {"dense", 1_000, 5}})
	target := censusReport(t, []row{{"repack", 1_000, 10}, {"dense", 3_000, 5}})
	diff, err := diffGapReports(base, target)
	if err != nil {
		t.Fatal(err)
	}
	if got := names(diff.Kernels.Relocated); len(got) != 0 {
		t.Errorf("relocated = %v, want none: only a duration changed", got)
	}
	if got := names(diff.Kernels.Unchanged); len(got) != 2 {
		t.Errorf("unchanged = %v, want both kernels", got)
	}
	if diff.Kernels.UnchangedNanos != 10_000 {
		t.Errorf("unchanged nanos = %d, want +10000: the slowdown is still reported", diff.Kernels.UnchangedNanos)
	}
}

func names(ds []KernelDelta) []string {
	var out []string
	for _, d := range ds {
		out = append(out, d.Name)
	}
	return out
}

// orderedCensus builds a capture whose kernel launches occur in exactly the
// given order, one launch per entry, all the same duration. An empty name is
// an unnamed kernel record -- a launch the census can count but not attribute.
func orderedCensus(t *testing.T, order []string) *GapReport {
	t.Helper()
	b := newJournalBuilder()
	var at uint64 = 1_000
	var corr uint32
	for _, name := range order {
		corr++
		if name == "" {
			b.kernel(at, at+1_000, corr)
		} else {
			b.namedKernel(name, at, at+1_000, corr)
		}
		b.api(at-500, at-100, corr)
		at += 2_000
	}
	cap := testCapture(map[string][]byte{"capture.journal": b.finalize()}, completeStatus())
	report, err := analyzeGaps(cap, gapOptions{maxRecords: controllerMaxJournalRecords})
	if err != nil {
		t.Fatal(err)
	}
	return report
}

// TestPositionIsTheMedianLaunchNotTheMean is why the position is a median. The
// spread kernel keeps four of its five launches at the head of the run in both
// arms and sends the fifth to the far end in one of them. That one launch
// moves a mean by 14 points of the run -- three times the relocation
// threshold -- and reports a kernel as relocated when the bulk of its work
// never moved. The median stays where the launches are.
func TestPositionIsTheMedianLaunchNotTheMean(t *testing.T) {
	filler := func(n int) []string {
		out := make([]string, n)
		for i := range out {
			out[i] = "filler"
		}
		return out
	}
	base := orderedCensus(t, append(append([]string{"spread", "spread", "spread", "spread"}, filler(10)...), "spread"))
	target := orderedCensus(t, append([]string{"spread", "spread", "spread", "spread", "spread"}, filler(10)...))
	diff, err := diffGapReports(base, target)
	if err != nil {
		t.Fatal(err)
	}
	for _, d := range diff.Kernels.Relocated {
		if d.Name == "spread" {
			t.Errorf("spread is relocated (shift %+.3f): four of five launches never moved", d.PositionShift)
		}
	}
	// The control: filler really did move, so the flag is not simply off.
	if got := names(diff.Kernels.Relocated); len(got) != 1 || got[0] != "filler" {
		t.Errorf("relocated = %v, want filler alone", got)
	}
}

// TestUnnamedLaunchesHoldThePositionScale checks that a kernel record with no
// name still advances the launch order. It is a launch: work ran on the device
// at that point in the run. Leaving it out of the scale makes position mean
// "rank among the launches we could name", and a kernel that sat still while
// four unnamed launches moved past it then reports no shift at all.
func TestUnnamedLaunchesHoldThePositionScale(t *testing.T) {
	base := orderedCensus(t, []string{"a", "b", "", "", "", "", "c"})
	target := orderedCensus(t, []string{"a", "", "", "", "", "b", "c"})
	diff, err := diffGapReports(base, target)
	if err != nil {
		t.Fatal(err)
	}
	if got := names(diff.Kernels.Relocated); len(got) != 1 || got[0] != "b" {
		t.Fatalf("relocated = %v, want b alone: four unnamed launches moved past it", got)
	}
	if got := diff.Kernels.Relocated[0].PositionShift; got < 0.6 {
		t.Errorf("b's shift = %+.3f, want ~+0.667 of the run's launch order", got)
	}
}

// TestPositionCaveatPrintsAboveTheTable is errata 21.155: a whole-capture diff
// whose two arms disagree about WHERE work happened has to say so above the
// table, not in a section a reader reaches after the totals have been read.
//
// The section heading has carried that sentence since the three-class split
// landed. It was not enough: the reader who divides by a window has already
// divided by the time they get there.
func TestPositionCaveatPrintsAboveTheTable(t *testing.T) {
	base := orderedCensus(t, []string{"early", "early", "held", "late", "late"})
	target := orderedCensus(t, []string{"held", "late", "late", "early", "early"})
	diff, err := diffGapReports(base, target)
	if err != nil {
		t.Fatal(err)
	}
	k := diff.Kernels
	if k.PositionCaveat == "" {
		t.Fatalf("no position caveat; relocated = %v", names(k.Relocated))
	}
	text := diff.text()
	caveatAt := strings.Index(text, k.PositionCaveat)
	tableAt := strings.Index(text, "launch count changed:")
	if caveatAt < 0 || tableAt < 0 {
		t.Fatalf("caveat at %d, first section at %d, in:\n%s", caveatAt, tableAt, text)
	}
	if caveatAt > tableAt {
		t.Errorf("position caveat printed below the table (%d > %d):\n%s", caveatAt, tableAt, text)
	}
	if !strings.Contains(k.PositionCaveat, "window") {
		t.Errorf("caveat does not name the window, the thing to check: %q", k.PositionCaveat)
	}
	var md strings.Builder
	writeKernelCensusMarkdown(&md, k)
	if !strings.Contains(md.String(), k.PositionCaveat) {
		t.Errorf("markdown render dropped the position caveat:\n%s", md.String())
	}
}

// TestPositionCaveatNamesTheFurthestMover keeps the caveat from reporting
// whichever row sorted first. The rows are ranked by delta, so the kernel that
// moved furthest is generally not row zero, and naming the wrong one sends the
// reader to the wrong window.
func TestPositionCaveatNamesTheFurthestMover(t *testing.T) {
	base := orderedCensus(t, []string{"far", "far", "near", "near", "anchor", "anchor"})
	target := orderedCensus(t, []string{"near", "anchor", "anchor", "near", "far", "far"})
	diff, err := diffGapReports(base, target)
	if err != nil {
		t.Fatal(err)
	}
	k := diff.Kernels
	var worst KernelDelta
	for _, r := range k.Relocated {
		if math.Abs(r.PositionShift) > math.Abs(worst.PositionShift) {
			worst = r
		}
	}
	if worst.Name == "" {
		t.Fatalf("nothing relocated: %v", names(k.Relocated))
	}
	if !strings.Contains(k.PositionCaveat, worst.Name) {
		t.Errorf("caveat %q does not name the furthest mover %q (shift %.3f)", k.PositionCaveat, worst.Name, worst.PositionShift)
	}
}

// TestNoPositionCaveatWhenNothingMoved is the negative control. A caveat that
// prints on every diff is the unconditional preamble again, and says nothing
// about the pair in front of the reader.
func TestNoPositionCaveatWhenNothingMoved(t *testing.T) {
	order := []string{"a", "a", "b", "b", "c"}
	diff, err := diffGapReports(orderedCensus(t, order), orderedCensus(t, order))
	if err != nil {
		t.Fatal(err)
	}
	if got := diff.Kernels.PositionCaveat; got != "" {
		t.Errorf("position caveat on an unmoved pair: %q", got)
	}
	if strings.Contains(diff.text(), "disagree about where work happened") {
		t.Errorf("text names a position disagreement that did not occur:\n%s", diff.text())
	}
}

// prologueJournal is two "embed" kernels separated by prologueGap, then three
// identical "layer" kernels: a run whose startup length varies while its
// steady-state work does not.
func prologueJournal(prologueGap uint64) []byte {
	b := newJournalBuilder()
	start := uint64(10_000)
	var corr uint32
	for range 2 {
		corr++
		b.namedKernel("embed", start, start+1_000, corr)
		start += 1_000 + prologueGap
	}
	for range 3 {
		corr++
		b.namedKernel("layer", start, start+8_000, corr)
		start += 8_000 + 1_000
	}
	return b.finalize()
}

// TestGapsDiffWarmupUntil cuts both steady-state windows at the first kernel
// of the same name. Two runs that differ only in the length of their startup
// then have identical steady states, where a fixed warmup cuts them at
// different points in their work and reports a difference in the steady state
// that is really in the prologue.
func TestGapsDiffWarmupUntil(t *testing.T) {
	analyze := func(data []byte, opts gapOptions) *GapReport {
		t.Helper()
		opts.maxRecords = controllerMaxJournalRecords
		r, err := analyzeGaps(testCapture(map[string][]byte{"capture.journal": data}, completeStatus()), opts)
		if err != nil {
			t.Fatal(err)
		}
		return r
	}
	short, long := prologueJournal(1_000), prologueJournal(30_000)

	// Control: a fixed 0.02 ms warmup keeps a prologue gap in the long run's
	// window and not in the short one's, so the steady states differ.
	fixed := gapOptions{warmupMS: 0.02}
	diff, err := diffGapReports(analyze(short, fixed), analyze(long, fixed))
	if err != nil {
		t.Fatal(err)
	}
	if s := diff.Devices[0].SteadyState; s == nil || s.Span.Delta == 0 {
		t.Fatalf("fixed warmup reported steady-state %+v, want a span difference for the cut to remove", s)
	}

	until := gapOptions{warmupUntil: "^layer$"}
	diff, err = diffGapReports(analyze(short, until), analyze(long, until))
	if err != nil {
		t.Fatal(err)
	}
	s := diff.Devices[0].SteadyState
	if s == nil {
		t.Fatalf("steady-state window missing: %v", diff.Devices[0].Missing)
	}
	if s.Span.Delta != 0 || s.Idle.Delta != 0 || s.Active.Delta != 0 {
		t.Errorf("steady-state deltas = span %+d active %+d idle %+d, want zero: the runs differ only before the first layer", s.Span.Delta, s.Active.Delta, s.Idle.Delta)
	}
	if !strings.Contains(diff.text(), `before the first kernel matching "^layer$" excluded from both`) {
		t.Errorf("report does not name the warmup kernel:\n%s", diff.text())
	}

	none := gapOptions{warmupUntil: "^absent$"}
	diff, err = diffGapReports(analyze(short, none), analyze(long, none))
	if err != nil {
		t.Fatal(err)
	}
	if diff.Devices[0].SteadyState != nil || len(diff.Devices[0].Missing) == 0 || !strings.Contains(diff.Devices[0].Missing[0], `"^absent$"`) {
		t.Errorf("unmatched pattern: steady %+v, missing %v; want the window absent and the pattern named", diff.Devices[0].SteadyState, diff.Devices[0].Missing)
	}

	if _, err := diffGapReports(analyze(short, until), analyze(long, fixed)); err == nil {
		t.Error("diffGapReports accepted a kernel cut against a fixed one")
	}
}

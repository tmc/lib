package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/x/cudacapture"
)

// writeKernelTimeCapture writes a capture whose profile holds kernel_time
// samples for kernels and whose journal, when not nil, is journal.
func writeKernelTimeCapture(t *testing.T, name string, kernels map[string]int64, journal []byte) string {
	t.Helper()
	p := &profile.Profile{SampleType: []*profile.ValueType{{Type: "kernel_time", Unit: "nanoseconds"}}}
	id := uint64(1)
	for k, ns := range kernels {
		fn := &profile.Function{ID: id, Name: k}
		loc := &profile.Location{ID: id, Line: []profile.Line{{Function: fn}}}
		id++
		p.Function = append(p.Function, fn)
		p.Location = append(p.Location, loc)
		p.Sample = append(p.Sample, &profile.Sample{Location: []*profile.Location{loc}, Value: []int64{ns}})
	}
	files := map[string][]byte{}
	if journal != nil {
		files["capture.journal"] = journal
	}
	cap := testCapture(files, completeStatus())
	cap.Profile = p
	path := filepath.Join(t.TempDir(), name)
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

// TestRunDiffExplainSummedKernelHeadline checks the headline of
// "cudaprof diff --explain". Its first case is the ds4 headline in miniature:
// both arms ran the same device timeline, but the target's kernel records
// cover a tenth of its device-active time (its graph replays were opaque), so
// its summed kernel time is a tenth of the base's. Before the headline read
// coverage and span it printed "90.0% reduction, 10.00× speedup" there.
func TestRunDiffExplainSummedKernelHeadline(t *testing.T) {
	paced := pacedJournal(8_000, 1_000) // five 8 us kernels: 40 us active, 44 us span
	tests := []struct {
		name            string
		base, target    map[string]int64
		baseJ, targetJ  []byte
		args            []string
		want, forbidden []string
	}{
		{
			name: "coverage mismatch",
			base: map[string]int64{"k": 40_000}, baseJ: paced,
			target: map[string]int64{"k": 4_000}, targetJ: paced,
			want:      []string{"coverage 100.00%", "coverage 10.00%", "captured span 0.044 ms", "no ratio: kernel coverage differs by 90.00 pp", "Summed kernel time is not compared"},
			forbidden: []string{"10.00×", "90.0% reduction", "-90.0%", "speedup", "Runtime", "Performance improvement", "pipeline balance"},
		},
		{
			name: "span moves against summed kernel time",
			base: map[string]int64{"k": 20_000, "j": 20_000}, baseJ: paced,
			target: map[string]int64{"k": 15_000, "j": 20_000}, targetJ: pacedJournal(7_000, 5_000),
			want:      []string{"coverage 100.00%", "no ratio: summed kernel time moved -5.000 us but captured span moved +11.000 us", "| `k` | 20.00 | 15.00 | -5.00 | withheld |"},
			forbidden: []string{"1.14×", "-12.5%"},
		},
		{
			name:      "no journal",
			base:      map[string]int64{"k": 40_000},
			target:    map[string]int64{"k": 30_000},
			want:      []string{"UNREAD", "no ratio: base coverage UNREAD"},
			forbidden: []string{"1.33×", "-25.0%"},
		},
		{
			// The whole sum and the span both fall, so the arms agree, but
			// the kernel --kernel selects rose. Its ratio is not one the
			// span supports.
			name: "selected kernel moves against span",
			base: map[string]int64{"k1": 30_000, "k2": 10_000}, baseJ: paced,
			target: map[string]int64{"k1": 10_000, "k2": 20_000}, targetJ: pacedJournal(6_000, 1_000),
			args:      []string{"--kernel=k2"},
			want:      []string{"summed kernel time of kernels matching \"k2\" +10.00 µs; no ratio: the selected kernels' summed time moved +10.000 us but captured span moved -10.000 us"},
			forbidden: []string{"0.50×", "in agreement"},
		},
		{
			name: "arms agree",
			base: map[string]int64{"k": 40_000}, baseJ: paced,
			target: map[string]int64{"k": 30_000}, targetJ: pacedJournal(6_000, 1_000),
			want:      []string{"coverage 100.00%", "summed kernel time -10.00 µs (-25.0%, base/target 1.33×)", "Summed kernel time fell by 0.010 ms"},
			forbidden: []string{"no ratio", "speedup", "Runtime"},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			base := writeKernelTimeCapture(t, "base.pprof", tc.base, tc.baseJ)
			target := writeKernelTimeCapture(t, "target.pprof", tc.target, tc.targetJ)
			var buf bytes.Buffer
			args := append(append([]string{"--explain"}, tc.args...), base, target)
			if err := runDiff(&buf, args); err != nil {
				t.Fatal(err)
			}
			out := buf.String()
			for _, want := range append([]string{"## Summed Kernel Time", "summed kernel time"}, tc.want...) {
				if !strings.Contains(out, want) {
					t.Errorf("output missing %q:\n%s", want, out)
				}
			}
			for _, bad := range tc.forbidden {
				if strings.Contains(out, bad) {
					t.Errorf("output contains %q:\n%s", bad, out)
				}
			}
		})
	}
}

// TestRunDiffExplainOneSidedKernelRows: a kernel present in one arm has no
// ratio. The table used to print 1.00× for it.
func TestRunDiffExplainOneSidedKernelRows(t *testing.T) {
	paced := pacedJournal(8_000, 1_000)
	base := writeKernelTimeCapture(t, "base.pprof", map[string]int64{"shared": 20_000, "gone": 20_000}, paced)
	target := writeKernelTimeCapture(t, "target.pprof", map[string]int64{"shared": 10_000, "new": 30_000}, paced)
	var buf bytes.Buffer
	if err := runDiff(&buf, []string{"--explain", base, target}); err != nil {
		t.Fatal(err)
	}
	out := buf.String()
	for _, want := range []string{
		"| `gone` | 20.00 | 0.00 | -20.00 | only in base |",
		"| `new` | 0.00 | 30.00 | +30.00 | only in target |",
		"| `shared` | 20.00 | 10.00 | -10.00 | 2.00× |",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("output missing row %q:\n%s", want, out)
		}
	}
	for _, line := range strings.Split(out, "\n") {
		if strings.Contains(line, "only in") && strings.Contains(line, "×") {
			t.Errorf("a one-sided row printed a ratio: %s", line)
		}
	}
}

// TestSummedKernelVerdict checks when two arms' summed kernel times may be
// compared as a ratio. The ds4 row is the C and Go prefill captures of
// 2026-09-22.
func TestSummedKernelVerdict(t *testing.T) {
	const ms = 1_000_000
	arm := func(summed, active, span int64) armTotals {
		return armTotals{summedNanos: summed, activeNanos: active, spanNanos: span}
	}
	tests := []struct {
		name         string
		base, target armTotals
		wantOK       bool
		wantReason   string
	}{
		{
			name:       "ds4 c vs go prefill",
			base:       arm(2_739_210_000, 2_740_236_000, 2_748_589_000),
			target:     arm(338_690_000, 3_235_140_000, 3_410_950_000),
			wantReason: "kernel coverage differs by 89.49 pp (base 99.96%, target 10.47%)",
		},
		{name: "a/a", base: arm(10*ms, 10*ms, 12*ms), target: arm(10*ms, 10*ms, 12*ms), wantOK: true},
		{name: "faster and shorter", base: arm(10*ms, 10*ms, 12*ms), target: arm(8*ms, 8*ms, 10*ms), wantOK: true},
		{name: "summed moves, span flat", base: arm(10*ms, 10*ms, 12*ms), target: arm(8*ms, 8*ms, 12*ms), wantOK: true},
		{
			name:       "no journal",
			base:       armTotals{summedNanos: 10 * ms, unread: "no capture.journal member"},
			target:     arm(8*ms, 8*ms, 10*ms),
			wantReason: "base coverage UNREAD: no capture.journal member",
		},
		{
			name:       "sign disagreement at equal coverage",
			base:       arm(10*ms, 10*ms, 12*ms),
			target:     arm(8*ms, 8*ms, 14*ms),
			wantReason: "summed kernel time moved -2.000 ms but captured span moved +2.000 ms",
		},
		{name: "coverage skew 1.0 pp", base: arm(1_000, 1_000, 2_000), target: arm(990, 1_000, 2_000), wantOK: true},
		{
			name:       "coverage skew 1.1 pp",
			base:       arm(1_000, 1_000, 2_000),
			target:     arm(989, 1_000, 2_000),
			wantReason: "kernel coverage differs by 1.10 pp",
		},
		{
			name:       "zero active",
			base:       arm(10*ms, 10*ms, 12*ms),
			target:     arm(8*ms, 0, 0),
			wantReason: "target device-active time is zero",
		},
		{
			name:       "zero summed",
			base:       arm(0, 10*ms, 12*ms),
			target:     arm(8*ms, 8*ms, 10*ms),
			wantReason: "base recorded no kernel time",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ok, reason := summedKernelVerdict(tc.base, tc.target)
			if ok != tc.wantOK {
				t.Errorf("ok = %v, want %v (reason %q)", ok, tc.wantOK, reason)
			}
			if tc.wantOK && reason != "" {
				t.Errorf("reason = %q, want none", reason)
			}
			if !strings.Contains(reason, tc.wantReason) {
				t.Errorf("reason = %q, want it to contain %q", reason, tc.wantReason)
			}
		})
	}
}

// TestReadArmTotals reads each arm's device-active time and span from a gap
// report: active is summed across devices, span is the longest device's, an
// opaque graph capture says so, and a failed analysis is UNREAD.
func TestReadArmTotals(t *testing.T) {
	multi := newJournalBuilder()
	multi.kernelOn(0, 1, 1_000, 3_000, 1)
	multi.kernelOn(1, 2, 1_000, 2_000, 2)
	multi.kernelOn(1, 2, 9_000, 10_000, 3)
	opaque := newJournalBuilder()
	opaque.kernelOn(0, 1, 1_000, 2_000, 1)
	opaque.graphTrace(0, 3_000, 8_000, 5, 2)
	tests := []struct {
		name    string
		journal []byte
		want    armTotals
	}{
		{name: "two devices", journal: multi.finalize(), want: armTotals{summedNanos: 7, activeNanos: 4_000, spanNanos: 9_000}},
		{name: "opaque graphs", journal: opaque.finalize(), want: armTotals{summedNanos: 7, activeNanos: 6_000, spanNanos: 7_000, opaqueGraphs: true}},
		{name: "no journal", want: armTotals{summedNanos: 7, unread: "capture contains no capture.journal member"}},
	}
	p := &profile.Profile{SampleType: []*profile.ValueType{{Type: "kernel_time", Unit: "nanoseconds"}}, Sample: []*profile.Sample{{Value: []int64{7}}}}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			files := map[string][]byte{}
			if tc.journal != nil {
				files["capture.journal"] = tc.journal
			}
			r, err := analyzeGaps(testCapture(files, completeStatus()), gapOptions{maxRecords: controllerMaxJournalRecords})
			got := readArmTotals(p, r, err)
			if tc.want.unread != "" {
				if !strings.HasPrefix(got.unread, tc.want.unread) {
					t.Errorf("unread = %q, want prefix %q", got.unread, tc.want.unread)
				}
				got.unread = tc.want.unread
			}
			if got != tc.want {
				t.Errorf("readArmTotals = %+v, want %+v", got, tc.want)
			}
			var sb strings.Builder
			writeArmLine(&sb, "target", got)
			if line := sb.String(); strings.Contains(line, "graph replays were recorded as envelopes only") != tc.want.opaqueGraphs {
				t.Errorf("arm line opaque note = %v, want %v:\n%s", !tc.want.opaqueGraphs, tc.want.opaqueGraphs, line)
			}
		})
	}
}

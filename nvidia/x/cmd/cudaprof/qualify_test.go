package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// campaign builds qualify entries from per-block steady spans in ms, in run
// order C1, G1, G2, C2. The whole span is the steady span plus 10 ms.
func campaign(blocks ...[4]float64) []qualifyEntry {
	var entries []qualifyEntry
	for i, b := range blocks {
		for j, arm := range qualifyArms {
			ns := int64(math.Round(b[j] * 1e6))
			entries = append(entries, qualifyEntry{Block: fmt.Sprintf("B%d", i+1), Arm: arm, SteadyNanos: ns, WholeNanos: ns + 10e6})
		}
	}
	return entries
}

func TestQualifyVerdict(t *testing.T) {
	// ds4Min is the 0923 ds4-min qualification on GB10, steady spans read
	// from the captures by cudaprof gaps.
	ds4Min := campaign(
		[4]float64{2151.443, 2261.305, 2249.368, 2169.013},
		[4]float64{2154.578, 2254.325, 2252.962, 2168.365},
		[4]float64{2188.634, 2290.717, 2251.514, 2173.358},
		[4]float64{2154.571, 2259.494, 2267.928, 2157.263},
		[4]float64{2160.343, 2264.582, 2261.837, 2156.793},
	)
	tests := []struct {
		name       string
		entries    []qualifyEntry
		multiplier float64
		floor      float64
		means      []float64
		median     float64
		madC, madG float64
		noise      string
		sign       string
	}{
		{
			name:    "ds4-min",
			entries: ds4Min, multiplier: 3, floor: 1,
			means:  []float64{95.108, 92.172, 90.119, 107.794, 104.641},
			median: 95.108, madC: 11.095, madG: 9.193,
			noise: "PASS", sign: "PASS",
		},
		{
			name:    "ds4-min at 10x",
			entries: ds4Min, multiplier: 10, floor: 1,
			means:  []float64{95.108, 92.172, 90.119, 107.794, 104.641},
			median: 95.108, madC: 11.095, madG: 9.193,
			noise: "FAIL", sign: "PASS",
		},
		{
			// Identical repeats have zero MAD, so the floor decides: 2 ms is
			// under 3 x 1 ms.
			name: "floor decides",
			entries: campaign(
				[4]float64{100, 102, 102, 100},
				[4]float64{100, 102, 102, 100},
				[4]float64{100, 102, 102, 100},
			),
			multiplier: 3, floor: 1,
			means: []float64{2, 2, 2}, median: 2,
			noise: "FAIL", sign: "PASS",
		},
		{
			name: "clears the floor",
			entries: campaign(
				[4]float64{100, 104, 104, 100},
				[4]float64{100, 104, 104, 100},
				[4]float64{100, 104, 104, 100},
			),
			multiplier: 3, floor: 1,
			means: []float64{4, 4, 4}, median: 4,
			noise: "PASS", sign: "PASS",
		},
		{
			// G faster: the effect is negative and must be gated by its size.
			name: "negative effect",
			entries: campaign(
				[4]float64{100, 90, 90, 100},
				[4]float64{100, 91, 90, 101},
				[4]float64{101, 90, 91, 100},
			),
			multiplier: 3, floor: 1,
			means: []float64{-10, -10, -10}, median: -10, madC: 1, madG: 1,
			noise: "PASS", sign: "PASS",
		},
		{
			name: "one pair crosses zero",
			entries: campaign(
				[4]float64{100, 120, 120, 100},
				[4]float64{100, 120, 120, 100},
				[4]float64{100, 99, 120, 100},
			),
			multiplier: 3, floor: 1,
			means: []float64{20, 20, 9.5}, median: 20, madG: 0,
			noise: "PASS", sign: "FAIL",
		},
		{
			name: "one pair is zero",
			entries: campaign(
				[4]float64{100, 120, 120, 100},
				[4]float64{100, 100, 120, 100},
				[4]float64{100, 120, 120, 100},
			),
			multiplier: 3, floor: 1,
			means: []float64{20, 10, 20}, median: 20,
			noise: "PASS", sign: "FAIL",
		},
		{
			// A MAD, not a standard deviation: one wild C2 moves the
			// within-arm spread by nothing.
			name: "outlier does not set the floor",
			entries: campaign(
				[4]float64{100, 110, 110, 100},
				[4]float64{100, 110, 110, 101},
				[4]float64{100, 110, 110, 180},
			),
			multiplier: 3, floor: 1,
			means: []float64{10, 9.5, -30}, median: 9.5, madC: 1,
			noise: "PASS", sign: "FAIL",
		},
		{
			// Noisy arms: each arm repeats within ±10 ms, so a 10 ms effect
			// does not clear 3 x 10 ms.
			name: "noise decides",
			entries: campaign(
				[4]float64{100, 110, 120, 110},
				[4]float64{110, 120, 110, 100},
				[4]float64{100, 110, 110, 100},
			),
			multiplier: 3, floor: 1,
			means: []float64{10, 10, 10}, median: 10, madC: 10, madG: 10,
			noise: "FAIL", sign: "PASS",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			q := qualify(tt.entries, tt.multiplier, tt.floor, 0)
			if len(q.Blocks) != len(tt.means) {
				t.Fatalf("got %d blocks, want %d", len(q.Blocks), len(tt.means))
			}
			for i, b := range q.Blocks {
				if !near(b.Steady.MeanMS, tt.means[i]) {
					t.Errorf("block %s mean = %.3f, want %.3f", b.Block, b.Steady.MeanMS, tt.means[i])
				}
			}
			if !near(q.Steady.MedianMS, tt.median) {
				t.Errorf("median = %.3f, want %.3f", q.Steady.MedianMS, tt.median)
			}
			if !near(q.Steady.MADCMS, tt.madC) || !near(q.Steady.MADGMS, tt.madG) {
				t.Errorf("MAD C, G = %.3f, %.3f, want %.3f, %.3f", q.Steady.MADCMS, q.Steady.MADGMS, tt.madC, tt.madG)
			}
			if q.NoiseGate != tt.noise || q.SignGate != tt.sign {
				t.Errorf("noise, sign = %s, %s, want %s, %s", q.NoiseGate, q.SignGate, tt.noise, tt.sign)
			}
			if want := gate(tt.noise == "PASS" && tt.sign == "PASS"); q.Verdict != want {
				t.Errorf("verdict = %s, want %s", q.Verdict, want)
			}
			// The whole span is every steady span plus the same 10 ms, so it
			// reads the same differences and must not be what was gated.
			if !near(q.Whole.MedianMS, q.Steady.MedianMS) {
				t.Errorf("whole median = %.3f, want %.3f", q.Whole.MedianMS, q.Steady.MedianMS)
			}
		})
	}
}

// TestQualifyDS4MinSummary pins the reported summary of the ds4-min
// campaign, which was computed independently from gaps --json span_ns.
func TestQualifyDS4MinSummary(t *testing.T) {
	q := qualify(campaign(
		[4]float64{2151.443, 2261.305, 2249.368, 2169.013},
		[4]float64{2154.578, 2254.325, 2252.962, 2168.365},
		[4]float64{2188.634, 2290.717, 2251.514, 2173.358},
		[4]float64{2154.571, 2259.494, 2267.928, 2157.263},
		[4]float64{2160.343, 2264.582, 2261.837, 2156.793},
	), 3, 1, 0)
	for _, c := range []struct {
		name      string
		got, want float64
	}{
		{"pair min", q.Steady.PairMinMS, 78.156},
		{"pair max", q.Steady.PairMaxMS, 110.665},
		{"noise floor", q.NoiseFloorMS, 11.095},
		{"noise threshold", q.NoiseThresholdMS, 33.284},
	} {
		if !near(c.got, c.want) {
			t.Errorf("%s = %.3f, want %.3f", c.name, c.got, c.want)
		}
	}
}

func near(a, b float64) bool { return math.Abs(a-b) < 0.002 }

func TestParseQualifyManifest(t *testing.T) {
	full := "B1 C1 a\nB1 G1 b\nB1 G2 c\nB1 C2 d\n"
	tests := []struct {
		name    string
		text    string
		wantErr string
	}{
		{"complete", full, ""},
		{"any order", "B1 C2 d\n# note\n\nB1 G2 c\nB1 G1 b\nB1 C1 a\n", ""},
		{"empty", "# nothing\n", "no captures"},
		{"missing arm", "B1 C1 a\nB1 G1 b\nB1 G2 c\n", "block B1 has no arm C2"},
		{"duplicate arm", full + "B1 G1 e\n", "names arm G1 twice"},
		{"unknown arm", full + "B2 G3 e\n", `arm "G3"`},
		{"short line", full + "B2 C1\n", "want \"block arm path\""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			entries, err := parseQualifyManifest(strings.NewReader(tt.text), "/m")
			if tt.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
					t.Fatalf("err = %v, want %q", err, tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			var got []string
			for _, e := range entries {
				got = append(got, e.Arm+"="+e.Path)
			}
			if want := "C1=/m/a G1=/m/b G2=/m/c C2=/m/d"; strings.Join(got, " ") != want {
				t.Errorf("entries = %s, want %s", strings.Join(got, " "), want)
			}
		})
	}
}

func TestParseQualifyArgs(t *testing.T) {
	tests := []struct {
		name    string
		args    []string
		wantErr string
	}{
		{"until", []string{"-warmup-until=k", "m.txt"}, ""},
		{"ms", []string{"--warmup-ms=0", "m.txt"}, ""},
		{"no window", []string{"m.txt"}, "must be named"},
		{"both windows", []string{"-warmup-ms=5", "-warmup-until=k", "m.txt"}, "exclusive"},
		{"bad pattern", []string{"-warmup-until=(", "m.txt"}, "warmup-until"},
		{"no manifest", []string{"-warmup-until=k"}, "usage"},
		{"zero multiplier", []string{"-warmup-until=k", "-noise-multiplier=0", "m.txt"}, "positive"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := parseQualifyArgs(tt.args, nil)
			if tt.wantErr == "" {
				if err != nil {
					t.Fatal(err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("err = %v, want %q", err, tt.wantErr)
			}
		})
	}
}

// qualifyJournal is a run with a 1000 ns prologue kernel, then the anchor
// kernel and span ns of work after it.
func qualifyJournal(span uint64) []byte {
	b := newJournalBuilder()
	b.namedKernel("load_weights", 10_000, 11_000, 1)
	b.namedKernel("hc_expand", 15_000, 16_000, 2)
	b.namedKernel("matmul", 16_000, 15_000+span, 3)
	return b.finalize()
}

func TestQualifyCLI(t *testing.T) {
	dir := t.TempDir()
	// Block steady spans: C 1 ms, G 1.5 ms, so every pair is +0.5 ms.
	spans := map[string]uint64{"C1": 1e6, "G1": 1.5e6, "G2": 1.5e6, "C2": 1e6}
	var manifest strings.Builder
	for _, block := range []string{"B1", "B2"} {
		for _, arm := range qualifyArms {
			name := block + "-" + arm + ".pprof"
			writeTestCaptureFile(t, filepath.Join(dir, name), testCapture(map[string][]byte{"capture.journal": qualifyJournal(spans[arm])}, completeStatus()))
			fmt.Fprintf(&manifest, "%s %s %s\n", block, arm, name)
		}
	}
	mpath := filepath.Join(dir, "manifest.txt")
	if err := os.WriteFile(mpath, []byte(manifest.String()), 0o644); err != nil {
		t.Fatal(err)
	}

	var out strings.Builder
	err := runQualifyCommand([]string{"-warmup-until=hc_expand", "-noise-floor-ms=0.1", "-json", mpath}, &out)
	if err != nil {
		t.Fatalf("qualify: %v\n%s", err, out.String())
	}
	var q Qualification
	if err := json.Unmarshal([]byte(out.String()), &q); err != nil {
		t.Fatal(err)
	}
	// The steady window starts at hc_expand, so the prologue's 5 us drop out
	// of the steady span and stay in the whole one.
	if c := q.Captures[0]; c.SteadyNanos != 1e6 || c.WholeNanos != 1e6+5_000 {
		t.Errorf("C1 steady, whole = %d, %d ns, want 1000000, 1005000", c.SteadyNanos, c.WholeNanos)
	}
	if q.Cudaprof.Revision == "" || q.Cudaprof.GoVersion != runtime.Version() {
		t.Errorf("cudaprof = %+v, want this build", q.Cudaprof)
	}
	if !near(q.Steady.MedianMS, 0.5) || q.Verdict != "PASS" {
		t.Errorf("median %.3f ms verdict %s, want 0.500 PASS", q.Steady.MedianMS, q.Verdict)
	}

	// These captures predate the identity and range records.
	if q.Identity == "verified" || q.Order == "verified" {
		t.Errorf("identity %q order %q, want both unverified", q.Identity, q.Order)
	}

	// The default 1 ms floor sets a 3 ms threshold that 0.5 ms cannot
	// clear: the verdict fails, and the exit status says so.
	out.Reset()
	err = runQualifyCommand([]string{"-warmup-until=hc_expand", mpath}, &out)
	var exit *exitCodeError
	if !errors.As(err, &exit) || exit.Code != 2 {
		t.Fatalf("err = %v, want exit status 2", err)
	}
	if !strings.Contains(out.String(), "noise gate             FAIL") {
		t.Errorf("text report does not show the failed gate:\n%s", out.String())
	}

	// An unmatched pattern leaves no window: refused, not scored over the
	// whole timeline.
	out.Reset()
	err = runQualifyCommand([]string{"-warmup-until=hc_cached_expand", mpath}, &out)
	if err == nil || !strings.Contains(err.Error(), "no steady-state window") {
		t.Fatalf("err = %v, want a refusal naming the missing window", err)
	}
	if errors.As(err, &exit) {
		t.Errorf("refusal exits %d, the status of a failed verdict", exit.Code)
	}

	// A G capture that ran another executable is refused before scoring.
	status := completeStatus()
	status["target_environment"] = map[string]any{"executable": map[string]any{"path": "/bin/other", "sha256": "bb"}}
	writeTestCaptureFile(t, filepath.Join(dir, "B2-G2.pprof"), testCapture(map[string][]byte{"capture.journal": qualifyJournal(spans["G2"])}, status))
	status = completeStatus()
	status["target_environment"] = map[string]any{"executable": map[string]any{"path": "/bin/g", "sha256": "aa"}}
	writeTestCaptureFile(t, filepath.Join(dir, "B1-G1.pprof"), testCapture(map[string][]byte{"capture.journal": qualifyJournal(spans["G1"])}, status))
	out.Reset()
	err = runQualifyCommand([]string{"-warmup-until=hc_expand", mpath}, &out)
	if err == nil || !strings.Contains(err.Error(), "arm G ran two executables") {
		t.Fatalf("err = %v, want a refusal naming arm G", err)
	}
}

func TestCheckQualifyRuns(t *testing.T) {
	// entries builds one block per element of exe and start, in run order.
	entries := func(exe [][4]string, start [][4]uint64) []qualifyEntry {
		var out []qualifyEntry
		for i := range exe {
			for j, arm := range qualifyArms {
				out = append(out, qualifyEntry{Block: fmt.Sprintf("B%d", i+1), Arm: arm, Executable: exe[i][j], Start: start[i][j]})
			}
		}
		return out
	}
	cg := [4]string{"c", "g", "g", "c"}
	inOrder := [4]uint64{10, 20, 30, 40}
	tests := []struct {
		name            string
		entries         []qualifyEntry
		identity, order string
		err             string
	}{
		{"verified", entries([][4]string{cg, cg}, [][4]uint64{inOrder, {1, 2, 3, 4}}), "verified", "verified", ""},
		{"one binary for both arms", entries([][4]string{{"x", "x", "x", "x"}}, [][4]uint64{inOrder}), "verified", "verified", ""},
		{"no records", entries([][4]string{{}}, [][4]uint64{{}}), "unverified", "unverified", ""},
		{"arm C changed", entries([][4]string{cg, {"c2", "g", "g", "c"}}, [][4]uint64{inOrder, inOrder}), "", "", "arm C ran two executables"},
		{"arm G changed within a block", entries([][4]string{{"c", "g", "g2", "c"}}, [][4]uint64{inOrder}), "", "", "arm G ran two executables"},
		{"G2 ran before G1", entries([][4]string{cg}, [][4]uint64{{10, 30, 20, 40}}), "", "", "block B1 ran out of manifest order: G2"},
		{"partial start records", entries([][4]string{cg}, [][4]uint64{{10, 0, 30, 40}}), "verified", "unverified", ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			identity, order, err := checkQualifyRuns(tt.entries)
			if tt.err != "" {
				if err == nil || !strings.Contains(err.Error(), tt.err) {
					t.Fatalf("err = %v, want %q", err, tt.err)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if !strings.HasPrefix(identity, tt.identity) || !strings.HasPrefix(order, tt.order) {
				t.Errorf("identity %q order %q, want %s and %s", identity, order, tt.identity, tt.order)
			}
		})
	}
}

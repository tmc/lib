package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"
)

func TestParseHostLog(t *testing.T) {
	const (
		good = "2026-09-24T03:12:07.5Z\t41822\tpython3\t327.0\t0\tds4-qualify\n"
		beat = "2026-09-24T03:12:08Z heartbeat ds4-qualify\n"
	)
	tests := []struct {
		name       string
		text       string
		violations int
		heartbeats int
		wantErr    string
	}{
		{"tabs", good, 1, 0, ""},
		{"spaces and comments", "# watcher v1\n\n2026-09-24T03:12:07-07:00 7 go.test 12.5 1536 -\n" + good, 2, 0, ""},
		{"heartbeats", beat + good + "2026-09-24T03:12:09Z heartbeat -\n", 1, 2, ""},
		{"empty", "", 0, 0, ""},
		{"short line", good + "2026-09-24T03:12:08Z 41822 python3 327.0 0\n", 0, 0, "line 2: want"},
		{"comm with a space", "2026-09-24T03:12:08Z 41822 tmux: server 1.0 0 -\n", 0, 0, "line 1: want"},
		{"heartbeat without holder", beat + "2026-09-24T03:12:09Z heartbeat\n", 0, 0, "line 2: want"},
		{"heartbeat with extra field", "2026-09-24T03:12:09Z heartbeat ds4-qualify 1\n", 0, 0, "line 1: want"},
		{"three fields, not a heartbeat", "2026-09-24T03:12:09Z heartbeats -\n", 0, 0, "line 1: want"},
		{"heartbeat bad time", "\n\nnow heartbeat -\n", 0, 0, "line 3: time"},
		{"bad time", "\n2026-09-24 41822 python3 327.0 0 -\n", 0, 0, "line 2: time"},
		{"zero pid", "2026-09-24T03:12:08Z 0 python3 327.0 0 -\n", 0, 0, "line 1: invalid pid"},
		{"negative cpu", "2026-09-24T03:12:08Z 1 python3 -1 0 -\n", 0, 0, "line 1: invalid cpu"},
		{"NaN cpu", "2026-09-24T03:12:08Z 1 python3 NaN 0 -\n", 0, 0, "line 1: invalid cpu"},
		{"bad gpu", "2026-09-24T03:12:08Z 1 python3 1 lots -\n", 0, 0, "line 1: invalid gpu"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			log, err := parseHostLog(strings.NewReader(tt.text))
			if tt.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
					t.Fatalf("err = %v, want %q", err, tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if len(log.Violations) != tt.violations || len(log.Heartbeats) != tt.heartbeats {
				t.Fatalf("got %d violations, %d heartbeats, want %d, %d", len(log.Violations), len(log.Heartbeats), tt.violations, tt.heartbeats)
			}
		})
	}
	log, err := parseHostLog(strings.NewReader(good))
	if err != nil {
		t.Fatal(err)
	}
	want := hostViolation{Time: time.Date(2026, 9, 24, 3, 12, 7, 5e8, time.UTC), PID: 41822, Comm: "python3", CPUPct: 327, GPUMiB: 0, Holder: "ds4-qualify"}
	if v := log.Violations[0]; !v.Time.Equal(want.Time) || v.PID != want.PID || v.Comm != want.Comm || v.CPUPct != want.CPUPct || v.GPUMiB != want.GPUMiB || v.Holder != want.Holder {
		t.Errorf("violation = %+v, want %+v", v, want)
	}
}

// heartbeats returns a heartbeat every step from first to last inclusive,
// leaving out those in skip.
func heartbeats(first, last time.Time, step time.Duration, skip ...time.Time) []time.Time {
	var hb []time.Time
	for t := first; !t.After(last); t = t.Add(step) {
		if !slices.ContainsFunc(skip, t.Equal) {
			hb = append(hb, t)
		}
	}
	return hb
}

func TestHostLogUncovered(t *testing.T) {
	// The widened window is [lo, hi]; the margin is also the longest
	// allowed gap.
	lo := time.Date(2026, 9, 24, 3, 0, 0, 0, time.UTC)
	hi := lo.Add(10 * time.Second)
	const margin = 2 * time.Second
	s := func(n int) time.Time { return lo.Add(time.Duration(n) * time.Second) }
	tests := []struct {
		name string
		hb   []time.Time
		want string // "" when covered
	}{
		{"covered", heartbeats(s(-1), s(11), time.Second), ""},
		{"heartbeats on the edges", heartbeats(s(0), s(10), time.Second), ""},
		{"gap of exactly the margin", heartbeats(s(0), s(10), 2*time.Second), ""},
		{"gap outside the span", append([]time.Time{s(-30)}, heartbeats(s(-1), s(11), time.Second)...), ""},
		{"gap inside", heartbeats(s(-1), s(11), time.Second, s(4), s(5)), "watcher did not cover 2026-09-24T03:00:03Z to 2026-09-24T03:00:06Z: 3s between heartbeats"},
		{"none before the start", heartbeats(s(1), s(11), time.Second), "watcher did not cover the window start: no heartbeat at or before 2026-09-24T03:00:00Z"},
		{"none after the end", heartbeats(s(-1), s(9), time.Second), "watcher did not cover the window end: no heartbeat at or after 2026-09-24T03:00:10Z"},
		{"no heartbeats", nil, "window start"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := hostLog{Heartbeats: tt.hb}.uncovered(lo, hi, margin)
			if tt.want == "" && got != "" || tt.want != "" && !strings.Contains(got, tt.want) {
				t.Errorf("uncovered = %q, want %q", got, tt.want)
			}
		})
	}
	if got := (hostLog{Heartbeats: heartbeats(s(0), s(10), 500*time.Millisecond)}).heartbeatInterval(); got != 500*time.Millisecond {
		t.Errorf("heartbeat interval = %s, want 500ms", got)
	}
}

// TestQualifyHostLoad runs a campaign whose B2 C1 ran 60 ms slow, as a
// loaded arm does. Scored, it flips the sign gate; excluded, it cannot.
func TestQualifyHostLoad(t *testing.T) {
	base := time.Date(2026, 9, 24, 3, 0, 0, 0, time.UTC)
	const margin = 2 * time.Second
	// Entry i ran [start(i), start(i)+5s]; entry 4 is B2 C1.
	start := func(i int) time.Time { return base.Add(time.Duration(i) * 10 * time.Second) }
	stop := func(i int) time.Time { return start(i).Add(5 * time.Second) }
	violation := func(at time.Time) hostViolation {
		return hostViolation{Time: at, PID: 41822, Comm: "python3", CPUPct: 327, Holder: "ds4-qualify"}
	}
	tests := []struct {
		name      string
		log       []hostViolation // nil: no log given
		noBeats   bool            // the log has no heartbeats
		unchecked []int
		minBlocks int

		verdict string
		dropped string
		loaded  string
		pairMin float64
	}{
		{
			name:    "no log",
			verdict: verdictFail, pairMin: -50,
		},
		{
			name:    "clean log",
			log:     []hostViolation{violation(stop(11).Add(time.Hour))},
			verdict: verdictFail, pairMin: -50,
		},
		{
			name:      "inside the outlier",
			log:       []hostViolation{violation(start(4).Add(time.Second))},
			minBlocks: 2,
			verdict:   verdictPass, dropped: "B2", loaded: "B2 C1", pairMin: 10,
		},
		{
			name:      "at the margin",
			log:       []hostViolation{violation(stop(4).Add(margin))},
			minBlocks: 2,
			verdict:   verdictPass, dropped: "B2", loaded: "B2 C1", pairMin: 10,
		},
		{
			name:      "just outside the margin",
			log:       []hostViolation{violation(stop(4).Add(margin + time.Nanosecond))},
			minBlocks: 2,
			verdict:   verdictFail, pairMin: -50,
		},
		{
			name:      "before the margin",
			log:       []hostViolation{violation(start(4).Add(-margin - time.Nanosecond))},
			minBlocks: 2,
			verdict:   verdictFail, pairMin: -50,
		},
		{
			// Unplaceable is excluded, not passed.
			name:      "unchecked arm",
			log:       []hostViolation{},
			unchecked: []int{4},
			minBlocks: 2,
			verdict:   verdictPass, dropped: "B2", pairMin: 10,
		},
		{
			// Violations the watcher saw, but no heartbeats saying when it
			// was looking: nothing is checked, so nothing is scored.
			name:      "violations without heartbeats",
			log:       []hostViolation{violation(start(4).Add(time.Second))},
			noBeats:   true,
			minBlocks: 1,
			verdict:   verdictRefused, dropped: "B1 B2 B3",
		},
		{
			name:    "every block required",
			log:     []hostViolation{violation(start(4).Add(time.Second))},
			verdict: verdictRefused, dropped: "B2", loaded: "B2 C1",
		},
		{
			name:      "too many exclusions",
			log:       []hostViolation{violation(start(1)), violation(start(4))},
			unchecked: []int{10},
			minBlocks: 2,
			verdict:   verdictRefused, dropped: "B1 B2 B3", loaded: "B1 G1 B2 C1",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			entries := campaign(
				[4]float64{100, 110, 110, 100},
				[4]float64{160, 110, 110, 100},
				[4]float64{100, 110, 110, 100},
			)
			for i := range entries {
				entries[i].Window = &qualifyWindow{Start: start(i), Stop: stop(i)}
			}
			for _, i := range tt.unchecked {
				entries[i].Window, entries[i].Unchecked = nil, "capture records no wall clock"
			}
			if tt.log != nil {
				log := hostLog{Violations: tt.log}
				if !tt.noBeats {
					log.Heartbeats = heartbeats(start(0).Add(-time.Minute), stop(11).Add(time.Minute), time.Second)
				}
				h := markHostLoad(entries, log, margin)
				if tt.noBeats && h.Unchecked != len(entries) {
					t.Errorf("%d arms unchecked, want all %d", h.Unchecked, len(entries))
				}
			}
			q := qualify(entries, 3, 1, tt.minBlocks)
			if q.Verdict != tt.verdict {
				t.Fatalf("verdict = %s (%s), want %s", q.Verdict, q.Reason, tt.verdict)
			}
			if got := strings.Join(q.DroppedBlocks, " "); got != tt.dropped {
				t.Errorf("dropped = %q, want %q", got, tt.dropped)
			}
			var loaded []string
			for _, e := range q.Captures {
				if e.HostLoad == hostLoaded {
					loaded = append(loaded, e.Block+" "+e.Arm)
					if e.Violations[0].PID != 41822 {
						t.Errorf("%s %s names pid %d, want 41822", e.Block, e.Arm, e.Violations[0].PID)
					}
				}
			}
			if got := strings.Join(loaded, " "); got != tt.loaded {
				t.Errorf("loaded = %q, want %q", got, tt.loaded)
			}
			if q.Verdict == verdictRefused {
				if len(q.Blocks) >= max(q.MinBlocks, 1) || q.Reason == "" {
					t.Errorf("refused with %d blocks of %d needed, reason %q", len(q.Blocks), q.MinBlocks, q.Reason)
				}
				return
			}
			// The exclusion must change what is scored, not only what is
			// reported: the outlier's -50 ms pair is gone.
			if !near(q.Steady.PairMinMS, tt.pairMin) {
				t.Errorf("pair min = %.3f, want %.3f", q.Steady.PairMinMS, tt.pairMin)
			}
			for _, b := range q.Blocks {
				if strings.Contains(tt.dropped, b.Block) {
					t.Errorf("dropped block %s was scored", b.Block)
				}
			}
		})
	}
}

// writeHostLoadCampaign writes a two-block campaign whose C arms span 1 ms
// and G arms 1.5 ms. Capture i's range runs from 1 s to 3 s on its wall
// clock, which reads base+10i s; a capture named in noClock records none.
func writeHostLoadCampaign(t *testing.T, base time.Time, noClock ...string) string {
	t.Helper()
	dir := t.TempDir()
	spans := map[string]uint64{"C1": 1e6, "G1": 1.5e6, "G2": 1.5e6, "C2": 1e6}
	var manifest strings.Builder
	i := 0
	for _, block := range []string{"B1", "B2"} {
		for _, arm := range qualifyArms {
			status := completeStatus()
			status["phase_receipt"] = map[string]any{
				"status": "MEASURED",
				"edges": []map[string]any{
					{"sequence": 5, "timestamp": 1e9, "edge": "start", "api": "cuProfilerStart"},
					{"sequence": 9, "timestamp": 3e9, "edge": "stop", "api": "cuProfilerStop"},
				},
			}
			if name := block + " " + arm; !strings.Contains(strings.Join(noClock, ","), name) {
				status["wall_clock"] = map[string]any{"monotonic_nanos": 0, "unix_nanos": base.Add(time.Duration(i) * 10 * time.Second).UnixNano()}
			}
			name := block + "-" + arm + ".pprof"
			writeTestCaptureFile(t, filepath.Join(dir, name), testCapture(map[string][]byte{"capture.journal": qualifyJournal(spans[arm])}, status))
			fmt.Fprintf(&manifest, "%s %s %s\n", block, arm, name)
			i++
		}
	}
	mpath := filepath.Join(dir, "manifest.txt")
	if err := os.WriteFile(mpath, []byte(manifest.String()), 0o644); err != nil {
		t.Fatal(err)
	}
	return mpath
}

func TestQualifyCLIHostLoad(t *testing.T) {
	base := time.Date(2026, 9, 24, 3, 0, 0, 0, time.UTC)
	mpath := writeHostLoadCampaign(t, base)
	logs := t.TempDir()
	writeLog := func(name, text string) string {
		path := filepath.Join(logs, name)
		if err := os.WriteFile(path, []byte(text), 0o644); err != nil {
			t.Fatal(err)
		}
		return path
	}
	// The captures' widened windows span base-1s to base+75s.
	var beats strings.Builder
	for _, hb := range heartbeats(base.Add(-10*time.Second), base.Add(80*time.Second), time.Second) {
		fmt.Fprintf(&beats, "%s heartbeat -\n", hb.Format(time.RFC3339))
	}
	clean := writeLog("clean.log", "# no violations\n"+beats.String())
	// B1 G1 is capture 1: its range is base+11s to base+13s.
	loaded := writeLog("loaded.log", beats.String()+base.Add(12*time.Second).Format(time.RFC3339)+"\t41822\tpython3\t327.0\t0\tds4-qualify\n")
	malformed := writeLog("malformed.log", "# header\n2026-09-24T03:00:12Z 41822 python3\n")
	args := []string{"-warmup-until=hc_expand", "-noise-floor-ms=0.1"}

	run := func(extra ...string) (string, error) {
		var out strings.Builder
		err := runQualifyCommand(append(append([]string(nil), args...), append(extra, mpath)...), &out)
		return out.String(), err
	}

	// Without a log the report says host load went unchecked.
	out, err := run()
	if err != nil {
		t.Fatalf("qualify: %v\n%s", err, out)
	}
	if !strings.Contains(out, "host load: not checked") {
		t.Errorf("report does not say host load was not checked:\n%s", out)
	}

	// A clean log leaves the verdict alone and says it checked.
	out, err = run("-violations="+clean, "-json")
	if err != nil {
		t.Fatalf("qualify: %v\n%s", err, out)
	}
	var q Qualification
	if err := json.Unmarshal([]byte(out), &q); err != nil {
		t.Fatal(err)
	}
	if q.Verdict != verdictPass || !q.HostLoad.Checked || q.HostLoad.Clean != 8 || len(q.Blocks) != 2 || q.HostLoad.HeartbeatIntervalMS != 1000 {
		t.Errorf("clean log: verdict %s, host load %+v, %d blocks; want PASS, 8 clean arms, 1000 ms heartbeats, 2 blocks", q.Verdict, q.HostLoad, len(q.Blocks))
	}
	if w := q.Captures[1].Window; w == nil || !w.Start.Equal(base.Add(11*time.Second)) || !w.Stop.Equal(base.Add(13*time.Second)) {
		t.Errorf("B1 G1 window = %+v, want base+11s to base+13s", w)
	}

	// A violation in B1 G1 drops B1, and names the process.
	out, err = run("-violations="+loaded, "-min-blocks=1", "-json")
	if err != nil {
		t.Fatalf("qualify: %v\n%s", err, out)
	}
	q = Qualification{}
	if err := json.Unmarshal([]byte(out), &q); err != nil {
		t.Fatal(err)
	}
	g1 := q.Captures[1]
	if q.Verdict != verdictPass || strings.Join(q.DroppedBlocks, " ") != "B1" || g1.HostLoad != hostLoaded || len(g1.Violations) != 1 || g1.Violations[0].Comm != "python3" {
		t.Errorf("loaded log: verdict %s, dropped %v, B1 G1 %s %+v", q.Verdict, q.DroppedBlocks, g1.HostLoad, g1.Violations)
	}
	out, _ = run("-violations="+loaded, "-min-blocks=1")
	if want := "B1 G1 excluded: 2026-09-24T03:00:12Z pid 41822 python3 327.0% CPU 0 MiB GPU, lock holder ds4-qualify"; !strings.Contains(out, want) {
		t.Errorf("text report does not name the violation %q:\n%s", want, out)
	}

	// With every block required, the same log refuses the verdict: exit
	// status 1, not the 2 of a failed one.
	out, err = run("-violations=" + loaded)
	var exit *exitCodeError
	if err == nil || !strings.Contains(err.Error(), "REFUSED") || errors.As(err, &exit) {
		t.Fatalf("err = %v, want a REFUSED verdict that is not exit status 2", err)
	}
	if !strings.Contains(out, "verdict                REFUSED: 1 of 2 blocks remain") {
		t.Errorf("text report does not show the refusal:\n%s", out)
	}

	_, err = run("-violations=" + malformed)
	if err == nil || !strings.Contains(err.Error(), "line 2:") {
		t.Errorf("err = %v, want a malformed-line error naming line 2", err)
	}

	// A capture with no wall clock cannot be checked, so it is not passed.
	mpath = writeHostLoadCampaign(t, base, "B2 C1")
	out, err = run("-violations=" + clean)
	if err == nil || !strings.Contains(err.Error(), "REFUSED") {
		t.Fatalf("err = %v, want REFUSED", err)
	}
	if !strings.Contains(out, "B2 C1 excluded, cannot check: capture records no wall clock") {
		t.Errorf("text report does not name the unchecked arm:\n%s", out)
	}

	for _, bad := range [][]string{{"-min-blocks=1"}, {"-violation-margin=1s"}, {"-violations=" + clean, "-min-blocks=3"}} {
		if _, err := run(bad...); err == nil {
			t.Errorf("qualify %v: no error", bad)
		}
	}
}

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

// breakdownJournal is a run with a 1 ms prologue kernel, then the anchor and
// attn and ffn kernels of the given durations, back to back.
func breakdownJournal(attn, ffn uint64) []byte {
	b := newJournalBuilder()
	b.namedKernel("load_weights", 10_000, 1_010_000, 1)
	at := uint64(2_000_000)
	b.namedKernel("hc_expand", at, at+1e6, 2)
	at += 1e6
	b.namedKernel("attn_kernel", at, at+attn, 3)
	at += attn
	b.namedKernel("ffn_kernel", at, at+ffn, 4)
	return b.finalize()
}

// breakdownBase is the wall time of writeBreakdownCampaign's first capture.
var breakdownBase = time.Date(2026, 9, 24, 3, 0, 0, 0, time.UTC)

// writeBreakdownCampaign writes a campaign of len(dAttn)/2 blocks in which C
// always runs attn 1 ms and ffn 2 ms and G runs them longer by the pair's
// dAttn and dFFN, in pair order G1-C1, G2-C2 of each block. Capture i's
// range runs from breakdownBase+10i+1 s to +3 s in wall time. A capture for
// which executable returns a nonempty SHA-256 records it; executable may be
// nil. It returns the manifest's path.
func writeBreakdownCampaign(t *testing.T, dAttn, dFFN []int64, executable func(block, arm string) string) string {
	t.Helper()
	dir := t.TempDir()
	var manifest strings.Builder
	i := 0
	for b := 0; b < len(dAttn)/2; b++ {
		block := fmt.Sprintf("B%d", b+1)
		durs := map[string][2]int64{
			"C1": {1e6, 2e6},
			"G1": {1e6 + dAttn[2*b], 2e6 + dFFN[2*b]},
			"G2": {1e6 + dAttn[2*b+1], 2e6 + dFFN[2*b+1]},
			"C2": {1e6, 2e6},
		}
		for _, arm := range qualifyArms {
			name := block + "-" + arm + ".pprof"
			d := durs[arm]
			j := breakdownJournal(uint64(d[0]), uint64(d[1]))
			status := completeStatus()
			// The receipt records edges but no measured range, so compare
			// takes the whole capture span, as it did before.
			status["phase_receipt"] = map[string]any{
				"edges": []map[string]any{
					{"sequence": 5, "timestamp": 1e9, "edge": "start", "api": "cuProfilerStart"},
					{"sequence": 9, "timestamp": 3e9, "edge": "stop", "api": "cuProfilerStop"},
				},
			}
			status["wall_clock"] = map[string]any{"monotonic_nanos": 0, "unix_nanos": breakdownBase.Add(time.Duration(i) * 10 * time.Second).UnixNano()}
			if executable != nil {
				if sum := executable(block, arm); sum != "" {
					status["target_environment"] = map[string]any{"executable": map[string]any{"path": "/bin/" + arm[:1], "sha256": sum}}
				}
			}
			writeTestCaptureFile(t, filepath.Join(dir, name), testCapture(map[string][]byte{"capture.journal": j}, status))
			fmt.Fprintf(&manifest, "%s %s %s\n", block, arm, name)
			i++
		}
	}
	for name, text := range map[string]string{
		"manifest.txt": manifest.String(),
		"ops.txt":      "^attn attn\n^ffn ffn\n^hc_ anchor\n",
		"stages.txt":   "^hc_ pre\n^ffn mlp\n",
	} {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(text), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return filepath.Join(dir, "manifest.txt")
}

func TestQualifyBreakdown(t *testing.T) {
	// Six pairs. attn's deltas are 0.1 .. 0.6 ms; ffn's are zero except two
	// pairs 1 ms faster, so its median is 0 while the kernel total's is
	// 0.25 ms, not 0.35 + 0: row medians do not add.
	mpath := writeBreakdownCampaign(t,
		[]int64{0.1e6, 0.2e6, 0.3e6, 0.4e6, 0.5e6, 0.6e6},
		[]int64{-1e6, 0, 0, 0, 0, -1e6},
		nil,
	)
	dir := filepath.Dir(mpath)
	var out strings.Builder
	err := runQualifyCommand([]string{"-warmup-until=hc_expand", "-json",
		"-ops", filepath.Join(dir, "ops.txt"), "-stages", filepath.Join(dir, "stages.txt"), mpath}, &out)
	var exit *exitCodeError
	if err != nil && !errors.As(err, &exit) {
		t.Fatalf("qualify: %v", err)
	}
	var q Qualification
	if err := json.Unmarshal([]byte(out.String()), &q); err != nil {
		t.Fatal(err)
	}
	bd := q.Breakdown
	if bd == nil || bd.Pairs != 6 || !strings.Contains(bd.Note, "do not sum") {
		t.Fatalf("breakdown = %+v, want 6 pairs and the note that medians do not sum", bd)
	}
	rows := map[string]qualifyRow{"kernel exclusive": bd.Kernel}
	for _, r := range append(bd.Ops, bd.Stages...) {
		rows[r.Name] = r
	}
	tests := []struct {
		name                    string
		c, g, delta, dmin, dmax float64
		launchesC, launchesG    [2]int
	}{
		{"kernel exclusive", 4, 4.25, 0.25, -0.9, 0.5, [2]int{3, 3}, [2]int{3, 3}},
		{"attn", 1, 1.35, 0.35, 0.1, 0.6, [2]int{1, 1}, [2]int{1, 1}},
		{"ffn", 2, 2, 0, -1, 0, [2]int{1, 1}, [2]int{1, 1}},
		{"anchor", 1, 1, 0, 0, 0, [2]int{1, 1}, [2]int{1, 1}},
		{"pre", 2, 2.35, 0.35, 0.1, 0.6, [2]int{2, 2}, [2]int{2, 2}},
		{"mlp", 2, 2, 0, -1, 0, [2]int{1, 1}, [2]int{1, 1}},
	}
	for _, tt := range tests {
		r, ok := rows[tt.name]
		if !ok {
			t.Errorf("no row %s in %+v", tt.name, bd)
			continue
		}
		if !near(r.BaseMS, tt.c) || !near(r.TargetMS, tt.g) || !near(r.DeltaMS, tt.delta) || !near(r.DeltaMinMS, tt.dmin) || !near(r.DeltaMaxMS, tt.dmax) {
			t.Errorf("%s = C %.3f G %.3f delta %.3f [%.3f, %.3f], want %.3f %.3f %.3f [%.3f, %.3f]",
				tt.name, r.BaseMS, r.TargetMS, r.DeltaMS, r.DeltaMinMS, r.DeltaMaxMS, tt.c, tt.g, tt.delta, tt.dmin, tt.dmax)
		}
		if r.BaseLaunches != tt.launchesC || r.TargetLaunches != tt.launchesG {
			t.Errorf("%s launches C %v G %v, want %v %v", tt.name, r.BaseLaunches, r.TargetLaunches, tt.launchesC, tt.launchesG)
		}
	}
	if len(bd.Ops) != 3 || bd.Ops[0].Name != "attn" {
		t.Errorf("ops = %+v, want attn, the largest |delta median|, first of 3", bd.Ops)
	}
	if len(bd.Stages) != 2 || bd.Stages[0].Name != "pre" {
		t.Errorf("stages = %+v, want pre then mlp", bd.Stages)
	}

	out.Reset()
	err = runQualifyCommand([]string{"-warmup-until=hc_expand", "-ops", filepath.Join(dir, "ops.txt"), mpath}, &out)
	if err != nil && !errors.As(err, &exit) {
		t.Fatalf("qualify: %v", err)
	}
	for _, want := range []string{
		"breakdown over 6 pairs",
		"do not sum to the kernel total's",
		"1.000  1.350  +0.350  +0.100  +0.600  attn",
	} {
		if !strings.Contains(out.String(), want) {
			t.Errorf("text output lacks %q:\n%s", want, out.String())
		}
	}
	if strings.Contains(out.String(), "stage") {
		t.Errorf("text output has a stage table without -stages:\n%s", out.String())
	}
}

func TestQualifyBreakdownDroppedBlocks(t *testing.T) {
	// attn's deltas are 0.1 .. 0.6 ms, two to a block. B3 G2 is capture 10,
	// whose window is breakdownBase+101 s to +103 s.
	dAttn := []int64{0.1e6, 0.2e6, 0.3e6, 0.4e6, 0.5e6, 0.6e6}
	dFFN := make([]int64, 6)
	mpath := writeBreakdownCampaign(t, dAttn, dFFN, nil)
	dir := filepath.Dir(mpath)
	writeLog := func(name, text string) string {
		path := filepath.Join(dir, name)
		if err := os.WriteFile(path, []byte(text), 0o644); err != nil {
			t.Fatal(err)
		}
		return path
	}
	clean := writeLog("clean.log", "# no violations\n")
	loaded := writeLog("loaded.log", breakdownBase.Add(102*time.Second).Format(time.RFC3339)+" 41822 python3 327.0 0 -\n")

	tests := []struct {
		name      string
		log       string
		minBlocks string
		dropped   []string
		pairs     int
		attnMaxMS float64
		refused   bool
	}{
		{"clean", clean, "2", nil, 6, 0.6, false},
		{"B3 loaded", loaded, "2", []string{"B3"}, 4, 0.4, false},
		{"B3 loaded, every block required", loaded, "0", []string{"B3"}, 0, 0, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var out strings.Builder
			err := runQualifyCommand([]string{"-warmup-until=hc_expand", "-json", "-noise-floor-ms=0.01",
				"-violations", tt.log, "-min-blocks=" + tt.minBlocks,
				"-ops", filepath.Join(dir, "ops.txt"), "-stages", filepath.Join(dir, "stages.txt"), mpath}, &out)
			var exit *exitCodeError
			if tt.refused {
				if err == nil || errors.As(err, &exit) || !strings.Contains(err.Error(), "REFUSED") {
					t.Fatalf("err = %v, want REFUSED", err)
				}
			} else if err != nil && !errors.As(err, &exit) {
				t.Fatalf("qualify: %v", err)
			}
			var q Qualification
			if err := json.Unmarshal([]byte(out.String()), &q); err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(q.DroppedBlocks, tt.dropped) {
				t.Errorf("dropped blocks = %v, want %v", q.DroppedBlocks, tt.dropped)
			}
			if tt.refused {
				if q.Breakdown != nil {
					t.Errorf("REFUSED verdict has a breakdown: %+v", q.Breakdown)
				}
				return
			}
			bd := q.Breakdown
			if bd == nil || bd.Pairs != tt.pairs {
				t.Fatalf("breakdown = %+v, want %d pairs", bd, tt.pairs)
			}
			if bd.Pairs != 2*len(q.Blocks) {
				t.Errorf("breakdown over %d pairs, verdict over %d blocks", bd.Pairs, len(q.Blocks))
			}
			found := false
			for _, r := range bd.Ops {
				if r.Name == "attn" {
					found = true
					if !near(r.DeltaMaxMS, tt.attnMaxMS) {
						t.Errorf("attn delta max = %.3f ms, want %.3f", r.DeltaMaxMS, tt.attnMaxMS)
					}
				}
			}
			if !found {
				t.Errorf("no attn row in %+v", bd.Ops)
			}
		})
	}

	// Identity is checked over every capture, dropped blocks included: B3
	// C2 ran another executable, and B3 is dropped for host load.
	mpath = writeBreakdownCampaign(t, dAttn, dFFN, func(block, arm string) string {
		if block == "B3" && arm == "C2" {
			return "cc"
		}
		return map[byte]string{'C': "aa", 'G': "bb"}[arm[0]]
	})
	var out strings.Builder
	err := runQualifyCommand([]string{"-warmup-until=hc_expand", "-violations", loaded, "-min-blocks=2", mpath}, &out)
	if err == nil || !strings.Contains(err.Error(), "arm C ran two executables") {
		t.Fatalf("err = %v, want arm C ran two executables", err)
	}
}

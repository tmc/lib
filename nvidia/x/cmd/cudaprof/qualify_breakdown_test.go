package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
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

// writeBreakdownCampaign writes a campaign of len(dAttn)/2 blocks in which C
// always runs attn 1 ms and ffn 2 ms and G runs them longer by the pair's
// dAttn and dFFN, in pair order G1-C1, G2-C2 of each block. It returns the
// manifest's path.
func writeBreakdownCampaign(t *testing.T, dAttn, dFFN []int64) string {
	t.Helper()
	dir := t.TempDir()
	var manifest strings.Builder
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
			writeTestCaptureFile(t, filepath.Join(dir, name), testCapture(map[string][]byte{"capture.journal": j}, completeStatus()))
			fmt.Fprintf(&manifest, "%s %s %s\n", block, arm, name)
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

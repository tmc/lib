package okcprofile

import (
	"math"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/pairedtiming"
)

const sample = `kernel                                                           GPU time  share  calls
void cutlass::device_kernel<flash::enable_sm90_or_later<f...       8.1 µs  52.7%      1
(anonymous namespace)::rmsnorm_sigmoid_gate_kernel(__nv_b...       1.2 µs   7.5%      1
_align_kernel                                                      7.1 µs   0.4%      3
... 11 more
total                                                             15.3 µs
GPU busy (kernels overlap: PDL or streams)                        15.0 µs
an overlapping kernel's time includes any wait for the one before it, so the rows add up to more
`

func TestParseReport(t *testing.T) {
	r, err := ParseReport(sample)
	if err != nil {
		t.Fatal(err)
	}
	want := []Row{
		{"void cutlass::device_kernel<flash::enable_sm90_or_later<f", true, 8100, 52.7, 1},
		{"(anonymous namespace)::rmsnorm_sigmoid_gate_kernel(__nv_b", true, 1200, 7.5, 1},
		{"_align_kernel", false, 7100, 0.4, 3},
	}
	if len(r.Rows) != len(want) {
		t.Fatalf("got %d rows, want %d", len(r.Rows), len(want))
	}
	for i := range want {
		if r.Rows[i] != want[i] {
			t.Errorf("row %d = %+v, want %+v", i, r.Rows[i], want[i])
		}
	}
	if r.Total != 15300 || r.Busy != 15000 || !r.Overlap || r.Hidden != 11 {
		t.Errorf("total %v busy %v overlap %v hidden %d", r.Total, r.Busy, r.Overlap, r.Hidden)
	}
	if n, exact := r.Launches(); n != 5 || exact {
		t.Errorf("Launches() = %d, %v; want 5, false", n, exact)
	}
}

func TestParseReportUnits(t *testing.T) {
	tests := []struct {
		line string
		want time.Duration
	}{
		{"k  12 ns  1.0%  1", 12},
		{"k  1.5 us  1.0%  1", 1500},
		{"k  1.5 µs  1.0%  1", 1500},
		{"k  2.25 ms  1.0%  1", 2250000},
		{"k  1.0 s  1.0%  1", time.Second},
	}
	for _, tt := range tests {
		r, err := ParseReport(tt.line + "\ntotal 1 s\n")
		if err != nil {
			t.Errorf("%q: %v", tt.line, err)
			continue
		}
		if got := r.Rows[0].Time; got != tt.want {
			t.Errorf("%q: time %v, want %v", tt.line, got, tt.want)
		}
		if r.Overlap || r.Busy != 0 {
			t.Errorf("%q: overlap %v busy %v without a busy line", tt.line, r.Overlap, r.Busy)
		}
	}
}

func TestParseReportErrors(t *testing.T) {
	tests := []struct {
		name, text, want string
	}{
		{"no total", "k 1 µs 100.0% 1\n", "needs a total"},
		{"no rows", "total 1 µs\n", "needs a total"},
		{"duplicate total", "k 1 µs 100.0% 1\ntotal 1 µs\ntotal 1 µs\n", "line 3: duplicate total"},
		{"duplicate busy", "k 1 µs 100.0% 1\ntotal 1 µs\nGPU busy 1 µs\nGPU busy 1 µs\n", "duplicate GPU busy"},
		{"unknown unit", "k 1 min 100.0% 1\ntotal 1 µs\n", `unrecognized report line "k 1 min 100.0% 1"`},
		{"new line kind", "k 1 µs 100.0% 1\ntotal 1 µs\nidle 3 µs\n", `line 3: unrecognized`},
	}
	for _, tt := range tests {
		_, err := ParseReport(tt.text)
		if err == nil || !strings.Contains(err.Error(), tt.want) {
			t.Errorf("%s: error %v, want %q", tt.name, err, tt.want)
		}
	}
}

func readFixture(t *testing.T) *File {
	t.Helper()
	fd, err := os.Open("testdata/prefill.json")
	if err != nil {
		t.Fatal(err)
	}
	defer fd.Close()
	f, err := Decode(fd)
	if err != nil {
		t.Fatal(err)
	}
	return f
}

// TestJudgeScore recomputes the judge's own score from its samples. The
// judge computed it independently, so agreement checks the block pairing and
// the intercept, not just this package against itself.
func TestJudgeScore(t *testing.T) {
	f := readFixture(t)
	series, err := f.Series()
	if err != nil {
		t.Fatal(err)
	}
	sc, _, err := pairedtiming.ScoreSeries(series, pairedtiming.Geometric)
	if err != nil {
		t.Fatal(err)
	}
	if d := math.Abs(sc.Speedup - f.Summary.Speedup); d > 1e-7 {
		t.Errorf("score %.10f, judge %.10f", sc.Speedup, f.Summary.Speedup)
	}
	if f.Benchmarks[0].Case.Params["q_len"] == nil {
		t.Errorf("case params lost q_len: %v", f.Benchmarks[0].Case.Params)
	}
}

func TestKernels(t *testing.T) {
	f := readFixture(t)
	tab, err := Kernels(f)
	if err != nil {
		t.Fatal(err)
	}
	if tab.Hidden == 0 || !tab.Overlap {
		t.Errorf("hidden %d overlap %v; the fixture has both", tab.Hidden, tab.Overlap)
	}
	var loads []string
	share := map[string]float64{}
	var merged bool
	for _, k := range tab.Kernels {
		if len(loads) == 0 || loads[len(loads)-1] != k.Load {
			loads = append(loads, k.Load)
		}
		share[k.Load] += k.Share
		merged = merged || (k.Merged && strings.Contains(k.Kernel, "norm_residual"))
	}
	if got := strings.Join(loads, ","); got != "turn 1,turn 4,first turn" {
		t.Errorf("loads %s", got)
	}
	for l, s := range share {
		// Rows overlap and some are hidden, so the visible shares neither
		// reach 1 by construction nor stay below it; they must be near it.
		if s < 0.9 || s > 1.1 {
			t.Errorf("load %s: visible shares sum to %.3f", l, s)
		}
	}
	if !merged {
		t.Error("the two norm_residual instantiations truncate to one name and must be marked Merged")
	}
}

func TestKernelsRejectsBadReport(t *testing.T) {
	f := &File{Profiles: []Profile{{Case: Case{Load: "a", Weight: 1}, Report: "nonsense\n"}}}
	if _, err := Kernels(f); err == nil || !strings.Contains(err.Error(), "profile 0:") {
		t.Errorf("err = %v", err)
	}
	f = &File{Profiles: []Profile{{Case: Case{Load: "a"}, Report: "k 1 µs 100.0% 1\ntotal 1 µs\n"}}}
	if _, err := Kernels(f); err == nil || !strings.Contains(err.Error(), "weight") {
		t.Errorf("err = %v", err)
	}
}

func TestKernelsWeighting(t *testing.T) {
	f := &File{Profiles: []Profile{
		{Case: Case{Load: "d", Weight: 3}, Report: "A 10 µs 100.0% 1\ntotal 10 µs\n"},
		{Case: Case{Load: "d", Weight: 1}, Report: "B... 15 µs 75.0% 2\nB... 5 µs 25.0% 1\ntotal 20 µs\n"},
	}}
	tab, err := Kernels(f)
	if err != nil {
		t.Fatal(err)
	}
	want := []Kernel{
		{Load: "d", Kernel: "A", Cases: 1, Calls: 1, Weighted: 7500, Share: 0.6},
		{Load: "d", Kernel: "B", Truncated: true, Cases: 1, Calls: 3, Weighted: 5000, Share: 0.4, Merged: true},
	}
	if len(tab.Kernels) != len(want) {
		t.Fatalf("got %+v", tab.Kernels)
	}
	for i, w := range want {
		g := tab.Kernels[i]
		if math.Abs(g.Share-w.Share) > 1e-12 {
			t.Errorf("%s share %v, want %v", g.Kernel, g.Share, w.Share)
		}
		g.Share = w.Share
		if g != w {
			t.Errorf("row %d = %+v, want %+v", i, g, w)
		}
	}
}

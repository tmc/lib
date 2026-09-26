package main

import (
	"bytes"
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/internal/cudainject/wire"
)

// gemmLine is a cuBLASLt log line for an MxNxK bf16 matmul with the given
// beta and algorithm id.
func gemmLine(m, n, k int, beta string, algo int) string {
	return strings.NewReplacer("$M", itoa(m), "$N", itoa(n), "$K", itoa(k), "$BETA", beta, "$ALGO", itoa(algo)).Replace(
		"A=0X0 Adesc=[type=R_16BF rows=$K cols=$M ld=$K] B=0X0 Bdesc=[type=R_16BF rows=$K cols=$N ld=$K] " +
			"C=0X0 Cdesc=[type=R_16BF rows=$M cols=$N ld=$M] D=0X0 Ddesc=[type=R_16BF rows=$M cols=$N ld=$M] " +
			"computeDesc=[computeType=COMPUTE_32F scaleType=R_32F transa=OP_T] " +
			"algo=[algoId=$ALGO tile=MATMUL_TILE_64x64 stages=MATMUL_STAGES_64x1] beta=$BETA stream=0X0")
}

func itoa(n int) string { b, _ := json.Marshal(n); return string(b) }

// gemmCall is one synthetic cuBLASLt call and the kernels it launched.
type gemmCall struct {
	m, n, k int
	beta    string
	algo    int
	kernels []rec
}

// gemmCapture writes a capture of calls, each call's kernels launched by
// that call, and returns its path.
func gemmCapture(t *testing.T, calls []gemmCall) string {
	t.Helper()
	b := newJournalBuilder()
	corr := uint32(0)
	for i, c := range calls {
		id := uint64(i + 1)
		b.append(wire.KindGEMM, map[string]any{"kind": "gemm", "id": id, "thread": 1, "api": "cublasLtMatmul", "log": gemmLine(c.m, c.n, c.k, c.beta, c.algo)})
		for _, k := range c.kernels {
			corr++
			b.append(wire.KindGEMM, map[string]any{"kind": "gemm_launch", "id": id, "correlation": corr})
			b.append(wire.KindActivity, map[string]any{
				"kind": "activity", "subtype": k.subtype, "name": k.name, "valid": true,
				"start": k.start, "end": k.end, "correlation": corr,
			})
		}
	}
	// A kernel no call launched, so the window is not only GEMMs.
	b.append(wire.KindActivity, map[string]any{"kind": "activity", "subtype": "kernel", "name": "elementwise", "valid": true, "start": 5000, "end": 5100, "correlation": 999})
	path := filepath.Join(t.TempDir(), "gemms.pprof")
	writeTestCaptureFile(t, path, testCapture(map[string][]byte{"capture.journal": b.finalize()}, completeStatus()))
	return path
}

func TestGEMMCensus(t *testing.T) {
	path := gemmCapture(t, []gemmCall{
		{768, 512, 1024, "0", 21, []rec{{subtype: "kernel", name: "nvjet_a", start: 100, end: 400}}},
		{768, 512, 1024, "0", 21, []rec{{subtype: "kernel", name: "nvjet_a", start: 500, end: 700}}},
		// Same shape, accumulating: its own row.
		{768, 512, 1024, "1", 21, []rec{{subtype: "kernel", name: "nvjet_a", start: 800, end: 900}}},
		// Split-K: two kernels, one call.
		{64, 64, 4096, "0", 6, []rec{
			{subtype: "kernel", name: "gemm_splitk", start: 1000, end: 1600},
			{subtype: "kernel", name: "splitk_reduce", start: 1600, end: 1700},
		}},
		// A call whose launch left no kernel record.
		{32, 32, 32, "0", 1, nil},
	})
	c, err := readGEMMCensus(path, compareOptions{phase: -1})
	if err != nil {
		t.Fatal(err)
	}
	if c.Calls != 4 || c.Joined != 4 || c.Unparsed != 0 {
		// The kernel-less call was logged at time 0, before the window.
		t.Errorf("calls, joined, unparsed = %d, %d, %d; want 4, 4, 0", c.Calls, c.Joined, c.Unparsed)
	}
	type row struct {
		shape, beta string
		calls       int
		nanos, per  int64
	}
	var got []row
	for i := range c.Rows {
		r := &c.Rows[i]
		got = append(got, row{r.Shape(), r.Beta, r.Calls, r.KernelNanos, r.PerCall()})
	}
	want := []row{
		{"64x64x4096", "0", 1, 700, 700},
		{"768x512x1024", "0", 2, 500, 250},
		{"768x512x1024", "1", 1, 100, 100},
	}
	if len(got) != len(want) {
		t.Fatalf("rows = %+v, want %+v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("row %d = %+v, want %+v", i, got[i], want[i])
		}
	}
	if k := c.Rows[0].Kernel(); k != "gemm_splitk (+1 more)" {
		t.Errorf("split-K row kernel = %q", k)
	}
	var out bytes.Buffer
	if err := c.write(&out); err != nil {
		t.Fatal(err)
	}
	for _, s := range []string{"4 in window, 4 joined", "768x512x1024", "TN", "R_16BF,R_16BF->R_16BF", "6 64x64/64x1", "nvjet_a"} {
		if !strings.Contains(out.String(), s) {
			t.Errorf("output lacks %q:\n%s", s, out.String())
		}
	}
}

func TestGEMMCensusRefusesWithoutGEMMs(t *testing.T) {
	path := filepath.Join(t.TempDir(), "plain.pprof")
	writeTestCaptureFile(t, path, compareCapture([]rec{{subtype: "kernel", name: "k", start: 1, end: 2}}, completeStatus()))
	_, err := readGEMMCensus(path, compareOptions{phase: -1})
	if err == nil || !strings.Contains(err.Error(), "capture with --gemms") {
		t.Errorf("error = %v, want a pointer to --gemms", err)
	}
}

func TestCompareGEMMs(t *testing.T) {
	base := gemmCapture(t, []gemmCall{
		{768, 512, 1024, "0", 21, []rec{{subtype: "kernel", name: "nvjet_a", start: 100, end: 400}}},
		{64, 64, 64, "0", 1, []rec{{subtype: "kernel", name: "small", start: 500, end: 510}}},
	})
	target := gemmCapture(t, []gemmCall{
		{768, 512, 1024, "1", 7, []rec{{subtype: "kernel", name: "cutlass_b", start: 100, end: 200}}},
		{768, 512, 1024, "1", 7, []rec{{subtype: "kernel", name: "cutlass_b", start: 300, end: 400}}},
		{128, 64, 64, "0", 1, []rec{{subtype: "kernel", name: "other", start: 500, end: 520}}},
	})
	var out bytes.Buffer
	if err := runCompareCommand([]string{"-gemms", base, target}, &out); err != nil {
		t.Fatal(err)
	}
	s := out.String()
	// base: 1 call of 300ns; target: 2 calls of 100ns each.
	row := "768x512x1024 TN R_16BF,R_16BF->R_16BF 1 2 0.3 0.1 -0.2 0.000 0.000 -0.000"
	found := false
	for _, line := range strings.Split(s, "\n") {
		found = found || strings.Join(strings.Fields(line), " ") == row
	}
	if !found {
		t.Errorf("output lacks row %q:\n%s", row, s)
	}
	for _, want := range []string{
		"base    1x beta=0 DEFAULT algo 21 64x64/64x1: nvjet_a",
		"target  2x beta=1 DEFAULT algo 7 64x64/64x1: cutlass_b",
		"64x64x64",
		"128x64x64",
	} {
		if !strings.Contains(s, want) {
			t.Errorf("output lacks %q:\n%s", want, s)
		}
	}

	out.Reset()
	if err := runCompareCommand([]string{"-gemms", "-json", base, target}, &out); err != nil {
		t.Fatal(err)
	}
	var c GEMMComparison
	if err := json.Unmarshal(out.Bytes(), &c); err != nil {
		t.Fatal(err)
	}
	if len(c.Shapes) != 3 || c.Shapes[0].Shape != "768x512x1024" || c.Shapes[0].Base.Calls != 1 || c.Shapes[0].Target.Calls != 2 {
		t.Errorf("JSON shapes = %+v", c.Shapes)
	}
}
